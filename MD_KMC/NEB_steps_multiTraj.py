#!/usr/bin/env python
# coding: utf-8

# In[5]:
import numpy as np
import pickle
import h5py
import subprocess
import sys
import time
from numba import jit, float64, int64
from ase.io.lammpsdata import write_lammps_data, read_lammps_data
from scipy.constants import physical_constants

kB = physical_constants["Boltzmann constant in eV/K"][0]

args = list(sys.argv)
T = float(args[1])
Nsteps = int(args[2])
NImage = int(args[3])
TestCode = bool(int(args[4]))
ProcPerImage = 1

__test__ = TestCode

with open("lammpsBox.txt", "r") as fl:
    Initlines = fl.readlines()

# Load the lammps cartesian positions and neighborhoods - pre-prepared
SiteIndToPos = np.load("SiteIndToLmpCartPos.npy")  # lammps pos of sites
SiteIndToNgb = np.load("siteIndtoNgbSiteInd.npy")  # Nsites x z array of site neighbors

def write_input_files(Ntr):
    for traj in range(Ntr):
        with open("in.neb_{0}".format(traj), "w") as fl:
            fl.write("units \t metal\n")
            fl.write("atom_style \t atomic\n")
            fl.write("atom_modify \t map array\n")
            fl.write("boundary \t p p p\n")
            fl.write("atom_modify \t sort 0 0.0\n")
            fl.write("read_data \t initial_{0}.data\n".format(traj))
            fl.write("pair_style \t meam\n")
            fl.write("pair_coeff \t * * pot/library.meam Co Ni Cr Fe Mn pot/params.meam Co Ni Cr Fe Mn\n")
            fl.write("fix \t 1 all neb 1.0\n")
            fl.write("timestep \t 0.01\n")
            fl.write("min_style \t quickmin\n")
            fl.write("neb \t 1e-5 0.0 500 500 10 final final_{0}.data".format(traj))

def write_init_states(SiteIndToSpec, vacSiteInd, TopLines):
    Ntr = vacSiteInd.shape[0]
    for traj in range(Ntr):
        with open("initial_{}.data".format(traj), "w") as fl:
            fl.writelines(TopLines[:12])
            counter = 1
            for idx in range(SiteIndToSpec.shape[1]):
                spec = SiteIndToSpec[traj, idx]
                if spec == -1:
                    assert idx == vacSiteInd[traj], "{} {}".format(idx, SiteIndToSpec[traj, idx])
                    continue
                pos = SiteIndToPos[idx]
                fl.write("{} {} {} {} {}\n".format(counter, spec, pos[0], pos[1], pos[2]))
                counter += 1

def write_final_states(SiteIndToPos, vacSiteInd, siteIndToNgb, jInd):
    Ntr = vacSiteInd.shape[0]
    for traj in range(Ntr):
        with open("final_{}.data".format(traj), "w") as fl:
            fl.write("{}\n".format(SiteIndToPos.shape[0] - 1))
            counter = 1
            for siteInd in range(len(SiteIndToPos)):
                if siteInd == vacSiteInd[traj]:
                    continue
                # the jumping atom will have vac site as the new position
                if siteInd == siteIndToNgb[vacSiteInd[traj], jInd]:
                    pos = SiteIndToPos[vacSiteInd[traj]]
                else:
                    pos = SiteIndToPos[siteInd]
                fl.write("{} {} {} {}\n".format(counter, pos[0], pos[1], pos[2]))
                counter += 1

# @jit(nopython=True)
def getJumpSelects(rates):
    Ntr = rates.shape[0]
    timeStep = 1./np.sum(rates, axis=1)
    ratesProb = rates*timeStep.reshape(Ntr, 1)
    ratesProbSum = np.cumsum(ratesProb, axis=1)
    rn = np.random.rand(Ntr)
    jumpID = np.zeros(Ntr, dtype=int)
    for tr in range(Ntr):
        jSelect = np.searchsorted(ratesProbSum[tr, :], rn[tr])
        jumpID[tr] = jSelect
    # jumpID, rateProbs, ratesCum, rndNums, time_step
    return jumpID, ratesProb, ratesProbSum, rn, timeStep

# @jit(nopython=True)
def updateStates(SiteIndToNgb, Nspec,  SiteIndToSpec, vacSiteInd, jumpID, dxList):
    Ntraj = jumpID.shape[0]
    jumpAtomSelectArray = np.zeros(Ntraj, dtype=int)
    X = np.zeros((Ntraj, Nspec, 3), dtype=float)
    for tr in range(Ntraj):
        jumpSiteSelect = SiteIndToNgb[vacSiteInd[tr], jumpID[tr]]
        jumpAtomSelect = SiteIndToSpec[tr, jumpSiteSelect]
        jumpAtomSelectArray[tr] = jumpAtomSelect
        SiteIndToSpec[tr, vacSiteInd[tr]] = jumpAtomSelect
        SiteIndToSpec[tr, jumpSiteSelect] = -1  # The next vacancy site
        vacSiteInd[tr] = jumpSiteSelect
        X[tr, 0, :] = dxList[jumpID[tr]]
        X[tr, jumpAtomSelect, :] = -dxList[jumpID[tr]]
        
    return jumpAtomSelectArray, X

with open("CrysDat/jnetFCC.pkl", "rb") as fl:
    jnetFCC = pickle.load(fl)
dxList = np.array([dx*3.59 for (i, j), dx in jnetFCC[0]])


# Load the starting data for the trajectories
SiteIndToSpec = np.load("SiteIndToSpec.npy") # Ntraj x Nsites array of occupancies
vacSiteInd = np.load("vacSiteInd.npy") # Ntraj size array: contains where the vac is in each traj.
specs, counts = np.unique(SiteIndToSpec[0], return_counts=True)
Nspec = len(specs)  # including the vacancy
Ntraj = SiteIndToSpec.shape[0]

Nsites = SiteIndToSpec.shape[1]

Initlines[2] = "{} \t atoms\n".format(Nsites - 1)
Initlines[3] = "{} atom types\n".format(Nspec-1)

try:
    X_steps = np.load("X_steps.npy")
    t_steps = np.load("X_steps.npy")
    stepsLast = np.load("steps_last.npy")[0]
except FileNotFoundError:
    X_steps = np.zeros((Ntraj, Nspec, Nsteps + 1, 3)) # 0th position will store vacancy jumps
    t_steps = np.zeros((Ntraj, Nsteps + 1))
    stepsLast = 0
    
stepCount = np.zeros(1, dtype=int)
# Before starting, write the lammps input files
write_input_files(Ntraj)
print("Running {0} steps at {1} K on {2} trajectories".format(Nsteps, T, Ntraj))
print("Previously Done : {} steps".format(stepsLast))
print("Testing : {}".format(__test__))
start = time.time()
if __test__:
    rates_steps = np.zeros((Nsteps, Ntraj, SiteIndToNgb.shape[1]))
    barrier_steps = np.zeros((Nsteps, Ntraj, SiteIndToNgb.shape[1]))
    rateProb_steps = np.zeros((Nsteps, Ntraj, SiteIndToNgb.shape[1]))
    rateCsum_steps = np.zeros((Nsteps, Ntraj, SiteIndToNgb.shape[1]))
    randNums_steps = np.zeros((Nsteps, Ntraj))

for step in range(Nsteps - stepsLast):
    # Write the initial states from last accepted state
    write_init_states(SiteIndToSpec, vacSiteInd, Initlines)

    rates = np.zeros((Ntraj, SiteIndToNgb.shape[1]))
    barriers = np.zeros((Ntraj, SiteIndToNgb.shape[1]))
    for jumpInd in range(SiteIndToNgb.shape[1]):
        # Write the final states in NEB format for lammps
        write_final_states(SiteIndToPos, vacSiteInd, SiteIndToNgb, jumpInd)
        if __test__:
            # Store the final data for each traj, at each step and for each jump
            for traj in range(Ntraj):
                cmd = subprocess.Popen("cp final_{0}.data final_{0}_{1}_{2}.data".format(traj, step, jumpInd), shell=True)
                rt = cmd.wait()
                assert rt == 0

        # Then run lammps
        commands = [
            "mpirun -np {0} --oversubscribe $LMPPATH/lmp -p {0}x1 -in in.neb_{1} > out_{1}.txt".format(NImage, traj)
            for traj in range(Ntraj)
        ]
        cmdList = [subprocess.Popen(cmd, shell=True) for cmd in commands]
        
        # wait for the lammps commands to complete
        for c in cmdList:
            rt_code = c.wait()
            assert rt_code == 0  # check for system errors
        
        # Then read the forward barrier -> ebf
        for traj in range(Ntraj):
            with open("out_{0}.txt".format(traj), "r") as fl:
                for line in fl:
                    continue
            ebfLine = line.split()
            ebf = float(ebfLine[6])
            rates[traj, jumpInd] = np.exp(-ebf/(kB*T))
            barriers[traj, jumpInd] = ebf

    # Then do selection
    jumpID, rateProbs, ratesCsum, rndNums, time_step = getJumpSelects(rates)
    
    # Then do the final exchange
    jumpAtomSelectArray, X_traj = updateStates(SiteIndToNgb, Nspec, SiteIndToSpec, vacSiteInd, jumpID, dxList)
    # def updateStates(SiteIndToNgb, Nspec,  SiteIndToSpec, vacSiteInd, jumpID, dxList):
    # Note the displacements and the time
    X_steps[:, :, step + stepsLast + 1, :] = X_traj[:, :, :]
    t_steps[:, step + stepsLast + 1] = time_step
    stepCount[0] = step + stepsLast + 1
    
    # save arrays for next step
    if not __test__:
        np.save("SiteIndToSpec.npy", SiteIndToSpec)
        np.save("vacSiteInd.npy", vacSiteInd)
        np.save("X_steps.npy", X_steps)
        np.save("t_steps.npy", t_steps)
        np.save("steps_last.npy", stepCount)
    else:
        print(rates)
        # Store the initial state for each traj, at each step
        for traj in range(Ntraj):
            cmd = subprocess.Popen("cp initial_{0}.data initial_{0}_{1}.data".format(traj, step), shell=True)
            rt = cmd.wait()
            assert rt == 0
        barrier_steps[step, :, :] = barriers[:, :]
        rates_steps[step, :, :] = rates[:, :]
        rateProb_steps[step, : , :] = rateProbs[:, :]
        rateCsum_steps[step, :, :] = ratesCsum[:, :]
        randNums_steps[step, :] = rndNums[:]
        np.save("SiteIndToSpec_{}.npy".format(step + stepsLast + 1), SiteIndToSpec)
        np.save("vacSiteInd_{}.npy".format(step + stepsLast + 1), vacSiteInd)
        np.save("JumpSelects_{}.npy".format(step + stepsLast + 1), jumpAtomSelectArray)
        np.save("X_steps.npy", X_steps)
        np.save("t_steps.npy", t_steps)
        np.save("steps_last.npy", stepCount)

end = time.time()
if __test__:
    np.save("Rates_steps_test.npy", rates_steps)
    np.save("Barriers_steps_test.npy", barrier_steps)
    np.save("rateProb_steps.npy", rateProb_steps)
    np.save("rateCumulSum_steps.npy", rateCsum_steps)
    np.save("randNums_test.npy", randNums_steps)

print("Time Per Step: {:.4f} seconds".format(end - start))
