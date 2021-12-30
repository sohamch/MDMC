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
Nseg = int(args[3])
NProc = int(args[4])
NImage = int(args[5])
ProcPerImage = 1

if NProc%(NImage*ProcPerImage) != 0:
    raise ValueError("Processors cannot be divided integrally across trajectories")

# Processors per trajectory = NImage*ProcPerImage
Ntraj = NProc//(NImage*ProcPerImage)
    
print("Running {0} steps at {1}K".format(Nsteps, T))
print("Segments at every {}th step".format(Nseg))

with open("lammpsBox.txt","r") as fl:
    Initlines = fl.readlines()

def write_input_files():
    pass

def write_init_states(SiteIndToSpec, vacSiteInd):
    Ntraj = vacSiteInd.shape[0]
    for traj in range(Ntraj):
        with open("initial_{}.data".format(traj), "w") as fl:
            fl.writelines(Initlines[:12])
            counter = 1
            for idx in range(SiteIndToSpec.shape[1]):
                spec = SiteIndToSpec[traj, idx]
                if spec == -1:
                    assert idx == vacSiteInd[traj]
                    continue
                pos = SiteIndToPos[idx]
                fl.write("{} {} {} {} {}\n".format(counter, spec, pos[0], pos[1], pos[2]))
                counter += 1

def write_final_states(SiteIndToSpec, SiteIndToPos, vacSiteInd, jumpSiteInd):
    Ntraj = vacSiteInd.shape[0]
    for traj in range(Ntraj):
        with open("final_{}.data".format(traj), "w") as fl:
            fl.write("{}\n".format(SiteIndToPos.shape[1] - 1))
            counter = 1
            for siteInd in range(len(SiteIndToPos)):
                if siteInd == vacSiteInd[traj]:
                    continue
                if siteInd == jumpSiteInd:  # the jumping atom will have vac site as the new position
                    pos = SiteIndToPos[vacSiteInd[traj]]
                else:
                    pos = SiteIndToPos[siteInd]
                fl.write("{} {} {} {}\n".format(counter, pos[0], pos[1], pos[2]))
                counter += 1

@jit(nopython=True)
def getJumpSelects(rates):
    Ntraj = rates.shape[0]
    timeStep = 1./np.sum(rates, axis=1)
    ratesProb = rates*timeStep.reshape(Ntraj, 1)
    ratesProbSum = np.cumsum(ratesProb, axis=1)
    rn = np.random.rand(Ntraj)
    jumpID = np.zeros(Ntraj, dtype=int)
    for tr in range(Ntraj):
        jSelect = np.searchsorted(ratesProbSum[tr, :], rn[tr])
        jumpID[tr] = jSelect
    return jumpID, timeStep

@jit(nopython=True)
def updateStates(SiteIndToNgb, Nspec, SiteIndToSpec, vacSiteInd, jumpID, dxList):
    Ntraj = jumpID.shape[0]
    jumpAtomSelectArray = np.zeros(Ntraj, dtype=int64)
    X = np.zeros((Ntraj, Nspec, 3))
    for tr in range(Ntraj):
        jumpSiteSelect = SiteIndToNgb[vacSiteInd[tr], jumpID[tr]]
        jumpAtomSelect = SiteIndToSpec[tr, jumpSiteSelect]
        jumpAtomSelectArray[tr] = jumpAtomSelect
        SiteIndToSpec[tr, vacSiteInd] = jumpAtomSelect
        SiteIndToSpec[tr, jumpSiteSelect] = -1 # The next vacancy site
        vacSiteInd[tr] = jumpSiteSelect
        X_traj[tr, 0, :] = dxList[jumpID[tr]]
        X_traj[tr, jumpAtomSelect, :] = -dxList[jumpID[tr]]
        
    return jumpAtomSelectArray, X_traj

with open("CrysDat/jnetFCC.pkl", "rb") as fl:
    jnetFCC = pickle.load(fl)
dxList = np.array([dx*3.59 for (i, j), dx in jnetFCC[0]])
print(dxList)

cmd = [
    "mpirun -np {0} $LMPPATH/lmp -p {1}x{2} -in in.neb > out.txt".format(NProc, NImage, ProcPerImage)
]

start = time.time()
neb_Time = 0.
neb_count = 0

print("Running {} trajectories on {} processors".format(Ntraj, NProc))

# Load the starting data for the trajectories
SiteIndToSpec = np.load("SiteIndToSpec.npy") # Ntraj x Nsites array of occupancies
SiteIndToPos = np.load("SiteIndToLmpCartPos.npy") # lammps pos of sites
SiteIndToNgb = np.load("siteIndtoNgbSiteInd.npy") # Nsites x z array of site neighbors
vacSiteInd = np.load("vacSiteInd.npy") # Ntraj size array: contains where the vac is in each traj.
specs, counts = np.unique(SiteIndToSpec[0], return_counts=True)
Nspec = len(specs)

try:
    X_steps = np.load("X_steps.npy")
    t_steps = np.load("X_steps.npy")
    stepsLast = np.load("steps_last.npy")[0]
except:
    X_steps = np.zeros((Ntraj, Nspec, Nsteps + 1, 3)) # 0th position will store vacancy jumps
    t_steps = np.zeros(Ntraj, Nsteps + 1)
    stepsLast = 0
    
stepCount = np.zeros(1, dtype=int)

for step in range(Nsteps - stepsLast):
    # Write the initial states from last accepted state
    write_cmd = subprocess.Popen(
            "mpirun -np {0} python write_init_states.py".format(Ntraj),
            shell=True
        )
    write_cmd.wait()
    
    rates = np.zeros(Ntraj, SiteIndToNgb.shape[1])
    for jumpInd in enumerate(SiteIndToNgb.shape[1]):
        # Write the final states in NEB format for lammps
        write_cmd = subprocess.Popen(
            ["mpirun -np {0} python write_final_state.py {1}".format(Ntraj, jumpInd)],
            shell=True
        )
        write_cmd.wait()
        
        # Then run lammps
        commands = [
            "mpirun -np {0} $LMPPATH/lmp -p {0}x1 -in in.neb_{1} > out_{1}.txt".format(NImage, traj)
            for traj in range(Ntraj)
        ]
        cmdList = [subprocess.Popen(cmd, shell=True)]
        
        # wait for the lammps commands to complete
        for c in cmdList:
            rt_code = c.wait()
            assert rt_code == 0 # check for system errors
        
        # Then read the forward barrier -> ebf
        for traj in range(Ntraj):
            with open("out_{0}.txt".format(traj), "r") as fl:
                for line in fl:
                    continue
            ebfLine = line.split()
            ebf = float(ebfLine[6])
            rates[traj, jumpInd] = np.exp(-ebf/(kB*T))
    
    # Then do selection 
    jumpID, time_step = getJumpSelects(rates)
    
    # Then do the final exchange
    jumpAtomSelectArray, X_traj = updateStates(SiteIndToNgb, SiteIndToSpec, vacSiteInd, jumpID, dxList)
        
    # Note the displacements and the time
    X_steps[:, :, step + stepsLast + 1, :] = X_traj[:, :, :]
    t_steps[step + stepsLast + 1] = time_step
    stepCount[0] = step + stepsLast
    
    # save arrays for next step
    np.save("SiteIndToSpec.npy", SiteIndToSpec)
    np.save("vacSiteInd.npy", vacSiteInd)
    np.save("X_steps.npy", X_steps)
    np.save("t_steps.npy", t_steps)
    np.save("steps_last.npy", stepCount)