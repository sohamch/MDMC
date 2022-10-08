#!/usr/bin/env python
# coding: utf-8

# In[5]:
import numpy as np
import pickle
import h5py
import subprocess
import sys
import time
import collections
import pickle
import h5py
from numba import jit, float64, int64
from ase.io.lammpsdata import write_lammps_data, read_lammps_data
from scipy.constants import physical_constants
from KMC_funcs import  *
import os

args = list(sys.argv)
T = int(args[1])
stepsLast = int(args[2])
Nsteps = int(args[3])
SampleStart = int(args[4])
batchSize = int(args[5])
storeRates = bool(int(args[6])) # store the rates? 0 if False.
if len(args) > 7:
    MainPath = args[7] # The path where the potentail file is found
else:
    MainPath = "/home/sohamc2/HEA_FCC/MDMC/"

InitPath, _ = os.path.split(os.path.realpath(__file__))[0] # the directory where the main script is
InitPath += "/"

# Need to get rid of these argument
NImage = 3
ProcPerImage = 1
RunPath = os.getcwd()+'/'
print("Running from : " + RunPath)

with open(MainPath + "lammpsBox.txt", "r") as fl:
    Initlines = fl.readlines()

# Load the lammps cartesian positions and neighborhoods - pre-prepared
SiteIndToPos = np.load(MainPath + "SiteIndToLmpCartPos.npy")  # lammps pos of sites
SiteIndToNgb = np.load(MainPath + "CrysDat_FCC/NNsites_sitewise.npy")[1:, :].T  # Nsites x z array of site neighbors

with open(MainPath + "CrysDat_FCC/jnetFCC.pkl", "rb") as fl:
    jnetFCC = pickle.load(fl)
dxList = np.array([dx*3.59 for (i, j), dx in jnetFCC[0]])

assert SiteIndToNgb.shape[0] = SiteIndToPos.shape[0]
assert SiteIndToNgb.shape[1] = dxList.shape[0]

# load the data
try:
    SiteIndToSpec = np.load(RunPath + "chkpt/StatesEnd_{}_{}.npy".format(SampleStart, stepsLast))
    vacSiteInd = np.load(RunPath + "chkpt/vacSiteIndEnd_{}_{}.npy".format(SampleStart, stepsLast))
    print("Starting from checkpointed step {}".format(stepsLast))
except:
    print("checkpoint not found or last step zero indicated. Starting from step zero.")
    allStates = np.load(InitPath + "states_{}_0.npy".format(T))
    try:
        perm = np.load("perm_{}.npy".format(T))
    except:
        perm = np.random.permuation(allStates.shape[0])
        np.save("perm_{}.npy".format(T), perm)

    # Load the starting data for the trajectories
    SiteIndToSpec = allStates[perm][SampleStart: SampleStart + batchSize]
    vacSiteInd = np.zeros(SiteIndToSpec.shape[0], dtype = int)
    assert np.all(SiteIndToSpec[:, 0] == 0)

specs, counts = np.unique(SiteIndToSpec[0], return_counts=True)
Nspec = len(specs)  # including the vacancy
Ntraj = SiteIndToSpec.shape[0]

Nsites = SiteIndToSpec.shape[1]

Initlines[2] = "{} \t atoms\n".format(Nsites - 1)
Initlines[3] = "{} atom types\n".format(Nspec-1)

X_steps = np.zeros((Ntraj, Nspec, Nsteps, 3)) # 0th position will store vacancy jumps
t_steps = np.zeros((Ntraj, Nsteps))
JumpSelection = np.zeros((Ntraj, Nsteps), dtype=np.int8)

ratesTest = np.zeros((Ntraj, Nsteps, SiteIndToNgb.shape[1]))
barriersTest = np.zeros((Ntraj, Nsteps, SiteIndToNgb.shape[1]))
randNumsTest = np.zeros((Ntraj, Nsteps))

kB = physical_constants["Boltzmann constant in eV/K"][0]
# Before starting, write the lammps input files
write_input_files(Ntraj, potPath=MainPath)

start = time.time()
NEB_count = 0
for step in range(Nsteps):
    # Write the initial states from last accepted state
    write_init_states(SiteIndToSpec, SiteIndToPos, vacSiteInd, Initlines)

    rates = np.zeros((Ntraj, SiteIndToNgb.shape[1]))
    barriers = np.zeros((Ntraj, SiteIndToNgb.shape[1]))
    for jumpInd in range(SiteIndToNgb.shape[1]):
        # Write the final states in NEB format for lammps
        write_final_states(SiteIndToPos, vacSiteInd, SiteIndToNgb, jumpInd)

        # Then run lammps
        commands = [
            "mpirun -np {0} $LMPPATH/lmp -p {0}x1 -in in.neb_{1} > out_{1}.txt".format(NImage, traj)
            for traj in range(Ntraj)
        ]
        cmdList = [subprocess.Popen(cmd, shell=True) for cmd in commands]
        # wait for the lammps commands to complete
        for c in cmdList:
            rt_code = c.wait()
            assert rt_code == 0  # check for system errors
        NEB_count += Ntraj
        
        # Then read the forward barrier -> ebf
        for traj in range(Ntraj):
            with open("out_{0}.txt".format(traj), "r") as fl:
                for line in fl:
                    continue
            ebfLine = line.split()
            ebf = float(ebfLine[6])
            rates[traj, jumpInd] = np.exp(-ebf/(kB*T))
            barriers[traj, jumpInd] = ebf

            # get the jumping species and store the barrier for later use
            vInd = vacSiteInd[traj]
            vacNgb = SiteIndToNgb[vInd, jumpInd]
            jAtom = SiteIndToSpec[traj, vacNgb]

    # Then do selection
    jumpID, rateProbs, ratesCsum, rndNums, time_step = getJumpSelects(rates)
    # store the selected jump
    JumpSelection[:, step] = jumpID[:]
    # Then do the final exchange
    jumpAtomSelectArray, X_traj = updateStates(SiteIndToNgb, Nspec, SiteIndToSpec, vacSiteInd, jumpID, dxList)
    # updateStates args : (SiteIndToNgb, Nspec,  SiteIndToSpec, vacSiteInd, jumpID, dxList)
    # Note the displacements and the time
    X_steps[:, :, step, :] = X_traj[:, :, :]
    t_steps[:, step] = time_step
    
    ratesTest[:, step, :] = rates[:, :]
    barriersTest[:, step, :] = barriers[:, :]
    randNumsTest[:, step] = rndNums[:]

    if (step+1)%10 == 0:
        with open("timing_{}.txt".format(SampleStart), "a") as fl:
            fl.write("Time for {} steps : {:.4f} seconds\n".format(step+1, time.time()-start))

    if step % 10 == 0:
        np.save(RunPath + "chkpt/StatesEnd_{}_{}.npy".format(SampleStart, stepsLast + step), SiteIndToSpec)
        np.save(RunPath + "chkpt/vacSiteIndEnd_{}_{}.npy".format(SampleStart, stepsLast + step), vacSiteInd)
        np.save(RunPath + "chkpt/Xsteps_{}_{}.npy".format(SampleStart, stepsLast + step), X_steps[:, :, :step, :])
        np.save(RunPath + "chkpt/tsteps_{}_{}.npy".format(SampleStart, stepsLast + step), t_steps[:, :step])
        np.save(RunPath + "chkpt/JumpSelects_{}_{}.npy".format(SampleStart, stepsLast + step), JumpSelection[:, :step])
        if step == 10:
            np.save(RunPath + "chkpt/ratesTest_{}_{}.npy".format(SampleStart, stepsLast + step), ratesTest[:, :step, :])
            np.save(RunPath + "chkpt/barriersTest_{}_{}.npy".format(SampleStart, stepsLast + step), barriersTest[:, :step, :])
            np.save(RunPath + "chkpt/randNumsTest_{}_{}.npy".format(SampleStart, stepsLast + step), randNumsTest[:, :step])


end = time.time()
print("time per step : {:.4f} seconds".format((end-start)/Nsteps))
print("time per NEB calculation : {:.4f} seconds".format((end-start)/NEB_count))

# save the end results.
np.save(RunPath + "StatesEnd_{}_{}.npy".format(SampleStart, stepsLast+Nsteps), SiteIndToSpec)
np.save(RunPath + "vacSiteIndEnd_{}_{}.npy".format(SampleStart, stepsLast+Nsteps), vacSiteInd)
np.save(RunPath + "Xsteps_{}_{}.npy".format(SampleStart, stepsLast+Nsteps), X_steps)
np.save(RunPath + "tsteps_{}_{}.npy".format(SampleStart, stepsLast+Nsteps), t_steps)
np.save(RunPath + "JumpSelects_{}_{}.npy".format(SampleStart, stepsLast+Nsteps), JumpSelection)
if storeRates:
    np.save(RunPath + "ratesTest_{}_{}.npy".format(SampleStart, stepsLast+Nsteps), ratesTest)
    np.save(RunPath + "randNumsTest_{}_{}.npy".format(SampleStart, stepsLast+Nsteps), randNumsTest)
    np.save(RunPath + "barriersTest_{}_{}.npy".format(SampleStart, stepsLast + Nsteps), barriersTest)
print("All done")
