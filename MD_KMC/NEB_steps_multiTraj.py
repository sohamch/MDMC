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
    MainPath = args[7]
else:
    MainPath = "/home/sohamc2/HEA_FCC/MDMC/"

InitPath = MainPath+"MD_KMC/" 
# Need to get rid of these argument
NImage = 3
ProcPerImage = 1
RunPath = os.getcwd()+'/'
print("Running from : " + RunPath)

__test__ = False

with open(MainPath + "lammpsBox.txt", "r") as fl:
    Initlines = fl.readlines()

# Load the lammps cartesian positions and neighborhoods - pre-prepared
SiteIndToPos = np.load(MainPath + "SiteIndToLmpCartPos.npy")  # lammps pos of sites
SiteIndToNgb = np.load(MainPath + "siteIndtoNgbSiteInd.npy")  # Nsites x z array of site neighbors

with open(MainPath + "CrysDat/jnetFCC.pkl", "rb") as fl:
    jnetFCC = pickle.load(fl)
dxList = np.array([dx*3.59 for (i, j), dx in jnetFCC[0]])

# load the data
try:
    SiteIndToSpec = np.load(RunPath + "StatesEnd_{}_{}.npy".format(SampleStart, stepsLast))
    vacSiteInd = np.load(RunPath + "vacSiteIndEnd_{}_{}.npy".format(SampleStart, stepsLast))
    print("Starting from checkpointed step {}".format(stepsLast))
except:
    print("checkpoint not found or last step zero indicated. Starting from step zero.")
    allStates = np.load(InitPath + "states_{}_0.npy".format(T))
    perm = np.arange(allStates.shape[0])
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

if storeRates:
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
    
    if (step+1)%10 == 0:
        with open("timing_{}.txt".format(SampleStart), "a") as fl:
            fl.write("Time for {} steps : {:.4f} seconds\n".format(step+1, time.time()-start))

    if storeRates:
        ratesTest[:, step, :] = rates[:, :]
        barriersTest[:, step, :] = barriers[:, :]
        randNumsTest[:, step] = rndNums[:]

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
print()
