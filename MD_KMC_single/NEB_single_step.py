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
from numba import jit, float64, int64


from scipy.constants import physical_constants
kB = physical_constants["Boltzmann constant in eV/K"][0]

MainPath = "/home/sohamc2/HEA_FCC/MDMC/MD_KMC_single/"
KMC_funcs_path = "/home/sohamc2/HEA_FCC/MDMC/MD_KMC/"

args = list(sys.argv)
T = int(args[1])
Ntraj = int(args[2]) # how many trajectories we want to simulate
startIndex = int(args[3])
batchSize = int(args[4]) # we'll evaluate the single-step trajectories in batches
NImage = int(args[5])

if len(args) == 7:
    MainPath = args[7]

if len(args) == 8:
    KMC_funcs_path = args[8]

import sys
sys.path.append(KMC_funcs_path)
from KMC_funcs import *

ProcPerImage = 1

if Ntraj%batchSize != 0:
    raise ValueError("batchSize does not divide Ntraj integrally.")

with open(MainPath+"lammpsBox.txt", "r") as fl:
    Initlines = fl.readlines()

# Load the lammps cartesian positions and neighborhoods - pre-prepared
SiteIndToPos = np.load(MainPath+"SiteIndToLmpCartPos.npy")  # lammps pos of sites
SiteIndToNgb = np.load(MainPath+"siteIndtoNgbSiteInd.npy")  # Nsites x z array of site neighbors

with open(MainPath+"CrysDat/jnetFCC.pkl", "rb") as fl:
    jnetFCC = pickle.load(fl)
dxList = np.array([dx*3.59 for (i, j), dx in jnetFCC[0]])


# Load the starting data for the trajectories
SiteIndToSpecAll = np.load(MainPath + "states_{}.npy".format(T))[startIndex : startIndex + Ntraj].astype(np.int16) # Ntraj x Nsites array of occupancies
assert np.all(SiteIndToSpecAll[:, 0] == 0) # check that the vacancy is always at the 0th site in the initial states
vacSiteIndAll = np.zeros(Ntraj, dtype=int) # Ntraj size array: contains where the vac is in each traj.
SiteIndToSpecAll[:, 0] = -1 # set vacancy occupancy to -1 to match the functions

specs, counts = np.unique(SiteIndToSpecAll[0], return_counts=True)
Nspec = len(specs)  # including the vacancy

Nsites = SiteIndToSpecAll.shape[1]

Initlines[2] = "{} \t atoms\n".format(Nsites - 1)
Initlines[3] = "{} atom types\n".format(Nspec-1)

FinalStates = np.zeros_like(SiteIndToSpecAll).astype(np.int16)
FinalVacSites = np.zeros(Ntraj).astype(np.int16)
SpecDisps = np.zeros((Ntraj, Nspec, 3))
tarr = np.zeros(Ntraj)
JumpSelects = np.zeros(Ntraj, dtype=np.int8) # which jump is chosen for each trajectory

# rates will be stored for the first batch for testing
TestRates = np.zeros((batchSize, 12)) # store all the rates to be tested
TestBarriers = np.zeros((batchSize, 12)) # store all the barriers to be tested
TestRandomNums = np.zeros(batchSize) # store the random numbers used in the test trajectories

# write the input file
Nbatch = Ntraj//batchSize

Barriers_Spec = collections.defaultdict(list)

write_input_files(batchSize)
start_timer = time.time()
for batch in range(Nbatch):
    # Write the initial states from last accepted state
    sampleStart = batch*batchSize
    sampleEnd = (batch+1)*batchSize
    SiteIndToSpec = SiteIndToSpecAll[sampleStart : sampleEnd].copy()
    vacSiteInd = vacSiteIndAll[sampleStart : sampleEnd].copy()
    
    write_init_states(SiteIndToSpec, SiteIndToPos, vacSiteInd, Initlines)
    
    rates = np.zeros((batchSize, SiteIndToNgb.shape[1]))
    barriers = np.zeros((batchSize, SiteIndToNgb.shape[1]))
    for jumpInd in range(SiteIndToNgb.shape[1]):
        # Write the final states in NEB format for lammps
        write_final_states(SiteIndToPos, vacSiteInd, SiteIndToNgb, jumpInd)

        # store the final lammps files for the first batch of states
        if batch == 0:
            # Store the final data for each traj, at each step and for each jump
            for traj in range(batchSize):
                cmd = subprocess.Popen("cp final_{0}.data final_{0}_{1}.data".format(traj, jumpInd), shell=True)
                rt = cmd.wait()
                assert rt == 0

        # Then run lammps
        commands = [
            "mpirun -np {0} $LMPPATH/lmp -p {0}x1 -in in.neb_{1} > out_{1}.txt".format(NImage, traj)
            for traj in range(batchSize)
        ]
        cmdList = [subprocess.Popen(cmd, shell=True) for cmd in commands]
        
        # wait for the lammps commands to complete
        for c in cmdList:
            rt_code = c.wait()
            assert rt_code == 0  # check for system errors
        
        # Then read the forward barrier -> ebf
        for traj in range(batchSize):
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
            Barriers_Spec[jAtom].append(ebf)

    if batch == 0:
        TestRates[:, :] = rates[:, :]
        TestBarriers[:, :] = barriers[:, :]

    # Then do selection
    jumpID, rateProbs, ratesCsum, rndNums, time_step = getJumpSelects(rates)
    # store the selected jumps
    JumpSelects[sampleStart : sampleEnd] = jumpID[:]

    # store the random numbers for the first set of jump
    if batch == 0:
        TestRandomNums[:] = rndNums[:]

    # Then do the final exchange
    jumpAtomSelectArray, X_traj = updateStates(SiteIndToNgb, Nspec, SiteIndToSpec, vacSiteInd, jumpID, dxList)
    # def updateStates(SiteIndToNgb, Nspec,  SiteIndToSpec, vacSiteInd, jumpID, dxList):
    
    # save final states, displacements and times
    FinalStates[sampleStart : sampleEnd, :] = SiteIndToSpec[:, :]
    SpecDisps[sampleStart:sampleEnd, :, :] = X_traj[:, :, :]
    tarr[sampleStart:sampleEnd] = time_step[:]
    with open("BatchTiming.txt", "a") as fl:
        fl.write("batch {0} of {1} completed in : {2} seconds\n".format(batch+1, Nbatch, time.time()-start_timer))

end_timer = time.time()
with open("SpecBarriers.pkl", "wb") as fl:
    pickle.dump(Barriers_Spec, fl)

# Next, save all the arrays in an hdf5 file
with h5py.File("data_{0}_{1}.h5".format(T, startIndex), "w") as fl:
    fl.create_dataset("FinalStates", data=FinalStates)
    fl.create_dataset("SpecDisps", data=SpecDisps)
    fl.create_dataset("times", data=tarr)
    fl.create_dataset("JumpSelects", data=JumpSelects)
    fl.create_dataset("TestRandNums", data=TestRandomNums)
    fl.create_dataset("TestRates", data=TestRates)
    fl.create_dataset("TestBarriers", data=TestBarriers)

print("Execution time: {}".format(end_timer-start_timer))
