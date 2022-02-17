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

args = list(sys.argv)
T = float(args[1])
Nsteps = int(args[2])
SampleStart = int(args[3])
batchSize = int(args[4])

# Need to get rid of these argument
NImage = 3
ProcPerImage = 1

__test__ = False

with open("lammpsBox.txt", "r") as fl:
    Initlines = fl.readlines()

# Load the lammps cartesian positions and neighborhoods - pre-prepared
SiteIndToPos = np.load("SiteIndToLmpCartPos.npy")  # lammps pos of sites
SiteIndToNgb = np.load("siteIndtoNgbSiteInd.npy")  # Nsites x z array of site neighbors


with open("CrysDat/jnetFCC.pkl", "rb") as fl:
    jnetFCC = pickle.load(fl)
dxList = np.array([dx*3.59 for (i, j), dx in jnetFCC[0]])

# load the data
allStates = np.load("states_{}.npy".format(T))
perm = np.load("perm_{}.npy".format(T))
# Load the starting data for the trajectories
SiteIndToSpec = allStates[perm][SampleStart: SampleStart + batchSize]
vacSiteInd = np.load("vacSiteInd.npy")[perm][SampleStart: SampleStart + batchSize]
specs, counts = np.unique(SiteIndToSpec[0], return_counts=True)
Nspec = len(specs)  # including the vacancy
Ntraj = SiteIndToSpec.shape[0]

# save the starting states for this set
np.save("StatesStart_{}.npy".format(SampleStart))
np.save("vacSiteInd_{}.npy".format(SampleStart))

Nsites = SiteIndToSpec.shape[1]

Initlines[2] = "{} \t atoms\n".format(Nsites - 1)
Initlines[3] = "{} atom types\n".format(Nspec-1)

try:
    X_steps = np.load("X_steps.npy")
    t_steps = np.load("t_steps.npy")
    stepsLast = np.load("steps_last.npy")[0]
except FileNotFoundError:
    X_steps = np.zeros((Ntraj, Nspec, Nsteps + 1, 3)) # 0th position will store vacancy jumps
    t_steps = np.zeros((Ntraj, Nsteps + 1))
    stepsLast = 0
    
stepCount = np.zeros(1, dtype=int)
# Before starting, write the lammps input files
write_input_files(Ntraj)

start = time.time()
kB = physical_constants["Boltzmann constant in eV/K"][0]

for step in range(Nsteps - stepsLast):
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
    
    # Then do the final exchange
    jumpAtomSelectArray, X_traj = updateStates(SiteIndToNgb, Nspec, SiteIndToSpec, vacSiteInd, jumpID, dxList)
    # updateStates args : (SiteIndToNgb, Nspec,  SiteIndToSpec, vacSiteInd, jumpID, dxList)
    # Note the displacements and the time
    X_steps[:, :, step + stepsLast + 1, :] = X_traj[:, :, :]
    t_steps[:, step + stepsLast + 1] = time_step
    stepCount[0] = step + stepsLast + 1

end = time.time()
