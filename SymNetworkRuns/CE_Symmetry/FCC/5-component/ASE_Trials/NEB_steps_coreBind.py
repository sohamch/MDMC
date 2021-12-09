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

if len(args) == 6:
    ProcPerImage = 1
else:
    ProcPerImage = int(args[6])

if NProc%(NImage*ProcPerImage) != 0:
    raise ValueError("Processors cannot be divided integrally across trajectories")

# Processors per trajectory = NImage*ProcPerImage
Ntraj = NProc//(NImage*ProcPerImage)
    
print("Running {0} steps at {1}K".format(Nsteps, T))
print("Segments at every {}th step".format(Nseg))

# Load the starting data for the trajectories
with h5py.File("KMCStartDat.h5", "r") as fl:
    SiteIndToSpec = np.array(fl["SiteIndToSpec"])
    SiteIndToPos = np.array(fl["SiteIndToLmpCartPos"])
    SiteIndToNgb = np.array(fl["siteIndtoNgbSiteInd"])

specs, counts = np.unique(SiteIndToSpec[0], return_counts=True)
Nspec = len(specs) - 1

# These functions need to work on all the trajectories
# maybe parallelize them?

with open("ForlammpsCoord.txt","r") as fl:
    Initlines = fl.readlines()

def lammps_nebFin_write(SiteIndToPos, vacSiteInd, jumpSiteInd):
    with open("final.data", "w") as fl:
        fl.write("{}\n".format(len(SiteIndToPos)-1))
        counter = 1
        for siteInd in range(len(SiteIndToPos)):
            if siteInd == vacSiteInd:
                continue
            if siteInd == jumpSiteInd: # the jumping atom will have vac site as the new position
                pos = SiteIndToPos[vacSiteInd]
            else:
                pos = SiteIndToPos[siteInd]
            fl.write("{} {} {} {}\n".format(counter, pos[0], pos[1], pos[2]))
            counter += 1

def lammps_init_write(SiteIndtoSpec, SiteIndToPos):
    with open("initial.data", "w") as fl:
        fl.writelines(Initlines[:12])
        # fl.write("\n")
        counter = 1
        for idx in range(SiteIndtoSpec.shape[0]):
            spec = SiteIndToSpec[idx]
            if spec == -1:
                continue
            pos = SiteIndToPos[idx]
            fl.write("{} {} {} {} {}\n".format(counter, spec, pos[0], pos[1], pos[2]))
            counter += 1

@jit(nopython=True)
def getJumpSelects(ratesProbSum, rn):
    Ntraj = ratesProbSum.shape[0]
    jumpID = np.zeros(Ntraj, dtype=int)
    for tr in range(Ntraj):
        jSelect = np.searchsorted(ratesProbSum[tr, :], rn[tr])
        jumpID[tr] = jSelect
    return jumpID

@jit(nopython=True)
def updateStates(SiteIndToNgb, SiteIndToSpec, vacSiteInd, jumpID):
    Ntraj = jumpID.shape[0]
    jumpAtomSelectArray = np.zeros(Ntraj, dtype=int64)
    for tr in range(Ntraj):
        jumpSiteSelect = SiteIndToNgb[vacSiteInd[tr], jumpID[tr]]
        jumpAtomSelect = SiteIndToSpec[tr, jumpSiteSelect]
        jumpAtomSelectArray[]
        SiteIndToSpec[tr, vacSiteInd] = jumpAtomSelect
        SiteIndToSpec[tr, jumpSiteSelect] = -1 # The next vacancy site
        vacSiteInd[tr] = jumpSiteSelect

with open("CrysDat/jnetFCC.pkl", "rb") as fl:
    jnetFCC = pickle.load(fl)
dxList = np.array([dx*3.59 for (i, j), dx in jnetFCC[0]])
print(dxList)

vacSiteInd = 0 # initially the vacancy is at site 0 by design
X_steps = np.zeros((Nspec+1, Nseg, 3)) # 0th position will store vacancy jumps
t_steps = np.zeros(Nseg)

cmd = [
    "mpirun -np {0} $LMPPATH/lmp -p {1}x{2} -in in.neb > out.txt".format(NProc, NImage, ProcPerImage)
]

start = time.time()
neb_Time = 0.
neb_count = 0

print("Running {} trajectories on {} processors".format(Ntraj, NProc))

for step in range(Nsteps + 1):
    # Write the initial state from last accepted state
    # The function does so for all trajectories
    lammps_init_write(SiteIndToSpec, SiteIndToPos)
    
    # Store the arrays in a matrix
    rates = np.zeros(Ntraj, SiteIndToNgb.shape[1])
    cmdList = [] # this will store all the subprocess commands
    for jumpInd, jumpSiteInd in enumerate(SiteIndToNgb[vacSiteInd]):
        
        # Now write the final state in NEB format for lammps
        lammps_nebFin_write(SiteIndToPos, vacSiteInd, jumpSiteInd)
        st = time.time()
        
        # Then run lammps
        c = subprocess.Popen(cmd, shell=True)
        c.wait() # wait for the lammps command to complete
        
        neb_Time += time.time()-st
        neb_count += 1
        
        # Then read the forward barrier -> ebf
        with open("log.lammps", "r") as fl:
            for line in fl:
                continue
        
        ebfLine = line[-1].split()
        ebf = float(ebfLine[6])
        rates[jumpInd] = np.exp(-ebf/(kB*T))
    
    # Then do selection
    timeStep = 1./np.sum(rates, axis=1)
    ratesProb = rates*timeStep.reshape(Ntraj, 1)
    ratesProbSum = np.cumsum(ratesProb, axis=1)
    rn = np.random.rand(Ntraj)    
    jumpID = getJumpSelects(ratesProbSum, rn)
    
    # Then do the final exchange
    updateStates(SiteIndToNgb, SiteIndToSpec, vacSiteInd, jumpID)
    # jumpSiteSelect = SiteIndToNgb[vacSiteInd, jumpID]
    # jumpAtomSelect = SiteIndToSpec[jumpSiteSelect]
    # SiteIndToSpec[vacSiteInd] = jumpAtomSelect
    # SiteIndToSpec[jumpSiteSelect] = -1 # The next vacancy site
    # vacSiteInd = jumpSiteSelect
    
    # Note the displacements and the time
    X_steps[jumpAtomSelect, step%Nseg, :] = -dxList[jumpID, :]
    X_steps[0 , step%Nseg, :] = dxList[jumpID, :]
    t_steps[step%Nseg] = timeStep
    
    # Checkpoint after every Nseg steps
    if (step+1)%Nseg == 0:
        # Save the following as numpy arrays to a checkpoint folder
        np.save("chkpts/SiteIndToSpec_{}.npy".format(step), SiteIndToSpec)
        np.save("chkpts/X_steps_{}.npy".format(step), X_steps)
        np.save("chkpts/t_steps_{}.npy".format(step), t_steps)
        end_Nseg = time.time()-start
        print("Time Per Step: {:.4f}".format(end_Nseg/Nseg))
        start = time.time()
print("time per neb: {}".format(neb_Time/(1.0*neb_count)))
