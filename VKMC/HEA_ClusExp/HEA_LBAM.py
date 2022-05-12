#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("/home/sohamc2/HEA_FCC/MDMC/VKMC/")
import os
RunPath = os.getcwd() + "/"
CrysDatPath = "/home/sohamc2/HEA_FCC/MDMC/" 

from onsager import crystal, supercell, cluster
import numpy as np
import scipy as sp
import Transitions
import Cluster_Expansion
import MC_JIT
import pickle
import h5py
from tqdm import tqdm
from numba import jit, int64, float64

# ## Set up the vector cluster expander with the saved supercell
import sys
args = list(sys.argv)
T = int(args[1])
MaxOrder = int(args[2])
clustCut = float(args[3])
SpecExpand = float(args[4])

# Load the data set
with h5py.File("CrysDatPath/MD_KMC_single/singleStep_{0}.h5".format(T), "r") as fl:
    perm = np.array(fl["Permutation"])
    state1List = np.array(fl["InitStates"])[perm]
    state2List = np.array(fl["FinStates"])[perm]
    AllJumpRates = np.array(fl["AllJumpRates"])[perm]
    rateList = np.array(fl["rates"])[perm]
    dispList = np.array(fl["SpecDisps"])[perm]
    jumpSelects = np.array(fl["JumpSelects"])[perm].astype(np.int8)

AllSpecs = np.unique(state1List[0])
NSpec = AllSpecs.shape[0]
Nsamples = state1List.shape[0]
SpecExpand = NSpec - 1 - SpecExpand

N_units = 8 # No. of unit cells along each axis in the supercell
            # The HEA simulations were all done on 8x8x8 supercells
            # So we restrict ourselves to that

# Load all the crystal data
a0 = 3.59
jList = np.load("CrysDatPath/CrysDat_FCC/jList.npy")
dxList = np.load("CrysDatPath/CrysDat_FCC/dxList.npy") * a0
RtoSiteInd = np.load("CrysDatPath/CrysDat_FCC/RtoSiteInd.npy")
siteIndtoR = np.load("CrysDatPath/CrysDat_FCC/SiteIndtoR.npy")
jumpNewIndices = np.load("CrysDatPath/CrysDat_FCC/JumpNewSiteIndices.npy")

with open("CrysDatPath/CrysDat_FCC/supercellFCC.pkl", "rb") as fl:
    superFCC = pickle.load(fl)
crys = superFCC.crys

with open("CrysDatPath/CrysDat_FCC/jnetFCC.pkl", "rb") as fl:
    jnetFCC = pickle.load(fl)

vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
vacsiteInd = superFCC.index(np.zeros(3, dtype=int), (0, 0))[0]
assert vacsiteInd == 0

TScombShellRange = 1  # upto 1nn combined shell
TSnnRange = 4
TScutoff = np.sqrt(2)  # 4th nn cutoff - must be the same as TSnnRange

clusexp = cluster.makeclusters(crys, clustCut, MaxOrder)

# We'll create a dummy KRA expander anyway since the MC_JIT module is designed to accept transition arrays
# However, this dummy KEA expander will never get used
VclusExp = Cluster_Expansion.VectorClusterExpansion(superFCC, clusexp, NSpec, vacsite, MaxOrder, TclusExp=True,
                                                    TScutoff=TScutoff, TScombShellRange=TScombShellRange,
                                                    TSnnRange=TSnnRange, jumpnetwork=jnetFCC,
                                                    OrigVac=False, zeroClusts=True)

VclusExp.generateSiteSpecInteracts()
# Generate the basis vectors for the clusters
VclusExp.genVecClustBasis(VclusExp.SpecClusters)
VclusExp.indexVclus2Clus()  # Index vector cluster list to cluster symmetry groups
VclusExp.indexClustertoVecClus()
NVclus = len(VclusExp.vecVec)

# First, we have to generate all the arrays
# Lattice gas Like -  set all energies to zero
# All the rates are known to us anyway - they are the ones that are going to get used
Energies = np.zeros(len(VclusExp.SpecClusters))
KRAEnergies = [np.zeros(len(KRAClusterDict)) for (key, KRAClusterDict) in
               VclusExp.KRAexpander.clusterSpeciesJumps.items()]

# First, the chemical data
numSitesInteracts, SupSitesInteracts, SpecOnInteractSites,\
Interaction2En, numInteractsSiteSpec, SiteSpecInterArray = VclusExp.makeJitInteractionsData(Energies)

# Next, the vector basis data
numVecsInteracts, VecsInteracts, VecGroupInteracts = VclusExp.makeJitVectorBasisData()

# Note : The KRA expansion works only for binary alloys
# Right now we don't need them, since we already know the rates
# However, we create a dummy one since the JIT MC calculator requires the arrays
KRACounterSpec = 1
TsInteractIndexDict, Index2TSinteractDict, numSitesTSInteracts, TSInteractSites,\
TSInteractSpecs, jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups,\
numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng = VclusExp.KRAexpander.makeTransJitData(KRACounterSpec, KRAEnergies)

# Make the MC class
KRASpecConstants = np.random.rand(NSpec-1)
MCJit = MC_JIT.MCSamplerClass(
    numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En,
    numInteractsSiteSpec, SiteSpecInterArray,
    numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
    FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups,
    JumpInteracts, Jump2KRAEng, KRASpecConstants
)

# Verify that the displacements, selected jumps and state changes are consistent
print("Verifying displacements")
for samp in tqdm(range(Nsamples), position=0, leave=True):
    jSelect = jumpSelects[samp]
    dispVac = dispList[samp, 0, :]
    dxJump = dxList[jSelect]
    assert np.allclose(dxJump, dispVac)
    state1 = state1List[samp]
    state2 = state2List[samp]
    
    specJump = state1[jList[jSelect]]
    assert np.allclose(dispList[samp, specJump, :], -dxJump)
    
    assert np.array_equal(state2, state1[jumpNewIndices[jSelect]])

# test the offsite counting
print("Verifying active cluster counting")
state_test = NSpec - 1 - state1List[0]
offsc = MC_JIT.GetOffSite(state_test, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites)
offCounts = np.zeros_like(offsc)
for interactInd in tqdm(range(100000), position=0, leave=True):
    off_count_interact = 0
    for siteInd in range(numSitesInteracts[interactInd]):
        supSite = SupSitesInteracts[interactInd, siteInd]
        spec = SpecOnInteractSites[interactInd, siteInd]
        if state_test[supSite] != spec:
            off_count_interact += 1
    offCounts[interactInd] = off_count_interact
    assert off_count_interact == offsc[interactInd]

# Get a dummy TS offsite counts
TSOffSc = np.zeros(numSitesTSInteracts.shape[0], dtype=np.int8)

# Then we write the expansion loop
totalW = np.zeros((NVclus, NVclus))
totalB = np.zeros(NVclus)

print("Calculating rate and velocity expansions")

for samp in tqdm(range(Nsamples//2), position=0, leave=True):
    
    # In the cluster expander, the vacancy is the highest labelled species,
    # In our case, it is the lowest
    # So we'll change the numbering so that the vacancy is labelled 5
    state = NSpec - 1 - state1List[samp]
    
    offsc = MC_JIT.GetOffSite(state, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites)
    
    WBar, bBar, rates_used = MCJit.Expand(state, jList, dxList, SpecExpand, offsc,
                                          TSOffSc, numVecsInteracts, VecGroupInteracts, VecsInteracts,
                                          NVclus, 0, vacsiteInd, AllJumpRates[samp])
    
    assert np.allclose(rates_used, AllJumpRates[samp])
    totalW += WBar
    totalB += bBar

totalW /= Nsamples
totalB /= Nsamples

np.save(RunPath + "Wbar_{}.npy".format(T), totalW)
np.save(RunPath + "Bbar_{}.npy".format(T), totalB)

# verify symmetry
for i in tqmd(range(totalW.shape[0]), position=0, leave=True):
    for j in range(i):
        assert np.allclose(totalW[i,j], totalW[j,i])

Gbar = sp.linalg.pinvh(totalW, rtol=1e-8)

# Check pseudo-inverse relations
assert np.allclose(Gbar @ totalW @ Gbar, Gbar)
assert np.allclose(totalW @ Gbar @ totalW, totalW)

# Compute relaxation expansion
etaBar = -np.dot(Gbar, totalB)

# Get the new training set result
print("Calculating Training set transport coefficients.")
L = 0.
for samp in tqdm(range(Nsamples//2), position=0, leave=True):
    state = NSpec - 1 - state1List[samp]
    
    offsc = MC_JIT.GetOffSite(state, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites)
    jSelect = jumpSelects[samp]
    jSite = jList[jSelect]
    
    del_lamb = MCJit.getDelLamb(state, offsc, vacsiteInd, jSite, NVclus,
                                numVecsInteracts, VecGroupInteracts, VecsInteracts)
    
    disp_sp = dispList[samp, SpecExpand, :]
    if state[jList[jSelect]] == 0:
        assert np.allclose(disp_sp, -dxList[jSelect])
    else:
        assert np.allclose(disp_sp, 0.)
    
    # Get the change in y
    del_y = del_lamb.T @ etaBar
    
    # Modify the total displacement
    disp_sp_mod = disp_sp + del_y
    
    L += rateList[samp] * np.linalg.norm(disp_sp_mod)**2 /6.0

L_train = L/(Nsamples//2)

# Get the new validation set result
print("Calculating Validation set transport coefficients.")
L = 0.
for samp in tqdm(range(Nsamples//2, Nsamples), position=0, leave=True):
    state = NSpec - 1 - state1List[samp]
    
    offsc = MC_JIT.GetOffSite(state, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites)
    jSelect = jumpSelects[samp]
    jSite = jList[jSelect]
    
    del_lamb = MCJit.getDelLamb(state, offsc, vacsiteInd, jSite, NVclus,
                                numVecsInteracts, VecGroupInteracts, VecsInteracts)
    
    disp_sp = dispList[samp, SpecExpand, :]
    if state[jList[jSelect]] == 0:
        assert np.allclose(disp_sp, -dxList[jSelect])
    else:
        assert np.allclose(disp_sp, 0.)
    
    del_y = del_lamb.T @ etaBar
    
    disp_sp_mod = disp_sp + del_y
    
    L += rateList[samp] * np.linalg.norm(disp_sp_mod)**2 /6.0

L_val = L/(Nsamples - Nsamples//2)

np.save(RunPath + "L{0}{0}_{1}.npy".format(SpecExpand, T), np.array(L_train, L_val))
