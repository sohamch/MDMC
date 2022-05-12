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

N_units = 8 # No. of unit cells along each axis in the supercell
MaxOrder = 2

# Load all the crystal data
jList = np.load("CrysDatPath/CrysDat_FCC/jList.npy")
dxList = np.load("CrysDatPath/CrysDat_FCC/dxList.npy")
RtoSiteInd = np.load("CrysDatPath/CrysDat_FCC/RtoSiteInd.npy")
siteIndtoR = np.load("CrysDatPath/CrysDat_FCC/SiteIndtoR.npy")
jumpNewIndices = np.load("CrysDatPath/CrysDat_FCC/JumpNewSiteIndices.npy")

a0 = 1.0
with open("CrysDatPath/CrysDat_FCC/supercellFCC.pkl", "rb") as fl:
    superFCC = pickle.load(fl)
crys = superFCC.crys

with open("CrysDatPath/CrysDat_FCC/jnetFCC.pkl", "rb") as fl:
    jnetFCC = pickle.load(fl)


NSpec = 6
vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
vacsiteInd = superFCC.index(np.zeros(3, dtype=int), (0, 0))[0]
assert vacsiteInd == 0


TScombShellRange = 1  # upto 1nn combined shell
TSnnRange = 4
TScutoff = np.sqrt(2) * a0  # 4th nn cutoff - must be the same as TSnnRange

clustCut = 1.01*np.sqrt(2)*a0
clusexp = cluster.makeclusters(crys, clustCut, MaxOrder)

# We'll create a dummy KRA expander anyway since the MC_JIT module is designed to accept transition arrays
# However, this dummy KEA expander will never get used
VclusExp = Cluster_Expansion.VectorClusterExpansion(superFCC, clusexp, NSpec, vacsite, MaxOrder, TclusExp=True,
                                                    TScutoff=TScutoff, TScombShellRange=TScombShellRange,
                                                    TSnnRange=TSnnRange, jumpnetwork=jnetFCC,
                                                    OrigVac=False, zeroClusts=True)


len([cl for clList in VclusExp.SpecClusters for cl in clList])


# In[10]:


VclusExp.generateSiteSpecInteracts()
# Generate the basis vectors for the clusters
VclusExp.genVecClustBasis(VclusExp.SpecClusters)
VclusExp.indexVclus2Clus()  # Index vector cluster list to cluster symmetry groups
VclusExp.indexClustertoVecClus()


# In[11]:


len([cl for clList in VclusExp.SpecClusters for cl in clList])


# In[12]:


# First, we have to generate all the arrays
# Lattice gas - set all energies to zero
Energies = np.zeros(len(VclusExp.SpecClusters))
KRAEnergies = [np.zeros(len(KRAClusterDict)) for (key, KRAClusterDict) in
               VclusExp.KRAexpander.clusterSpeciesJumps.items()]

# First, the chemical data
numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numInteractsSiteSpec, SiteSpecInterArray = VclusExp.makeJitInteractionsData(Energies)

# Next, the vector basis data
numVecsInteracts, VecsInteracts, VecGroupInteracts = VclusExp.makeJitVectorBasisData()

# Note : The KRA expansion works only for binary alloys
# Right now we don't need them, since we already know the rates
KRACounterSpec = 1
TsInteractIndexDict, Index2TSinteractDict, numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng =     VclusExp.KRAexpander.makeTransJitData(KRACounterSpec, KRAEnergies)


# In[16]:


numSitesInteracts.shape


# In[15]:


(numSitesInteracts.nbytes)/(1000 * 1000)


# In[ ]:


# save the required arrays and the cluster expansion
with open("ClusExpData_3body_4nn/VclusExp.pkl", "wb") as fl:
    pickle.dump(VclusExp, fl)


# In[ ]:


with h5py.File("ClusExpData_3body_4nn/JitVectors.h5", "w") as fl:
    fl.create_dataset("numSitesInteracts", data = numSitesInteracts)
    fl.create_dataset("SupSitesInteracts", data = SupSitesInteracts)
    fl.create_dataset("SpecOnInteractSites", data = SpecOnInteractSites)
    fl.create_dataset("Interaction2En", data = Interaction2En)
    fl.create_dataset("numInteractsSiteSpec", data = numInteractsSiteSpec)
    fl.create_dataset("SiteSpecInterArray", data = SiteSpecInterArray)
    fl.create_dataset("numSitesTSInteracts", data = numSitesTSInteracts)
    fl.create_dataset("TSInteractSites", data = TSInteractSites)
    fl.create_dataset("TSInteractSpecs", data = TSInteractSpecs)
    fl.create_dataset("jumpFinSites", data = jumpFinSites)
    fl.create_dataset("jumpFinSpec", data = jumpFinSpec)
    fl.create_dataset("FinSiteFinSpecJumpInd", data = FinSiteFinSpecJumpInd)
    fl.create_dataset("numJumpPointGroups", data = numJumpPointGroups)
    fl.create_dataset("numTSInteractsInPtGroups", data = numTSInteractsInPtGroups)
    fl.create_dataset("JumpInteracts", data = JumpInteracts)
    fl.create_dataset("Jump2KRAEng", data = Jump2KRAEng)


# In[7]:


len(VclusExp.vecClus)


# In[8]:


# Make the MC class
KRASpecConstants = np.random.rand(NSpec-1)
MCJit = MC_JIT.MCSamplerClass(
    numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En,
    numInteractsSiteSpec, SiteSpecInterArray,
    numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
    FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups,
    JumpInteracts, Jump2KRAEng, KRASpecConstants
)


# In[9]:


# Load the data set and the saved y vectors
T = 1073
with h5py.File("CrysDatPath/MD_KMC_single/singleStep_{0}.h5".format(T), "r") as fl:
    perm = np.array(fl["Permutation"])
    state1List = np.array(fl["InitStates"])[perm]
    state2List = np.array(fl["FinStates"])[perm]
    AllJumpRates = np.array(fl["AllJumpRates"])[perm]
    rateList = np.array(fl["rates"])[perm]
    dispList = np.array(fl["SpecDisps"])[perm]
    jumpSelects = np.array(fl["JumpSelects"])[perm].astype(np.int8)


# In[10]:


# Verify that the displacements, selected jumps and state changes are consistent
Nsamples = state1List.shape[0]
for samp in tqdm(range(Nsamples), position=0, leave=True):
    jSelect = jumpSelects[samp]
    dispVac = dispList[samp, 0, :]
    dxJump = dxList[jSelect] * 3.59
    assert np.allclose(dxJump, dispVac)
    state1 = state1List[samp]
    state2 = state2List[samp]
    
    specJump = state1[jList[jSelect]]
    assert np.allclose(dispList[samp, specJump, :], -dxJump)
    
    assert np.array_equal(state2, state1[jumpNewIndices[jSelect]])


# In[11]:


NVclus = len(VclusExp.vecVec)


# In[12]:


# test the offsite counting
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


# In[13]:


# Get a dummy TS offsite counts
TSOffSc = np.zeros(numSitesTSInteracts.shape[0], dtype=np.int8)


# In[15]:


# Construct the modified Mn displacements out of all jumps from the states
Mn_Jumps = np.zeros((Nsamples * dxList.shape[0], 3))
for samp in range(Nsamples):
    state1 = state1List[samp]
    for jInd in range(dxList.shape[0]):
        specJump = state1[jList[jInd]]
        if specJump == 5:
            Mn_Jumps[samp * dxList.shape[0] + jInd] =            -dxList[jInd]*3.59


# In[19]:


totalW = np.zeros((NVclus, NVclus))
totalB = np.zeros(NVclus)

for samp in tqdm(range(Nsamples//2), position=0, leave=True):
    
    # In the cluster expander, the vacancy is the highest labelled species,
    # In our case, it is the lowest
    # So we'll change the numbering so that the vacancy is labelled 5
    state1 = NSpec - 1 - state1List[samp]
    
    offsc = MC_JIT.GetOffSite(state1, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites)
    
    WBar, bBar, rates_used = MCJit.Expand(state1, jList, dxList * 3.59, 0, offsc,
                                          TSOffSc, numVecsInteracts, VecGroupInteracts, VecsInteracts,
                                          NVclus, 0, 0, AllJumpRates[samp])
    
    assert np.allclose(rates_used, AllJumpRates[samp])
    totalW += WBar
    totalB += bBar

totalW /= Nsamples
totalB /= Nsamples


# In[20]:


(dxList * 3.59).shape


# In[21]:


eigvals, eigvecs = np.linalg.eigh(totalW)


# In[22]:


np.where(eigvals < 1e-8)[0].shape


# In[23]:


for i in range(totalW.shape[0]):
    for j in range(totalW.shape[0]):
        assert np.allclose(totalW[i,j], totalW[j,i])


# In[24]:


Gbar = sp.linalg.pinvh(totalW, rtol=1e-8)


# In[25]:


# Check pseudo-inverse relations
assert np.allclose(Gbar @ totalW @ Gbar, Gbar)
assert np.allclose(totalW @ Gbar @ totalW, totalW)
# assert np.allclose(np.matmul(totalW, Gbar), np.eye(totalW.shape[0]))


# In[31]:


etaBar = -np.dot(Gbar, totalB)


# In[29]:


# Get the new training set result
L = 0.
for samp in tqdm(range(Nsamples//2), position=0, leave=True):
    state1 = NSpec - 1 - state1List[samp]
    
    offsc = MC_JIT.GetOffSite(state1, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites)
    lamb1 = MCJit.getLambda(offsc, NVclus, numVecsInteracts, VecGroupInteracts, VecsInteracts)
    
    jSelect = jumpSelects[samp]
    state2 = state1[jumpNewIndices[jSelect]]
    
    assert np.array_equal(state2, NSpec - 1 - state2List[samp])

    offsc2 = MC_JIT.GetOffSite(state2, numSitesInteracts,
                                   SupSitesInteracts, SpecOnInteractSites)

    lamb2 = MCJit.getLambda(offsc2, NVclus, numVecsInteracts, VecGroupInteracts, VecsInteracts)
    
    disp_Mn = dispList[samp, 5, :]
    if state1[jList[jSelect]] == 0:
        assert np.allclose(disp_Mn, -dxList[jSelect] * 3.59)
    
    # Get the second order modification
    del_y = (lamb2.T @ etaBar) - (lamb1.T @ etaBar)
    
    # Modify the total displacement
    disp_Mn_mod = disp_Mn + del_y
    
    L += rateList[samp] * np.linalg.norm(disp_Mn_mod)**2 /6.0


# In[33]:


L_train = L/(Nsamples//2)
L_train


# In[42]:


# Get the new validation set result
L = 0.
for samp in tqdm(range(Nsamples//2, Nsamples), position=0, leave=True):
    state = NSpec - 1 - state1List[samp]
    
    offsc = MC_JIT.GetOffSite(state, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites)
    jSelect = jumpSelects[samp]
    jSite = jList[jSelect]
    
    del_lamb = MCJit.getDelLamb(state, offsc, vacsiteInd, jSite, NVclus,
                                numVecsInteracts, VecGroupInteracts, VecsInteracts)
    
    disp_Mn = dispList[samp, 5, :]
    if state[jList[jSelect]] == 0:
        assert np.allclose(disp_Mn, -dxList[jSelect] * 3.59)
    else:
        assert np.allclose(disp_Mn, 0.)
    
    # Get the second order modification
    del_y = del_lamb.T @ etaBar
    
    # Modify the total displacement
    disp_Mn_mod = disp_Mn + del_y
    
    L += rateList[samp] * np.linalg.norm(disp_Mn_mod)**2 /6.0


# In[43]:


L_val = L/(Nsamples//2)
L_val


# In[44]:


np.save("ClusExpData_3body_2nn/etaBar_3body_2nn_{}.npy".format(T), etaBar)
np.save("ClusExpData_3body_2nn/LMnMn_{}.npy".format(T), np.array(L_train, L_val))


# In[ ]:




