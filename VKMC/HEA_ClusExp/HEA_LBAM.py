#!/usr/bin/env python
# coding: utf-8

import sys
# This is the path to the cluster expansion modules
sys.path.append("/home/sohamc2/HEA_FCC/MDMC/VKMC/")

# This is the path to the GCNetRun module - from which data loaders will be used
sys.path.append("/home/sohamc2/HEA_FCC/MDMC/ML_runs/")
import os
RunPath = os.getcwd() + "/"

CrysDatPath = "/home/sohamc2/HEA_FCC/MDMC/" 
DataPath = "/home/sohamc2/HEA_FCC/MDMC/ML_runs/DataSets/"

from onsager import crystal, supercell, cluster
import numpy as np
import scipy.linalg as spla
import Cluster_Expansion
import MC_JIT
import pickle
import h5py
from tqdm import tqdm
from numba import jit, int64, float64
from GCNetRun import Load_Data 
import gc


N_units = 8 # No. of unit cells along each axis in the supercell
            # The HEA simulations were all done on 8x8x8 supercells
            # So we restrict ourselves to that

# Load all the crystal data
def Load_crys_Data(typ="FCC")
    print("Loading {} Crystal data".format(typ))
    jList = np.load(CrysDatPath+"CrysDat_{}/jList.npy".format(typ))
    dxList = np.load(CrysDatPath+"CrysDat_{}/dxList.npy".format(typ))
    jumpNewIndices = np.load(CrysDatPath+"CrysDat_{}/JumpNewSiteIndices.npy".format(typ))

    with open(CrysDatPath+"CrysDat_{}/supercellFCC.pkl".format(typ), "rb") as fl:
        superFCC = pickle.load(fl)

    with open(CrysDatPath+"CrysDat_{}/jnetFCC.pkl".format(typ), "rb") as fl:
        jnetFCC = pickle.load(fl)

    vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
    vacsiteInd = superFCC.index(np.zeros(3, dtype=int), (0, 0))[0]
    assert vacsiteInd == 0
    return jList, dxList, jumpNewIndices, superFCC, jnetFCC, vacsite, vacsiteInd

def makeVClusExp(superCell, clustCut, MaxOrder, NSpec, vacsite)
    TScombShellRange = 1  # upto 1nn combined shell
    TSnnRange = 4
    TScutoff = np.sqrt(2)  # 4th nn cutoff - must be the same as TSnnRange


    print("Creating cluster expansion.")
    crys = superCell.crys
    clusexp = cluster.makeclusters(crys, clustCut, MaxOrder)

    # We'll create a dummy KRA expander anyway since the MC_JIT module is designed to accept transition arrays
    # However, this dummy KEA expander will never get used
    VclusExp = Cluster_Expansion.VectorClusterExpansion(superFCC, clusexp, NSpec, vacsite, MaxOrder, TclusExp=True,
                                                    TScutoff=TScutoff, TScombShellRange=TScombShellRange,
                                                    TSnnRange=TSnnRange, jumpnetwork=jnetFCC,
                                                    OrigVac=False, zeroClusts=True)

    print("generating interaction and vector basis data.")
    VclusExp.generateSiteSpecInteracts()
    # Generate the basis vectors for the clusters
    VclusExp.genVecClustBasis(VclusExp.SpecClusters)
    VclusExp.indexVclus2Clus()  # Index vector cluster list to cluster symmetry groups
    VclusExp.indexClustertoVecClus()

    return VclusExp

def CreateJitCalculator(VclusExp):
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

    return MCJit, numVecsInteracts, VecsInteracts, VecGroupInteracts


def Expand(T, state1List, Nsamples, dxList, SpecExpand, AllJumpRates, MCJit, NVclus,
        numVecsInteracts, VecsInteracts, VecGroupInteracts):

    # Get a dummy TS offsite counts
    TSOffSc = np.zeros(MCJit.numSitesTSInteracts.shape[0], dtype=np.int8)

    # Then we write the expansion loop
    totalW = np.zeros((NVclus, NVclus))
    totalB = np.zeros(NVclus)

    print("Calculating rate and velocity expansions")
    for samp in tqdm(range(Nsamples), position=0, leave=True):
    
        # In the cluster expander, the vacancy is the highest labelled species,
        # In our case, it is the lowest
        # So we'll change the numbering so that the vacancy is labelled 5
        state = state1List[samp]
    
        offsc = MC_JIT.GetOffSite(state, MCJit.numSitesInteracts, MCJit.SupSitesInteracts, MCJit.SpecOnInteractSites)
    
        WBar, bBar, rates_used , _, _ = MCJit.Expand(state, jList, dxList, SpecExpand, offsc,
                                          TSOffSc, numVecsInteracts, VecGroupInteracts, VecsInteracts,
                                          NVclus, 0, vacsiteInd, AllJumpRates[samp])
    
        assert np.allclose(rates_used, AllJumpRates[samp])
        totalW += WBar
        totalB += bBar

    totalW /= Nsamples
    totalB /= Nsamples

    # verify symmetry
    print("verifying symmetry of rate expansion")
    for i in tqdm(range(totalW.shape[0]), position=0, leave=True):
        for j in range(i):
            assert np.allclose(totalW[i,j], totalW[j,i])

    Gbar = spla.pinvh(totalW, rcond=1e-8)

    # Check pseudo-inverse relations
    assert np.allclose(Gbar @ totalW @ Gbar, Gbar)
    assert np.allclose(totalW @ Gbar @ totalW, totalW)

    # Compute relaxation expansion
    etaBar = -np.dot(Gbar, totalB)
    
    np.save(RunPath + "Wbar_{}.npy".format(T), totalW)
    np.save(RunPath + "Gbar_{}.npy".format(T), Gbar)
    np.save(RunPath + "etabar_{}.npy".format(T), etaBar)
    np.save(RunPath + "Bbar_{}.npy".format(T), totalB)

    return totalW, totalB, Gbar, etaBar


# Get the Transport coefficients
def Calculate_L(state1List, SpecExpand, rateList, dispList, jumpSelects,
        jList, vacsiteInd, NVclus, MCJit, etaBar, start, end):

    L = 0.
    for samp in tqdm(range(start, End), position=0, leave=True):
        state = state1List[samp]
    
        offsc = MC_JIT.GetOffSite(state, MCJit.numSitesInteracts, MCJit.SupSitesInteracts, MCJit.SpecOnInteractSites)
        jSelect = jumpSelects[samp]
        jSite = jList[jSelect]
    
        del_lamb = MCJit.getDelLamb(state, offsc, vacsiteInd, jSite, NVclus,
                                    numVecsInteracts, VecGroupInteracts, VecsInteracts)
    
        disp_sp = dispList[samp, SpecExpand, :]
        if state[jSite] == SpecExpand:
            assert np.allclose(disp_sp, -dxList[jSelect]), "{}\n {}\n {}\n".format(dxList, jSelect, disp_sp)
        else:
            assert np.allclose(disp_sp, 0.)
    
        # Get the change in y
        del_y = del_lamb.T @ etaBar
    
        # Modify the total displacement
        disp_sp_mod = disp_sp + del_y
    
        L += rateList[samp] * np.linalg.norm(disp_sp_mod)**2 /6.0

    L/= (end-start)

    return L

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
    
    disp_sp = dispList[samp, NSpec - 1 - SpecExpand, :]
    if state[jList[jSelect]] == 0:
        assert np.allclose(disp_sp, -dxList[jSelect])
    else:
        assert np.allclose(disp_sp, 0.)
    
    del_y = del_lamb.T @ etaBar
    
    disp_sp_mod = disp_sp + del_y
    
    L += rateList[samp] * np.linalg.norm(disp_sp_mod)**2 /6.0

L_val = L/(Nsamples - Nsamples//2)
np.save(RunPath + "L{0}{0}_{1}.npy".format(SpecExpand, T), np.array([L_train, L_val]))

if __name__ == "__main__":

    args = list(sys.argv)
    count = 1
    T = int(args[count])
    count += 1

    FileName = args[count]
    count += 1

    MaxOrder = int(args[count])
    count += 1

    clustCut = float(args[count])
    count += 1

    SpecExpand = int(args[count])
    count += 1

    VacSpec = int(args[count])


    # Load Data
    state1List, state2List, dispList, rateList, AllJumpRates = Load_Data(FileName)
    AllSpecs = np.unique(state1List[0])
    NSpec = AllSpecs.shape[0]
    Nsamples = state1List.shape[0]

    if VacSpec == 0:
        # Convert data so that vacancy is highest 
        state1List = NSpec - 1 - state1List
        state2List = NSpec - 1 - state2List
        dispListNew = np.zeros_like(dispList)
        for spec in range(NSpec):
            dispListNew[:, NSpec - 1 - spec, :] = dispList[:, spec, :]
        del(dispList)
        gc.collect()

        dispList = dispListNew

        SpecExpand = NSpec - 1 - SpecExpand


    # Load Crystal Data

    # Load vecClus - if existing, else create new
    NVclus = len(VclusExp.vecVec)

    # Make MCJIT

    # Expand W and B

    # Get Gbar and etabar

    # Calculate transport coefficients


