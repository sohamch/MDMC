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
import gc


N_units = 8 # No. of unit cells along each axis in the supercell
            # The HEA simulations were all done on 8x8x8 supercells
            # So we restrict ourselves to that

# Load all the crystal data
def Load_crys_Data(typ="FCC"):
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

def Load_Data(FileName):
    with h5py.File(DataPath + FileName, "r") as fl:
        try:
            perm = fl["Permutation"]
            print("found permuation")
        except:
            perm = np.arange(len(fl["InitStates"]))

        state1List = np.array(fl["InitStates"])[perm]
        state2List = np.array(fl["FinStates"])[perm]
        dispList = np.array(fl["SpecDisps"])[perm]
        rateList = np.array(fl["rates"])[perm]

        try:
            AllJumpRates = np.array(fl["AllJumpRates"])[perm]
        except:
            raise ValueError("All Jump Rates not provided in data set.")

        try:
            jumpSelects = np.array(fl["JumpSelects"])[perm].astype(np.int8)
        except:
            jumpSelects = np.array(fl["JumpSelection"])[perm].astype(np.int8)

    return state1List, state2List, dispList, rateList, AllJumpRates, jumpSelects


def makeVClusExp(superCell, jnet, clustCut, MaxOrder, NSpec, vacsite):
    TScombShellRange = 1  # upto 1nn combined shell
    TSnnRange = 4
    TScutoff = np.sqrt(2)  # 4th nn cutoff - must be the same as TSnnRange


    print("Creating cluster expansion.")
    crys = superCell.crys
    clusexp = cluster.makeclusters(crys, clustCut, MaxOrder)

    # We'll create a dummy KRA expander anyway since the MC_JIT module is designed to accept transition arrays
    # However, this dummy KEA expander will never get used
    VclusExp = Cluster_Expansion.VectorClusterExpansion(superCell, clusexp, NSpec, vacsite, MaxOrder, TclusExp=True,
                                                    TScutoff=TScutoff, TScombShellRange=TScombShellRange,
                                                    TSnnRange=TSnnRange, jumpnetwork=jnet,
                                                    OrigVac=False, zeroClusts=True)

    print("generating interaction and vector basis data.")
    VclusExp.generateSiteSpecInteracts()
    # Generate the basis vectors for the clusters
    VclusExp.genVecClustBasis(VclusExp.SpecClusters)
    VclusExp.indexVclus2Clus()  # Index vector cluster list to cluster symmetry groups
    VclusExp.indexClustertoVecClus()

    return VclusExp

def CreateJitCalculator(VclusExp, NSpec, T, scratch=True, save=True):
    if scratch:
        # First, we have to generate all the arrays
        # Lattice gas Like -  set all energies to zero
        # All the rates are known to us anyway - they are the ones that are going to get used
        Energies = np.zeros(len(VclusExp.SpecClusters))
        KRAEnergies = [np.zeros(len(KRAClusterDict)) for (key, KRAClusterDict) in 
                VclusExp.KRAexpander.clusterSpeciesJumps.items()]

        KRASpecConstants = np.zeros(NSpec-1)

        # First, the chemical data
        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites,\
        Interaction2En, numInteractsSiteSpec, SiteSpecInterArray = VclusExp.makeJitInteractionsData(Energies)

        # Next, the vector basis data
        numVecsInteracts, VecsInteracts, VecGroupInteracts = VclusExp.makeJitVectorBasisData()
        
        NVclus = len(VclusExp.vecClus)

        # Note : The KRA expansion works only for binary alloys
        # Right now we don't need them, since we already know the rates
        # However, we create a dummy one since the JIT MC calculator requires the arrays
        KRACounterSpec = 1
        TsInteractIndexDict, Index2TSinteractDict, numSitesTSInteracts, TSInteractSites,\
        TSInteractSpecs, jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups,\
        numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng = VclusExp.KRAexpander.makeTransJitData(KRACounterSpec, KRAEnergies)
    
    if save:
        print("Saving JIT arrays")
        with h5py.File(RunPath+"JitArrays.h5", "w") as fl:
            fl.create_dataset("numSitesInteracts", data = numSitesInteracts)
            fl.create_dataset("SupSitesInteracts", data = SupSitesInteracts)
            fl.create_dataset("SpecOnInteractSites", data=SpecOnInteractSites)
            fl.create_dataset("numInteractsSiteSpec", data=numInteractsSiteSpec)
            fl.create_dataset("SiteSpecInterArray", data=SiteSpecInterArray)
            fl.create_dataset("numVecsInteracts", data=numVecsInteracts)
            fl.create_dataset("VecsInteracts", data=VecsInteracts)
            fl.create_dataset("VecGroupInteracts", data=VecGroupInteracts)
            
            fl.create_dataset("numSitesTSInteracts", data=numSitesTSInteracts)
            fl.create_dataset("TSInteractSites", data=TSInteractSites)
            fl.create_dataset("TSInteractSpecs", data=TSInteractSpecs)
            fl.create_dataset("jumpFinSites", data=jumpFinSites)
            fl.create_dataset("jumpFinSpec", data=jumpFinSpec)
            fl.create_dataset("FinSiteFinSpecJumpInd", data=FinSiteFinSpecJumpInd)
            fl.create_dataset("numJumpPointGroups", data=numJumpPointGroups)
            fl.create_dataset("numTSInteractsInPtGroups", data=numTSInteractsInPtGroups)
            fl.create_dataset("JumpInteracts", data=JumpInteracts)
            fl.create_dataset("Jump2KRAEng", data=Jump2KRAEng)
            fl.create_dataset("KRASpecConstants", data=KRASpecConstants)
            fl.create_dataset("NVclus", data=np.array([NVclus], dtype=int))

    
    else:
        print("Attempting to load arrays")
        with h5py.File(RunPath+"JitArrays.h5", "r") as fl:
            numSitesInteracts = np.array(fl["numSitesInteracts"])
            SupSitesInteracts = np.array(fl["SupSitesInteracts"])
            SpecOnInteractSites = np.array(fl["SpecOnInteractSites"])
            numInteractsSiteSpec = np.array(fl["numInteractsSiteSpec"])
            SiteSpecInterArray = np.array(fl["SiteSpecInterArray"])
            numVecsInteracts = np.array(fl["numVecsInteracts"])
            VecsInteracts = np.array(fl["VecsInteracts"])
            VecGroupInteracts = np.array(fl["VecGroupInteracts"])

            numSitesTSInteracts = np.array(fl["numSitesTSInteracts"])
            TSInteractSites = np.array(fl["TSInteractSites"])
            TSInteractSpecs = np.array(fl["TSInteractSpecs"])
            jumpFinSites = np.array(fl["jumpFinSites"])
            jumpFinSpec = np.array(fl["jumpFinSpec"])
            FinSiteFinSpecJumpInd = np.array(fl["FinSiteFinSpecJumpInd"])
            numJumpPointGroups = np.array(fl["numJumpPointGroups"])
            numTSInteractsInPtGroups = np.array(fl["numTSInteractsInPtGroups"])
            JumpInteracts = np.array(fl["JumpInteracts"])
            Jump2KRAEng = np.array(fl["Jump2KRAEng"])
            KRASpecConstants = np.array(fl["KRASpecConstants"])
            NVclus = np.array(fl["NVclus"])[0]
    
    # Make the MC class
    Interaction2En = np.zeros_like(numSitesInteracts, dtype=float)
    MCJit = MC_JIT.MCSamplerClass(
        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En,
        numInteractsSiteSpec, SiteSpecInterArray,
        numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
        FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups,
        JumpInteracts, Jump2KRAEng, KRASpecConstants
    )
    
    # The vector expansion data are not explicitly part of MCJit, so we'll return them separately
    return MCJit, numVecsInteracts, VecsInteracts, VecGroupInteracts, NVclus


def Expand(T, state1List, vacsiteInd, Nsamples, jList, dxList, SpecExpand, AllJumpRates, MCJit, NVclus,
        numVecsInteracts, VecsInteracts, VecGroupInteracts):

    # Get a dummy TS offsite counts
    TSOffSc = np.zeros(MCJit.numSitesTSInteracts.shape[0], dtype=np.int8)

    # Then we write the expansion loop
    totalW = np.zeros((NVclus, NVclus))
    totalB = np.zeros(NVclus)
    
    assert np.all(state1List[:, vacsiteInd] == MCJit.Nspecs - 1)

    print("Calculating rate and velocity expansions")
    for samp in tqdm(range(Nsamples), position=0, leave=True):
    
        # In the cluster expander, the vacancy is the highest labelled species,
        # In our case, it is the lowest
        # So we'll change the numbering so that the vacancy is labelled 5
        state = state1List[samp].copy()
    
        offsc = MC_JIT.GetOffSite(state, MCJit.numSitesInteracts, MCJit.SupSitesInteracts, MCJit.SpecOnInteractSites)
    
        WBar, bBar, rates_used , _, _ = MCJit.Expand(state, jList, dxList, SpecExpand, offsc,
                                          TSOffSc, numVecsInteracts, VecGroupInteracts, VecsInteracts,
                                          NVclus, 0, vacsiteInd, AllJumpRates[samp])
        
        assert np.array_equal(state, state1List[samp]) # assert revertions
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


def Expand_1jmp(T, state1List, vacsiteInd, Nsamples, jSiteList, jSelectList, dispList, SpecExpand, rates, MCJit, NVclus,
        numVecsInteracts, VecsInteracts, VecGroupInteracts):

    # Get a dummy TS offsite counts
    TSOffSc = np.zeros(MCJit.numSitesTSInteracts.shape[0], dtype=np.int8)

    # Then we write the expansion loop
    totalW = np.zeros((NVclus, NVclus))
    totalB = np.zeros(NVclus)
    
    assert np.all(state1List[:, vacsiteInd] == MCJit.Nspecs - 1)

    print("Calculating rate and velocity expansions")
    for samp in tqdm(range(Nsamples), position=0, leave=True):
    
        # In the cluster expander, the vacancy is the highest labelled species,
        # In our case, it is the lowest
        # So we'll change the numbering so that the vacancy is labelled 5
        state = state1List[samp].copy()
        
        jList = np.array([jSiteList[jSelectList[samp]]], dtype=int)
        dxList = np.array([dispList[samp, -1, :]], dtype=float)
        Rate = np.array([rates[samp]], dtype=float)

        offsc = MC_JIT.GetOffSite(state, MCJit.numSitesInteracts, MCJit.SupSitesInteracts, MCJit.SpecOnInteractSites)
    
        WBar, bBar, rates_used , _, _ = MCJit.Expand(state, jList, dxList, SpecExpand, offsc,
                                          TSOffSc, numVecsInteracts, VecGroupInteracts, VecsInteracts,
                                          NVclus, 0, vacsiteInd, Rate)
        
        assert np.array_equal(state, state1List[samp]) # assert revertions
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
        jList, dxList, vacsiteInd, NVclus, MCJit, etaBar, start, end,
        numVecsInteracts, VecGroupInteracts, VecsInteracts):

    L = 0.
    for samp in tqdm(range(start, end), position=0, leave=True):
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

    L /= (end-start)

    return L


def main(args):
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
    count += 1

    N_train = int(args[count])
    count += 1

    from_scratch = bool(int(args[count]))
    count += 1

    saveClusExp = bool(int(args[count]))
    count += 1

    saveJit = bool(int(args[count]))
    count += 1
    
    singleJump_Train = bool(int(args[count]))
    count += 1

    CrysType = "FCC" if count == len(args) else args[count]

    # Load Data
    specExpOriginal = SpecExpand
    state1List, state2List, dispList, rateList, AllJumpRates, jumpSelects = Load_Data(FileName)

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
    jList, dxList, jumpNewIndices, superCell, jnet, vacsite, vacsiteInd = Load_crys_Data(typ=CrysType)

    if from_scratch:
        print("Generating New cluster expansion")
        VclusExp = makeVClusExp(superCell, jnet, clustCut, MaxOrder, NSpec, vacsite)
        if saveClusExp:
            with open(RunPath+"VclusExp.pkl", "wb") as fl:
                pickle.dump(VclusExp, fl)

    else:
        VclusExp = None
        saveJit = False

    # Make MCJIT
    MCJit, numVecsInteracts, VecsInteracts, VecGroupInteracts, NVclus = CreateJitCalculator(VclusExp, NSpec, T, scratch=from_scratch, save=saveJit) 
    
    # Expand W and B
    # We need to scale displacements properly first
    a0 = np.linalg.norm(dispList[0, NSpec -1 , :])/np.linalg.norm(dxList[0])
    
    #Expand(T, state1List, vacsiteInd, Nsamples, dxList, SpecExpand, AllJumpRates, MCJit, NVclus,
    #    numVecsInteracts, VecsInteracts, VecGroupInteracts)
    
    if singleJump_Train:
        print("Training to 1 jump")

        #Expand_1jmp(T, state1List, vacsiteInd, Nsamples, jSiteList, jSelectList, dispList, SpecExpand, rates, MCJit, NVclus,
            #numVecsInteracts, VecsInteracts, VecGroupInteracts):
        Wbar, Bbar, Gbar, etaBar = Expand_1jmp(T, state1List, vacsiteInd, N_train, jList, jumpSelects, dispList, 
                SpecExpand, rateList, MCJit, NVclus, numVecsInteracts, VecsInteracts, VecGroupInteracts) 

    else:
        print("Training to all jumps")
        Wbar, Bbar, Gbar, etaBar = Expand(T, state1List, vacsiteInd, N_train, jList, dxList*a0, 
                SpecExpand, AllJumpRates, MCJit, NVclus, numVecsInteracts, VecsInteracts, VecGroupInteracts) 


    # Calculate transport coefficients
    print("Computing Transport coefficients")
    L_train = Calculate_L(state1List, SpecExpand, rateList, 
            dispList, jumpSelects, jList, dxList*a0,
            vacsiteInd, NVclus, MCJit, 
            etaBar, 0, N_train,
            numVecsInteracts, VecGroupInteracts, VecsInteracts)

    L_val = Calculate_L(state1List, SpecExpand, rateList, 
            dispList, jumpSelects, jList, dxList*a0,
            vacsiteInd, NVclus, MCJit, 
            etaBar, N_train, state1List.shape[0],
            numVecsInteracts, VecGroupInteracts, VecsInteracts)

    np.save("L{0}{0}_{1}.npy".format(specExpOriginal, T), np.array([L_train, L_val]))

    print("All Done \n\n")

if __name__ == "__main__":
    main(list(sys.argv))
