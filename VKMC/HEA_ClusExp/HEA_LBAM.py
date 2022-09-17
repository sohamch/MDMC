#!/usr/bin/env python
# coding: utf-8

import sys
# This is the path to the cluster expansion modules
sys.path.append("/mnt/WorkPartition/Work/Research/UIUC/MDMC/VKMC")

import os
RunPath = os.getcwd() + "/"

from onsager import crystal, supercell, cluster
import numpy as np
import scipy.linalg as spla
import Cluster_Expansion
import MC_JIT
import pickle
import h5py
from tqdm import tqdm
import argparse
import gc


N_units = 8 # No. of unit cells along each axis in the supercell
            # The HEA simulations were all done on 8x8x8 supercells
            # So we restrict ourselves to that

# Load all the crystal data
def Load_crys_Data(CrysDatPath, typ="FCC"):
    print("Loading {} Crystal data".format(typ))
    jList = np.load(CrysDatPath+"CrysDat_{}/jList.npy".format(typ))
    dxList = np.load(CrysDatPath+"CrysDat_{}/dxList.npy".format(typ))
    jumpNewIndices = np.load(CrysDatPath+"CrysDat_{}/JumpNewSiteIndices.npy".format(typ))

    with open(CrysDatPath+"CrysDat_{0}/supercell{0}.pkl".format(typ), "rb") as fl:
        superFCC = pickle.load(fl)

    with open(CrysDatPath+"CrysDat_{0}/jnet{0}.pkl".format(typ), "rb") as fl:
        jnetFCC = pickle.load(fl)

    vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
    vacsiteInd = superFCC.index(np.zeros(3, dtype=int), (0, 0))[0]
    assert vacsiteInd == 0
    return jList, dxList, jumpNewIndices, superFCC, jnetFCC, vacsite, vacsiteInd

def Load_Data(DataPath):
    with h5py.File(DataPath, "r") as fl:
        try:
            perm = np.array(fl["Permutation"])
            print("found permuation")
        except:
            perm = np.arange(len(fl["InitStates"]))

        state1List = np.array(fl["InitStates"])[perm]
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

    return state1List, dispList, rateList, AllJumpRates, jumpSelects


def makeVClusExp(superCell, jnet, jList, clustCut, MaxOrder, NSpec, vacsite, AllInteracts=False):
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

    vacSiteInd, _ = superCell.index(vacsite.R, vacsite.ci)
    reqSites = [vacSiteInd] + list(jList) if not AllInteracts else None
    print("generating interactions with required sites : {}".format(reqSites))
    VclusExp.generateSiteSpecInteracts(reqSites=reqSites)
    print("No. of interactions : {}".format(len(VclusExp.Id2InteractionDict)))
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
                fl.create_dataset("numSitesInteracts", data=numSitesInteracts)
                fl.create_dataset("SupSitesInteracts", data=SupSitesInteracts)
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


def Expand(T, state1List, vacsiteInd, Nsamples, jList, dxList, AllJumpRates,
           jSelectList, dispSelects, ratesSelect, SpecExpand, MCJit, NVclus,
           numVecsInteracts, VecsInteracts, VecGroupInteracts, aj=True):


    # Get a dummy TS offsite counts
    TSOffSc = np.zeros(MCJit.numSitesTSInteracts.shape[0], dtype=np.int8)

    # Then we write the expansion loop
    totalW = np.zeros((NVclus, NVclus))
    totalB = np.zeros(NVclus)
    
    assert np.all(state1List[:, vacsiteInd] == MCJit.Nspecs - 1)

    state1ListCpy = state1List.copy()

    print("Calculating rate and velocity expansions")
    for samp in tqdm(range(Nsamples), position=0, leave=True):
    
        # In the cluster expander, the vacancy is the highest labelled species,
        # In our case, it is the lowest
        # So we'll change the numbering so that the vacancy is labelled 5
        state = state1ListCpy[samp]
    
        offsc = MC_JIT.GetOffSite(state, MCJit.numSitesInteracts, MCJit.SupSitesInteracts, MCJit.SpecOnInteractSites)

        if aj:
            WBar, bBar, rates_used, _, _ = MCJit.Expand(state, jList, dxList, SpecExpand, offsc,
                                              TSOffSc, numVecsInteracts, VecGroupInteracts, VecsInteracts,
                                              NVclus, 0, vacsiteInd, AllJumpRates[samp])

            assert np.array_equal(state, state1List[samp])  # assert revertions
            assert np.allclose(rates_used, AllJumpRates[samp])

        else:
            jList = np.array([jList[jSelectList[samp]]], dtype=int)
            dxList = np.array([dispSelects[samp, -1, :]], dtype=float) # The last one is the vacancy jump
            Rate = np.array([ratesSelect[samp]], dtype=float)
            WBar, bBar, rates_used, _, _ = MCJit.Expand(state, jList, dxList, SpecExpand, offsc,
                                                        TSOffSc, numVecsInteracts, VecGroupInteracts, VecsInteracts,
                                                        NVclus, 0, vacsiteInd, Rate)
            assert np.array_equal(state, state1List[samp])  # assert revertions
            assert np.allclose(rates_used, Rate)

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

    T = args.Temp
    DataPath = args.DataPath
    MaxOrder = args.MaxOrder
    clustCut = args.ClustCut
    SpecExpand = args.SpecExpand
    VacSpec = args.VacSpec
    N_train = args.NTrain
    from_scratch = args.Scratch
    saveClusExp = args.SaveCE
    saveJit = args.SaveJitArrays
    CrysDatPath = args.CrysDatPath
    CrysType = args.CrysType
    # Load Data
    specExpOriginal = SpecExpand
    state1List, dispList, rateList, AllJumpRates, jumpSelects = Load_Data(DataPath)

    AllSpecs = np.unique(state1List[0])
    NSpec = AllSpecs.shape[0]

    if VacSpec == 0:
        # Convert data so that vacancy is highest
        # This is the convenience chosen in the MC_JIT code
        state1List = NSpec - 1 - state1List
        dispListNew = np.zeros_like(dispList)
        for spec in range(NSpec):
            dispListNew[:, NSpec - 1 - spec, :] = dispList[:, spec, :]
        del(dispList)
        gc.collect()

        dispList = dispListNew
        SpecExpand = NSpec - 1 - SpecExpand

    # Load Crystal Data
    jList, dxList, jumpNewIndices, superCell, jnet, vacsite, vacsiteInd = Load_crys_Data(CrysDatPath, typ=CrysType)

    if from_scratch:
        print("Generating New cluster expansion with vacancy at {}, {}".format(vacsite.ci, vacsite.R))
        VclusExp = makeVClusExp(superCell, jnet, jList, clustCut, MaxOrder, NSpec, vacsite,
                                AllInteracts=args.AllInteracts)
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
    a0 = np.linalg.norm(dispList[0, NSpec -1, :])/np.linalg.norm(dxList[0])

    print("Training to all jumps.")
    Wbar, Bbar, Gbar, etaBar = Expand(T, state1List, vacsiteInd, N_train, jList, dxList*a0,
                                      AllJumpRates, jumpSelects, dispList, rateList,SpecExpand, MCJit, NVclus,
                                      numVecsInteracts, VecsInteracts, VecGroupInteracts, aj=args.AllJumps)


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

    parser = argparse.ArgumentParser(description="Input parameters for using GCnets", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-T", "--Temp", metavar="eg. 1073/50", type=int,
                        help="Temperature of data set (or composition for the binary alloys.")
    parser.add_argument("-DP", "--DataPath", metavar="/path/to/data", type=str,
                        help="Path to Data file.")
    parser.add_argument("-cr", "--CrysDatPath", metavar="/path/to/crys/dat", type=str,
                        help="Path to crystal Data.")
    parser.add_argument("-ct", "--CrysType", metavar="FCC/BCC", default=None, type=str,
                        help="Type of crystal.")
    parser.add_argument("-mo", "--MaxOrder", metavar="eg. 3", type=int, default=2,
                        help="Maximum sites to consider in a cluster.")
    parser.add_argument("-cc", "--ClustCut", metavar="eg. 2.0", type=float, default=5.0,
                        help="Maximum distance between sites to consider in a cluster.")
    parser.add_argument("-sp", "--SpecExpand", metavar="eg. 0", type=int, default=5,
                        help="Which species to expand.")
    parser.add_argument("-vsp", "--VacSpec", metavar="eg. 0", type=int, default=0,
                        help="Index of vacancy species.")
    parser.add_argument("-nt", "--NTrain", metavar="eg. 10000", type=int, default=10000,
                        help="No. of training samples.")
    parser.add_argument("-aj", "--AllJumps", action="store_true",
                        help="Whether to train on all jumps or train KMC-style.")
    parser.add_argument("-scr", "--Scratch", action="store_true",
                        help="Whether to create new network and start from scratch")
    parser.add_argument("-ait", "--AllInteracts", action="store_true",
                        help="Whether to consider all interactions, or just the ones that contain the jump sites.")
    parser.add_argument("-svc", "--SaveCE", action="store_true",
                        help="Whether to save the cluster expansion.")
    parser.add_argument("-svj", "--SaveJitArrays", action="store_true",
                        help="Whether to store arrays for JIT calculations.")
    parser.add_argument("-d", "--DumpArgs", action="store_true",
                        help="Whether to dump arguments in a file")
    parser.add_argument("-dpf", "--DumpFile", metavar="F", type=str,
                        help="Name of file to dump arguments to (can be the jobID in a cluster for example).")


    args = parser.parse_args()
    if args.DumpArgs:
        print("Dumping arguments to: {}".format(args.DumpFile))
        opts = vars(args)
        with open(RunPath + args.DumpFile, "w") as fl:
            for key, val in opts.items():
                fl.write("{}: {}\n".format(key, val))

    main(args)