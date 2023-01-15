#!/usr/bin/env python
# coding: utf-8

# import sys
# This is the path to the cluster expansion modules
# sys.path.append("/mnt/WorkPartition/Work/Research/UIUC/MDMC/VCE")
# Commenting out - add this path to $PYTHONPATH instead.

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
import time

# Load all the crystal data
def Load_crys_Data(CrysDatPath):
    print("Loading Crystal data at {}".format(CrysDatPath))

    with h5py.File(CrysDatPath, "r") as fl:
        lattice = np.array(fl["Lattice_basis_vectors"])
        superlatt = np.array(fl["SuperLatt"])
        dxList = np.array(fl["dxList_1nn"])
        NNList = np.array(fl["NNsiteList_sitewise"])
        jumpNewIndices = np.array(fl["JumpSiteIndexPermutation"])

    jList = NNList[1:, 0]

    crys = crystal.Crystal(lattice=lattice, basis=[[np.array([0., 0., 0.])]], chemistry=["A"])
    superCell = supercell.ClusterSupercell(crys, superlatt)

    jnet = [[((0, 0), dxList[jInd]) for jInd in range(dxList.shape[0])]]

    vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
    vacsiteInd = superCell.index(np.zeros(3, dtype=int), (0, 0))[0]
    assert vacsiteInd == 0
    return jList, dxList, jumpNewIndices, superCell, jnet, vacsite, vacsiteInd

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
        AllJumpRates = np.array(fl["AllJumpRates_Init"])[perm]
        jmpSelects = np.array(fl["JumpSelects"])[perm]

    return state1List, dispList, rateList, AllJumpRates, jmpSelects


def makeVClusExp(superCell, jnet, jList, clustCut, MaxOrder, NSpec, vacsite, vacSpec, AllInteracts=False):
    TScombShellRange = 1  # upto 1nn combined shell
    TSnnRange = 4
    TScutoff = np.sqrt(2)  # 4th nn cutoff - must be the same as TSnnRange


    print("Creating cluster expansion.")
    crys = superCell.crys
    clusexp = cluster.makeclusters(crys, clustCut, MaxOrder)

    # We'll create a dummy KRA expander anyway since the MC_JIT module is designed to accept transition arrays
    # However, this dummy KEA expander will never get used
    # TODO: Remove KRA Expander or at least make it optional.
    VclusExp = Cluster_Expansion.VectorClusterExpansion(superCell, clusexp, NSpec, vacsite, vacSpec, MaxOrder, TclusExp=True,
                                                    TScutoff=TScutoff, TScombShellRange=TScombShellRange,
                                                    TSnnRange=TSnnRange, jumpnetwork=jnet)

    vacSiteInd, _ = superCell.index(vacsite.R, vacsite.ci)
    reqSites = [vacSiteInd] + list(jList) if not AllInteracts else None
    print("Generating interactions with required sites : {}".format(reqSites))
    start = time.time()
    VclusExp.generateSiteSpecInteracts(reqSites=reqSites)
    end = time.time()
    print("No. of interactions : {}. Time : {:.4f} seconds".format(len(VclusExp.Id2InteractionDict), end-start))
    # Generate the basis vectors for the clusters
    print("Generating and indexing vector basis data.")
    VclusExp.genVecClustBasis(VclusExp.SpecClusters)
    VclusExp.indexVclus2Clus()  # Index vector cluster list to cluster symmetry groups
    VclusExp.indexClustertoVecClus()
    print("No. of vector groups : {}. Time : {:.4f} seconds".format(len(VclusExp.vecClus), end - start))

    return VclusExp

def CreateJitCalculator(VclusExp, NSpec, scratch=True, save=True):
    if scratch:
        # First, we have to generate all the arrays
        # Lattice gas Like -  set all energies to zero
        # All the rates are known to us anyway - they are the ones that are going to get used
        Energies = np.zeros(len(VclusExp.SpecClusters))
        KRAEnergies = [np.zeros(len(KRAClusterDict)) for (key, KRAClusterDict) in 
                VclusExp.KRAexpander.clusterSpeciesJumps.items()]

        KRASpecConstants = np.zeros(NSpec)

        # First, the chemical data
        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites,\
        Interaction2En, numInteractsSiteSpec, SiteSpecInterArray = VclusExp.makeJitInteractionsData(Energies)

        # Next, the vector basis data
        numVecsInteracts, VecsInteracts, VecGroupInteracts = VclusExp.makeJitVectorBasisData()
        
        NVclus = len(VclusExp.vecClus)
        vacSpec = VclusExp.vacSpec
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
                fl.create_dataset("vacSpec", data=np.array([vacSpec], dtype=int))

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
            vacSpec = np.array(fl["vacSpec"])[0]
    
    # Make the MC class
    print("vacancy species: {}".format(vacSpec))
    Interaction2En = np.zeros_like(numSitesInteracts, dtype=float)
    MCJit = MC_JIT.MCSamplerClass(
        vacSpec, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En,
        numInteractsSiteSpec, SiteSpecInterArray,
        numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
        FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups,
        JumpInteracts, Jump2KRAEng, KRASpecConstants
    )
    
    # The vector expansion data are not explicitly part of MCJit, so we'll return them separately
    return MCJit, numVecsInteracts, VecsInteracts, VecGroupInteracts, NVclus


def Expand(T, state1List, vacsiteInd, Nsamples, jSiteList, dxList, AllJumpRates,
           jSelectList, dispSelects, ratesEscape, SpecExpand, MCJit, NVclus,
           numVecsInteracts, VecsInteracts, VecGroupInteracts, aj, rcond=1e-8):


    # Get a dummy TS offsite counts
    TSOffSc = np.zeros(MCJit.numSitesTSInteracts.shape[0], dtype=np.int8)

    # Then we write the expansion loop
    totalW = np.zeros((NVclus, NVclus))
    totalB = np.zeros(NVclus)
    
    assert np.all(state1List[:, vacsiteInd] == MCJit.vacSpec)

    state1ListCpy = state1List.copy()

    print("Calculating rate and velocity (species {0}) expansions with first {1} samples".format(SpecExpand, Nsamples))
    offscTime = 0.
    expandTime = 0.
    for samp in tqdm(range(Nsamples), position=0, leave=True):
    
        # In the cluster expander, the vacancy is the highest labelled species,
        # In our case, it is the lowest
        # So we'll change the numbering so that the vacancy is labelled 5
        state = state1ListCpy[samp]

        start = time.time()
        offsc = MC_JIT.GetOffSite(state, MCJit.numSitesInteracts, MCJit.SupSitesInteracts, MCJit.SpecOnInteractSites)
        end = time.time()
        offscTime += end - start

        start = time.time()
        if aj:
            WBar, bBar, rates_used, _, _ = MCJit.Expand(state, jSiteList, dxList, SpecExpand, offsc,
                                              TSOffSc, numVecsInteracts, VecGroupInteracts, VecsInteracts,
                                              NVclus, 0, vacsiteInd, AllJumpRates[samp])

            assert np.array_equal(state, state1List[samp])  # assert revertions
            assert np.allclose(rates_used, AllJumpRates[samp])

        else:
            jList = np.array([jSiteList[jSelectList[samp]]], dtype=int)
            dxList = np.array([dispSelects[samp, MCJit.vacSpec, :]], dtype=float) # The last one is the vacancy jump
            Rate = np.array([ratesEscape[samp]], dtype=float)
            WBar, bBar, rates_used, _, _ = MCJit.Expand(state, jList, dxList, SpecExpand, offsc,
                                                        TSOffSc, numVecsInteracts, VecGroupInteracts, VecsInteracts,
                                                        NVclus, 0, vacsiteInd, Rate)
            assert np.array_equal(state, state1List[samp])  # assert revertions
            assert np.allclose(rates_used, Rate)

        end = time.time()
        expandTime += end - start

        totalW += WBar
        totalB += bBar

    totalW /= Nsamples
    totalB /= Nsamples

    np.save(RunPath + "Bbar_{}.npy".format(T), totalB)

    np.save(RunPath + "Wbar_{}.npy".format(T), totalW)
    # save eigvals
    vals, _ = np.linalg.eigh(totalW)
    np.save("W_eigs_{}.npy".format(T), vals)

    # Compute relaxation expansion
    etaBar, residues, rank, singVals = spla.lstsq(totalW, -totalB, cond=rcond)
    np.save(RunPath + "etabar_{}.npy".format(T), etaBar)
    np.save("W_singvals_{}.npy".format(T), singVals)

    return totalW, totalB, etaBar, offscTime / Nsamples, expandTime / Nsamples


# Get the Transport coefficients
def Calculate_L(state1List, SpecExpand, VacSpec, rateList, dispList, jumpSelects,
        jList, dxList, vacsiteInd, NVclus, MCJit, etaBar, start, end,
        numVecsInteracts, VecGroupInteracts, VecsInteracts):

    L = 0.
    assert vacsiteInd == 0
    Lsamp = np.zeros(end - start)
    print("Computing Species {0} transport coefficients for samples {1} to {2}".format(SpecExpand, start, end))
    for samp in tqdm(range(start, end), position=0, leave=True):
        state = state1List[samp]
    
        offsc = MC_JIT.GetOffSite(state, MCJit.numSitesInteracts, MCJit.SupSitesInteracts, MCJit.SpecOnInteractSites)
        jSelect = jumpSelects[samp]
        jSite = jList[jSelect]

        assert state[vacsiteInd] == MCJit.vacSpec

        del_lamb = MCJit.getDelLamb(state, offsc, vacsiteInd, jSite, NVclus,
                                    numVecsInteracts, VecGroupInteracts, VecsInteracts)
    
        disp_sp = dispList[samp, SpecExpand, :]

        # Check displacements
        if SpecExpand != VacSpec:
            if state[jSite] == SpecExpand:
                assert np.allclose(disp_sp, -dxList[jSelect]), "{}\n {}\n {}\n".format(dxList, jSelect, disp_sp)
            else:
                assert np.allclose(disp_sp, 0.)

        else:
            assert np.allclose(disp_sp, dxList[jSelect])

    
        # Get the change in y
        del_y = del_lamb.T @ etaBar
    
        # Modify the total displacement
        disp_sp_mod = disp_sp + del_y
        ls = rateList[samp] * np.linalg.norm(disp_sp_mod)**2 /6.0
        L += ls
        Lsamp[samp - start] = ls

    L /= (end-start)

    return L, Lsamp

def main(args):

    # Load Data
    specExpOriginal = args.SpecExpand
    state1List, dispList, rateList, AllJumpRates, jumpSelects = Load_Data(args.DataPath)

    AllSpecs = np.unique(state1List[0])
    NSpec = AllSpecs.shape[0]

    SpecExpand = args.SpecExpand

    # Load Crystal Data
    jList, dxList, jumpNewIndices, superCell, jnet, vacsite, vacsiteInd = Load_crys_Data(args.CrysDatPath)
    
    # Let's find which jump was selected
    a0 = np.linalg.norm(dispList[0, args.VacSpec]) / np.linalg.norm(dxList[0])

    print("Checking displacements and jump indexing.", flush=True)
    print("Computed lattice parameter: {}.".format(a0), flush=True)
    for stateInd in tqdm(range(state1List.shape[0]), position=0, leave=True):
        dxVac = dispList[stateInd, args.VacSpec, :]
        count = 0
        jmpInd = None
        for jInd in range(dxList.shape[0]):
            if np.allclose(dxList[jInd] * a0, dxVac):
                count += 1
                jmpInd = jInd
        assert count == 1
        
        assert np.allclose(dxList[jmpInd] * a0, dxVac)
        
        sp = state1List[stateInd, jList[jmpInd]]
        assert np.allclose(-dxVac, dispList[stateInd, sp, :])
        
        assert jumpSelects[stateInd] == jmpInd

    saveJit = args.SaveJitArrays
    if args.Scratch:
        # (superCell, jList, clustCut, MaxOrder, NSpec, vacsite, AllInteracts=False):
        print("Generating New cluster expansion with vacancy at {}, {}".format(vacsite.ci, vacsite.R))
        VclusExp = makeVClusExp(superCell, jnet, jList, args.ClustCut, args.MaxOrder, NSpec, vacsite,
                                args.VacSpec, AllInteracts=args.AllInteracts)
        if args.SaveCE:
            with open(RunPath+"VclusExp.pkl", "wb") as fl:
                pickle.dump(VclusExp, fl)

    else:
        VclusExp = None
        saveJit = False

    # Make MCJIT
    MCJit, numVecsInteracts, VecsInteracts, VecGroupInteracts, NVclus = CreateJitCalculator(VclusExp, NSpec,
                                                                                            scratch=args.Scratch,
                                                                                            save=saveJit)
    if args.Scratch and saveJit and args.ArrayOnly:
        # If we only want to save the Jit arrays (so that later jobs can be run in parallel)
        print("Created arrays. Terminating.")
        return
    
    if not args.NoExpand:
        print("Expanding.")
        Wbar, Bbar, etaBar, offscTime, expandTime = Expand(args.Temp, state1List, vacsiteInd, args.NTrain, jList, dxList*a0,
                                          AllJumpRates, jumpSelects, dispList, rateList, SpecExpand, MCJit, NVclus,
                                          numVecsInteracts, VecsInteracts, VecGroupInteracts, args.AllJumps, args.rcond)

        print("Off site counting time per sample: {}".format(offscTime))
        print("Expansion time per sample: {}".format(expandTime))


        if args.ExpandOnly:
            # If we only want to save the Jit arrays (so that later jobs can be run in parallel)
            print("Expansion complete. Terminating.")
            return

    # Calculate transport coefficients
    print("Computing Transport coefficients")
    etaBar = np.load(RunPath + "etabar_{}.npy".format(args.Temp))
    L_train, L_train_samples = Calculate_L(state1List, SpecExpand, args.VacSpec, rateList,
            dispList, jumpSelects, jList, dxList*a0,
            vacsiteInd, NVclus, MCJit, 
            etaBar, 0, args.NTrain,
            numVecsInteracts, VecGroupInteracts, VecsInteracts)

    L_val, L_val_samples = Calculate_L(state1List, SpecExpand, args.VacSpec, rateList,
            dispList, jumpSelects, jList, dxList*a0,
            vacsiteInd, NVclus, MCJit, 
            etaBar, args.NTrain, state1List.shape[0],
            numVecsInteracts, VecGroupInteracts, VecsInteracts)

    np.save("L{0}{0}_{1}.npy".format(specExpOriginal, args.Temp), np.array([L_train, L_val]))
    np.save("L_trSamps_{0}{0}_{1}.npy".format(specExpOriginal, args.Temp), L_train_samples)
    np.save("L_valSamps_{0}{0}_{1}.npy".format(specExpOriginal, args.Temp), L_val_samples)

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
    
    parser.add_argument("-mo", "--MaxOrder", metavar="eg. 3", type=int, default=None,
                        help="Maximum sites to consider in a cluster.")
    
    parser.add_argument("-cc", "--ClustCut", metavar="eg. 2.0", type=float, default=None,
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
    
    parser.add_argument("-ao", "--ArrayOnly", action="store_true",
                        help="Use the run to only generate the Jit arrays - no transport calculation.")
    
    parser.add_argument("-eo", "--ExpandOnly", action="store_true",
                        help="Use the run to only generate the Jit arrays - no transport calculation.")
    
    parser.add_argument("-rc", "--rcond", type=float, default=1e-8,
                        help="Threshold for zero singular values.")
    
    parser.add_argument("-nex", "--NoExpand", action="store_true",
                        help="Use the run to only generate the Jit arrays - no transport calculation.")

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
