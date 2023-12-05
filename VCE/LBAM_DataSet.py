import os
RunPath = os.getcwd() + "/"

from onsager import crystal, supercell, cluster
import numpy as np
import scipy.linalg as spla
import Cluster_Expansion
import pickle
import h5py
from tqdm import tqdm
import argparse
from numba import jit, int64
import time

@jit(nopython=True)
def GetOffSite(state, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites):
    """
    :param state: State for which to count off sites of interactions
    :return: OffSiteCount array (N_interaction x 1)
    """
    OffSiteCount = np.zeros(numSitesInteracts.shape[0], dtype=int64)
    for interactIdx in range(numSitesInteracts.shape[0]):
        for intSiteind in range(numSitesInteracts[interactIdx]):
            if state[SupSitesInteracts[interactIdx, intSiteind]] != SpecOnInteractSites[interactIdx, intSiteind]:
                OffSiteCount[interactIdx] += 1
    return OffSiteCount

def make_siteMap_non_prim_to_prim(superCell, superCell_primitve):

    crys = superCell.crys
    crys_primitve = superCell_primitve.crys
    assert len(superCell_primitve.mobilepos) == len(superCell.mobilepos)
    Nsites = len(superCell.mobilepos)

    siteMap_nonPrimitive_to_primitive = np.zeros(Nsites, dtype=int)
    primtive_site_encountered = set([])
    for siteInd in range(Nsites):
        # First, determine the cartesian position from the non-primitive lattice
        ciSite, Rsite = superCell.ciR(siteInd)
        xSite = crys.pos2cart(Rsite, ciSite)

        # Now get the lattice position with the primitive cell
        Rsite_prim, ciSite_prim = crys_primitve.cart2pos(xSite)
        siteInd_primitive = superCell_primitve.index(Rsite_prim, ciSite_prim)

        # check that there is no collision
        assert siteInd_primitive not in primtive_site_encountered
        primtive_site_encountered.add(siteInd_primitive)

        # Then store it in the indexing array
        siteMap_nonPrimitive_to_primitive[siteInd_primitive] = siteInd

    return siteMap_nonPrimitive_to_primitive

# Load all the crystal data
def Load_crys_Data(CrysDatPath, ReduceToPrimitve=False):
    print("Loading Crystal data at {}".format(CrysDatPath))

    with h5py.File(CrysDatPath, "r") as fl:
        lattice = np.array(fl["Lattice_basis_vectors"])
        superlatt = np.array(fl["SuperLatt"])

        try:
            basis_sites = np.array(fl["basis_sites"])
            basis = [[b for b in basis_sites]]
        except KeyError:
            basis = [[np.array([0., 0., 0.])]]

        dxList = np.array(fl["dxList_1nn"])
        NNList = np.array(fl["NNsiteList_sitewise"])

    crys = crystal.Crystal(lattice=lattice, basis=basis, chemistry=["A"], noreduce=True)
    superCell = supercell.ClusterSupercell(crys, superlatt)

    if ReduceToPrimitve:
        print("Reducing crystal to primitive form.")
        crys_primitve = crystal.Crystal(lattice=lattice, basis=basis, chemistry=["A"], noreduce=False)

        superlatt_transf = np.dot(np.linalg.inv(crys_primitve.lattice), superlatt).round(decimals=0)
        superCell_primitve = supercell.ClusterSupercell(crys_primitve, superlatt_transf.astype(int))

        # Make a mapping of the site indices
        siteMap_nonPrimitive_to_primitive = make_siteMap_non_prim_to_prim(superCell, superCell_primitve)

        superCell = superCell_primitve

    else:
        siteMap_nonPrimitive_to_primitive = None

    dxVac = np.zeros(3)
    Rvac, civac = superCell.crys.cart2pos(dxVac)
    vacsite = cluster.ClusterSite(ci=civac, R=Rvac)
    vacsiteInd = superCell.index(Rvac, civac)[0]
    assert vacsiteInd == 0

    # construct the jump neighbor list
    jList = np.zeros(dxList.shape[0], dtype=int)
    for jumpInd in range(dxList.shape[0]):
        dx = dxList[jumpInd]
        Rsite, ciSite = superCell.crys.cart2pos(dx)
        siteInd = superCell.index(Rsite, ciSite)[0]
        jList[jumpInd] = siteInd

    if not ReduceToPrimitve:
        assert np.all(jList == NNList[1:, vacsiteInd])

    print("Working Crystal:")
    print(superCell.crys)
    print()

    return jList, dxList, superCell, vacsite, vacsiteInd, siteMap_nonPrimitive_to_primitive

def Load_Data(DataPath, siteMap_nonPrimitive_to_primitive):
    with h5py.File(DataPath, "r") as fl:
        state1List = np.array(fl["InitStates"])
        dispList = np.array(fl["SpecDisps"])
        rateList = np.array(fl["rates"])
        AllJumpRates = np.array(fl["AllJumpRates_Init"])
        jmpSelects = np.array(fl["JumpSelects"])

        if siteMap_nonPrimitive_to_primitive is not None:
            print("Re-indexing sites from non-primitive to primitive lattice")
            for stateInd in range(state1List.shape[0]):
                state = state1List[stateInd].copy()
                state1List[stateInd, :] = state[siteMap_nonPrimitive_to_primitive][:]

    return state1List, dispList, rateList, AllJumpRates, jmpSelects


def makeVClusExp(superCell, jList, clustCut, MaxOrder, NSpec, vacsite, vacSpec, AllInteracts=False):

    print("Creating cluster expansion.")
    crys = superCell.crys
    clusexp = cluster.makeclusters(crys, clustCut, MaxOrder)

    # sup, clusexp, NSpec, vacSite, vacSpec, maxorder, Nvac = 1, chemExpand = 0, MadeSpecClusts = None, zeroClusts = True
    VclusExp = Cluster_Expansion.VectorClusterExpansion(superCell, clusexp, NSpec, vacsite, vacSpec, MaxOrder)

    print("Vacancy species : {}".format(VclusExp.vacSpec))
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

        # First, the chemical data
        # numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, \
        # numInteractsSiteSpec, SiteSpecInterArray

        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites,\
        Interaction2En, numInteractsSiteSpec, SiteSpecInterArray = VclusExp.makeJitInteractionsData(Energies)

        # Next, the vector basis data
        numVecsInteracts, VecsInteracts, VecGroupInteracts = VclusExp.makeJitVectorBasisData()
        
        NVclus = len(VclusExp.vecClus)
        vacSpec = VclusExp.vacSpec
        # Note : The KRA expansion works only for binary alloys
        # Right now we don't need them, since we already know the rates
        # However, we create a dummy one since the JIT MC calculator requires the arrays
    
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

            NVclus = np.array(fl["NVclus"])[0]
            vacSpec = np.array(fl["vacSpec"])[0]
    
    # Make the MC class
    print("vacancy species: {}".format(vacSpec))
    Interaction2En = np.zeros_like(numSitesInteracts, dtype=float)
    # vacSpec, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites,
    # Interaction2En, numInteractsSiteSpec, SiteSpecInterArray
    JitExpander = Cluster_Expansion.JITExpanderClass(
        vacSpec, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En,
        numInteractsSiteSpec, SiteSpecInterArray
    )
    
    # The vector expansion data are not explicitly part of JitExpander, so we'll return them separately
    return JitExpander, numVecsInteracts, VecsInteracts, VecGroupInteracts, NVclus


def Expand(T, state1List, vacsiteInd, Nsamples, jSiteList, dxList, AllJumpRates,
           jSelectList, dispSelects, ratesEscape, SpecExpand, JitExpander, NVclus,
           numVecsInteracts, VecsInteracts, VecGroupInteracts, aj, rcond=1e-8):

    # Then we write the expansion loop
    totalW = np.zeros((NVclus, NVclus))
    totalB = np.zeros(NVclus)
    
    assert np.all(state1List[:, vacsiteInd] == JitExpander.vacSpec)

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
        offsc = GetOffSite(state, JitExpander.numSitesInteracts, JitExpander.SupSitesInteracts, JitExpander.SpecOnInteractSites)
        end = time.time()
        offscTime += end - start

        start = time.time()
        # Expand(self, state, jList, dxList, spec, OffSiteCount,
        #        numVecsInteracts, VecGroupInteracts, VecsInteracts,
        #        lenVecClus, vacSiteInd, RateList)
        if aj:
            WBar, bBar, rates_used = JitExpander.Expand(state, jSiteList, dxList, SpecExpand, offsc,
                                              numVecsInteracts, VecGroupInteracts, VecsInteracts,
                                              NVclus, vacsiteInd, AllJumpRates[samp])

            assert np.array_equal(state, state1List[samp])  # assert revertions
            assert np.allclose(rates_used, AllJumpRates[samp])

        else:
            jList = np.array([jSiteList[jSelectList[samp]]], dtype=int)
            dxList = np.array([dispSelects[samp, JitExpander.vacSpec, :]], dtype=float)
            Rate = np.array([ratesEscape[samp]], dtype=float)
            # (self, state, jList, dxList, spec, OffSiteCount,
            #  numVecsInteracts, VecGroupInteracts, VecsInteracts,
            #  lenVecClus, vacSiteInd, RateList):

            WBar, bBar, rates_used = JitExpander.Expand(state, jList, dxList, SpecExpand, offsc,
                                                        numVecsInteracts, VecGroupInteracts, VecsInteracts,
                                                        NVclus, vacsiteInd, Rate)
            assert np.array_equal(state, state1List[samp])  # assert revertions
            assert np.allclose(rates_used, Rate)

        end = time.time()
        expandTime += end - start

        totalW += WBar
        totalB += bBar

    totalW /= Nsamples
    totalB /= Nsamples

    # Compute relaxation expansion
    etaBar, residues, rank, singVals = spla.lstsq(totalW, -totalB, cond=rcond)
    np.save(RunPath + "etabar_{}.npy".format(T), etaBar)
    np.save("W_singvals_{}.npy".format(T), singVals)

    return totalW, totalB, etaBar, offscTime / Nsamples, expandTime / Nsamples

# Get the Transport coefficients
def Calculate_L(state1List, SpecExpand, VacSpec, rateList, dispList, jumpSelects,
        jList, dxList, vacsiteInd, NVclus, JitExpander, etaBar, start, end,
        numVecsInteracts, VecGroupInteracts, VecsInteracts):

    L = 0.
    assert vacsiteInd == 0
    Lsamp = np.zeros(end - start)
    print("Computing Species {0} transport coefficients for samples {1} to {2}".format(SpecExpand, start, end))
    for samp in tqdm(range(start, end), position=0, leave=True):
        state = state1List[samp]
    
        offsc = GetOffSite(state, JitExpander.numSitesInteracts, JitExpander.SupSitesInteracts, JitExpander.SpecOnInteractSites)
        jSelect = jumpSelects[samp]
        jSite = jList[jSelect]

        assert state[vacsiteInd] == JitExpander.vacSpec

        del_lamb = JitExpander.getDelLamb(state, offsc, vacsiteInd, jSite, NVclus,
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

    # Load Crystal Data
    jList, dxList, superCell, vacsite, vacsiteInd, siteMap_nonPrimitive_to_primitive = \
        Load_crys_Data(args.CrysDatPath, ReduceToPrimitve=args.ReduceToPrimitve)

    # Load Data
    specExpOriginal = args.SpecExpand
    state1List, dispList, rateList, AllJumpRates, jumpSelects =\
        Load_Data(args.DataPath, siteMap_nonPrimitive_to_primitive)

    AllSpecs = np.unique(state1List[0])
    NSpec = AllSpecs.shape[0]

    SpecExpand = args.SpecExpand
    
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
        assert state1List[stateInd, vacsiteInd] == args.VacSpec
        assert np.allclose(-dxVac, dispList[stateInd, sp, :])
        
        assert jumpSelects[stateInd] == jmpInd

    saveJit = args.SaveJitArrays
    if args.Scratch:
        # (superCell, jList, clustCut, MaxOrder, NSpec, vacsite, AllInteracts=False):
        print("Generating New cluster expansion with vacancy at {}, {}".format(vacsite.ci, vacsite.R))
        VclusExp = makeVClusExp(superCell, jList, args.ClustCut, args.MaxOrder, NSpec, vacsite,
                                args.VacSpec, AllInteracts=args.AllInteracts)
        if args.SaveCE:
            with open(RunPath+"VclusExp.pkl", "wb") as fl:
                pickle.dump(VclusExp, fl)

    else:
        VclusExp = None
        saveJit = False

    # Make MCJIT
    JitExpander, numVecsInteracts, VecsInteracts, VecGroupInteracts, NVclus = CreateJitCalculator(VclusExp, NSpec,
                                                                                            scratch=args.Scratch,
                                                                                            save=saveJit)
    if args.Scratch and saveJit and args.ArrayOnly:
        # If we only want to save the Jit arrays (so that later jobs can be run in parallel)
        print("Created arrays. Terminating.")
        return
    
    if not args.NoExpand:
        print("Expanding.")
        Wbar, Bbar, etaBar, offscTime, expandTime = Expand(args.Temp, state1List, vacsiteInd, args.NTrain, jList, dxList*a0,
                                          AllJumpRates, jumpSelects, dispList, rateList, SpecExpand, JitExpander, NVclus,
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
            vacsiteInd, NVclus, JitExpander,
            etaBar, 0, args.NTrain,
            numVecsInteracts, VecGroupInteracts, VecsInteracts)

    L_val, L_val_samples = Calculate_L(state1List, SpecExpand, args.VacSpec, rateList,
            dispList, jumpSelects, jList, dxList*a0,
            vacsiteInd, NVclus, JitExpander,
            etaBar, args.NTrain, state1List.shape[0],
            numVecsInteracts, VecGroupInteracts, VecsInteracts)

    np.save("L{0}{0}_{1}.npy".format(specExpOriginal, args.Temp), np.array([L_train, L_val]))
    np.save("L_trSamps_{0}{0}_{1}.npy".format(specExpOriginal, args.Temp), L_train_samples)
    np.save("L_valSamps_{0}{0}_{1}.npy".format(specExpOriginal, args.Temp), L_val_samples)

    print("All Done \n\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Input parameters for cluster expansion transport coefficient prediction",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-T", "--Temp", metavar="int", type=int,
                        help="Temperature of data set (or composition for the binary alloys.")
    
    parser.add_argument("-DP", "--DataPath", metavar="/path/to/data", type=str,
                        help="Path to Data file.")
    
    parser.add_argument("-cr", "--CrysDatPath", metavar="/path/to/crys/dat", type=str,
                        help="Path to crystal Data.")

    parser.add_argument("-red", "--ReduceToPrimitve", action="store_true",
                        help="Whether to reduce the crystal from the crystal data file to a primitive crystal.")

    parser.add_argument("-mo", "--MaxOrder", metavar="int", type=int, default=None,
                        help="Maximum sites to consider in a cluster.")
    
    parser.add_argument("-cc", "--ClustCut", metavar="float", type=float, default=None,
                        help="Maximum distance between sites to consider in a cluster in lattice parameter units.")
    
    parser.add_argument("-sp", "--SpecExpand", metavar="int", type=int, default=5,
                        help="Which species to expand.")
    
    parser.add_argument("-vsp", "--VacSpec", metavar="int", type=int, default=0,
                        help="Index of vacancy species.")
    
    parser.add_argument("-nt", "--NTrain", metavar="int", type=int, default=10000,
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
                        help="Use the run to only generate and save rate and velocity expansions to compute transport calculations"
                             "- no transport calculation. These expansions can then be loaded and reused later on if needed.")
    
    parser.add_argument("-rc", "--rcond", type=float, default=1e-8,
                        help="Threshold for zero singular values.")
    
    parser.add_argument("-nex", "--NoExpand", action="store_true",
                        help="Use the run to use saved rate and velocity expansions to compute transport calculations\n."
                        "without having to generate them again.")

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
