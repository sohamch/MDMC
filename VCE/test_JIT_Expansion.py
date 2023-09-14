from onsager import crystal, supercell, cluster
import numpy as np
import h5py
from numba import jit, int64
import Cluster_Expansion
import unittest
import warnings
from tqdm import tqdm

mo=3
vsp = 0

@jit(nopython=True)
def GetOffSite(state, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites):
    """
    :param state: State for which to count off sites of interactions
    :return: OffSiteCount array (N_interaction x 1)
    """
    OffSiteCount = np.zeros(numSitesInteracts.shape[0], dtype=int64)
    for interactIdx in range(numSitesInteracts.shape[0]):
        for intSiteind in range(numSitesInteracts[interactIdx]):
            if state[SupSitesInteracts[interactIdx, intSiteind]] != \
                    SpecOnInteractSites[interactIdx, intSiteind]:
                OffSiteCount[interactIdx] += 1
    return OffSiteCount

class Test_Jit(unittest.TestCase):

    def test_getLambda_symmetry(self):

        initState = self.initState
        NVclus = len(self.VclusExp.vecClus)
        print(NVclus)

        offsc1 = GetOffSite(initState, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)
        lamb1 = self.JitExpander.getLambda(offsc1, NVclus, self.numVecsInteracts, self.VecGroupInteracts,
                                             self.VecsInteracts)

        print("max, min: ", np.max(lamb1), np.min(lamb1))
        # check lamda_1 by computing explicitly

        for g in tqdm(self.VclusExp.sup.crys.G, ncols=65, position=0, leave=True):
            stateG = np.zeros_like(initState)

            for site in range(initState.shape[0]):
                ciSite, Rsite = self.VclusExp.sup.ciR(site)
                Rnew, ciNew = self.VclusExp.sup.crys.g_pos(g, Rsite, ciSite)
                siteIndNew, _ = self.VclusExp.sup.index(Rnew, ciNew)
                stateG[siteIndNew] = initState[site]

            # Now get their off site counts and basis vectors
            offsc2 = GetOffSite(stateG, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)

            lamb2 = self.JitExpander.getLambda(offsc2, NVclus, self.numVecsInteracts, self.VecGroupInteracts,
                                                 self.VecsInteracts)

            # Rotate lamb1 by the group operation
            lamb1G = np.dot(g.cartrot, lamb1.T).T
            self.assertTrue(np.allclose(lamb1G, lamb2))

    def test_getLambda_translation(self):
        initState = self.initState
        R_trans = np.array([4, 2, 1])
        initState_trans = np.zeros_like(initState)

        for siteInd in range(initState.shape[0]):
            ciSite, RSite = self.VclusExp.sup.ciR(siteInd)
            RNew = RSite + R_trans
            siteNew = self.VclusExp.sup.index(RNew, ciSite)
            initState_trans[siteNew] = initState[siteInd]

        NVclus = len(self.VclusExp.vecVec)

        offsc1 = GetOffSite(initState, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)
        lamb1 = self.JitExpander.getLambda(offsc1, NVclus, self.numVecsInteracts, self.VecGroupInteracts,
                                           self.VecsInteracts)

        offsc2 = GetOffSite(initState_trans, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)
        lamb2 = self.JitExpander.getLambda(offsc2, NVclus, self.numVecsInteracts, self.VecGroupInteracts,
                                           self.VecsInteracts)

        self.assertTrue(np.allclose(lamb1, lamb2))

    def test_getDelLamb(self):
        initState = self.initState
        NVclus = len(self.VclusExp.vecClus)
        print(NVclus)

        offsc1 = GetOffSite(initState, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)
        lamb1 = self.JitExpander.getLambda(offsc1, NVclus, self.numVecsInteracts, self.VecGroupInteracts,
                                           self.VecsInteracts)

        siteA = np.random.randint(0, initState.shape[0])
        siteB = np.random.randint(0, initState.shape[0])
        while initState[siteA] == initState[siteB]:
            siteB = np.random.randint(0, initState.shape[0])

        # swap and update
        state2 = initState.copy()
        temp = state2[siteA]
        state2[siteA] = state2[siteB]
        state2[siteB] = temp

        offsc2 = GetOffSite(state2, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)
        lamb2 = self.JitExpander.getLambda(offsc2, NVclus, self.numVecsInteracts, self.VecGroupInteracts,
                                           self.VecsInteracts)

        del_lamb = lamb2 - lamb1

        del_lamb_calc = self.JitExpander.getDelLamb(initState, offsc1, siteA, siteB, NVclus,
                                                      self.numVecsInteracts, self.VecGroupInteracts,
                                                      self.VecsInteracts)

        assert np.allclose(del_lamb, del_lamb_calc)

    def test_expansion(self):
        """
        To test if Wbar and Bbar are computed correctly
        """
        # Compute them with the dicts and match with what comes out from the code.
        # Wbar_test = np.zeros()

        initState = self.initState
        dxList = self.dxList
        jList = self.jList
        lenVecClus = len(self.VclusExp.vecClus)

        # Now, do the expansion
        offscjit = GetOffSite(initState, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)
        state = initState.copy()
        spec = 1

        # self, state, jList, dxList, spec, OffSiteCount,
        # numVecsInteracts, VecGroupInteracts, VecsInteracts,
        # lenVecClus, vacSiteInd, RateList

        RateList = np.random.rand(len(dxList))
        Wbar, Bbar, rates_used =\
            self.JitExpander.Expand(state, jList, dxList, spec, offscjit,
                                 self.numVecsInteracts, self.VecGroupInteracts, self.VecsInteracts,
                                 lenVecClus, self.vacsiteInd, RateList)

        assert np.array_equal(RateList, rates_used)

        print(len(self.numVecsInteracts))

        if len(self.crys.basis[self.chem]) > 1:
            rng = lenVecClus // 4
        else:
            rng = lenVecClus

        print("Bbar max, min: {:.4f} {:.4f}".format(np.max(Bbar[:rng]), np.min(Bbar[:rng])))
        print("Wbar max, min: {:.4f} {:.4f}".format(np.max(Wbar[:rng, :rng]), np.min(Wbar[:rng, :rng])))

        offscjit2 = GetOffSite(initState, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)

        # Check that the offsitecounts have been correctly reverted and state is unchanged.

        self.assertTrue(np.array_equal(offscjit2, offscjit))
        self.assertTrue(np.array_equal(state, initState))

        self.assertTrue(np.array_equal(self.SiteSpecInterArray, self.JitExpander.SiteSpecInterArray))
        self.assertTrue(np.array_equal(self.numInteractsSiteSpec, self.JitExpander.numInteractsSiteSpec))
        self.assertTrue(np.allclose(self.Interaction2En, self.JitExpander.Interaction2En))

        print("Starting LBAM tests")
        # Check the symmetry
        self.assertEqual(Wbar.shape[0], Wbar.shape[1])
        for i in range(Wbar.shape[0]):
            for j in range(i):
                self.assertEqual(Wbar[i, j], Wbar[j, i])

        # Now test the rate expansion by explicitly constructing it
        Wbar_test = np.zeros_like(Wbar, dtype=float)
        Bbar_test = np.zeros_like(Bbar, dtype=float)
        for vs1 in tqdm(range(rng), position=0, leave=True, ncols=65):
            for vs2 in range(rng):
                # Go through all the jumps
                for TInd in range(len(jList)):
                    # For every jump, reset the offsite count
                    rate = RateList[TInd]
                    offscjit = offscjit2.copy()

                    vec1 = np.zeros(3, dtype=float)
                    vec2 = np.zeros(3, dtype=float)

                    siteB = jList[TInd]
                    siteA = self.vacsiteInd
                    # Check that the initial site is always the vacancy
                    specA = state[siteA]  # the vacancy
                    self.assertEqual(specA, self.vacSpec)
                    specB = state[siteB]

                    for interactnum in range(self.numInteractsSiteSpec[siteA, specA]):
                        interactInd = self.SiteSpecInterArray[siteA, specA, interactnum]
                        repClusInd = self.VclusExp.InteractionId2ClusId[interactInd]
                        repClus = self.VclusExp.Num2Clus[repClusInd]
                        repClustSymListInd = self.VclusExp.clust2SpecClus[repClus][0]
                        if self.VclusExp.clus2LenVecClus[repClustSymListInd] == 0:
                            self.assertEqual(self.numVecsInteracts[interactInd], -1)
                            vecList = []
                        else:
                            vecList = self.VclusExp.clust2vecClus[repClus]
                            self.assertEqual(self.numVecsInteracts[interactInd], len(vecList))

                        for i in range(self.numVecsInteracts[interactInd]):
                            self.assertTrue(np.allclose(self.VecsInteracts[interactInd, i, :],
                                                        self.VclusExp.vecVec[vecList[i][0]][vecList[i][1]]))
                            self.assertEqual(self.VecGroupInteracts[interactInd, i], vecList[i][0])

                        if offscjit[interactInd] == 0:
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs1:
                                    vec1 -= self.VecsInteracts[interactInd, tupInd, :]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs2:
                                    vec2 -= self.VecsInteracts[interactInd, tupInd, :]
                        offscjit[interactInd] += 1

                    for interactnum in range(self.numInteractsSiteSpec[siteB, specB]):
                        interactInd = self.SiteSpecInterArray[siteB, specB, interactnum]
                        repClusInd = self.VclusExp.InteractionId2ClusId[interactInd]
                        repClus = self.VclusExp.Num2Clus[repClusInd]
                        repClustSymListInd = self.VclusExp.clust2SpecClus[repClus][0]
                        if self.VclusExp.clus2LenVecClus[repClustSymListInd] == 0:
                            self.assertEqual(self.numVecsInteracts[interactInd], -1)
                            vecList = []
                        else:
                            vecList = self.VclusExp.clust2vecClus[repClus]
                            self.assertEqual(self.numVecsInteracts[interactInd], len(vecList))

                        for i in range(self.numVecsInteracts[interactInd]):
                            self.assertTrue(np.allclose(self.VecsInteracts[interactInd, i, :],
                                                        self.VclusExp.vecVec[vecList[i][0]][vecList[i][1]]))
                            self.assertEqual(self.VecGroupInteracts[interactInd, i], vecList[i][0])

                        if offscjit[interactInd] == 0:
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs1:
                                    vec1 -= self.VecsInteracts[interactInd, tupInd, :]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs2:
                                    vec2 -= self.VecsInteracts[interactInd, tupInd, :]
                        offscjit[interactInd] += 1

                    for interactnum in range(self.numInteractsSiteSpec[siteA, specB]):
                        interactInd = self.SiteSpecInterArray[siteA, specB, interactnum]
                        repClusInd = self.VclusExp.InteractionId2ClusId[interactInd]
                        repClus = self.VclusExp.Num2Clus[repClusInd]
                        repClustSymListInd = self.VclusExp.clust2SpecClus[repClus][0]
                        if self.VclusExp.clus2LenVecClus[repClustSymListInd] == 0:
                            self.assertEqual(self.numVecsInteracts[interactInd], -1)
                            vecList = []
                        else:
                            vecList = self.VclusExp.clust2vecClus[repClus]
                            self.assertEqual(self.numVecsInteracts[interactInd], len(vecList))

                        for i in range(self.numVecsInteracts[interactInd]):
                            self.assertTrue(np.allclose(self.VecsInteracts[interactInd, i, :],
                                                        self.VclusExp.vecVec[vecList[i][0]][vecList[i][1]]))
                            self.assertEqual(self.VecGroupInteracts[interactInd, i], vecList[i][0])

                        offscjit[interactInd] -= 1
                        if offscjit[interactInd] == 0:
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs1:
                                    vec1 += self.VecsInteracts[interactInd, tupInd, :]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs2:
                                    vec2 += self.VecsInteracts[interactInd, tupInd, :]

                    for interactnum in range(self.numInteractsSiteSpec[siteB, specA]):
                        interactInd = self.SiteSpecInterArray[siteB, specA, interactnum]
                        repClusInd = self.VclusExp.InteractionId2ClusId[interactInd]
                        repClus = self.VclusExp.Num2Clus[repClusInd]
                        repClustSymListInd = self.VclusExp.clust2SpecClus[repClus][0]
                        if self.VclusExp.clus2LenVecClus[repClustSymListInd] == 0:
                            self.assertEqual(self.numVecsInteracts[interactInd], -1)
                            vecList = []
                        else:
                            vecList = self.VclusExp.clust2vecClus[repClus]
                            self.assertEqual(self.numVecsInteracts[interactInd], len(vecList))

                        for i in range(self.numVecsInteracts[interactInd]):
                            self.assertTrue(np.allclose(self.VecsInteracts[interactInd, i, :],
                                                        self.VclusExp.vecVec[vecList[i][0]][vecList[i][1]]))
                            self.assertEqual(self.VecGroupInteracts[interactInd, i], vecList[i][0])

                        offscjit[interactInd] -= 1
                        if offscjit[interactInd] == 0:
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs1:
                                    self.assertEqual(self.VecGroupInteracts[interactInd, tupInd], vs1)
                                    vec1 += self.VecsInteracts[interactInd, tupInd, :]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs2:
                                    self.assertEqual(self.VecGroupInteracts[interactInd, tupInd], vs2)
                                    vec2 += self.VecsInteracts[interactInd, tupInd, :]

                    # get the dot product
                    dot = np.dot(vec1, vec2)

                    Wbar_test[vs1, vs2] += rate*dot
                    if vs1 == 0:
                        if spec == specA:
                            Bbar_test[vs2] += rate*np.dot(dxList[TInd], vec2)
                        elif spec == specB:
                            Bbar_test[vs2] += rate * np.dot(-dxList[TInd], vec2)

                self.assertAlmostEqual(Wbar[vs1, vs2], Wbar_test[vs1, vs2], 8,
                                       msg="\n{} {}".format(Wbar[vs1, vs2], Wbar_test[vs1, vs2]))

        self.assertTrue(np.allclose(Bbar[:rng], Bbar_test[:rng], rtol=0, atol=1e-8))

class Test_JIT_FCC(Test_Jit):

    def setUp(self):
        self.Nvac = 1
        self.MaxOrder = mo
        a0 = 1.0
        self.a0 = a0
        self.crys = crystal.Crystal.FCC(a0, chemistry="A")
        self.chem = 0
        with h5py.File("../CrysDat_FCC/CrystData.h5", "r") as fl:
            dxList = np.array(fl["dxList_1nn"])
            NNList = np.array(fl["NNsiteList_sitewise"])

        self.superlatt = 8 * np.eye(3, dtype=int)
        self.superFCC = supercell.ClusterSupercell(self.crys, self.superlatt)
        # get the number of sites in the supercell - should be 8x8x8
        numSites = len(self.superFCC.mobilepos)
        assert numSites == NNList.shape[1]
        self.vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
        self.vacsiteInd = self.superFCC.index(np.zeros(3, dtype=int), (0, 0))[0]
        self.dxList = dxList
        self.jList = NNList[1:, self.vacsiteInd]

        self.clusexp = cluster.makeclusters(self.crys, 1.01 * a0, self.MaxOrder)
        print("No. of site Cluster symmetry groups: ", len(self.clusexp))
        self.NSpec = 3
        self.vacSpec = vsp
        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(self.superFCC, self.clusexp, self.NSpec, self.vacsite,
                                                                 self.vacSpec, self.MaxOrder)
        print("No. of Species Cluster symmetry groups: ", len(self.VclusExp.SpecClusters))
        self.VclusExp.generateSiteSpecInteracts()

        self.VclusExp.genVecClustBasis(self.VclusExp.SpecClusters)
        self.VclusExp.indexVclus2Clus()  # Index vector cluster list to cluster symmetry groups
        self.VclusExp.indexClustertoVecClus()

        self.Energies = np.random.rand(len(self.VclusExp.SpecClusters))

        self.MakeJITs()
        print("Done setting up FCC data.")

    def MakeJITs(self):
        # First, the chemical data
        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, \
        numInteractsSiteSpec, SiteSpecInterArray = self.VclusExp.makeJitInteractionsData(self.Energies)

        self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites, self.Interaction2En, \
        self.numInteractsSiteSpec, self.SiteSpecInterArray= \
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, \
            numInteractsSiteSpec, SiteSpecInterArray

        # Next, the vector basis data
        self.numVecsInteracts, self.VecsInteracts, self.VecGroupInteracts = self.VclusExp.makeJitVectorBasisData()

        Nsites = self.VclusExp.Nsites
        N_units = self.VclusExp.sup.superlatt[0, 0]

        self.Nsites = Nsites
        self.N_units = N_units

        # Make a random state
        initState = np.zeros(len(self.VclusExp.sup.mobilepos), dtype=int)
        # Now assign random species (excluding the vacancy)
        speclabels = np.array([i for i in range(self.NSpec) if i != self.vacSpec])
        for i in range(len(self.VclusExp.sup.mobilepos)):
            initState[i] = speclabels[np.random.randint(0, speclabels.shape[0])]

        # Now put in the vacancy at the vacancy site
        initState[self.vacsiteInd] = self.vacSpec

        self.initState = initState

        self.JitExpander = Cluster_Expansion.JITExpanderClass(
            self.vacSpec, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites,
            Interaction2En, numInteractsSiteSpec, SiteSpecInterArray
        )


class Test_JIT_FCC_orthogonal(Test_Jit):

    def setUp(self):
        self.Nvac = 1
        self.MaxOrder = mo
        a0 = 1.0
        self.a0 = a0
        # load crystal data
        with h5py.File("../CrysDat_FCC/CrystData_ortho_5_cube.h5", "r") as fl:
            lattice = np.array(fl["Lattice_basis_vectors"])
            superlatt = np.array(fl["SuperLatt"])
            basis_cubic = np.array(fl["basis_sites"])
            dxList = np.array(fl["dxList_1nn"])
            NNList = np.array(fl["NNsiteList_sitewise"])

        crys = crystal.Crystal(lattice=lattice, basis=[[b for b in basis_cubic]], chemistry=["A"], noreduce=True)
        self.crys = crys

        self.superlatt = superlatt
        self.superFCC = supercell.ClusterSupercell(crys, superlatt)

        self.chem = 0
        assert len(self.crys.basis[self.chem]) == 4
        self.N_units = 5

        dxVac = np.zeros(3)
        Rvac, civac = self.crys.cart2pos(dxVac)

        self.vacsite = cluster.ClusterSite(civac, Rvac)
        self.vacsiteInd = self.superFCC.index(Rvac, civac)[0]
        print(self.vacsiteInd)
        self.dxList = dxList
        self.jList = NNList[1:, self.vacsiteInd]

        self.clusexp = cluster.makeclusters(self.crys, 1.01 * a0, self.MaxOrder)
        print("No. of site Cluster symmetry groups: ", len(self.clusexp))

        self.NSpec = 3
        self.vacSpec = vsp
        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(self.superFCC, self.clusexp, self.NSpec, self.vacsite,
                                                                 self.vacSpec, self.MaxOrder)
        print("No. of Species Cluster symmetry groups: ",len(self.VclusExp.SpecClusters))
        self.VclusExp.generateSiteSpecInteracts()

        self.VclusExp.genVecClustBasis(self.VclusExp.SpecClusters)
        self.VclusExp.indexVclus2Clus()  # Index vector cluster list to cluster symmetry groups
        self.VclusExp.indexClustertoVecClus()

        self.Energies = np.random.rand(len(self.VclusExp.SpecClusters))

        self.MakeJITs()
        print("Done setting up FCC data.")

    def MakeJITs(self):
        # First, the chemical data
        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, \
        numInteractsSiteSpec, SiteSpecInterArray = self.VclusExp.makeJitInteractionsData(self.Energies)

        self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites, self.Interaction2En, \
        self.numInteractsSiteSpec, self.SiteSpecInterArray = \
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, \
            numInteractsSiteSpec, SiteSpecInterArray

        # Next, the vector basis data
        self.numVecsInteracts, self.VecsInteracts, self.VecGroupInteracts = self.VclusExp.makeJitVectorBasisData()

        Nsites = self.VclusExp.Nsites
        N_units = self.VclusExp.sup.superlatt[0, 0]

        self.Nsites = Nsites
        self.N_units = N_units

        # Make a random state
        initState = np.zeros(len(self.VclusExp.sup.mobilepos), dtype=int)
        # Now assign random species (excluding the vacancy)
        speclabels = np.array([i for i in range(self.NSpec) if i != self.vacSpec])
        for i in range(len(self.VclusExp.sup.mobilepos)):
            initState[i] = speclabels[np.random.randint(0, speclabels.shape[0])]

        # Now put in the vacancy at the vacancy site
        initState[self.vacsiteInd] = self.vacSpec

        self.initState = initState

        self.JitExpander = Cluster_Expansion.JITExpanderClass(
            self.vacSpec, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites,
            Interaction2En, numInteractsSiteSpec, SiteSpecInterArray
        )