from onsager import crystal, supercell, cluster
import numpy as np
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

                assert ciNew == ciSite == (0, 0)
                siteIndNew, _ = self.VclusExp.sup.index(Rnew, (0, 0))
                stateG[siteIndNew] = initState[site]

            # Now get their off site counts and basis vectors
            offsc2 = GetOffSite(stateG, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)

            lamb2 = self.JitExpander.getLambda(offsc2, NVclus, self.numVecsInteracts, self.VecGroupInteracts,
                                                 self.VecsInteracts)

            # Rotate lamb1 by the group operation
            lamb1G = np.dot(g.cartrot, lamb1.T).T
            self.assertTrue(np.allclose(lamb1G, lamb2))

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
        state2[siteB] = state2[siteA]
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

        MCSampler_Jit = self.JitExpander

        TransOffSiteCount = np.zeros(len(self.TSInteractSites), dtype=int)

        # Build TS offsites
        for TsInteractIdx in range(len(TransOffSiteCount)):
            for Siteind in range(self.numSitesTSInteracts[TsInteractIdx]):
                if initState[self.TSInteractSites[TsInteractIdx, Siteind]] != self.TSInteractSpecs[TsInteractIdx, Siteind]:
                    TransOffSiteCount[TsInteractIdx] += 1

        ijList = np.array(self.VclusExp.KRAexpander.jList)
        dxList = np.array(self.VclusExp.KRAexpander.dxList)
        lenVecClus = len(self.VclusExp.vecClus)

        # Now, do the expansion
        offscjit = GetOffSite(initState, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)
        state = initState.copy()
        spec = 1
        beta = 1.0
        Wbar, Bbar, rates_used, delEJumps, delEKRAJumps =\
            MCSampler_Jit.Expand(state, ijList, dxList, spec, offscjit, TransOffSiteCount,
                                 self.numVecsInteracts, self.VecGroupInteracts, self.VecsInteracts,
                                 lenVecClus, beta, self.vacsiteInd, None)

        np.save("Wbar_unit_test.npy", Wbar)
        np.save("Bbar_unit_test.npy", Bbar)

        offscjit2 = GetOffSite(initState, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)

        # Check that the offsitecounts have been correctly reverted and state is unchanged.

        self.assertTrue(np.array_equal(offscjit2, offscjit))
        self.assertTrue(np.array_equal(state, initState))

        self.assertTrue(np.array_equal(self.SiteSpecInterArray, MCSampler_Jit.SiteSpecInterArray))
        self.assertTrue(np.array_equal(self.numInteractsSiteSpec, MCSampler_Jit.numInteractsSiteSpec))
        self.assertTrue(np.allclose(self.Interaction2En, MCSampler_Jit.Interaction2En))

        print("Starting LBAM tests")
        Wbar_test = np.zeros_like(Wbar, dtype=float)
        Bbar_test = np.zeros_like(Bbar, dtype=float)

        # Now test the rate expansion by explicitly constructing it
        for vs1 in tqdm(range(len(self.VclusExp.vecVec)), position=0, leave=True):
            for vs2 in range(len(self.VclusExp.vecVec)):
                # Go through all the jumps
                for TInd in range(len(ijList)):
                    # For every jump, reset the offsite count
                    offscjit = offscjit2.copy()
                    TSOffCount = TransOffSiteCount.copy()
                    vec1 = np.zeros(3, dtype=float)
                    vec2 = np.zeros(3, dtype=float)

                    siteB = ijList[TInd]
                    siteA = self.vacsiteInd
                    # Check that the initial site is always the vacancy
                    specA = state[siteA]  # the vacancy
                    self.assertEqual(specA, self.vacSpec)
                    specB = state[siteB]
                    delEKRA = self.KRASpecConstants[specB]
                    # get the index of this transition
                    jumpInd = self.FinSiteFinSpecJumpInd[siteB, specB]
                    # get the KRA energy for this jump in this state
                    for ptgrpInd in range(self.numJumpPointGroups[jumpInd]):
                        for ptGpInteractInd in range(self.numTSInteractsInPtGroups[jumpInd, ptgrpInd]):
                            # See if the interaction is on
                            offcount = TSOffCount[self.JumpInteracts[jumpInd, ptgrpInd, ptGpInteractInd]]
                            if offcount == 0:
                                delEKRA += self.Jump2KRAEng[jumpInd, ptgrpInd, ptGpInteractInd]

                    # Now do the site swaps and calculate the energy
                    delE = 0.0

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
                            delE -= self.Interaction2En[interactInd]
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
                            delE -= self.Interaction2En[interactInd]
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
                            delE += self.Interaction2En[interactInd]
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
                            delE += self.Interaction2En[interactInd]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs1:
                                    self.assertEqual(self.VecGroupInteracts[interactInd, tupInd], vs1)
                                    vec1 += self.VecsInteracts[interactInd, tupInd, :]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs2:
                                    self.assertEqual(self.VecGroupInteracts[interactInd, tupInd], vs2)
                                    vec2 += self.VecsInteracts[interactInd, tupInd, :]
                    # get the rate
                    rate = np.exp(-(0.5 * delE + delEKRA))
                    # test the rate
                    self.assertAlmostEqual(delEKRA, delEKRAJumps[TInd], 8,
                                           msg="{} {} {}".format(delEKRA, delEKRAJumps[TInd], TInd))
                    self.assertAlmostEqual(delE, delEJumps[TInd], 8,
                                           msg="{} {} {}".format(delE, delEJumps[TInd], TInd))
                    self.assertAlmostEqual(rate, rates_used[TInd], 8)

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

        self.assertTrue(np.allclose(Bbar, Bbar_test))
        print("Bbar max, min: {:.4f} {:.4f}".format(np.max(Bbar), np.min(Bbar)))
        print("Wbar max, min: {:.4f} {:.4f}".format(np.max(Wbar), np.min(Wbar)))

class Test_JIT_FCC(Test_Jit):

    def setUp(self):
        self.Nvac = 1
        self.MaxOrder = mo
        a0 = 1.0
        self.a0 = a0
        self.crys = crystal.Crystal.FCC(a0, chemistry="A")
        self.chem = 0
        self.superlatt = 8 * np.eye(3, dtype=int)
        self.superFCC = supercell.ClusterSupercell(self.crys, self.superlatt)
        # get the number of sites in the supercell - should be 8x8x8
        numSites = len(self.superFCC.mobilepos)
        self.vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
        self.vacsiteInd = self.superFCC.index(np.zeros(3, dtype=int), (0, 0))[0]
        self.clusexp = cluster.makeclusters(self.crys, 1.01 * a0, self.MaxOrder)
        self.NSpec = 3
        self.vacSpec = vsp
        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(self.superFCC, self.clusexp, self.NSpec, self.vacsite,
                                                                 self.vacSpec, self.MaxOrder)

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

        self.MCSampler_Jit = Cluster_Expansion.JITExpanderClass(
            self.vacSpec, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites,
            Interaction2En, numInteractsSiteSpec, SiteSpecInterArray
        )


class Test_JIT_FCC_orthogonal(Test_Jit):

    def setUp(self):
        self.Nvac = 1
        self.MaxOrder = mo
        a0 = 1.0
        self.a0 = a0
        basis_cube_fcc = [np.array([0, 0, 0]), np.array([0, 0.5, 0.5]), np.array([0.5, 0., 0.5]),
                          np.array([0.5, 0.5, 0.])]
        self.crys = crystal.Crystal(lattice=np.eye(3) * a0, basis=[basis_cube_fcc], chemistry=["A"], noreduce=True)

        self.chem = 0
        assert len(self.crys.basis[self.chem]) == 4

        self.N_units = 5
        self.superlatt = self.N_units * np.eye(3, dtype=int)
        self.superFCC = supercell.ClusterSupercell(self.crys, self.superlatt)
        self.vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
        self.vacsiteInd = self.superFCC.index(np.zeros(3, dtype=int), (0, 0))[0]
        self.clusexp = cluster.makeclusters(self.crys, 1.01 * a0, self.MaxOrder)
        self.NSpec = 3
        self.vacSpec = vsp
        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(self.superFCC, self.clusexp, self.NSpec, self.vacsite,
                                                                 self.vacSpec, self.MaxOrder)

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

        self.MCSampler_Jit = Cluster_Expansion.JITExpanderClass(
            self.vacSpec, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites,
            Interaction2En, numInteractsSiteSpec, SiteSpecInterArray
        )