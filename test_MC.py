from onsager import crystal, supercell, cluster
import numpy as np
import itertools
import Transitions
import Cluster_Expansion
import MC_JIT
import unittest
import time

class Test_MC_Arrays(unittest.TestCase):

    def setUp(self):
        self.NSpec = 4
        self.MaxOrder = 3
        self.MaxOrderTrans = 3
        self.crys = crystal.Crystal.BCC(0.2836, chemistry="A")
        self.jnetBCC = self.crys.jumpnetwork(0, 0.26)
        self.superlatt = 8 * np.eye(3, dtype=int)
        self.superBCC = supercell.ClusterSupercell(self.crys, self.superlatt)
        # get the number of sites in the supercell - should be 8x8x8
        numSites = len(self.superBCC.mobilepos)
        self.vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
        self.vacsiteInd = self.superBCC.index(np.zeros(3, dtype=int), (0, 0))[0]
        self.mobOccs = np.zeros((self.NSpec, numSites), dtype=int)
        for site in range(1, numSites):
            spec = np.random.randint(0, self.NSpec-1)
            self.mobOccs[spec][site] = 1
        self.mobOccs[-1, self.vacsiteInd] = 1
        self.mobCountList = [np.sum(self.mobOccs[i]) for i in range(self.NSpec)]
        self.clusexp = cluster.makeclusters(self.crys, 0.284, self.MaxOrder)
        self.Tclusexp = cluster.makeclusters(self.crys, 0.29, self.MaxOrderTrans)
        self.KRAexpander = Transitions.KRAExpand(self.superBCC, 0, self.jnetBCC, self.Tclusexp, self.Tclusexp, self.mobCountList,
                                                 self.vacsite)
        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(self.superBCC, self.clusexp, self.Tclusexp, self.jnetBCC,
                                                                 self.mobCountList, self.vacsite, self.MaxOrder,
                                                                 self.MaxOrderTrans)

        self.Energies = np.random.rand(len(self.VclusExp.SpecClusters))
        self.KRAEnergies = [np.random.rand(len(val)) for (key, val) in self.VclusExp.KRAexpander.clusterSpeciesJumps.items()]
        print("Done setting up")

    def test_arrays(self):

        start = time.time()
        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts, VecsInteracts, \
        VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray, vacSiteInd, InteractionIndexDict, InteractionRepClusDict, \
        Index2InteractionDict, repClustCounter\
            = self.VclusExp.makeJitInteractionsData(self.Energies, self.KRAEnergies)

        # Check that each cluster has been translated as many times as there are sites in the supercell
        # Only then we have constructed every possible interaction
        print("Done creating arrays : {}".format(time.time() - start))
        # Now, we first test the interaction arrays - the ones to be used in the MC sweeps
        for repClust, count in repClustCounter.items():
            self.assertEqual(count, len(self.VclusExp.sup.mobilepos))
        # numSitesInteracts - the number of sites in an interaction
        for i in range(len(numSitesInteracts)):
            siteCountInArray = numSitesInteracts[i]
            # get the interaction
            interaction = Index2InteractionDict[i]
            # get the stored index for this interaction
            i_stored = InteractionIndexDict[interaction]
            self.assertEqual(i, i_stored)
            self.assertEqual(len(interaction), siteCountInArray)

        # Now, check the supercell sites and species stored for these interactions
        self.assertEqual(len(SupSitesInteracts), len(numSitesInteracts))
        self.assertEqual(len(SpecOnInteractSites), len(numSitesInteracts))
        for i in range(len(numSitesInteracts)):
            siteSpecSet = set([(SupSitesInteracts[i, j], SpecOnInteractSites[i, j])
                               for j in range(numSitesInteracts[i])])
            interaction = Index2InteractionDict[i]
            interactionSet = set(interaction)
            self.assertEqual(siteSpecSet, interactionSet)

        # Now, test the vector basis and energy information for the clusters
        for i in range(len(numSitesInteracts)):
            # get the interaction
            interaction = Index2InteractionDict[i]
            # Now, get the representative cluster
            repClus = InteractionRepClusDict[interaction]

            # test the energy index
            enIndex = self.VclusExp.clust2SpecClus[repClus][0]
            self.assertEqual(Interaction2En[i], self.Energies[enIndex])

            # get the vector basis info for this cluster
            vecList = self.VclusExp.clust2vecClus[repClus]
            # check the number of vectors
            self.assertEqual(numVecsInteracts[i], len(vecList))
            # check that the correct vector have been stored, in the same order as in vecList (not necessary but short testing)
            for vecind in range(len(vecList)):
                vec = self.VclusExp.vecVec[vecList[vecind][0]][vecList[vecind][1]]
                self.assertTrue(np.allclose(vec, VecsInteracts[i, vecind, :]))

        # Next, test the interactions each (site, spec) is a part of
        self.assertEqual(numInteractsSiteSpec.shape[0], len(self.superBCC.mobilepos))
        self.assertEqual(numInteractsSiteSpec.shape[1], self.NSpec)
        for site in range(len(self.superBCC.mobilepos)):
            for spec in range(self.NSpec):
                numInteractStored = numInteractsSiteSpec[site, spec]
                # get the actual count
                ci, R = self.VclusExp.sup.ciR(site)
                clsite = cluster.ClusterSite(ci=ci, R=R)
                self.assertEqual(len(self.VclusExp.SiteSpecInteractions[(clsite, spec)]), numInteractStored)
                for IdxOfInteract in range(numInteractStored):
                    interactMainIndex = SiteSpecInterArray[site, spec, IdxOfInteract]
                    self.assertEqual(Index2InteractionDict[interactMainIndex],
                                     self.VclusExp.SiteSpecInteractions[(clsite, spec)][IdxOfInteract][0])

    def testTransArrays(self):
        # Now, we start testing the jump arrays
        # jumpFinSites, jumpFinSpec, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng

        TsInteractIndexDict, Index2TSinteractDict, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec, \
        FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng =\
            self.VclusExp.KRAexpander.makeTransJitData(self.KRAEnergies)

        for jumpInd, (jumpkey, TSptGrps) in zip(itertools.count(), self.VclusExp.KRAexpander.clusterSpeciesJumps.items()):
            FinSite = jumpkey[1]
            FinSpec = jumpkey[2]
            # Check that the correct initial and final states have been stored
            self.assertEqual(jumpFinSites[jumpInd], FinSite)
            self.assertEqual(jumpFinSpec[jumpInd], FinSpec)

            # Check that the correct number of point groups are stored
            NptGrp = len(TSptGrps)
            self.assertEqual(numJumpPointGroups[jumpInd], NptGrp)

            # Check that in for each point group, the correct interactions are stored.
            for TsPtGpInd, (spectup, TSinteractList) in zip(itertools.count(), TSptGrps):
                self.assertEqual(numTSInteractsInPtGroups[jumpInd, TsPtGpInd], len(TSinteractList))
                specList = [self.NSpec - 1, FinSpec] + [spec for spec in spectup]
                for interactInd, TSClust in enumerate(TSinteractList):
                    interact = tuple([(self.VclusExp.sup.index(site.R, site.ci)[0], spec)
                                      for site, spec in zip(TSClust.sites, specList)])
                    interactStored = Index2TSinteractDict[JumpInteracts[jumpInd, TsPtGpInd, interactInd]]

                    self.assertEqual(set(interact), set(interactStored))
                    self.assertEqual(Jump2KRAEng[jumpInd, TsPtGpInd, interactInd], self.KRAEnergies[jumpInd][TsPtGpInd])

class Test_MC(Test_MC_Arrays):

    def test_MC_step(self):
        # First, create a random state
        initState = np.zeros(len(self.VclusExp.sup.mobilepos), dtype=int)
        # Now assign random species (excluding the vacancy)
        for i in range(len(self.VclusExp.sup.mobilepos)):
            initState[i] = np.random.randint(0, self.NSpec-1)

        # Now put in the vacancy at the vacancy site
        initState[self.vacsiteInd] = self.NSpec - 1

        initCopy = initState.copy()

        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts, VecsInteracts, \
        VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray, vacSiteInd, InteractionIndexDict, InteractionRepClusDict, \
        Index2InteractionDict, repClustCounter =\
            self.VclusExp.makeJitInteractionsData(self.Energies)

        TsInteractIndexDict, Index2TSinteractDict, numSitesTSInteracts, TSInteractSites, TSInteractSpecs, \
        jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, \
        JumpInteracts, Jump2KRAEng =\
            self.VclusExp.KRAexpander.makeTransJitData(self.KRAEnergies)

        # Initiate the MC sampler
        MCSampler = Cluster_Expansion.MCSamplerClass(
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts,
            VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray,
            numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
            FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng,
            vacSiteInd, initState
        )

        # First check that the initial OffsiteCount is computed correctly
        for (interaction, interactionInd) in InteractionIndexDict.items():
            offsiteCount = 0
            for (site, spec) in interaction:
                if initState[site] != spec:
                    offsiteCount += 1
            self.assertEqual(MCSampler.OffSiteCount[interactionInd], offsiteCount)

        # calculate Initial energy
        InitEn = 0.
        for i in range(len(MCSampler.OffSiteCount)):
            if MCSampler.OffSiteCount[i] == 0:
                InitEn += Interaction2En[i]

        # Do MC single metropolis step that updates the state
        offsc = MCSampler.OffSiteCount.copy()
        TransOffSiteCountNew = np.zeros(len(TSInteractSites), dtype=int)
        Nsites = len(self.VclusExp.sup.mobilepos)
        Nswaptrials = 1  # We are only testing a single step here
        swaptrials = np.zeros((Nswaptrials, 2), dtype=int)

        count = 0
        while count < Nswaptrials:
            # first select two random sites to swap - for now, let's just select naively.
            siteA = np.random.randint(0, Nsites)
            siteB = np.random.randint(0, Nsites)

            # make sure we are swapping different atoms because otherwise we are in the same state
            if initState[siteA] == initState[siteB] or siteA == vacSiteInd or siteB == vacSiteInd:
                continue

            swaptrials[count, 0] = siteA
            swaptrials[count, 1] = siteB
            count += 1
        randarr = np.log(np.random.rand(Nswaptrials))

        MCSampler.makeMCsweep(initState, offsc, TransOffSiteCountNew, swaptrials, 1.0, randarr, Nswaptrials)

        # Now, first check
        offsc2 = MCSampler.OffSiteCount.copy()
        for intInd in range(numInteractsSiteSpec[siteA, initCopy[siteA]]):
            offsc2[SiteSpecInterArray[siteA, initCopy[siteA], intInd]] += 1

        for intInd in range(numInteractsSiteSpec[siteB, initCopy[siteB]]):
            offsc2[SiteSpecInterArray[siteB, initCopy[siteB], intInd]] += 1

        for intInd in range(numInteractsSiteSpec[siteA, initCopy[siteB]]):
            offsc2[SiteSpecInterArray[siteA, initCopy[siteB], intInd]] -= 1

        for intInd in range(numInteractsSiteSpec[siteB, initCopy[siteA]]):
            offsc2[SiteSpecInterArray[siteB, initCopy[siteA], intInd]] -= 1

        # Now calculate the final energy based on the new offsite counts
        FinEn = 0.
        for i in range(len(offsc2)):
            if offsc2[i] == 0:
                FinEn += Interaction2En[i]

        initJit = initCopy.copy()

        OffSiteCount = np.zeros_like(MCSampler.OffSiteCount)
        TSOffSiteCount2 = TransOffSiteCountNew.copy()
        MCSampler_Jit = MC_JIT.MCSamplerClass(
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts,
            VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray,
            numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
            FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng,
            vacSiteInd, initJit, OffSiteCount
        )
        offscjit = MCSampler_Jit.OffSiteCount.copy()
        MCSampler_Jit.makeMCsweep(initJit, offscjit, TSOffSiteCount2, swaptrials, 1.0, randarr, Nswaptrials)

        # Check that TS offsite counts are updated correctly.
        for TsInteractIdx in range(len(TransOffSiteCountNew)):
            offcount = 0
            for Siteind in range(numSitesTSInteracts[TsInteractIdx]):
                if initState[TSInteractSites[TsInteractIdx, Siteind]] != TSInteractSpecs[TsInteractIdx, Siteind]:
                    offcount += 1

            self.assertEqual(TransOffSiteCountNew[TsInteractIdx], offcount)

        # Check that the same results are found
        self.assertTrue(np.array_equal(initJit, initState))
        self.assertTrue(np.array_equal(offscjit, offsc))
        self.assertTrue(np.array_equal(TransOffSiteCountNew, TSOffSiteCount2))

        # test that energy calculation and site swaps are done correctly.
        if np.exp(-MCSampler.delE) > np.exp(randarr[0]):
            self.assertEqual(initState[siteA], initCopy[siteB])
            self.assertEqual(initState[siteB], initCopy[siteA])

            print("move was accepted")
        else:
            self.assertTrue(np.array_equal(initState, initCopy))
            print("move was not accepted")

        self.assertTrue(np.allclose(MCSampler.delE, FinEn - InitEn), msg="{}, {}".format(MCSampler.delE, FinEn - InitEn))




