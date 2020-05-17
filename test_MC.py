from onsager import crystal, supercell, cluster
import numpy as np
import itertools
import Transitions
import Cluster_Expansion
import unittest
import time

class Test_MC_Arrays(unittest.TestCase):

    def setUp(self):
        self.NSpec = 3
        self.MaxOrder = 3
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
        self.clusexp = cluster.makeclusters(self.crys, 0.29, self.MaxOrder)
        self.KRAexpander = Transitions.KRAExpand(self.superBCC, 0, self.jnetBCC, self.clusexp, self.mobCountList,
                                                 self.vacsite)
        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(self.superBCC, self.clusexp, self.jnetBCC,
                                                                 self.mobCountList, self.vacsite, self.MaxOrder)

        self.Energies = np.random.rand(len(self.VclusExp.SpecClusters))
        self.KRAEnergies = [np.random.rand(len(val)) for (key, val) in self.VclusExp.KRAexpander.clusterSpeciesJumps.items()]
        print("Done setting up")

    def test_arrays(self):

        start = time.time()
        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts, \
        VecsInteracts, numInteractsSiteSpec, SiteSpecInterArray, jumpFinSites, jumpFinSpec, \
        numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng, vacSiteInd, InteractionIndexDict,\
        InteractionRepClusDict, Index2InteractionDict, repClustCounter =\
            self.VclusExp.makeJitInteractionsData(self.Energies, self.KRAEnergies)

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

        # Now, we start testing the jump arrays
        # jumpFinSites, jumpFinSpec, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng

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
                    interactStored = Index2InteractionDict[JumpInteracts[jumpInd, TsPtGpInd, interactInd]]

                    self.assertEqual(set(interact), set(interactStored))
                    self.assertEqual(Jump2KRAEng[jumpInd, TsPtGpInd, interactInd], self.KRAEnergies[jumpInd][TsPtGpInd])

class Test_MC(Test_MC_Arrays):

    def test_MC_steps(self):
        # First, create a random state
        initState = np.zeros(len(self.VclusExp.sup.mobilepos), dtype=int)
        # Now assign random species (excluding the vacancy)
        for i in range(len(self.VclusExp.sup.mobilepos)):
            initState[i] = np.random.randint(0, self.NSpec-1)

        # Now put in the vacancy at the vacancy site
        initState[self.vacsiteInd] = self.NSpec - 1

        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts, \
        VecsInteracts, numInteractsSiteSpec, SiteSpecInterArray, jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd,\
        numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng, vacSiteInd, InteractionIndexDict, \
        InteractionRepClusDict, Index2InteractionDict, repClustCounter = \
            self.VclusExp.makeJitInteractionsData(self.Energies, self.KRAEnergies)

        # Initiate the MC sampler
        MCSampler = Cluster_Expansion.MCSamplerClass(
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts,
            VecsInteracts, numInteractsSiteSpec, SiteSpecInterArray, jumpFinSites, jumpFinSpec,
            numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng, vacSiteInd, initState
        )

        # First check that the initial OffsiteCount is computed correctly
        for (interaction, interactionInd) in InteractionIndexDict.items():
            offsiteCount = 0
            for (site, spec) in interaction:
                if initState[site] != spec:
                    offsiteCount += 1
            self.assertEqual(MCSampler.OffSiteCount[interactionInd], offsiteCount)

        # Now, we test some single MC steps
        start = time.time()
        siteA, siteB, delE, newstate, rand = MCSampler.makeMCsweep(1, 1.0, test_single=True)
        print("Single swap time : {}".format(time.time() - start))

        print(siteA, siteB)
        # evaluate the energy change manually
        # switch occupancies of the two sites
        stateNew = initState.copy()
        temp = stateNew[siteA]
        stateNew[siteA] = stateNew[siteB]
        stateNew[siteB] = temp

        # Now get the intial energies from all the interactions that are "on" in the initial state
        # get the off site counts for the interactions in the new state
        NewOffSiteCount = np.zeros_like(MCSampler.OffSiteCount)
        for (interaction, interactionInd) in InteractionIndexDict.items():
            # offsiteCount = 0
            for (site, spec) in interaction:
                if stateNew[site] != spec:
                    NewOffSiteCount[interactionInd] += 1
            # NewOffSiteCount[interactionInd] = offsiteCount

        # calculate Intial energy
        InitEn = 0.
        for i in range(len(MCSampler.OffSiteCount)):
            if MCSampler.OffSiteCount[i] == 0:
                InitEn += Interaction2En[i]

        # Now calculate the final energies
        FinEn = 0.
        for i in range(len(NewOffSiteCount)):
            if NewOffSiteCount[i] == 0:
                FinEn += Interaction2En[i]

        self.assertTrue(np.allclose(delE, FinEn - InitEn), msg="{}, {}".format(delE, FinEn - InitEn))

        if np.exp(-delE) > rand:
            self.assertTrue(np.array_equal(stateNew, newstate))
            print("move was accepted")
        else:
            self.assertTrue(np.array_equal(initState, newstate))
            print("move was not accepted")




