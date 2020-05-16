from onsager import crystal, supercell, cluster
import numpy as np
import collections
import itertools
import Transitions
import Cluster_Expansion
import unittest
import time

class MC_Arrays(unittest.TestCase):

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

        # return numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, \
        #        Interaction2En, numVecsInteracts, VecsInteracts, numInteractsSiteSpec, SiteSpecInterArray, \
        #        jumpFinSites, jumpFinSpec, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng, \
        #        vacSiteInd, InteractionIndexDict, InteractionRepClusDict, Index2InteractionDict

        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts, \
        VecsInteracts, numInteractsSiteSpec, SiteSpecInterArray, jumpFinSites, jumpFinSpec, \
        numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng, vacSiteInd, InteractionIndexDict,\
        InteractionRepClusDict, Index2InteractionDict =\
            self.VclusExp.makeJitInteractionsData(self.Energies, self.KRAEnergies)

        print("Done creating arrays : {}".format(time.time() - start))
        # Now, we first test the interaction arrays - the ones to be used in the MC sweeps

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
            siteSpecSet = set(((SupSitesInteracts[i, j], SpecOnInteractSites[i, j])
                               for j in range(numSitesInteracts[i])))
            interaction = Index2InteractionDict[i]
            interactionSet = set(((self.VclusExp.sup.index(site.R, site.ci)[0], spec) for site, spec in interaction))
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


