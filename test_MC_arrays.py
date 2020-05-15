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

    def test_arrays(self):

        Energies = np.random.rand(len(self.VclusExp.SpecClusters))

        KRAEnergies

        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts, \
        VecsInteracts, numInteractsSiteSpec, SiteSpecInterArray, jumpFinSites, jumpFinSpec, \
        numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, vacSiteInd =\
            self.VclusExp.makeJitInteractionsData()