from onsager import crystal, supercell, cluster
import numpy as np
import collections
import Transitions
import unittest

class testKRA(unittest.TestCase):

    def setUp(self):
        self.crys = crystal.Crystal.BCC(0.2836, chemistry="A")
        self.jnetBCC = self.crys.jumpnetwork(0, 0.26)
        self.superlatt = 8 * np.eye(3, dtype=int)
        self.superBCC = supercell.ClusterSupercell(self.crys, self.superlatt)
        # get the number of sites in the supercell - should be 8x8x8
        numSites = len(self.superBCC.mobilepos)
        vacsite = self.superBCC.index(np.zeros(3, dtype=int), (0, 0))[0]
        self.mobOccs = np.zeros((5, numSites), dtype=int)
        for site in range(1, numSites):
            spec = np.random.randint(0, 4)
            self.mobOccs[spec][site] = 1
        self.mobOccs[-1, 0] = 1
        self.mobCountList = [np.sum(self.mobOccs[i]) for i in range(5)]
        self.clusexp = cluster.makeclusters(self.crys, 0.29, 4)
        self.KRAexpander = Transitions.KRAExpand(self.superBCC, 0, self.jnetBCC, self.clusexp, self.mobCountList)

    def test_groupTrans(self):
        """
        To check if the group operations that form the clusters keep the transition sites unchanged.
        """
        for key, clusterLists in self.KRAexpander.SymTransClusters.items():

            ciA, RA = self.superBCC.ciR(key[0])
            ciB, RB = self.superBCC.ciR(key[1])
            siteA = cluster.ClusterSite(ci=ciA, R=RA)
            siteB = cluster.ClusterSite(ci=ciB, R=RB)

            for clist in clusterLists:
                cl0 = clist[0]
                for clust in clist:
                    count = 0
                    countSym = 0
                    for g in self.crys.G:
                        clNew = cl0.g(self.crys, g)
                        if clNew == clust:
                            count += 1
                            if siteA == siteA.g(self.crys, g) and siteB == siteB.g(self.crys, g):
                                countSym += 1
                    self.assertNotEqual(count, 0)
                    self.assertNotEqual(countSym, 0)

    def test_species_grouping(self):
        """
        The objective for this is to check that each clusterlist is repeated as many times as there should be
        species in its sites.
        """
        # First, count that every transition has every specJ at the end site
        clusterSpeciesJumps = self.KRAexpander.clusterSpeciesJumps

        counter = collections.defaultdict(int)
        for key, items in clusterSpeciesJumps.items():
            counter[(key[0], key[1])] += 1

        for key, item in counter.items():
            self.assertEqual(item, 4)

        # Now check that all possible atomic arrangements have been accounted for
        clusterCounts = collections.defaultdict(int)
        keyset = set()
        for key, SpeciesclusterLists in clusterSpeciesJumps.items():
            if (key[0], key[1]) in keyset:
                continue
            keyset.add((key[0], key[1]))
            for species, clusterList in SpeciesclusterLists:
                cl0 = clusterList[0]
                clusterCounts[cl0] += 1

        for cl0, count in clusterCounts.items():
            numTrue = 4**(cl0.Norder)
            self.assertEqual(numTrue, count, msg="{}, {}, {}".format(numTrue, count, cl0.Norder))



