from onsager import crystal, supercell, cluster
import numpy as np
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
        self.mobCountList = [(i, np.sum(self.mobOccs[i])) for i in range(5)]
        self.clusexp = cluster.makeclusters(self.crys, 0.29, 4)
        self.KRAexpander = Transitions.KRAExpand(self.superBCC, 0, self.jnetBCC, self.clusexp, self.mobCountList)

    def test_groupTrans(self):
        """
        To check if the group operations that form the clusters keep the transition sites unchanged.
        """
        for key, clusterLists in self.KRAexpander.SymTransClusters.items():
            siteA = self.superBCC.ciR(key[0])
            siteB = self.superBCC.ciR(key[1])
            for clist in clusterLists:
                cl0 = clist[0]
                for clust in clist:
                    count = 0
                    for g in self.crys.G:
                        clNew = cl0.g(self.crys, self.g)
                        if clNew == clust:
                            count += 1
                            self.assertEqual(siteA, siteA.g(self.crys, g))
                            self.assertEqual(siteB, siteB.g(self.crys, g))
                    self.assertNotEqual(count, 0)

