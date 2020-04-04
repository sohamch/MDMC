from onsager import crystal, supercell, cluster
import numpy as np
import collections
import itertools
import Transitions
import Cluster_Expansion
import unittest

class testKRA(unittest.TestCase):

    def setUp(self):
        self.crys = crystal.Crystal.BCC(0.2836, chemistry="A")
        self.jnetBCC = self.crys.jumpnetwork(0, 0.26)
        self.superlatt = 8 * np.eye(3, dtype=int)
        self.superBCC = supercell.ClusterSupercell(self.crys, self.superlatt)
        # get the number of sites in the supercell - should be 8x8x8
        numSites = len(self.superBCC.mobilepos)
        self.vacsite = self.superBCC.index(np.zeros(3, dtype=int), (0, 0))[0]
        self.mobOccs = np.zeros((5, numSites), dtype=int)
        for site in range(1, numSites):
            spec = np.random.randint(0, 4)
            self.mobOccs[spec][site] = 1
        self.mobOccs[-1, 0] = 1
        self.mobCountList = [np.sum(self.mobOccs[i]) for i in range(5)]
        self.clusexp = cluster.makeclusters(self.crys, 0.29, 4)
        self.vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
        self.KRAexpander = Transitions.KRAExpand(self.superBCC, 0, self.jnetBCC, self.clusexp, self.mobCountList,
                                                 self.vacsite)
        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(self.superBCC, self.clusexp, self.jnetBCC,
                                                                 self.mobCountList, self.vacsite)

    def test_groupTrans(self):
        """
        To check if the group operations that form the clusters keep the transition sites unchanged.
        """
        for key, clusterLists in self.KRAexpander.SymTransClusters.items():

            ciA, RA = self.superBCC.ciR(key[0])
            ciB, RB = self.superBCC.ciR(key[1])
            siteA = cluster.ClusterSite(ci=ciA, R=RA)
            siteB = cluster.ClusterSite(ci=ciB, R=RB)
            self.assertEqual(siteA, self.vacsite)
            clusterListCount = collections.defaultdict(int)  # each cluster should only appear in one list
            for clist in clusterLists:
                cl0 = clist[0]
                for clust in clist:
                    clusterListCount[clust] += 1
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
            for clust, count in clusterListCount.items():
                self.assertEqual(count, 1)

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
        for key, SpeciesclusterLists in clusterSpeciesJumps.items():
            clusterCounts = collections.defaultdict(int)
            for species, clusterList in SpeciesclusterLists:
                cl0 = clusterList[0]
                self.assertEqual(cl0.Norder, len(species))
                clusterCounts[cl0] += 1

            for cl0, count in clusterCounts.items():
                numTrue = 4**cl0.Norder
                self.assertEqual(numTrue, count, msg="{}, {}, {}".format(numTrue, count, cl0.Norder))

    def test_KRA(self):
        """
        Checking whether the KRA expansions are done correctly
        """
        # Go through each transition
        for transition, clusterLists in self.KRAexpander.clusterSpeciesJumps.items():
            # get the number of clusterLists, and generate that many coefficients
            KRACoeffs = np.array([np.random.rand() for i in range(len(clusterLists))])
            valOn = np.zeros(len(KRACoeffs))
            # Now go through the clusterLists and note which clusters are on
            for Idx, (tup, clList) in enumerate(clusterLists):
                for cl in clList:
                    # check if this cluster is On
                    prod = 1
                    for siteInd, site in enumerate(cl.sites):
                        siteIdx = self.superBCC.index(site.R, site.ci)[0]
                        if siteInd == 0:
                            # vacancy site is always occupied by vacancy
                            self.assertEqual(siteIdx, self.vacsite)
                            self.assertEqual(self.mobOccs[-1, siteInd], 1)
                        elif siteInd == 1:
                            # SpecJ
                            specJ = transition[2]
                            # check if this site is occupied
                            self.assertEqual(self.superBCC.index(site.R, site.ci)[0], transition[1])
                            if self.mobOccs[specJ][transition[1]] == 0:
                                continue  # this is not the transition we are looking for for this state
                        elif self.mobOccs[tup[siteInd - 2], siteIdx] == 0:
                            prod = 0

                    if prod == 1:
                        valOn[Idx] += KRACoeffs[Idx]

            KRAen = np.sum(valOn)
            KRAcalc = self.KRAexpander.GetKRA(transition, self.mobOccs, KRACoeffs)
            self.assertTrue(np.allclose(KRAen, KRAcalc), msg="{}, {}".format(KRAen, KRAcalc))
            print("Envalues : {}, {}".format(KRAcalc, KRAen))


class test_Vector_Cluster_Expansion(testKRA):

    def test_genvecs(self):
        """
        Here, we test if we have generated the vector cluster basis (site-based only) properly
        """

        # Test 1 - check that every clusterList is repeated exactly three times
        # And the that the vectors associated with the clusters are orthogonal to each other
        for clListInd, clList, vecList in zip(itertools.count(), self.VclusExp.vecClus,
                                              self.VclusExp.vecVec):
            self.assertEqual(len(clList), len(vecList))
            cl0, vec0 = clList[0], vecList[0]
            for clust, vec in zip(clList, vecList):
                # First check that symmetry operations are consistent
                count = 0
                for g in self.crys.G:
                    if cl0.g(self.crys, g) == clust:
                        count += 1
                        self.assertTrue(np.allclose(np.dot(g.cartrot, vec0), vec) or
                                        np.allclose(np.dot(g.cartrot, vec0) + vec, np.zeros(3)),
                                        msg="\n{}, {} \n{}, {}\n{}\n{}".format(vec0, vec, cl0, clust,
                                                                               self.crys.lattice, g.cartrot))
                self.assertGreater(count, 0)

    def test_indexing(self):

        for vclusListInd, clListInd in enumerate(self.VclusExp.Vclus2Clus):
            cl0 = self.VclusExp.vecClus[vclusListInd][0]
            self.assertEqual(cl0, self.VclusExp.SpecClusters[clListInd][0])




