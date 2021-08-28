from onsager import crystal, supercell, cluster
import numpy as np
import collections
import itertools
import Transitions
import Cluster_Expansion
import unittest
import time

class test_Vector_Cluster_Expansion(unittest.TestCase):

    def setUp(self):
        self.NSpec = 3
        self.Nvac = 1
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
        self.clusexp = cluster.makeclusters(self.crys, 0.284, self.MaxOrder)
        self.Tclusexp = cluster.makeclusters(self.crys, 0.29, self.MaxOrderTrans)
        self.KRAexpander = Transitions.KRAExpand(self.superBCC, 0, self.jnetBCC, self.Tclusexp, self.Tclusexp, self.NSpec,
                                                 self.Nvac, self.vacsite)
        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(self.superBCC, self.clusexp, self.Tclusexp, self.jnetBCC,
                                                                 self.NSpec, self.vacsite, self.MaxOrder,
                                                                 self.MaxOrderTrans)

        self.Energies = np.random.rand(len(self.VclusExp.SpecClusters))
        self.KRAEnergies = [np.random.rand(len(val)) for (key, val) in self.VclusExp.KRAexpander.clusterSpeciesJumps.items()]

        self.mobOccs = np.zeros((self.NSpec, numSites), dtype=int)
        for site in range(1, numSites):
            spec = np.random.randint(0, self.NSpec - 1)
            self.mobOccs[spec][site] = 1
        self.mobOccs[-1, self.vacsiteInd] = 1
        self.mobCountList = [np.sum(self.mobOccs[i]) for i in range(self.NSpec)]
        print("Done setting up cluster expansion tests.")

    def test_spec_assign(self):

        # check that every cluster appears just once in just one symmetry list
        clList = [cl for clList in self.VclusExp.SpecClusters for cl in clList]

        for cl1 in clList:
            count = 0
            for cl2 in clList:
                if cl1 == cl2:
                    count += 1
            assert count == 1 # check that the cluster occurs only once

        # let's test if the number of symmetric site clusters generated is the same
        sitetuples = set([])
        for clusterList in self.VclusExp.SpecClusters:
            for clust in clusterList:
                Rtrans = sum([site.R for site in clust.siteList]) // len(clust.siteList)
                sites_shift = tuple([site-Rtrans for site in clust.siteList])
                # check correct translations
                sitesBuilt = tuple([site for site, spec in clust.transPairs])
                if self.VclusExp.zeroClusts:
                    self.assertEqual(sites_shift, sitesBuilt)
                sitetuples.add(sites_shift)

        sitesFromClusexp = set([])
        for clSet in self.clusexp:
            for cl in list(clSet):
                for g in self.VclusExp.crys.G:
                    newsitelist = tuple([site.g(self.VclusExp.crys, g) for site in cl.sites])
                    Rtrans = sum([site.R for site in newsitelist])//len(newsitelist)
                    sitesFromClusexp.add(tuple([site - Rtrans for site in newsitelist]))

        total_reps = len(sitesFromClusexp)

        self.assertEqual(total_reps, len(sitetuples), msg="{}, {}".format(total_reps, len(sitetuples)))

        # This test is specifically for a BCC, 3-spec, 3-order, 2nn cluster expansion
        oneCounts = 0
        twocounts = 0
        threecounts = 0
        for siteList in sitesFromClusexp:
            ln=len(siteList)
            if ln == 1:
                oneCounts += 1
            elif ln == 2:
                twocounts += 1
            elif ln == 3:
                threecounts += 1

        self.assertEqual((oneCounts, twocounts, threecounts), (1, 14, 24))

        # Go through all the site clusters:
        total_spec_clusts = 0
        for clSites in sitesFromClusexp:
            ordr = len(clSites)
            # for singletons, any species assignment is acceptable
            if ordr == 1:
                total_spec_clusts += self.NSpec
                continue
            # Now, for two and three body clusters:
            # we can have max one vacancy at a site, and any other site can have any other occupancy.
            # or we can have no vacancy at all in which case it's just (Nspec - 1)**ordr
            # Also, Since we are in BCC lattices, we can symmetrically permute two sites in every two or three body cluster
            total_spec_clusts += (ordr * (self.NSpec - 1)**(ordr - 1) + (self.NSpec-1)**ordr)//2
        #
        total_code = sum([len(lst) for lst in self.VclusExp.SpecClusters])
        #
        self.assertEqual(total_code, total_spec_clusts, msg="{}, {}".format(total_code, total_spec_clusts))

        oneBody = 0
        TwoBody = 0
        ThreeBody = 0

        oneBodyList = []
        TwoBodyList = []
        ThreeBodyList = []
        # Check some of the numbers explicitly - this part is the definitive test
        for clusterList in self.VclusExp.SpecClusters:
            for clust in clusterList:
                order = len(clust.SiteSpecs)
                if order == 1:
                    oneBody += 1
                    oneBodyList.append(clust)
                if order == 2:
                    TwoBody += 1
                    TwoBodyList.append(clust)

                if order == 3:
                    ThreeBody += 1
                    ThreeBodyList.append(clust)

        self.assertEqual(oneBody, 3)
        self.assertEqual(TwoBody, 56)
        self.assertEqual(ThreeBody, 240)

        # Now let's check some two body numbers
        vacCount = 0
        nonVacCount = 0
        for clust in TwoBodyList:
            specList = clust.specList
            if any([spec == 2 for spec in specList]):
                vacCount += 1
            else:
                nonVacCount += 1

        self.assertEqual(vacCount, 2*2*4 + 2*2*3)  # The first term is for nearest neighbor, the second for second nearest neighbor
        self.assertEqual(nonVacCount, 2*2*4 + 2*2*3)

        # Now let's check some three body numbers
        vacCount = 0
        nonVacCount = 0
        for clust in ThreeBodyList:
            specList = clust.specList
            if any([spec == 2 for spec in specList]):
                vacCount += 1
            else:
                nonVacCount += 1

        self.assertEqual(vacCount, (3 * 2 * 2) * 12)
        self.assertEqual(nonVacCount, (2 * 2 * 2) * 12)
        print("Done assignment tests")


    def test_genvecs(self):
        """
        Here, we test if we have generated the vector cluster basis (site-based only) properly
        """
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

    def testcluster2vecClus(self):

        # check that every cluster appears just once in just one symmetry list
        clList = [cl for clList in self.VclusExp.SpecClusters for cl in clList]

        for cl1 in clList:
            count = 0
            for cl2 in clList:
                if cl1 == cl2:
                    count += 1
            self.assertEqual(count, 1)  # check that the cluster occurs only once

        for clListInd, clList in enumerate(self.VclusExp.SpecClusters):
            for clust in clList:
                if self.VclusExp.clus2LenVecClus[clListInd] == 0:
                    self.assertFalse(clust in self.VclusExp.clust2vecClus.keys())
                else:
                    self.assertTrue(self.VclusExp.clus2LenVecClus[clListInd] <= 3)
                    vecList = self.VclusExp.clust2vecClus[clust]
                    self.assertTrue(self.VclusExp.clus2LenVecClus[clListInd] == len(vecList))
                    for tup in vecList:
                        self.assertEqual(clust, self.VclusExp.vecClus[tup[0]][tup[1]])

    def test_indexing(self):
        for vclusListInd, clListInd in enumerate(self.VclusExp.Vclus2Clus):
            cl0 = self.VclusExp.vecClus[vclusListInd][0]
            self.assertEqual(cl0, self.VclusExp.SpecClusters[clListInd][0])

    def test_site_interactions(self):
        # test that every interaction is valid with the given Rtrans provided
        # The key site should be present only once
        interaction2RepClust = {}
        interactCounter = collections.defaultdict(int)
        self.assertEqual(len(self.VclusExp.SiteSpecInteractIds), self.NSpec*len(self.superBCC.mobilepos))
        for (key, interactIdList) in self.VclusExp.SiteSpecInteractIds.items():
            siteInd = key[0]
            sp = key[1]
            for Id in interactIdList:
                interaction = self.VclusExp.InteractionIdDict[Id]
                interactCounter[interaction] += 1
                count = 0
                for (site, spec) in interaction:
                    if site == siteInd and sp == spec:
                        count += 1
                self.assertEqual(count, 1)

        interactCounter.default_factory = None
        # Check that an interaction has occurred as many times as there are sites in it
        for interact, count in interactCounter.items():
            self.assertEqual(count, len(interact))

        # check that all representative clusters have been translated
        self.assertEqual(len(self.VclusExp.clust2InteractId), sum([len(clList) for clList in self.VclusExp.SpecClusters]),
                         msg="{}".format(len(self.VclusExp.clust2InteractId)))

        clAll = [cl for clList in self.VclusExp.SpecClusters for cl in clList]
        for clInd, cl in enumerate(clAll):
            self.assertEqual(clInd, self.VclusExp.Clus2Num[cl])

        for clustInd in self.VclusExp.clust2InteractId.keys():
            clust = self.VclusExp.Num2Clus[clustInd]
            self.assertTrue(clust in clAll)

        # check that each interaction corresponds to a single rep cluster
        InteractIdtoRepClust = collections.defaultdict(list)
        for repClust, interactIdList in self.VclusExp.clust2InteractId.items():
            for interId in interactIdList:
                InteractIdtoRepClust[interId].append(repClust)

        for key, item in InteractIdtoRepClust.items():
            self.assertEqual(len(item), 1)

    def test_trans_count(self):
        # test that all translations of all representative clusters are considered
        allSpCl = [SpCl for SpClList in self.VclusExp.SpecClusters for SpCl in SpClList]
        for (key, interactIdList) in self.VclusExp.SiteSpecInteractIds.items():
            siteInd = key[0]
            ci = self.VclusExp.sup.ciR(siteInd)[0]
            sp = key[1]
            count = 0
            # For each species at a site, check that all translations of a cluster are considered.
            for SpecClus in allSpCl:
                for (site, spec) in SpecClus.SiteSpecs:
                    if spec == sp and site.ci == ci:
                        count += 1  # a translation of this cluster should exist

            self.assertEqual(count, len(interactIdList), msg="\ncount {}, stored {}\n{}".format(count, len(interactIdList), key))

    def testcluster2SpecClus(self):

        for clListInd, clList in enumerate(self.VclusExp.SpecClusters):
            for clustInd, clust in enumerate(clList):
                tup = self.VclusExp.clust2SpecClus[clust]
                self.assertEqual(tup[0], clListInd)
                self.assertEqual(tup[1], clustInd)




    # def test_arrays(self):
    #
    #     start = time.time()
    #
    #     # Check that each cluster has been translated as many times as there are sites in the supercell
    #     # Only then we have constructed every possible interaction
    #     print("Done creating arrays : {}".format(time.time() - start))
    #     # Now, we first test the interaction arrays - the ones to be used in the MC sweeps
    #
    #     # check that every representative clusters has been translated as many times
    #     # as there are unit cells in the super cell.
    #     for repClust, count in self.repClustCounter.items():
    #         self.assertEqual(count, len(self.VclusExp.sup.mobilepos))
    #     # numSitesInteracts - the number of sites in an interaction
    #     for i in range(len(self.numSitesInteracts)):
    #         siteCountInArray = self.numSitesInteracts[i]
    #         # get the interaction
    #         interaction = self.Index2InteractionDict[i]
    #         # get the stored index for this interaction
    #         i_stored = self.InteractionIndexDict[interaction]
    #         self.assertEqual(i, i_stored)
    #         self.assertEqual(len(interaction), siteCountInArray)
    #
    #     # Now, check the supercell sites and species stored for these interactions
    #     # check cluster translations
    #     allSpCl = [SpCl for SpClList in self.VclusExp.SpecClusters for SpCl in SpClList]
    #
    #     SpClset = set(allSpCl)
    #
    #     allCount = len(self.VclusExp.sup.mobilepos) * len(allSpCl)
    #     self.assertEqual(len(self.InteractionIndexDict), allCount, msg="\n{}\n{}".format(len(self.InteractionIndexDict), allCount))
    #
    #     self.assertEqual(len(self.SupSitesInteracts), len(self.numSitesInteracts))
    #     self.assertEqual(len(self.SpecOnInteractSites), len(self.numSitesInteracts))
    #
    #     repclustCount = collections.defaultdict(int)
    #
    #     SpClset2 = set([])
    #
    #     for i in range(len(self.numSitesInteracts)):
    #         siteSpecSet = set([(self.SupSitesInteracts[i, j], self.SpecOnInteractSites[i, j])
    #                            for j in range(self.numSitesInteracts[i])])
    #         interaction = self.Index2InteractionDict[i]
    #         interactionSet = set(interaction)
    #         self.assertEqual(siteSpecSet, interactionSet)
    #
    #         # let's get the representative cluster of this interaction
    #         repClusStored = self.InteractionRepClusDict[interaction]
    #         siteList = []
    #         specList = []
    #
    #         for siteInd, spec in [(self.SupSitesInteracts[i, j], self.SpecOnInteractSites[i, j])
    #                               for j in range(self.numSitesInteracts[i])]:
    #             ci, R = self.VclusExp.sup.ciR(siteInd)
    #             # R %= self.N_units  # Bring it within the unit cell
    #             clSite = cluster.ClusterSite(ci=ci, R=R)
    #             siteList.append(clSite)
    #             specList.append(spec)
    #
    #         SpCl = Cluster_Expansion.ClusterSpecies(specList, siteList)
    #
    #         # apply periodicity to the siteList and rebuild
    #         siteListBuilt = [site for (site, spec) in SpCl.transPairs]
    #         specListBuilt = [spec for (site, spec) in SpCl.transPairs]
    #         siteListnew = []
    #         for site in siteListBuilt:
    #             R = site.R
    #             ci = site.ci
    #             R %= self.N_units
    #             siteListnew.append(cluster.ClusterSite(ci=ci, R=R))
    #
    #         SpCl = Cluster_Expansion.ClusterSpecies(specListBuilt, siteListnew)
    #         self.assertEqual(SpCl, repClusStored)
    #         self.assertTrue(SpCl in SpClset, msg="\n{}\n{}\n{}\n{}".format(SpCl, repClusStored, siteList, specList))
    #
    #         # Check that the correct energies have been assigned
    #         En = self.Energies[self.VclusExp.clust2SpecClus[repClusStored][0]]
    #         EnStored = self.Interaction2En[i]
    #         self.assertAlmostEqual(En, EnStored, 10)
    #         SpClset2.add(SpCl)
    #
    #         repclustCount[SpCl] += 1
    #
    #     # Check that all translations of repclusts were considered
    #     self.assertEqual(SpClset, SpClset2)
    #     for key, item in repclustCount.items():
    #         self.assertEqual(item, len(self.VclusExp.sup.mobilepos))
    #
    #
    #     print("checked interactions")
    #
    #     # Now, test the vector basis and energy information for the clusters
    #     for i in range(len(self.numSitesInteracts)):
    #         # get the interaction
    #         interaction = self.Index2InteractionDict[i]
    #         # Now, get the representative cluster
    #         repClus = self.InteractionRepClusDict[interaction]
    #
    #         # test the energy index
    #         enIndex = self.VclusExp.clust2SpecClus[repClus][0]
    #         self.assertEqual(self.Interaction2En[i], self.Energies[enIndex])
    #
    #         # if the vector basis is empty, continue
    #         if self.VclusExp.clus2LenVecClus[self.VclusExp.clust2SpecClus[repClus][0]] == 0:
    #             self.assertEqual(self.numVecsInteracts[i], -1)
    #             continue
    #         # get the vector basis info for this cluster
    #         vecList = self.VclusExp.clust2vecClus[repClus]
    #         # check the number of vectors
    #         self.assertEqual(self.numVecsInteracts[i], len(vecList))
    #         # check that the correct vector have been stored, in the same order as in vecList (not necessary but short testing)
    #         for vecind in range(len(vecList)):
    #             vec = self.VclusExp.vecVec[vecList[vecind][0]][vecList[vecind][1]]
    #             self.assertTrue(np.allclose(vec, self.VecsInteracts[i, vecind, :]))
    #
    #     # Next, test the interactions each (site, spec) is a part of
    #     InteractSet = set([])
    #     self.assertEqual(self.numInteractsSiteSpec.shape[0], len(self.superBCC.mobilepos))
    #     self.assertEqual(self.numInteractsSiteSpec.shape[1], self.NSpec)
    #     for siteInd in range(len(self.superBCC.mobilepos)):
    #         for spec in range(self.NSpec):
    #             numInteractStored = self.numInteractsSiteSpec[siteInd, spec]
    #             # get the actual count
    #             ci, R = self.VclusExp.sup.ciR(siteInd)
    #             clsite = cluster.ClusterSite(ci=ci, R=R)
    #             self.assertEqual(len(self.VclusExp.SiteSpecInteractions[(clsite, spec)]), numInteractStored)
    #             for IdxOfInteract in range(numInteractStored):
    #                 interactMainIndex = self.SiteSpecInterArray[siteInd, spec, IdxOfInteract]
    #                 interactMain = self.Index2InteractionDict[interactMainIndex]
    #                 InteractSet.add(interactMain)
    #                 self.assertEqual(interactMain, self.VclusExp.SiteSpecInteractions[(clsite, spec)][IdxOfInteract][0])
    #
    #                 # Now translate it back and look at the energies
    #                 siteList = []
    #                 specList = []
    #
    #                 i = interactMainIndex
    #                 count = 0
    #                 for siteNum, sp in [(self.SupSitesInteracts[i, j], self.SpecOnInteractSites[i, j])
    #                                       for j in range(self.numSitesInteracts[i])]:
    #                     ci, R = self.VclusExp.sup.ciR(siteNum)
    #                     # R %= self.N_units  # Bring it within the unit cell
    #                     cls = cluster.ClusterSite(ci=ci, R=R)
    #                     siteList.append(cls)
    #                     specList.append(sp)
    #                     if cls == clsite and sp == spec:
    #                         count += 1
    #                 self.assertEqual(count, 1)
    #
    #                 SpCl = Cluster_Expansion.ClusterSpecies(specList, siteList)
    #
    #                 # apply periodicity to the siteList and rebuild
    #                 siteListBuilt = [site for (site, spec) in SpCl.transPairs]
    #                 specListBuilt = [spec for (site, spec) in SpCl.transPairs]
    #                 siteListnew = []
    #                 for site in siteListBuilt:
    #                     R = site.R
    #                     ci = site.ci
    #                     R %= self.N_units
    #                     siteListnew.append(cluster.ClusterSite(ci=ci, R=R))
    #
    #                 SpCl = Cluster_Expansion.ClusterSpecies(specListBuilt, siteListnew)
    #                 En = self.Energies[self.VclusExp.clust2SpecClus[SpCl][0]]
    #                 EnStored = self.Interaction2En[interactMainIndex]
    #                 self.assertAlmostEqual(En, EnStored, 10)

        






