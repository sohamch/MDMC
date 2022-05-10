from onsager import crystal, supercell, cluster
import numpy as np
import collections
import itertools
from ClustSpec import ClusterSpecies
import Cluster_Expansion
import unittest
import time

class test_Vector_Cluster_Expansion(unittest.TestCase):

    def setUp(self):
        self.NSpec = 3
        self.Nvac = 1
        self.MaxOrder = 3
        self.MaxOrderTrans = 3
        a0 = 1
        self.a0 = a0
        self.crys = crystal.Crystal.BCC(a0, chemistry="A")
        jumpCutoff = 1.01*np.sqrt(3./4.)*a0
        self.jnetBCC = self.crys.jumpnetwork(0, jumpCutoff)
        self.BCC_active = True
        self.N_units = 8
        self.superlatt = self.N_units * np.eye(3, dtype=int)
        self.superBCC = supercell.ClusterSupercell(self.crys, self.superlatt)
        # get the number of sites in the supercell - should be 8x8x8
        numSites = len(self.superBCC.mobilepos)
        self.vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
        self.vacsiteInd = self.superBCC.index(np.zeros(3, dtype=int), (0, 0))[0]
        self.clusexp = cluster.makeclusters(self.crys, 1.01*a0, self.MaxOrder)

        TScombShellRange = 1  # upto 1nn combined shell
        TSnnRange = 4
        TScutoff = np.sqrt(3) * a0  # 5th nn cutoff

        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(self.superBCC, self.clusexp, self.NSpec, self.vacsite, self.MaxOrder,
        TScutoff=None, TScombShellRange=None, TSnnRange=None, jumpnetwork=None)

        self.VclusExp.generateSiteSpecInteracts()
        self.VclusExp.genVecClustBasis(self.VclusExp.SpecClusters)
        self.VclusExp.indexVclus2Clus()  # Index vector cluster list to cluster symmetry groups
        self.VclusExp.indexClustertoVecClus()

        self.Energies = np.random.rand(len(self.VclusExp.SpecClusters))

        self.mobOccs = np.zeros((self.NSpec, numSites), dtype=int)
        for site in range(1, numSites):
            spec = np.random.randint(0, self.NSpec - 1)
            self.mobOccs[spec][site] = 1
        self.mobOccs[-1, self.vacsiteInd] = 1
        self.mobCountList = [np.sum(self.mobOccs[i]) for i in range(self.NSpec)]
        self.Energies = np.random.rand(len(self.VclusExp.SpecClusters))
        print("Done setting up cluster expansion tests.")

    def test_spec_assign(self):

        # check that every cluster appears just once in just one symmetry list
        clList = [cl for clList in self.VclusExp.SpecClusters for cl in clList]
        self.assertEqual(len(clList), len(set(clList)))

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
        # Check some of the numbers explicitly - this part is for BCC crystals
        if self.BCC_active:
            # This test is specifically for a BCC, 3-spec, 3-order, 2nn cluster expansion
            oneCounts = 0
            twocounts = 0
            threecounts = 0
            for siteList in sitesFromClusexp:
                ln = len(siteList)
                if ln == 1:
                    oneCounts += 1
                elif ln == 2:
                    twocounts += 1
                elif ln == 3:
                    threecounts += 1

            self.assertEqual((oneCounts, twocounts, threecounts), (1, 14, 24))
            print("Checking cluster counts")
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
        # test that every interaction is valid
        # The key site should be present only once in every interaction stored for it
        interactCounter = collections.defaultdict(int)
        self.assertEqual(len(self.VclusExp.SiteSpecInteractIds), self.NSpec*len(self.superBCC.mobilepos))
        for (key, interactIdList) in self.VclusExp.SiteSpecInteractIds.items():
            siteInd = key[0]
            sp = key[1]
            for Id in interactIdList:
                interaction = self.VclusExp.Id2InteractionDict[Id]
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

    def test_arrays(self):
        # First, we generate the Jit arrays

        # First, the chemical data
        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, \
        numInteractsSiteSpec, SiteSpecInterArray = self.VclusExp.makeJitInteractionsData(self.Energies)

        # Next, the vector basis data
        numVecsInteracts, VecsInteracts, VecGroupInteracts = self.VclusExp.makeJitVectorBasisData()

        # check that every representative clusters has been translated as many times
        # as there are unit cells in the super cell.

        allSpCl = [SpCl for SpClList in self.VclusExp.SpecClusters for SpCl in SpClList]

        allCount = len(self.VclusExp.sup.mobilepos) * len(allSpCl)
        self.assertEqual(len(self.VclusExp.Id2InteractionDict), allCount,
                         msg="\n{}\n{}".format(len(self.VclusExp.Id2InteractionDict), allCount))
        allSpClset = set(allSpCl)
        self.assertEqual(len(allSpCl), len(allSpClset))
        self.assertEqual(allCount, len(numSitesInteracts))
        self.assertEqual(len(SupSitesInteracts), len(numSitesInteracts))
        self.assertEqual(len(SpecOnInteractSites), len(numSitesInteracts))

        repclustCount = collections.defaultdict(int)

        SpClset = set([])

        for i in range(len(numSitesInteracts)):
            siteSpecSet = set([(SupSitesInteracts[i, j], SpecOnInteractSites[i, j])
                               for j in range(numSitesInteracts[i])])
            interaction = self.VclusExp.Id2InteractionDict[i]
            interactionSet = set(interaction)
            self.assertEqual(siteSpecSet, interactionSet)

            # let's get the representative cluster of this interaction
            repClusStored = self.VclusExp.Num2Clus[self.VclusExp.InteractionId2ClusId[i]]
            siteList = []
            specList = []

            for siteInd, spec in [(SupSitesInteracts[i, j], SpecOnInteractSites[i, j])
                                  for j in range(numSitesInteracts[i])]:
                ci, R = self.VclusExp.sup.ciR(siteInd)
                clSite = cluster.ClusterSite(ci=ci, R=R)
                siteList.append(clSite)
                specList.append(spec)

            SpCl = ClusterSpecies(specList, siteList)

            # apply periodicity to the siteList and rebuild
            siteListBuilt = [site for (site, spec) in SpCl.transPairs]
            specListBuilt = [spec for (site, spec) in SpCl.transPairs]
            siteListnew = []
            for site in siteListBuilt:
                R = site.R
                ci = site.ci
                R %= self.N_units
                siteListnew.append(cluster.ClusterSite(ci=ci, R=R))

            SpCl = ClusterSpecies(specListBuilt, siteListnew)
            self.assertEqual(SpCl, repClusStored)
            self.assertTrue(SpCl in allSpClset, msg="\n{}\n{}\n{}\n{}".format(SpCl, repClusStored, siteList, specList))

            # Check that the correct energies have been assigned
            En = self.Energies[self.VclusExp.clust2SpecClus[repClusStored][0]]
            EnStored = Interaction2En[i]
            self.assertAlmostEqual(En, EnStored, 10)
            SpClset.add(SpCl)

            repclustCount[SpCl] += 1
    #
        # Check that all translations of repclusts were considered
        self.assertEqual(SpClset, allSpClset)
        for key, item in repclustCount.items():
            self.assertEqual(item, len(self.VclusExp.sup.mobilepos))


        print("checked interactions")

        # Now, test the vector basis and energy information for the clusters
        for i in range(len(numSitesInteracts)):
            # get the representative cluster
            repClus = self.VclusExp.Num2Clus[self.VclusExp.InteractionId2ClusId[i]]

            # test the energy index
            enIndex = self.VclusExp.clust2SpecClus[repClus][0]
            self.assertEqual(Interaction2En[i], self.Energies[enIndex])

            # if the vector basis is empty, continue
            if self.VclusExp.clus2LenVecClus[self.VclusExp.clust2SpecClus[repClus][0]] == 0:
                self.assertEqual(numVecsInteracts[i], -1)
                continue
            # get the vector basis info for this cluster
            vecList = self.VclusExp.clust2vecClus[repClus]
            # check the number of vectors
            self.assertEqual(numVecsInteracts[i], len(vecList))
            # check that the correct vector have been stored, in the same order as in vecList (not necessary but short testing)
            for vecind in range(len(vecList)):
                vec = self.VclusExp.vecVec[vecList[vecind][0]][vecList[vecind][1]]
                self.assertTrue(np.allclose(vec, VecsInteracts[i, vecind, :]))

        # Next, test the interactions each (site, spec) is a part of
        self.assertEqual(numInteractsSiteSpec.shape[0], len(self.VclusExp.sup.mobilepos))
        self.assertEqual(numInteractsSiteSpec.shape[1], self.NSpec)
        for siteInd in range(len(self.superBCC.mobilepos)):
            for spec in range(self.NSpec):
                numInteractStored = numInteractsSiteSpec[siteInd, spec]
                # get the actual count
                ci, R = self.VclusExp.sup.ciR(siteInd)
                self.assertEqual(len(self.VclusExp.SiteSpecInteractIds[(siteInd, spec)]), numInteractStored)
                for IdxOfInteract in range(numInteractStored):
                    interactMainIndex = SiteSpecInterArray[siteInd, spec, IdxOfInteract]
                    self.assertEqual(interactMainIndex, self.VclusExp.SiteSpecInteractIds[(siteInd, spec)][IdxOfInteract])

        






