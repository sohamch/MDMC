from onsager import cluster
import numpy as np
import collections
import itertools
# import Transitions - needs fixing
from KRA3Body import KRA3bodyInteractions
from ClustSpec import ClusterSpecies
import time
from tqdm import tqdm


class VectorClusterExpansion(object):
    """
    class to expand velocities and rates in vector cluster functions.
    """
    def __init__(self, sup, clusexp, TScutoff, TScombShellRange, TSnnRange, jumpnetwork, NSpec, vacSite, maxorder,
                 TclusExp=None, maxOrderTrans=None, zeroClusts=True, OrigVac=False):
        """
        :param sup : clusterSupercell object
        :param clusexp: cluster expansion about a single unit cell.
        :param Tclusexp: Transition state cluster expansion - will be added in later
        :param jumpnetwork: the single vacancy jump network in the lattice used to construct sup
        :param NSpec: no. of species to consider (including vacancies)
        :param vacSite: Index of the vacancy site (used in MC sampling and JIT construction)
        :param vacSite: the site of the vacancy as a clusterSite object. This does not change during the simulation.
        :param maxorder: the maximum order of a cluster in clusexp.
        :param zeroClusts: Same as parameter "zero" of ClusterSpecies class - whether to bring a cluster's centroid to zero or not.
        :param OrigVac: only vacancy-atom pairs with the vacancy at the centre will be considered. This will not use clusexp.
        In this type of simulations, we consider a solid with a single wyckoff set on which atoms are arranged.
        """
        self.chem = 0  # we'll work with a monoatomic basis
        self.sup = sup
        self.N_units = sup.superlatt[0, 0]
        self.Nsites = len(self.sup.mobilepos)
        self.crys = self.sup.crys
        # vacInd will always be the initial state in the transitions that we consider.
        self.clusexp = clusexp
        self.maxOrder = maxorder
        self.vacSpec = NSpec - 1
        self.Nvac = 1
        self.NSpec = NSpec
        self.mobList = list(range(NSpec))
        self.vacSite = vacSite  # This stays fixed throughout the simulation, so makes sense to store it.
        self.jumpnetwork = jumpnetwork
        self.zeroClusts = zeroClusts
        self.OrigVac = OrigVac

        start = time.time()
        if OrigVac:
            self.SpecClusters, self.SiteSpecInteractIds, self.InteractionIdDict,\
            self.clust2InteractId, self.maxinteractions = self.InteractsOrigVac()
        else:
            self.SpecClusters = self.recalcClusters()
            end1 = time.time()
            print("Built {} clusters:{:.4f} seconds".format(len([cl for clist in self.SpecClusters for cl in clist])
                                                               , end1 - start), flush=True)
            # print("Translating clusters in supercell:", flush=True)
            # self.SiteSpecInteractIds, self.Id2InteractionDict, self.Interaction2IdDict,\
            # self.clust2InteractId, self.InteractionId2ClusId, self.maxinteractions = self.generateSiteSpecInteracts()
            # print("Built interaction Data:{:.4f} seconds".format(time.time() - end1), flush=True)

        # start = time.time()
        # self.genVecClustBasis(self.SpecClusters)
        # print("Built vector bases for clusters : {:.4f}".format(time.time() - start))

        start = time.time()
        if TclusExp is None:
            self.KRAexpander = KRA3bodyInteractions(sup, jumpnetwork, self.chem, TScombShellRange, TSnnRange, TScutoff,
                                                    NSpec, self.Nvac, vacSite)
        print("Built KRA expander : {:.4f}".format(time.time() - start))

        start = time.time()
        self.IndexClusters()  # assign integer integer IDs to each cluster
        self.indexClustertoSpecClus()  # Index clusters to symmetry groups
        print("Built Indexing : {:.4f}".format(time.time() - start))

    def recalcClusters(self):
        """
        Intended to take in a site based cluster expansion and recalculate the clusters with species in them
        """
        allClusts = set()
        symClusterList = []
        self.SpecClust2Clus = {}
        for clSetInd, clSet in enumerate(self.clusexp):
            for clust in list(clSet):
                Nsites = len(clust.sites)
                occupancies = list(itertools.product(self.mobList, repeat=Nsites))
                for siteOcc in occupancies:
                    # Make the cluster site object
                    if self.OrigVac:
                        if siteOcc[0] != self.vacSpec:
                            continue
                    ClustSpec = ClusterSpecies(siteOcc, clust.sites, zero=self.zeroClusts)
                    # check if this has already been considered
                    if ClustSpec in allClusts:
                        continue
                    # Check if number of each species in the cluster is okay
                    mobcount = collections.Counter(siteOcc)
                    # Check if the number of vacancies is kept to the allowed number
                    if mobcount[self.vacSpec] > self.Nvac:
                        continue
                    # Otherwise, find all symmetry-grouped counterparts
                    if self.OrigVac:
                        newSymSet = set([ClusterSpecies.inSuperCell(ClustSpec.g(self.crys, g, zero=self.zeroClusts), self.N_units)
                                         for g in self.crys.G])
                    else:
                        newSymSet = set([ClustSpec.g(self.crys, g, zero=self.zeroClusts) for g in self.crys.G])

                    allClusts.update(newSymSet)
                    newList = list(newSymSet)
                    self.SpecClust2Clus[len(symClusterList)] = clSetInd
                    symClusterList.append(newList)

        return symClusterList

    def genVecClustBasis(self, specClusters):

        vecClustList = []
        vecVecList = []
        clus2LenVecClus = np.zeros(len(specClusters), dtype=int)
        for clListInd, clList in enumerate(specClusters):
            cl0 = clList[0]
            glist0 = []
            if not self.OrigVac:
                for g in self.crys.G:
                    if cl0.g(self.crys, g, zero=self.zeroClusts) == cl0:
                        glist0.append(g)
            else:
                for g in self.crys.G:
                    if ClusterSpecies.inSuperCell(cl0.g(self.crys, g, zero=self.zeroClusts), self.N_units) == cl0:
                        glist0.append(g)

            G0 = sum([g.cartrot for g in glist0])/len(glist0)
            vals, vecs = np.linalg.eig(G0)
            vecs = np.real(vecs)
            vlist = [vecs[:, i]/np.linalg.norm(vecs[:, i]) for i in range(3) if np.isclose(vals[i], 1.0)]
            clus2LenVecClus[clListInd] = len(vlist)

            if clus2LenVecClus[clListInd] == 0:  # If the vector basis is empty, don't consider the cluster
                # vecClustList.append(clList)
                # vecVecList.append([np.zeros(3) for i in range(len(clList))])
                continue

            for v in vlist:
                newClustList = [cl0]
                # The first cluster being the same helps in indexing
                newVecList = [v]
                for g in self.crys.G:
                    if not self.OrigVac:
                        cl1 = cl0.g(self.crys, g, zero=self.zeroClusts)
                    else:
                        cl1 = ClusterSpecies.inSuperCell(cl0.g(self.crys, g, zero=self.zeroClusts), self.N_units)
                    if cl1 in newClustList:
                        continue
                    newClustList.append(cl1)
                    newVecList.append(np.dot(g.cartrot, v))

                vecClustList.append(newClustList)
                vecVecList.append(newVecList)

        self.vecClus, self.vecVec, self.clus2LenVecClus = vecClustList, vecVecList, clus2LenVecClus

    def indexVclus2Clus(self):

        self.Vclus2Clus = np.zeros(len(self.vecClus), dtype=int)
        self.Clus2VClus = collections.defaultdict(list)
        for cLlistInd, clList in enumerate(self.SpecClusters):
            if self.clus2LenVecClus[cLlistInd] == 0:  # If the vector basis is empty, don't consider the cluster
                self.Clus2VClus[cLlistInd] = []
                continue
            cl0 = clList[0]
            for vClusListInd, vClusList in enumerate(self.vecClus):
                clVec0 = vClusList[0]
                if clVec0 == cl0:
                    self.Vclus2Clus[vClusListInd] = cLlistInd
                    self.Clus2VClus[cLlistInd].append(vClusListInd)

        self.Clus2VClus.default_factory = None

    def indexClustertoVecClus(self):
        """
        For a given cluster, store which vector cluster it belongs to
        """
        self.clust2vecClus = collections.defaultdict(list)
        for clListInd, clList in enumerate(self.SpecClusters):
            if self.clus2LenVecClus[clListInd] == 0:  # If the vector basis is empty, don't consider the cluster
                continue
            vecClusIndList = self.Clus2VClus[clListInd]
            for clust1 in clList:
                for vecClusInd in vecClusIndList:
                    for clust2Ind, clust2 in enumerate(self.vecClus[vecClusInd]):
                        if clust1 == clust2:
                            self.clust2vecClus[clust1].append((vecClusInd, clust2Ind))

        self.clust2vecClus.default_factory = None

    def indexClustertoSpecClus(self):
        """
        For a given cluster, store which vector cluster it belongs to
        """
        self.clust2SpecClus = {}
        for clListInd, clList in enumerate(self.SpecClusters):
            for clustInd, clust in enumerate(clList):
                self.clust2SpecClus[clust] = (clListInd, clustInd)

    def InteractsOrigVac(self):
        """
        NOTE : only works for monoatomic lattices for now
        """
        allClusts = set()
        symClusterList = []
        siteA = cluster.ClusterSite(ci=(0, 0), R=np.zeros(self.crys.dim, dtype=int))
        assert siteA == self.vacSite

        for siteInd in range(self.Nsites):
            ciSite, RSite = self.sup.ciR(siteInd)
            clSite = cluster.ClusterSite(ci=ciSite, R=RSite)
            if clSite == siteA:
                continue
            for spec in range(self.NSpec-1):
                siteList = [siteA, clSite]
                specList = [self.NSpec - 1, spec]
                SpCl = ClusterSpecies(specList, siteList, zero=self.zeroClusts)
                if SpCl in allClusts:
                    continue
                # Apply group operations
                newsymset = set([ClusterSpecies.inSuperCell(SpCl.g(self.crys, g, zero=self.zeroClusts), self.N_units)
                             for g in self.crys.G])
                allClusts.update(newsymset)
                symClusterList.append(list(newsymset))

        allSpCl = [cl for clSet in symClusterList for cl in clSet]

        # index the clusters
        clust2InteractId = collections.defaultdict(list)
        InteractionIdDict = {}
        for i, SpCl in enumerate(allSpCl):
            self.Clus2Num[SpCl] = i
            clust2InteractId[SpCl].append(i)
            self.Num2Clus[i] = SpCl
            InteractionIdDict[i] = tuple(sorted([(self.sup.index(st.R, st.ci)[0], spec)
                                                 for st, spec in SpCl.SiteSpecs],
                                                key=lambda x: x[0]))

        SiteSpecinteractIds = collections.defaultdict(list)
        for clSet in symClusterList:
            for cl in clSet:
                Id = self.Clus2Num[cl]
                site = cl.siteList[1]
                siteInd = self.sup.index(ci=site.ci, R=site.R)
                spec = cl.specList[1]
                SiteSpecinteractIds[(siteInd, spec)].append(Id)

        SiteSpecinteractIds.default_factory = None
        maxinteractions = max([len(lst) for key, lst in SiteSpecinteractIds.items()])

        return symClusterList, SiteSpecinteractIds, InteractionIdDict, clust2InteractId, maxinteractions

    def generateSiteSpecInteracts(self):
        """
        generate interactions for every site - for MC moves
        """
        allLatTransTuples = [self.sup.ciR(siteInd) for siteInd in range(self.Nsites)]
        Id2InteractionDict = {}
        Interaction2IdDict = {}
        SiteSpecInteractIds = collections.defaultdict(list)
        clust2InteractId = collections.defaultdict(list)
        InteractionId2ClusId = {}

        count = 0
        # Traverse through all the unit cells in the supercell
        for translateInd in tqdm(range(self.Nsites), position=0, leave=True):
            # Now, go through all the clusters and translate by each lattice translation
            for clID, cl in self.Num2Clus.items():
                # get the cluster site
                R = allLatTransTuples[translateInd][1]
                # translate all sites with this translation
                interactSupInd = tuple(sorted([(self.sup.index(st.R + R, st.ci)[0], spec)
                                               for st, spec in cl.SiteSpecs],
                                              key=lambda x: x[0]))
                if interactSupInd in Id2InteractionDict:
                    raise ValueError("Interaction encountered twice while either translating same cluster differently"
                                     "or different clusters.")
                # give the new interaction an Index
                Id2InteractionDict[count] = interactSupInd
                Interaction2IdDict[interactSupInd] = count


                # For every rep cluster, store which interactions they produce
                clust2InteractId[clID].append(count)
                InteractionId2ClusId[count] = clID

                # For every site and species, store which interaction they belong to
                for siteInd, spec in interactSupInd:
                    SiteSpecInteractIds[(siteInd, spec)].append(count)

                # Increment the index
                count += 1

        maxinteractions = max([len(lst) for key, lst in SiteSpecInteractIds.items()])

        SiteSpecInteractIds.default_factory = None
        clust2InteractId.default_factory = None

        self.SiteSpecInteractIds, self.Id2InteractionDict, self.Interaction2IdDict, \
        self.clust2InteractId, self.InteractionId2ClusId, self.maxinteractions = \
            SiteSpecInteractIds, Id2InteractionDict, Interaction2IdDict, clust2InteractId, InteractionId2ClusId, maxinteractions

    def IndexClusters(self):
        """
        Assign a unique integer to each representative cluster. To help identifying them in JIT operations
        """
        allSpCl = [SpCl for SpClList in self.SpecClusters for SpCl in SpClList]
        self.Clus2Num = {}
        self.Num2Clus = {}

        for i, SpCl in enumerate(allSpCl):
            self.Clus2Num[SpCl] = i
            self.Num2Clus[i] = SpCl

    def makeJitInteractionsData(self, Energies):
        """
        Function to represent all the data structures in the form of numpy arrays so that they can be accelerated with
        numba's jit compilations.
        Data structures to cast into numpy arrays:
        SiteInteractions
        KRAexpander.clusterSpeciesJumps - these correspond to transitions - We'll proceed with this later on
        """

        # first, we assign unique integers to interactions
        # while we're at it, let's also store which siteSpec contains which interact
        numInteractsSiteSpec = np.zeros((self.Nsites, self.NSpec), dtype=int)
        SiteSpecInterArray = np.full((self.Nsites, self.NSpec, self.maxinteractions), -1, dtype=int)

        for key, interactIdList in self.SiteSpecInteractIds.items():
            keySite = key[0]  # the "index" function applies PBC to sites outside sup.
            keySpec = key[1]
            numInteractsSiteSpec[keySite, keySpec] = len(interactIdList)
            for interactNum, interactId in enumerate(interactIdList):
                SiteSpecInterArray[keySite, keySpec, interactNum] = interactId

        print("Done Indexing interactions")
        # Now that we have integers assigned to all the interactions, let's store their data as numpy arrays
        numInteracts = len(self.Id2InteractionDict)

        # 1. Store chemical data
        # we'll need the number of sites in each interaction
        numSitesInteracts = np.zeros(numInteracts, dtype=int)

        # and the supercell sites in each interaction
        SupSitesInteracts = np.full((numInteracts, self.maxOrder), -1, dtype=int)

        # and the species on the supercell sites in each interaction
        SpecOnInteractSites = np.full((numInteracts, self.maxOrder), -1, dtype=int)

        for (key, interaction) in self.Id2InteractionDict.items():
            numSitesInteracts[key] = len(interaction)
            for idx, (interactSite, interactSpec) in enumerate(interaction):
                SupSitesInteracts[key, idx] = interactSite
                SpecOnInteractSites[key, idx] = interactSpec

        print("Done chemical data of interactions ")

        # 2. Store energy data and vector data
        Interaction2En = np.zeros(numInteracts, dtype=float)
        for repClusInd, interactionList in self.clust2InteractId.items():
            repClus = self.Num2Clus[repClusInd]
            for idx in interactionList:
            # get the energy index here
                Interaction2En[idx] = Energies[self.clust2SpecClus[repClus][0]]
        print("Done energy data for interactions")

        return numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En,\
               numInteractsSiteSpec, SiteSpecInterArray

    def makeJitVectorBasisData(self):
        numInteracts = len(self.Id2InteractionDict)
        numVecsInteracts = np.full(numInteracts, -1, dtype=int)
        VecsInteracts = np.zeros((numInteracts, 3, 3))
        VecGroupInteracts = np.full((numInteracts, 3), -1, dtype=int)
        for repClusInd, interactionList in self.clust2InteractId.items():
            repClus = self.Num2Clus[repClusInd]
            for idx in interactionList:
                # get the vector basis data here
                # if vector basis is empty, keep no of elements to -1.
                if self.clus2LenVecClus[self.clust2SpecClus[repClus][0]] == 0:
                    continue
                vecList = self.clust2vecClus[repClus]
                # store the number of vectors in the basis
                numVecsInteracts[idx] = len(vecList)
                # store the vector
                for vecidx, tup in enumerate(vecList):
                    VecsInteracts[idx, vecidx, :] = self.vecVec[tup[0]][tup[1]].copy()
                    VecGroupInteracts[idx, vecidx] = tup[0]
        print("Done Vector data for interactions")
        return numVecsInteracts, VecsInteracts, VecGroupInteracts

    def makeSiteIndToSite(self):
        Nsites = self.Nsites
        N_units = self.sup.superlatt[0, 0]
        siteIndtoR = np.zeros((Nsites, 3), dtype=int)
        RtoSiteInd = np.zeros((N_units, N_units, N_units), dtype=int)

        for siteInd in range(Nsites):
            R = self.sup.ciR(siteInd)[1]
            siteIndtoR[siteInd, :] = R
            RtoSiteInd[R[0], R[1], R[2]] = siteInd
        return siteIndtoR, RtoSiteInd