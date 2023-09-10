from onsager import cluster
import numpy as np
import collections
import itertools
from ClustSpec import ClusterSpecies
from onsager import crystal
import time
from tqdm import tqdm
from functools import reduce


class VectorClusterExpansion(object):
    """
    class to expand velocities and rates in vector cluster functions.
    """
    def __init__(self, sup, clusexp, NSpec, vacSite, vacSpec, maxorder, Nvac=1, MadeSpecClusts=None, zeroClusts=True):
        """
        Cluster expansion for mono-atomic lattices
        :param sup : clusterSupercell object
        :param clusexp: cluster expansion about a single unit cell.
        :param NSpec: no. of species to consider (including vacancies)
        :param vacSite: Index of the vacancy site (used in MC sampling and JIT construction)
        :param vacSite: the site of the vacancy as a clusterSite object. This does not change during the simulation.
        :param maxorder: the maximum order of a cluster in clusexp.
        :param MadeSpecClusts: an optional pre-made species cluster group list.
        :param zeroClusts: Same as parameter "zero" of ClusterSpecies class - whether to bring a cluster's centroid to zero or not.
        """
        self.chem = 0  # we'll work with a monoatomic basis
        self.sup = sup
        self.N_units = sup.superlatt[0, 0]
        self.Nsites = len(self.sup.mobilepos)
        self.crys = self.sup.crys
        # vacInd will always be the initial state in the transitions that we consider.
        self.clusexp = clusexp
        self.maxOrder = maxorder
        self.vacSpec = vacSpec
        self.Nvac = Nvac
        self.NSpec = NSpec
        if self.vacSpec >= self.NSpec:
            print(vacSpec, NSpec)
            raise ValueError("Vacancy label ({0}) must be less than the number of species ({1}).".format(self.vacSpec, self.NSpec))

        self.mobList = list(range(NSpec))
        self.vacSite = vacSite  # This stays fixed throughout the simulation, so makes sense to store it.
        self.zeroClusts = zeroClusts

        start = time.time()
        if MadeSpecClusts is not None:
            self.SpecClusters = MadeSpecClusts
        else:
            self.SpecClusters = self.recalcClusters()
        end1 = time.time()
        print("Built {} clusters:{:.4f} seconds".format(len([cl for clist in self.SpecClusters for cl in clist]),
                                                        end1 - start), flush=True)

        start = time.time()
        self.IndexClusters(self.SpecClusters)  # assign integer integer IDs to each cluster
        self.indexClustertoSpecClus(self.SpecClusters)  # Index clusters to symmetry groups
        print("Built Indexing : {:.4f}".format(time.time() - start))

    def recalcClusters(self):
        """
        Intended to take in a site based cluster expansion and recalculate the clusters with species in them
        """
        allClusts = set()
        symClusterList = []
        # self.SpecClust2Clus = {}
        for clSetInd, clSet in enumerate(self.clusexp):
            for clust in list(clSet):
                Nsites = len(clust.sites)
                occupancies = list(itertools.product(self.mobList, repeat=Nsites))
                for siteOcc in occupancies:
                    # Make the cluster site object
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
                    newSymSet = set([ClustSpec.g(self.crys, g) for g in self.crys.G])

                    allClusts.update(newSymSet)
                    newList = list(newSymSet)
                    # self.SpecClust2Clus[len(symClusterList)] = clSetInd
                    symClusterList.append(newList)

        AllAfterSym = [cl for clList in symClusterList for cl in clList]
        assert set(AllAfterSym) == allClusts

        return symClusterList

    def genVecClustBasis(self, specClusters):

        vecClustList = []
        vecVecList = []
        clus2LenVecClus = np.zeros(len(specClusters), dtype=int)
        for clListInd, clList in enumerate(specClusters):
            cl0 = clList[0]
            glist0 = []
            for g in self.crys.G:
                if cl0.g(self.crys, g) == cl0:
                    glist0.append(g)

            vb = reduce(crystal.CombineVectorBasis, [crystal.VectorBasis(*g.eigen()) for g in glist0])
            # Get orthonormal vectors
            vlist = self.crys.vectlist(vb)
            vlist = [vec/np.linalg.norm(vec) for vec in vlist]
            clus2LenVecClus[clListInd] = len(vlist)

            if clus2LenVecClus[clListInd] == 0:  # If the vector basis is empty, don't consider the cluster
                continue

            for v in vlist:
                newClustList = [cl0]
                # The first cluster being the same helps in indexing
                newVecList = [v]
                for g in self.crys.G:
                    cl1 = cl0.g(self.crys, g)
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

    def indexClustertoSpecClus(self, SpecClusters):
        """
        For a given cluster, store which vector cluster it belongs to
        """
        self.clust2SpecClus = {}
        for clListInd, clList in enumerate(SpecClusters):
            for clustInd, clust in enumerate(clList):
                self.clust2SpecClus[clust] = (clListInd, clustInd)

    def generateSiteSpecInteracts(self, reqSites=None):
        """
        generate interactions for every site - for MC moves
        :param reqSites : list of required sites - if None (default), all sites will be considered
        """
        allLatTransTuples = [self.sup.ciR(siteInd) for siteInd in range(self.Nsites)]
        Id2InteractionDict = {}
        Interaction2IdDict = {}
        SiteSpecInteractIds = collections.defaultdict(list)
        clust2InteractId = collections.defaultdict(list)
        InteractionId2ClusId = {}

        n_req = 0 if reqSites is None else len(reqSites)

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

                if n_req > 0 and not any([IntSite in reqSites for (IntSite, sp) in interactSupInd]):
                    # skip the interaction if it doesn't contain required sites
                    # beneficial for pre-made data sets, where only vacancy and its jump site need to be considered
                    continue

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

        SiteSpecInteractIds = dict(SiteSpecInteractIds)
        clust2InteractId = dict(clust2InteractId)

        maxinteractions = max([len(lst) for key, lst in SiteSpecInteractIds.items()])


        self.SiteSpecInteractIds, self.Id2InteractionDict, self.Interaction2IdDict, \
        self.clust2InteractId, self.InteractionId2ClusId, self.maxinteractions = \
            SiteSpecInteractIds, Id2InteractionDict, Interaction2IdDict, clust2InteractId, InteractionId2ClusId, maxinteractions

    def IndexClusters(self, SymmClusterList):
        """
        Assign a unique integer to each representative cluster. To help identifying them in JIT operations
        """
        allSpCl = [SpCl for SpClList in SymmClusterList for SpCl in SpClList]
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

        # print("Done Indexing interactions")
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

        # print("Done chemical data of interactions ")

        # 2. Store energy data and vector data
        Interaction2En = np.zeros(numInteracts, dtype=float)
        for repClusInd, interactionList in self.clust2InteractId.items():
            repClus = self.Num2Clus[repClusInd]
            for idx in interactionList:
            # get the energy index here
                Interaction2En[idx] = Energies[self.clust2SpecClus[repClus][0]]
        # print("Done energy data for interactions")

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
        # print("Done Vector data for interactions")
        return numVecsInteracts, VecsInteracts, VecGroupInteracts


from numba.experimental import jitclass
from numba import int64, float64

# Paste all the function definitions here as comments
JitSpec = [
    ("numInteractsSiteSpec", int64[:, :]),
    ("SiteSpecInterArray", int64[:, :, :]),
    ("numSitesInteracts", int64[:]),
    ("SupSitesInteracts", int64[:, :]),
    ("SpecOnInteractSites", int64[:, :]),
    ("Interaction2En", float64[:]),
    ("mobOcc", int64[:]),
    ("Nsites", int64),
    ("Nspecs", int64),
    ("OffSiteCount", int64[:]),
    ("vacSpec", int64),
]


@jitclass(JitSpec)
class JITExpanderClass(object):

    def __init__(self, vacSpec, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites,
                 Interaction2En, numInteractsSiteSpec, SiteSpecInterArray):

        self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites, self.Interaction2En, self.numInteractsSiteSpec,\
        self.SiteSpecInterArray = \
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numInteractsSiteSpec, SiteSpecInterArray

        self.vacSpec = vacSpec

        # check if proper sites and species data are entered
        self.Nsites, self.Nspecs = numInteractsSiteSpec.shape[0], numInteractsSiteSpec.shape[1]

    def DoSwapUpdate(self, state, siteA, siteB, lenVecClus, OffSiteCount,
                     numVecsInteracts, VecGroupInteracts, VecsInteracts):

        del_lamb = np.zeros((lenVecClus, 3))
        delE = 0.0
        # Switch required sites off
        for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteA]]):
            # check if an interaction is on
            interMainInd = self.SiteSpecInterArray[siteA, state[siteA], interIdx]
            if OffSiteCount[interMainInd] == 0:
                delE -= self.Interaction2En[interMainInd]
                # take away the vectors for this interaction
                if numVecsInteracts is not None:
                    for i in range(numVecsInteracts[interMainInd]):
                        del_lamb[VecGroupInteracts[interMainInd, i]] -= VecsInteracts[interMainInd, i, :]
            OffSiteCount[interMainInd] += 1

        for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
            interMainInd = self.SiteSpecInterArray[siteB, state[siteB], interIdx]
            if OffSiteCount[interMainInd] == 0:
                delE -= self.Interaction2En[interMainInd]
                if numVecsInteracts is not None:
                    for i in range(numVecsInteracts[interMainInd]):
                        del_lamb[VecGroupInteracts[interMainInd, i]] -= VecsInteracts[interMainInd, i, :]
            OffSiteCount[interMainInd] += 1

        # Next, switch required sites on
        for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
            interMainInd = self.SiteSpecInterArray[siteA, state[siteB], interIdx]
            OffSiteCount[interMainInd] -= 1
            if OffSiteCount[interMainInd] == 0:
                delE += self.Interaction2En[interMainInd]
                # add the vectors for this interaction
                if numVecsInteracts is not None:
                    for i in range(numVecsInteracts[interMainInd]):
                        del_lamb[VecGroupInteracts[interMainInd, i]] += VecsInteracts[interMainInd, i, :]

        for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
            interMainInd = self.SiteSpecInterArray[siteB, state[siteA], interIdx]
            OffSiteCount[interMainInd] -= 1
            if OffSiteCount[interMainInd] == 0:
                delE += self.Interaction2En[interMainInd]
                # add the vectors for this interaction
                # for interactions with zero vector basis, numVecsInteracts[interMainInd] = -1 and the
                # loop doesn't run
                if numVecsInteracts is not None:
                    for i in range(numVecsInteracts[interMainInd]):
                        del_lamb[VecGroupInteracts[interMainInd, i]] += VecsInteracts[interMainInd, i, :]

        return delE, del_lamb

    def revert(self, offsc, state, siteA, siteB):
        for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteA]]):
            offsc[self.SiteSpecInterArray[siteA, state[siteA], interIdx]] -= 1

        for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
            offsc[self.SiteSpecInterArray[siteB, state[siteB], interIdx]] -= 1

        for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
            offsc[self.SiteSpecInterArray[siteA, state[siteB], interIdx]] += 1

        for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
            offsc[self.SiteSpecInterArray[siteB, state[siteA], interIdx]] += 1

    def getLambda(self, offsc, NVclus, numVecsInteracts, VecGroupInteracts, VecsInteracts):
        lamb = np.zeros((NVclus, 3))
        for interactInd in range(offsc.shape[0]):
            if offsc[interactInd] == 0:
                for vGInd in range(numVecsInteracts[interactInd]):
                    vGroup = VecGroupInteracts[interactInd, vGInd]
                    vec = VecsInteracts[interactInd, vGInd]
                    lamb[vGroup, :] += vec
        return lamb

    def getDelLamb(self, state, offsc, siteA, siteB, lenVecClus,
                   numVecsInteracts, VecGroupInteracts, VecsInteracts):


        _, del_lamb = self.DoSwapUpdate(state, siteA, siteB, lenVecClus, offsc,
                                        numVecsInteracts, VecGroupInteracts, VecsInteracts)

        # Then revert back off site count to original values
        self.revert(offsc, state, siteA, siteB)

        return del_lamb

    def Expand(self, state, ijList, dxList, spec, OffSiteCount,
               numVecsInteracts, VecGroupInteracts, VecsInteracts,
               lenVecClus, vacSiteInd, RateList):

        del_lamb_mat = np.zeros((lenVecClus, lenVecClus, ijList.shape[0]))
        delxDotdelLamb = np.zeros((lenVecClus, ijList.shape[0]))
        ratelist = RateList.copy()

        siteA, specA = vacSiteInd, state[vacSiteInd]
        # go through all the transition

        for jumpInd in range(ijList.shape[0]):
            # Get the transition site and species
            siteB, specB = ijList[jumpInd], state[ijList[jumpInd]]

            # next, calculate the energy and basis function change due to site swapping
            delE, del_lamb = self.DoSwapUpdate(state, siteA, siteB, lenVecClus, OffSiteCount,
                                               numVecsInteracts, VecGroupInteracts, VecsInteracts)

            # Next, restore OffSiteCounts to original values for next jump
            self.revert(OffSiteCount, state, siteA, siteB)
            del_lamb_mat[:, :, jumpInd] = np.dot(del_lamb, del_lamb.T)

            # let's do the tensordot by hand (numba doesn't support np.tensordot)
            for i in range(lenVecClus):
                if spec == specA:
                    delxDotdelLamb[i, jumpInd] = np.dot(del_lamb[i, :], dxList[jumpInd, :])
                elif spec == specB:
                    delxDotdelLamb[i, jumpInd] = np.dot(del_lamb[i, :], -dxList[jumpInd, :])

        WBar = np.zeros((lenVecClus, lenVecClus))
        for i in range(lenVecClus):
            WBar[i, i] += np.dot(del_lamb_mat[i, i, :], ratelist)
            for j in range(i):
                WBar[i, j] += np.dot(del_lamb_mat[i, j, :], ratelist)
                WBar[j, i] = WBar[i, j]

        BBar = np.zeros(lenVecClus)
        for i in range(lenVecClus):
            BBar[i] = np.dot(ratelist, delxDotdelLamb[i, :])

        return WBar, BBar, ratelist