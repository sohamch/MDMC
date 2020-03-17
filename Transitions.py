from onsager import crystal, cluster, supercell
import numpy as np
import itertools
import collections


class KRAExpand(object):
    """
    Object that contains all information regarding the KRA expansion of a jumpnetwork in a supercell.
    """
    def __init__(self, sup, chem, jumpnetwork, clusexp, mobCountList):
        """
        :param sup: clusterSupercell Object
        :param chem: the sublattice index on which the jumpnetwork has been built.
        :param jumpnetwork: jumpnetwork to expand
        :param mobCountList : total count for each species in the supercell.
        :param clusexp: representative set of clusters - out put of make clusters function.
        """
        self.sup = sup
        self.chem = chem
        self.crys = self.sup.crys
        self.jumpnetwork = jumpnetwork
        self.clusexp = clusexp
        self.mobCountList = mobCountList

        # First, we reform the jumpnetwork
        self.TSClusters = cluster.makeTSclusters(sup.crys, chem, jumpnetwork, clusexp)
        self.SymTransClusters = self.GroupTransClusters()
        self.clusterSpeciesJumps = self.defineTransSpecies()
        self.clusterSpecies = self.defineSpecies()

    def GroupTransClusters(self):
        TransClustersAll = collections.defaultdict(list)
        TransClustersSym = {}

        for clList in self.TSClusters:
            for clust in clList:
                # get supercell indices of the jumps.
                siteA = clust.sites[0]
                siteB = clust.sites[1]

                IndA = self.sup.index(siteA.R, siteA.ci)
                IndB = self.sup.index(siteB.R, siteB.ci)

                TransClustersAll[(IndA, IndB)].append(clust)

        for key, clustList in TransClustersAll.items():
            Glist = []
            for g in self.crys.G:
                siteANew = key[0].g(self.crys, g)
                siteBNew = key[1].g(self.crys, g)

                if key[0] == siteANew and key[1] == siteBNew:
                    Glist.append(g)
            newSymList = []
            clusts_done = set()
            for clust in clustList:
                if clust not in clusts_done:
                    clusterSetnew = set([clust.g(self.crys, gop) for gop in Glist])
                    newSymList.append(list(clusterSetnew))
                    clusts_done.update(clusterSetnew)

            TransClustersSym[key] = newSymList

        return TransClustersSym

    def defineSpecies(self):
        """
        Used to assign chemical species to site clusters
        :return: symmetry grouped cluster expansions, with species assigned to sites
        """
        clusterSpecies = []
        for clistInd, clist in enumerate(self.clusexp):
            cl0 = clist[0]  # get the representative cluster
            # cluster.sites is a tuple, which maintains the order of the elements.
            Nmobile = len(cl0.sites)
            arrangemobs = itertools.product(range(len(self.mobCountList)), repeat=Nmobile)
            # arrange mobile sites on mobile species.

            for tup in arrangemobs:
                mobcount = collections.Counter(tup)
                # Check if the number of atoms of a given species does not exceed the total number of atoms of that
                # species in the solid.
                if any(j > self.mobCountList[i] for i, j in mobcount.items()):
                    continue
                # TODO - can we check for absence of vacancy at origin and ignore it?
                clusterSpecies.append([tup, clist])
        return clusterSpecies

    def defineTransSpecies(self):
        """
        Used to assign chemical species to the jump cluster expansions.

        :param: mobOccs - this is the starting/initial state and is only used to get the number of species of each kind.

        We'll have separate expansions for every species that occupies the final site of a carrier jump,
        and every cluster for a given jump will have species assigned to it.

        :returns  SpecClusterJumpList - a cluster expansion for KRA with species assigned to the clusters.
        """

        Nmobile = len(self.mobCountList)
        clusterJumps = getattr(self, "clusterJumps", None)
        if clusterJumps is None:
            raise ValueError("Need to generate cluster expansions for the jumps first.")

        mobileSpecs = tuple(range(Nmobile-1))  # the last species is the vacancy, so we are not considering it.
        clusterJumpsSpecies = {}
        for AB, clusterSymLists in self.SymTransClusters.items():
            # For this transition, first assign species to the clusters
            AtomicClusterSymList = []
            for clusterList in clusterSymLists:
                cl0 = clusterList[0].cluster
                # Get the order of the cluster and assign species to the sites
                Specs = itertools.product(mobileSpecs, repeat=cl0.Norder - 2)  # take away the first two sites
                for tup in Specs:
                    # Check if the number of atoms crosses the total number of atoms of a species.
                    MobNumber = collections.Counter(tup)
                    if any(self.mobCountList[i] < j for i, j in MobNumber.items()):
                        continue
                    AtomicClusterSymList.append([tup, clusterList])
            # use itertools.product like in normal cluster expansion.
            # Then, assign species to te final site of the jumps.
            for specJ in range(Nmobile-1):
                ABspecJ = (AB[0], AB[1], specJ)
                clusterJumpsSpecies[ABspecJ] = AtomicClusterSymList

        return clusterJumpsSpecies

    def GetKRA(self, transition, mobOcc, KRACoeffs):
        """
        Given a transition and a state, get the KRA activation value for that jump in that state.
        During testing, we'll assume that fitting KRA coefficients has already been done.
        :param transition: the transition in the form of ((I,J), dx) - supercell indices
        :param mobOcc: the state as an array of occupancy vectors.
        :param specJ: the species that exchanges with the vacancies.
        :param KRACoeffs: the KRA coefficients for that type of jump, arranged appropriately for each cluster.
        :return: The Calculated KRA activation energy.
        """
        I, J, dx = transition[0][0], transition[0][1], transition[1]
        specJ = sum([occ[J]*spec for spec, occ in zip(itertools.count(), mobOcc)])
        SymClusterlists = self.clusterSpeciesJumps[(I, J, specJ)]

        # SymClusterlists : Key=(A,B,SpecJ) :[[tup11, clusterlist1], [tup12, clusterlist1],...
        # ...,[tup21, clusterlist2], [tup22, clusterlist2],....]
        if not len(SymClusterlists) == len(KRACoeffs):  # Every clusterlist with a species must have its own
            # coefficients
            raise TypeError("Number of KRA coefficients entered does not match the number of clusters"
                            "for the transition")
        # Now, check which clusters are on and calculate the KRA values
        DelEKRA = 0
        # How do we speed this up?
        for interactIdx, (tup, clusterList) in zip(itertools.count(), SymClusterlists):
            for cluster in clusterList:
                if all([mobOcc[spec][site] == 1 for spec, site in zip(tup, cluster.sites[2:])]):
                    DelEKRA += KRACoeffs[interactIdx]

        return DelEKRA
        # Next, we need the contributions of the initial and final states.
        # Have to check which clusters are on and which are off
