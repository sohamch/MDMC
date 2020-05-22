from onsager import crystal, cluster, supercell
import numpy as np
import itertools
import collections


class KRAExpand(object):
    """
    Object that contains all information regarding the KRA expansion of a jumpnetwork in a supercell.
    """
    def __init__(self, sup, chem, jumpnetwork, clusexp, mobCountList, vacSite):
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
        self.vacSite = vacSite  # We'll be concerned with only those jumps where this is site A in A->B jumps.
                                # The reverse jumps are not jumps out of this state, we need not worry about them.
        self.mobCountList = mobCountList
        # Index the jump network into an array
        self.IndexJumps()
        # First, we reform the jumpnetwork
        self.TSClusters = cluster.makeTSclusters(sup.crys, chem, jumpnetwork, clusexp)
        self.SymTransClusters = self.GroupTransClusters()
        self.clusterSpeciesJumps = self.defineTransSpecies()

    def IndexJumps(self):
        jumpFinSite = []
        jumpdx = []
        for jump in [jmp for jmpList in self.jumpnetwork for jmp in jmpList]:

            siteA = cluster.ClusterSite(ci=(self.chem, jump[0][0]), R=np.zeros(3, dtype=int))
            if siteA != self.vacSite:
                # if the initial site of the vacancy is not the vacSite, it is not a jump out of this state.
                # Ignore it - removes reverse jumps from multi-site, single-Wyckoff lattices.
                continue

            # Rj + uj = ui + dx (since Ri is zero in the jumpnetwork)
            Rj, (c, cj) = self.crys.cart2pos(jump[1] + np.dot(self.crys.lattice, self.crys.basis[self.chem][jump[0][0]]))
            # check we have the correct site
            if not cj == jump[0][1]:
                raise ValueError("improper coordinate transformation, did not get same final jump site")

            siteB = cluster.ClusterSite(ci=(self.chem, jump[0][1]), R=Rj)

            indA = self.sup.index(siteA.R, siteA.ci)[0]
            assert indA == self.sup.index(self.vacSite.R, self.vacSite.ci)[0]
            indB = self.sup.index(siteB.R, siteB.ci)[0]

            jumpFinSite.append(indB)
            jumpdx.append(jump[1])

        # cast into numpy arrays - to be used in KRA expansions
        self.ijList = np.array(jumpFinSite, dtype=int)
        self.dxList = np.array(jumpdx, dtype=float)

    def GroupTransClusters(self):
        TransClustersAll = collections.defaultdict(list)
        TransClustersSym = {}

        for clList in self.TSClusters:
            for clust in clList:
                # get supercell indices of the jump sites.
                siteA = clust.sites[0]
                siteB = clust.sites[1]

                if siteA != self.vacSite:  # If the jump is not out of the vacancy site, don't consider it.
                    continue

                IndA = self.sup.index(siteA.R, siteA.ci)[0]
                IndB = self.sup.index(siteB.R, siteB.ci)[0]

                TransClustersAll[(IndA, IndB)].append(clust)

        for key, clustList in TransClustersAll.items():
            ciA, RA = self.sup.ciR(key[0])
            ciB, RB = self.sup.ciR(key[1])
            siteA = cluster.ClusterSite(ci=ciA, R=RA)
            siteB = cluster.ClusterSite(ci=ciB, R=RB)
            Glist = []
            for g in self.crys.G:
                siteANew = siteA.g(self.crys, g)
                siteBNew = siteB.g(self.crys, g)
                if siteA == siteANew and siteB == siteBNew:
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

    def defineTransSpecies(self):
        """
        Used to assign chemical species to the jump cluster expansions.
        We'll have separate expansions for every species that occupies the final site of a carrier jump,
        and every cluster for a given jump will have species assigned to it.

        :returns  SpecClusterJumpList - a cluster expansion for KRA with species assigned to the clusters.
        """

        Nmobile = len(self.mobCountList)

        mobileSpecs = tuple(range(Nmobile - 1))  # the last species is the vacancy, so we are not considering it.
        # print("mobilespecs :{} \n".format(mobileSpecs))
        clusterJumpsSpecies = {}
        for AB, clusterSymLists in self.SymTransClusters.items():
            # For this transition, first assign species to the clusters
            AtomicClusterSymList = []
            for clusterList in clusterSymLists:
                cl0 = clusterList[0]
                # Get the order of the cluster and assign species to the sites
                # remember that we have removed vacancies from being considered in mobileSpecs
                # Norder for TSClusters is 2 less than the actual site count, to exclude initial final sites from any
                # "touching" so to speak - specI is always vacancy, we'll assign specJ later on
                Specs = itertools.product(mobileSpecs, repeat=cl0.Norder)
                for specJ in range(Nmobile - 1):
                    ABspecJ = (AB[0], AB[1], specJ)
                    for tup in Specs:
                        # Check if the number of atoms crosses the total number of atoms of a species.
                        MobNumber = collections.Counter(tup)
                        if specJ in MobNumber:
                            MobNumber[specJ] += 1
                        else:
                            MobNumber[specJ] = 1
                        # print(len(self.mobCountList), MobNumber)
                        if any(self.mobCountList[i] < j for i, j in MobNumber.items()):
                            continue
                        AtomicClusterSymList.append([tup, clusterList])
                    clusterJumpsSpecies[ABspecJ] = AtomicClusterSymList
            # use itertools.product like in normal cluster expansion.
            # Then, assign species to the final site of the jumps.
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
        I, J, specJ = transition[0], transition[1], transition[2]
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
        for interactIdx, (tups, clusterList) in zip(itertools.count(), SymClusterlists):
            for cluster in clusterList:
                if all(mobOcc[spec, self.sup.index(site.R, site.ci)[0]] == 1 for spec, site in zip(tups, cluster.sites[2:])):
                    DelEKRA += KRACoeffs[interactIdx]
        return DelEKRA
        # Next, we need the contributions of the initial and final states.
        # Have to check which clusters are on and which are off

    # Next, build numpy arrays for jitting
    def makeTransJitData(self):
        pass
