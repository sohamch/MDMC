from onsager import crystal, cluster, supercell
import numpy as np

"""
This is to create KRA expansions of activation barriers. We assume for now that the expansion coefficients are given.
"""

class ClusterTranslation(object):

    def __init__(self, RepCluster, Rtrans):
        self.cluster = RepCluster
        self.R = Rtrans

    def __hash__(self):
        return hash((self.cluster, self.R[0], self.R[1], self.R[2]))

    def g(self, crys, g):
        Rnew = crys.dot(g.rot, self.R)
        clusterNew = self.cluster.g(crys, g)
        return self.__class__(clusterNew, Rnew)

class KRAExpand(object):
    """
    Object that contains all information regarding the KRA expansion of a jumpnetwork in a supercell.
    """
    def __init__(self, sup, chem, jumpnetwork, clusexp, InteractionCutoff):
        """
        :param sup: clusterSupercell Object
        :param chem: the sublattice index on which the jumpnetwork has been built.
        :param jumpnetwork: jumpnetwork to expand
        :param clusexp: representative set of clusters - out put of make clusters function.
        :param InteractionCutoff: the maximum distance of a cluster from a transition to consider
        """
        self.sup = sup
        self.chem = chem
        self.crys = self.sup.crys
        self.jumpnetwork = jumpnetwork
        self.clusexp = clusexp
        self.cutoff = InteractionCutoff

        # First, we reform the jumpnetwork
        self.clusterJumps = self.reDefineJumps()

    def reDefineJumps(self):
        """
        Redefining the jumps in terms of the initial and final states involved. Since KRA values are same for both
        forward and backward jumps, we will represent both with the same definition.
        :return - clusterJumps - all the jumps defined as two supercell sites (a,b) - the clusters associated with the KRA
        of the jumps, grouped together into lists so that all clusters in a list are related by transition-invariant
        symmetry operations (that is, those group operations that leave (a, b) unchanged.
        """
        # Go through the jumps in the jumpnetwork
        # Define Rvecs.
        Rvecs = []
        nmax = [int(np.round(self.cutoff*self.cutoff/self.crys.metric[i, i])) + 1 for i in range(3)]
        Rvecs = [np.array([n0, n1, n2])
                 for n0 in range(-nmax[0], nmax[0] + 1)
                 for n1 in range(-nmax[1], nmax[1] + 1)
                 for n2 in range(-nmax[2], nmax[2] + 1)]

        clusterjumplist = {}
        for jlist in self.jumpnetwork:
            for ((i, j), dx) in jlist:
                siteA = self.sup.index((self.chem, i), np.zeros(3, dtype=int))
                Rj, (c, cj) = self.crys.cart2pos(dx + np.dot(self.crys.lattice, self.crys.basis[self.chem][j]))
                # check we have the correct site
                if not cj == j:
                    raise ValueError("improper coordinate transformation, did not get same site")
                siteB = self.sup.index((self.chem, j), Rj)
                newtrans = [R + Rj for R in Rvecs] + Rvecs

                # build "point group" for this jump, from the space group of the crystal
                ijPtGroup = []
                for gop in self.crys.G:
                    siteAnew = siteA.g(self.crys, gop)
                    siteBnew = siteB.g(self.crys, gop)

                    if siteA == siteAnew and siteB==siteBnew:
                        ijPtGroup.append(gop)

                # start cluster translations
                ijClexp = []
                clTracker = set()
                for Rtrans in newtrans:
                    # take a cluster and translate it.
                    for cl in [clust for clustList in self.clusexp for clust in clustList]:
                        # translate all the sites in the cluster
                        newsiteList = [site + Rtrans for site in cl.sites]
                        # The new cluster should not contain the initial or final site of the jump.
                        if siteA in newsiteList or siteB in newsiteList:
                            continue
                        # Get distances of new sites from initial and final state
                        dists_i = [np.dot(self.crys.lattice, (site.R - siteA.R + self.crys.basis[site.ci[0]][site.ci[1]]
                                                              - self.crys.basis[siteA.ci[0]][siteA.ci[1]]))
                                   for site in newsiteList]
                        dists_j = [np.dot(self.crys.lattice, (site.R - siteB.R + self.crys.basis[site.ci[0]][site.ci[1]]
                                                              - self.crys.basis[siteB.ci[0]][siteB.ci[1]]))
                                   for site in newsiteList]
                        # Check if all the sites have crossed the cutoff distance - else keep this cluster
                        if any(dist*dist > self.cutoff*self.cutoff for dist in dists_i+dists_j):
                            continue
                        newClust = ClusterTranslation(cluster.Cluster(newsiteList), newsiteList[0].R)
                        if newClust not in clTracker:
                            newsymList = [newClust.g(self.crys, gop) for gop in ijPtGroup]
                            ijClexp.append(newsymList)
                            clTracker.update(newsymList)

                clusterjumplist[(siteA, siteB)] = ijClexp
                
        return clusterjumplist

    def defineSpecies(self, Nmobile):
        """
        Used to assign chemical species to the jump cluster expansions.
        :param Nmobile: number of mobile species in the lattice, including the carrier. It is assumed that the species
        are labelled with numbers and that the number Nmobile-1 is the label for the carrier.

        We'll have separate expansions for every species that occupies the final site of a carrier jump,
        and every cluster for a given jump will have species assigned to it.

        :returns  SpecClusterJumpList - a cluster expansion for KRA with species assigned to the clusters.
        """

        clusterJumps = getattr(self, "clusterJumps", None)
        if clusterJumps is None:
            raise ValueError("Need to have generated cluster expansions for the jumps first.")

        clusterJumpsSpecies = {}
        for AB, clusterSymLists in clusterJumps.items():
            # For this transition, first assign species to the clusters
            # use itertools.product like in normal cluster expansion.

            # Then, assign species to te final site of the jumps.
            for specJ in range(Nmobile-1):
                ABspecJ = (AB[0], AB[1], specJ)


