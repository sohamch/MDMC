from onsager import crystal, cluster, supercell
import numpy as np
import collections
import itertools


class VectorClusterExpansion(object):
    """
    class to expand velocities and rates in vector cluster functions.
    """
    def __init__(self, sup, clusexp, mobList, vacSite):
        """
        param sup : clusterSupercell object
        clusexp: cluster expansion about a single unit cell.
        mobList - list of labels for chemical species on mobile sites - in order as their occupancies are defined.

        In this type of simulations, we consider a solid with a single wyckoff set on which atoms are arranged.
        """
        self.chem = 0  # we'll work with a monoatomic basis

        self.vacSite = cluster.clusterSite((self.chem, vacSite), np.zeros(3, dtype=int))
        # vacSite is the basis site index in which the vacancy is fixed.
        self.sup = sup
        self.vacInd = sup.index((self.chem, vacSite), np.zeros(3, dtype=int))

        # vacInd will always be the initial state in the transitions that we consider.
        self.clusexp = clusexp
        self.mobList = mobList  # labels of the mobile species - the last label is for the vacancy.
        self.genVecs()
        self.FullClusterBasis = self.createFullBasis()  # Generate the complete cluster basis including the
        # arrangement of species on sites other than the vacancy site.
        self.index()

    def genVecs(self):
        """
        Function to generate a symmetry-grouped vector cluster expansion similar to vector stars in the onsager code.
        """
        sup = self.sup
        clusexp = self.clusexp
        Id3 = np.eye(3)
        self.VclusterList = []
        self.vecList = []
        for clist in clusexp:
            cl0 = clist[0]
            for vec in Id3:
                symclList = []
                symvecList = []
                for cl in clist:
                    for gop in sup.crys.G:
                        if cl0.g(sup.crys, gop) == cl:
                            if any(cl1 == cl for cl1 in symclList):
                                continue
                            symclList.append(cl0)
                            symvecList.append(np.dot(gop.cartrot, vec))
                self.VclusterList.append(symclList)
                self.vecList.append(symvecList)

    def index(self):
        """
        Index each site to a vector cluster list.
        """
        siteToVclusBasis = {}
        for BasisInd, BasisDat in enumerate(self.FullClusterBasis):
            for clInd, cl in enumerate(self.VclusterList[BasisDat[1]]):
                for siteInd, site in enumerate(cl.sites):
                    if site.ci not in siteToVclusBasis:
                        siteToVclusBasis[site.ci] = collections.defaultdict(list)
                    siteToVclusBasis[site.ci][BasisInd].append((clInd, siteInd))
        self.site2VclusBasis = siteToVclusBasis

    def indexSupInd2Clus(self):
        """
        Takes the sites in the clusters, get their indices in the supercell sitelist, and store the clusters they
        belong to, with these indices as keys.
        """
        siteToVclusBasis = collections.defaultdict(list)
        for BasisInd, BasisDat in enumerate(self.FullClusterBasis):
            for clInd, cl in enumerate(self.VclusterList[BasisDat[1]]):
                for siteInd, site in enumerate(cl.sites):
                    siteToVclusBasis[self.sup.index(site.ci, site.R)].append((BasisInd, clInd, siteInd))

        self.SupInd2Clus = siteToVclusBasis

    def createFullBasis(self):
        """
        Function to add in the species arrangements to the cluster basis functions.
        """

        lenMob = [np.sum(occArray) for occArray in self.mobList]

        clusterBasis = []
        for clistInd, clist in enumerate(self.clusexp):
            cl0 = clist[0]  # get the representative cluster
            # cluster.sites is a tuple, which maintains the order of the elements.
            Nmobile = len(cl0.sites)
            arrangemobs = itertools.product(self.mobList, repeat=Nmobile)  # arrange mobile sites on mobile species.

            for tup in arrangemobs:
                mobcount = collections.Counter(tup)
                # Check if the number of atoms of a given species does not exceed the total number of atoms of that
                # species in the solid.
                if any(j > lenMob[i] for i, j in mobcount.items()):
                    continue
                # Each cluster is associated with three vectors
                # Any cluster that contains a vacancy is shifted so that the vacancy is at the origin.
                clusterBasis.append((tup, clistInd*3))
                clusterBasis.append((tup, clistInd*3 + 1))
                clusterBasis.append((tup, clistInd*3 + 2))
        return clusterBasis

    def Expand(self, mobOccs, transitions):
        ijlist, ratelist, dxlist = transitions

        for (ij, rate, dx) in zip(ijlist, ratelist, dxlist):
            specJ = sum([occ[ij[1]]*label for occ, label in zip(mobOccs, self.mobList)])
            siteJ = self.sup.ciR(ij[1])  # get the lattice site where the jumping species initially sits

            # Get all the clusters that contain the vacancy at the vacancy site and/or specJ at ij[1]
            # and are On in the initial state.
            # noinspection PyUnresolvedReferences
            On_clusters_vac = [(bInd, clInd) for bInd, clInd, siteInd in self.SupInd2Clus[ij[0]]
                               if np.prod([mobOccs[species][self.sup.index(site.ci, site.R)]
                                           for species, site in zip(self.FullClusterBasis[bInd][0],
                                                                    self.VclusterList[bInd][clInd].sites)]) == 1
                               ]
            On_clusters_specJ = [(bInd, clInd) for bInd, clInd, siteInd in self.SupInd2Clus[ij[1]]
                                 if np.prod([mobOccs[species][self.sup.index(site.ci, site.R)]
                                            for species, site in zip(self.FullClusterBasis[bInd][0],
                                                                     self.VclusterList[bInd][clInd].sites)]) == 1
                                 ]

