from onsager import crystal, cluster, supercell
import numpy as np
import collections
import itertools


class VectorClusterExpansion(object):
    """
    class to expand velocities and rates in vector cluster functions.
    """
    def __init__(self, sup, clusexp, mobList):
        """
        param sup : clusterSupercell object
        clusexp: cluster expansion about a single unit cell.
        mobList - list of labels for chemical species on mobile sites - in order as their occupancies are defined.

        In this type of simulations, we consider a solid with a single wyckoff set on which atoms are arranged.
        """
        self.chem = 0  # we'll work with a monoatomic basis
        self.sup = sup
        # vacInd will always be the initial state in the transitions that we consider.
        self.clusexp = clusexp
        self.mobList = mobList  # labels of the mobile species - the last label is for the vacancy.
        self.genVecs()
        self.FullClusterBasis, self.ScalarBasis = self.createFullBasis()
        # Generate the complete cluster basis including the
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
        self.VclusterSupIndList = []
        self.vecList = []
        for clist in clusexp:
            cl0 = clist[0]
            for vec in Id3:
                symclList = []
                symcLSupIndList = []
                symvecList = []
                for cl in clist:
                    for gop in sup.crys.G:
                        if cl0.g(sup.crys, gop) == cl:
                            if any(cl1 == cl for cl1 in symclList):
                                continue
                            symclList.append(cl)
                            symcLSupIndList.append([site for site in cl.sites])
                            symvecList.append(np.dot(gop.cartrot, vec))
                self.VclusterList.append(symclList)
                self.vecList.append(symvecList)
                self.VclusterSupIndList.append(symcLSupIndList)

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

        site2ScalClusBasis = {}
        for BasisInd, BasisDat in enumerate(self.ScalarBasis):
            for clInd, cl in enumerate(self.clusexp[BasisDat[1]]):
                for siteInd, site in enumerate(cl.sites):
                    if site.ci not in site2ScalClusBasis:
                        siteToVclusBasis[site.ci] = collections.defaultdict(list)
                    siteToVclusBasis[site.ci][BasisInd].append((clInd, siteInd))
        self.site2ScalBasis = siteToVclusBasis

    def indexSupInd2Clus(self):
        """
        Takes the sites in the clusters, get their indices in the supercell sitelist, and store the clusters they
        belong to, with these indices as keys.
        """
        siteToVclusBasis = collections.defaultdict(list)
        for BasisInd, BasisDat in enumerate(self.FullClusterBasis):
            for clInd, cl in enumerate(self.VclusterList[BasisDat[1]]):
                for siteInd, site in enumerate(cl.sites):
                    siteToVclusBasis[self.sup.index(site.R, site.ci)].append((BasisInd, clInd, siteInd))
        self.SupInd2VClus = siteToVclusBasis

        supInd2scalBasis = {}
        for BasisInd, BasisDat in enumerate(self.ScalarBasis):
            for clInd, cl in enumerate(self.clusexp[BasisDat[1]]):
                for siteInd, site in enumerate(cl.sites):
                    supInd2scalBasis[self.sup.index(site.R, site.ci)].append((BasisInd, clInd, siteInd))
        self.supInd2scalBasis = supInd2scalBasis

    def createFullBasis(self):
        """
        Function to add in the species arrangements to the cluster basis functions.
        """

        lenMob = [np.sum(occArray) for occArray in self.mobList]

        FullclusterBasis = []
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
                clusterBasis.append((tup, clistInd))
                FullclusterBasis.append((tup, clistInd*3))
                FullclusterBasis.append((tup, clistInd*3 + 1))
                FullclusterBasis.append((tup, clistInd*3 + 2))
        return FullclusterBasis, clusterBasis

    def Expand(self, mobOccs, transitions):
        ijlist, ratelist, dxlist = transitions
        mobOccs_final = mobOccs.copy()

        del_lamb_mat = np.zeros((len(self.FullClusterBasis), len(self.FullClusterBasis), len(ijlist)))
        delxDotdelLamb = np.zeros((len(self.FullClusterBasis), len(ijlist)))

        delLambDx = np.zeros((len(self.FullClusterBasis), len(ijlist)))
        # To be tensor dotted with ratelist with axes = (0,1)

        for (jnum, ij, rate, dx) in zip(itertools.count(), ijlist, ratelist, dxlist):

            del_lamb = np.zeros((len(self.FullClusterBasis), 3))

            specJ = sum([occ[ij[1]]*label for occ, label in zip(mobOccs, self.mobList)])
            siteJ = self.sup.ciR(ij[1])  # get the lattice site where the jumping species initially sits

            # switch the occupancies in the final state
            mobOccs_final[specJ][ij[0]] = 1
            mobOccs_final[specJ][ij[1]] = 0

            mobOccs_final[-1][ij[0]] = 0
            mobOccs_final[specJ][ij[1]] = 1

            # delOcc = mobOccs_final - mobOccs  # Doesn't seem to be much we can do with this

            # Get all the clusters that contain the vacancy at the vacancy site and/or specJ at ij[1]
            # and are On in the initial state.

            InitOnClustersVac = [(bInd, clInd) for bInd, clInd, siteInd in self.SupInd2VClus[ij[0]]
                                 if np.prod(np.array([mobOccs[species][idx]
                                            for species, idx in zip(self.FullClusterBasis[bInd][0],
                                                                    self.VclusterSupIndList[bInd][clInd])])) == 1
                                 ]

            InitOnClustersSpecJ = [(bInd, clInd) for bInd, clInd, siteInd in self.SupInd2VClus[ij[1]]
                                   if np.prod(np.array([mobOccs[species][idx]
                                              for species, idx in zip(self.FullClusterBasis[bInd][0],
                                                                      self.VclusterSupIndList[bInd][clInd])])) == 1
                                   ]

            FinOnClustersVac = [(bInd, clInd) for bInd, clInd, siteInd in self.SupInd2VClus[ij[0]]
                                if np.prod(np.array([mobOccs_final[species][idx]
                                           for species, idx in zip(self.FullClusterBasis[bInd][0],
                                                                   self.VclusterSupIndList[bInd][clInd])])) == 1
                                ]

            FinOnClustersSpecJ = [(bInd, clInd) for bInd, clInd, siteInd in self.SupInd2VClus[ij[1]]
                                  if np.prod(np.array([mobOccs_final[species][idx]
                                             for species, idx in zip(self.FullClusterBasis[bInd][0],
                                                                     self.VclusterSupIndList[bInd][clInd])])) == 1
                                  ]

            # Turn of the On clusters
            for (bInd, clInd) in set(InitOnClustersVac).union(set(InitOnClustersSpecJ)):
                del_lamb[bInd] -= self.vecList[bInd][clInd]

            # Turn on the Off clusters
            for (bInd, clInd) in set(FinOnClustersVac).union(FinOnClustersSpecJ):
                del_lamb[bInd] += self.vecList[bInd][clInd]

            # Create the matrix to find Wbar
            del_lamb_mat[:, :, jnum] = np.dot(del_lamb, del_lamb.T)

            # Create the matrix to find Bbar
            delxDotdelLamb[:, jnum] = np.tensordot(del_lamb_mat, dx, axes=(1, 0))

        Wbar = np.tensordot(ratelist, del_lamb_mat, axes=(0, 0))
        Bbar = np.tensordot(ratelist, delxDotdelLamb, axes=(0, 1))

        return Wbar, Bbar

    def transitions(self, mobOcc, jumpnetwork):
        """
        Function to calculate the transitions and their rates out of a given state.
        :param mobOcc: occupancy vectors for mobile species
        :param jumpnetwork: vacancy jumpnetwork
        :return: (ijlist, ratelist, dxlist)
        """
        jumplist = [jump for jlist in jumpnetwork for jump in jlist]
        pass



