from onsager import crystal, cluster, supercell
import numpy as np
import collections
import itertools
import Transitions

class ClusterSpecies():

    def __init__(self, specList, siteList):
        if len(specList)!= len(siteList):
            raise ValueError("Species and site lists must have same length")
        if not all(isinstance(site, cluster.ClusterSite) for site in siteList):
            raise TypeError("The sites must be entered as clusterSite object instances")
        # Form (site, species) set
        # Calculate the translation to bring center of the sites to the origin unit cell
        self.specList = specList
        self.siteList = siteList
        Rtrans = sum([site.R for site in siteList])//len(siteList)
        self.SiteSpecs = set([(site-Rtrans, spec) for site, spec in zip(siteList, specList)])
        self.__hashcache__ = sum([hash((site-Rtrans, spec))for site, spec in zip(siteList, specList)])

    def __eq__(self, other):
        if self.SiteSpecs == other.SiteSpecs:
            return True
        return False

    def __hash__(self):
        return self.__hashcache__

    def g(self, crys, g):
        return self.__class__(self.specList, [site.g(crys, g) for site in self.siteList])

    def strRep(self):
        str= ""
        for site, spec in zip(self.siteList, self.specList):
            str += "Spec:{}, site:{},{}".format(spec, site.ci, site.R)
        return str

    def __repr__(self):
        return self.strRep()

    def __str__(self):
        return self.strRep()


class VectorClusterExpansion(object):
    """
    class to expand velocities and rates in vector cluster functions.
    """
    def __init__(self, sup, clusexp, jumpnetwork, mobCountList, vacSite):
        """
        :param sup : clusterSupercell object
        :param clusexp: cluster expansion about a single unit cell.
        :param mobList - list of labels for chemical species on mobile sites - in order as their occupancies are defined.
        :param sampleMobOccs - a starting mobile occupancy array to just count the number of each species
        :param vacSite - the site of the vacancy as a clusterSite object. This does not change during the simulation.
        In this type of simulations, we consider a solid with a single wyckoff set on which atoms are arranged.
        """
        self.chem = 0  # we'll work with a monoatomic basis
        self.sup = sup
        self.crys = self.sup.crys
        # vacInd will always be the initial state in the transitions that we consider.
        self.clusexp = clusexp
        self.mobCountList = mobCountList
        self.mobList = list(range(len(mobCountList)))
        self.vacSite = vacSite
        # TODO - think of a better way to do this.
        self.genVecs()
        self.ScalarBasis = self.createScalarBasis()
        # Generate the complete cluster basis including the
        # arrangement of species on sites other than the vacancy site.
        self.index()
        self.KRAexpander = Transitions.KRAExpand(sup, self.chem, jumpnetwork, clusexp, mobCountList)

    def recalcClusters(self):
        """
        Intended to take in a site based cluster expansion and recalculate the clusters with species in them
        """
        considered = set()
        for clList in self.clusexp:
            for clust in clList:
                Nsites = len(clust.sites)
                occupancies = itertools.product(self.mobList, repeat=Nsites)
                for siteOcc in occupancies:
                    # Make the cluster site object
                    ClustSpec = ClusterSpecies(siteOcc, clust.sites)
                    # check if this has already been considered
                    if ClustSpec in considered:
                        continue
                    # Otherwise, find all symmetry-grouped counterparts



    def genVecs(self):
        """
        Function to generate a symmetry-grouped vector cluster expansion similar to vector stars in the onsager code.
        """
        sup = self.sup
        clusexp = self.clusexp
        Id3 = np.eye(3)
        self.SymVecClustList = []
        self.VclusterSupIndList = []
        self.SymVecList = []
        for clist in clusexp:
            cl0 = list(clist)[0]
            for vec in Id3:
                symclList = [cl0]
                symcLSupIndList = []
                symvecList = [vec]
                for cl in clist:
                    for gop in sup.crys.G:
                        if cl0.g(sup.crys, gop) == cl:
                            if any(cl1 == cl for cl1 in symclList):
                                continue
                            symclList.append(cl)
                            symvecList.append(np.dot(gop.cartrot, vec))
                            symcLSupIndList.append([self.sup.index(site.R, site.ci) for site in cl.sites])

                self.SymVecClustList.append(symclList)
                self.SymVecList.append(symvecList)
                self.VclusterSupIndList.append(symcLSupIndList)

    # def index(self):
    #     """
    #     Index each site to a vector cluster list.
    #     """
    #     siteToVclusBasis = {}
    #     for BasisInd, BasisDat in enumerate(self.FullClusterBasis):
    #         for clInd, cl in enumerate(self.SymVecClustList[BasisDat[1]]):
    #             for siteInd, site in enumerate(cl.sites):
    #                 if site.ci not in siteToVclusBasis:
    #                     siteToVclusBasis[site.ci] = collections.defaultdict(list)
    #                 siteToVclusBasis[site.ci][BasisInd].append((clInd, siteInd))
    #     self.site2VclusBasis = siteToVclusBasis
    #
    #     site2ScalClusBasis = {}
    #     for BasisInd, BasisDat in enumerate(self.ScalarBasis):
    #         for clInd, cl in enumerate(self.clusexp[BasisDat[1]]):
    #             for siteInd, site in enumerate(cl.sites):
    #                 if site.ci not in site2ScalClusBasis:
    #                     siteToVclusBasis[site.ci] = collections.defaultdict(list)
    #                 siteToVclusBasis[site.ci][BasisInd].append((clInd, siteInd))
    #     self.site2ScalBasis = siteToVclusBasis
    #
    # def indexSupInd2Clus(self):
    #     """
    #     Takes the sites in the clusters, get their indices in the supercell sitelist, and store the clusters they
    #     belong to, with these indices as keys.
    #     """
    #     siteToVclusBasis = collections.defaultdict(list)
    #     for BasisInd, BasisDat in enumerate(self.FullClusterBasis):
    #         for clInd, cl in enumerate(self.SymVecClustList[BasisDat[1]]):
    #             for siteInd, site in enumerate(cl.sites):
    #                 siteToVclusBasis[self.sup.index(site.R, site.ci)].append((BasisInd, clInd, siteInd))
    #     self.SupInd2VClus = siteToVclusBasis
    #
    #     supInd2scalBasis = {}
    #     for BasisInd, BasisDat in enumerate(self.ScalarBasis):
    #         for clInd, cl in enumerate(self.clusexp[BasisDat[1]]):
    #             for siteInd, site in enumerate(cl.sites):
    #                 supInd2scalBasis[self.sup.index(site.R, site.ci)].append((BasisInd, clInd, siteInd))
    #     self.supInd2scalBasis = supInd2scalBasis

    def createScalarBasis(self):
        """
        Function to add in the species arrangements to the cluster basis functions.
        """
        clusterBasis = []
        for clistInd, clist in enumerate(self.clusexp):
            cl0 = list(clist)[0]  # get the representative cluster
            # cluster.sites is a tuple, which maintains the order of the elements.
            Nmobile = len(cl0.sites)
            arrangemobs = itertools.product(self.mobList, repeat=Nmobile)  # arrange mobile sites on mobile species.
            for tup in arrangemobs:
                mobcount = collections.Counter(tup)
                # Check if the number of atoms of a given species does not exceed the total number of atoms of that
                # species in the solid.
                if any(j > self.mobCountList[i] for i, j in mobcount.items()):
                    continue
                # Each cluster is associated with three vectors
                clusterBasis.append((tup, clistInd))
        return clusterBasis

    def Expand(self, beta, mobOccs, transitions, EnCoeffs, KRACoeffs):

        """
        :param beta : 1/KB*T
        :param mobOccs: the mobile occupancy in the current state
        :param transitions: the jumps out of the current state - supercell indices for initial and final sites
        :param EnCoeffs: energy interaction coefficients in a cluster expansion
        :param KRACoeffs: kinetic energy coefficients - pre-formed
        :return: Wbar, Bbar - rate and bias expansions in the cluster basis
        """

        ijlist, dxlist = transitions
        mobOccs_final = mobOccs.copy()

        del_lamb_mat = np.zeros((len(self.FullClusterBasis), len(self.FullClusterBasis), len(ijlist)))
        delxDotdelLamb = np.zeros((len(self.FullClusterBasis), len(ijlist)))

        # To be tensor dotted with ratelist with axes = (0,1)
        ratelist = np.zeros(len(ijlist))

        for (jnum, ij, dx) in zip(itertools.count(), ijlist, dxlist):

            del_lamb = np.zeros((len(self.FullClusterBasis), 3))

            specJ = sum([occ[ij[1]]*label for occ, label in zip(mobOccs, self.mobList)])
            # siteJ = self.sup.ciR(ij[1])  # get the lattice site where the jumping species initially sits

            # Get the KRA energy for this jump
            delEKRA = self.KRAexpander.GetKRA((ij, dx), mobOccs, KRACoeffs[(ij[0], ij[1], specJ)])
            delE = 0.0  # This will added to the KRA energy to get the activation barrier

            # switch the occupancies in the final state
            mobOccs_final[specJ][ij[0]] = 1
            mobOccs_final[specJ][ij[1]] = 0

            mobOccs_final[-1][ij[0]] = 0
            mobOccs_final[specJ][ij[1]] = 1

            # delOcc = mobOccs_final - mobOccs  # Doesn't seem to be much we can do with this

            # Get all the clusters that contain the vacancy at the vacancy site and/or specJ at ij[1]
            # and    are On in the initial state.

            InitOnClustersVac = [(bInd, clInd) for bInd, clInd, siteInd in self.SupInd2VClus[ij[0]]
                                 if all([mobOccs[species][idx] == 1
                                            for species, idx in zip(self.FullClusterBasis[bInd][0],
                                                                    self.VclusterSupIndList[bInd][clInd])])
                                 ]

            InitOnClustersSpecJ = [(bInd, clInd) for bInd, clInd, siteInd in self.SupInd2VClus[ij[1]]
                                   if all([mobOccs[species][idx] == 1
                                              for species, idx in zip(self.FullClusterBasis[bInd][0],
                                                                      self.VclusterSupIndList[bInd][clInd])])
                                   ]

            FinOnClustersVac = [(bInd, clInd) for bInd, clInd, siteInd in self.SupInd2VClus[ij[0]]
                                if all([mobOccs_final[species][idx] == 1
                                           for species, idx in zip(self.FullClusterBasis[bInd][0],
                                                                   self.VclusterSupIndList[bInd][clInd])])
                                ]

            FinOnClustersSpecJ = [(bInd, clInd) for bInd, clInd, siteInd in self.SupInd2VClus[ij[1]]
                                  if all([mobOccs_final[species][idx] == 1
                                             for species, idx in zip(self.FullClusterBasis[bInd][0],
                                                                     self.VclusterSupIndList[bInd][clInd])])
                                  ]

            # Turn of the On clusters
            for (bInd, clInd) in set(InitOnClustersVac).union(set(InitOnClustersSpecJ)):
                del_lamb[bInd] -= self.SymVecList[bInd][clInd]
                delE -= EnCoeffs[self.ScalarBasis[bInd//3]]

            # Turn on the Off clusters
            for (bInd, clInd) in set(FinOnClustersVac).union(FinOnClustersSpecJ):
                del_lamb[bInd] += self.SymVecList[bInd][clInd]
                delE += EnCoeffs[self.ScalarBasis[bInd//3]]

            # append to the rateList
            ratelist[jnum] = np.exp(-(0.5*delE + delEKRA))

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
        :param mobOcc: occupancy vectors for mobile species in the current state
        :param jumpnetwork: vacancy jumpnetwork
        :return: (ijlist, dxlist)
        """
        # TODO - ijlist, dxlist will be calculated from the lattice, ratelist will be calculated from Transitions

        ijList = []
        dxList = []
        for jump in [jmp for jList in jumpnetwork for jmp in jList]:
            siteA = self.sup.index((self.chem, jump[0][0]), np.zeros(3, dtype=int))
            Rj, (c, cj) = self.crys.cart2pos(jump[1] -
                                             np.dot(self.crys.lattice, self.crys.basis[self.chem][jump[0][1]]) -
                                             np.dot(self.crys.lattice, self.crys.basis[self.chem][jump[0][0]]))
            # check we have the correct site
            if not cj == jump[0][1]:
                raise ValueError("improper coordinate transformation, did not get same site")
            siteB = self.sup.index((self.chem, jump[0][1]), Rj)

            specJ = np.prod(np.array([mobOcc[spec][siteB] for spec in self.mobList]))

            ijList.append((siteA, siteB, jump[1]))
            dxList.append(jump[1])

        return ijList, dxList




