from onsager import crystal, cluster, supercell
import numpy as np
import collections
import itertools
import Transitions
import time


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
        self.transPairs = [(site-Rtrans, spec) for site, spec in zip(siteList, specList)]
        self.SiteSpecs = set(self.transPairs)
        self.__hashcache__ = int(np.prod(np.array([hash((site, spec)) for site, spec in self.transPairs]))) +\
                             sum([hash((site, spec)) for site, spec in self.transPairs])

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
        for site, spec in self.SiteSpecs:
            str += "Spec:{}, site:{},{} ".format(spec, site.ci, site.R)
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
        self.ScalarBasis = self.createScalarBasis()
        self.SpecClusters = self.recalcClusters()
        self.vecClus, self.vecVec = self.genVecClustBasis(self.SpecClusters)
        self.indexVclus2Clus()

        # Generate the complete cluster basis including the arrangement of species on sites other than the vacancy site.
        self.KRAexpander = Transitions.KRAExpand(sup, self.chem, jumpnetwork, clusexp, mobCountList)

    def recalcClusters(self):
        """
        Intended to take in a site based cluster expansion and recalculate the clusters with species in them
        """
        allClusts = set()
        symClusterList = []
        for clSet in self.clusexp:
            for clust in list(clSet):
                Nsites = len(clust.sites)
                occupancies = list(itertools.product(self.mobList, repeat=Nsites))
                for siteOcc in occupancies:
                    # Make the cluster site object
                    ClustSpec = ClusterSpecies(siteOcc, clust.sites)
                    # check if this has already been considered
                    if ClustSpec in allClusts:
                        continue
                    # Check if number of each species in the cluster is okay
                    mobcount = collections.Counter(siteOcc)
                    # Check if the number of atoms of a given species does not exceed the total number of atoms of that
                    # species in the solid.
                    if any(j > self.mobCountList[i] for i, j in mobcount.items()):
                        continue
                    # Otherwise, find all symmetry-grouped counterparts
                    newSymSet = set([ClustSpec.g(self.crys, g) for g in self.crys.G])
                    allClusts.update(newSymSet)
                    newList = list(newSymSet)
                    symClusterList.append(newList)

        return symClusterList

    def genVecClustBasis(self, specClusters):

        vecClustList = []
        vecVecList = []
        for clList in specClusters:
            cl0 = clList[0]
            glist0 = []
            for g in self.crys.G:
                if cl0.g(self.crys, g) == cl0:
                    glist0.append(g)

            G0 = sum([g.cartrot for g in glist0])/len(glist0)
            vals, vecs = np.linalg.eig(G0)
            vlist = [vecs[:, i]/np.linalg.norm(vecs[:, i]) for i in range(3) if np.isclose(vals[i], 1.0)]

            if len(vlist) == 0:  # For systems with inversion symmetry
                vecClustList.append(clList)
                vecVecList.append([np.zeros(3) for i in range(len(clList))])
                continue

            for v in vlist:
                newClustList = [cl0]
                # The first state being the same helps in indexing
                newVecList = [v]
                for g in self.crys.G:
                    cl1 = cl0.g(self.crys, g)
                    if cl1 in newClustList:
                        continue
                    newClustList.append(cl1)
                    newVecList.append(np.dot(g.cartrot, v))

                vecClustList.append(newClustList)
                vecVecList.append(newVecList)

        return vecClustList, vecVecList

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

    def indexVclus2Clus(self):

        self.Vclus2Clus = np.zeros(len(self.vecClus), dtype=int)
        for vClusListInd, vClusList in enumerate(self.vecClus):
            clVec0 = vClusList[0]
            for cLlistInd, clList in enumerate(self.SpecClusters):
                cl0 = clList[0]
                if clVec0 == cl0:
                    self.Vclus2Clus[vClusListInd] = cLlistInd

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

        del_lamb_mat = np.zeros((len(self.vecClus), len(self.vecClus), len(ijlist)))
        delxDotdelLamb = np.zeros((len(self.vecClus), len(ijlist)))

        # To be tensor dotted with ratelist with axes = (0,1)
        ratelist = np.zeros(len(ijlist))

        for (jnum, ij, dx) in zip(itertools.count(), ijlist, dxlist):

            del_lamb = np.zeros((len(self.vecClus), 3))

            specJ = sum([occ[ij[1]]*label for label, occ in enumerate(mobOccs)])
            siteJ, Rj = self.sup.ciR(ij[1])  # get the lattice site where the jumping species initially sits
            siteI, Ri = self.sup.ciR(ij[0])  # get the lattice site where the jumping species finally sits

            if siteI != self.vacSite.ci or not np.allclose(Ri, self.vacSite.R):
                raise ValueError("The initial site must be the vacancy site")

            # Get the KRA energy for this jump
            delEKRA = self.KRAexpander.GetKRA((ij, dx), mobOccs, KRACoeffs[(ij[0], ij[1], specJ)])
            delE = 0.0  # This will added to the KRA energy to get the activation barrier

            # switch the occupancies in the final state
            mobOccs_final = mobOccs.copy()
            mobOccs_final[-1][ij[0]] = 0
            mobOccs_final[-1][ij[1]] = 1
            mobOccs_final[specJ][ij[0]] = 1
            mobOccs_final[specJ][ij[1]] = 0

            for clListInd, clList in enumerate(self.vecClus):
                for clInd, clus in enumerate(clList):
                    for siteSpec in clus.SiteSpecs:
                        site, spec = siteSpec[0], siteSpec[1]
                        # First, we check for vacancies
                        # Check if any translation of this cluster needs to be turned off
                        if site.ci == siteI:
                            siteSpecNew = set([(clsite - site.R, spec) for clsite, spec in clus.SiteSpecs])
                            # Check if this translated cluster is on in the initial state
                            if all([mobOccs[spec][self.sup.index(clSite.R, clSite.ci)[0]] == 1
                                    for clSite, spec in siteSpecNew]):
                                # Turn in off
                                del_lamb[clListInd] -= self.vecVec[clListInd][clInd]
                                delE -= EnCoeffs[self.Vclus2Clus[clListInd]]

                            # Check if this cluster is on in the final state
                            if all([mobOccs_final[spec][self.sup.index(clSite.R, clSite.ci)[0]] == 1
                                    for clSite, spec in siteSpecNew]):
                                # Turn it on
                                del_lamb[clListInd] += self.vecVec[clListInd][clInd]
                                delE += EnCoeffs[self.Vclus2Clus[clListInd]]

                        # Next, we check for specJ
                        if site.ci == siteJ.ci:
                            # Bring this site to Rj instead of site.R
                            siteSpecNew = set([(clsite - site.R + siteJ.R, spec) for clsite, spec in clus.SiteSpecs])
                            # Check if this translated cluster is on in the initial state
                            if all([mobOccs[spec][self.sup.index(clSite.R, clSite.ci)[0]] == 1
                                    for clSite, spec in siteSpecNew]):
                                # Turn it off
                                del_lamb[clListInd] -= self.vecVec[clListInd][clInd]
                                delE -= EnCoeffs[self.Vclus2Clus[clListInd]]

                            # Check if this cluster is on in the final state
                            if all([mobOccs_final[spec][self.sup.index(clSite.R, clSite.ci)[0]] == 1
                                    for clSite, spec in siteSpecNew]):
                                # Turn it on
                                del_lamb[clListInd] -= self.vecVec[clListInd][clInd]
                                delE -= EnCoeffs[self.Vclus2Clus[clListInd]]

            # append to the rateList
            ratelist[jnum] = np.exp(-(0.5*delE + delEKRA))

            # Create the matrix to find Wbar
            del_lamb_mat[:, :, jnum] = np.dot(del_lamb, del_lamb.T)

            # Create the matrix to find Bbar
            delxDotdelLamb[:, jnum] = np.tensordot(del_lamb_mat, dx, axes=(1, 0))

        Wbar = np.tensordot(ratelist, del_lamb_mat, axes=(0, 0))
        Bbar = np.tensordot(ratelist, delxDotdelLamb, axes=(0, 1))

        return Wbar, Bbar

    def GenTransitions(self, mobOcc, jumpnetwork):
        """
        Function to calculate the transitions and their rates out of a given state.
        The rate for each jump is calculated in-situ while computing the vector expansion, since they require
        similar operations.
        :param mobOcc: occupancy vectors for mobile species in the current state
        :param jumpnetwork: vacancy jumpnetwork
        :return: (ijlist, dxlist)
        """

        ijList = []
        dxList = []
        for jump in [jmp for jList in jumpnetwork for jmp in jList]:
            siteA = self.sup.index((self.chem, jump[0][0]), np.zeros(3, dtype=int))
            Rj, (c, cj) = self.crys.cart2pos(jump[1] -
                                             np.dot(self.crys.lattice, self.crys.basis[self.chem][jump[0][1]]) +
                                             np.dot(self.crys.lattice, self.crys.basis[self.chem][jump[0][0]]))
            # check we have the correct site
            if not cj == jump[0][1]:
                raise ValueError("improper coordinate transformation, did not get same site")
            siteB = self.sup.index((self.chem, jump[0][1]), Rj)

            specJ = np.prod(np.array([mobOcc[spec][siteB] for spec in self.mobList]))

            ijList.append((siteA, siteB, jump[1]))
            dxList.append(jump[1])

        return ijList, dxList




