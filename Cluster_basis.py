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

        self.sup = sup
        self.clusexp = clusexp

        self.mobList = mobList

        self.genVecs()
        self.createFullBasis()  # Generate the complete cluster basis including the arrangement of species
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

    def createFullBasis(self):
        """
        Function to add in the species arrangements to the cluster basis functions.
        """
        # count the number of each species in the spectator list and mobile list.
        # This will be used to eliminate unrealistic clusters that are always "off".
        sup = self.sup
        clusexp = self.clusexp
        mobList = self.mobList
        lenMob = [np.sum(occArray) for occArray in mobList]

        clusterGates = []
        for clistInd, clist in enumerate(clusexp):
            cl0 = clist[0]  # get the representative cluster
            # cluster.sites is a tuple, which maintains the order of the elements.
            Nmobile = len(cl0.sites)

            arrangemobs = itertools.product(mobList, repeat=Nmobile)  # arrange mobile sites on mobile species.

            for tup1 in arrangemobs:
                mobcount = collections.Counter(tup1)
                if any(j > lenMob[i] for i, j in mobcount.items()):
                    continue
                # Each cluster is associated with three vectors
                clusterGates.append((tup1, clistInd*3))
                clusterGates.append((tup1, clistInd*3 + 1))
                clusterGates.append((tup1, clistInd*3 + 2))

        self.FullClusterBasis = clusterGates

    def countGatechange(self, translist, specOcc, mobOcc):

        ijlist, ratelist, dxlist = translist

        # This stores the changed in the basis functions due to all the jumps out of the current state.
        W = np.zeros((len(self.FullClusterBasis), len(self.FullClusterBasis)))
        # W is the contribution to Wbar by the current state

        for ij, rate, dx in zip(ijlist, ratelist, dxlist):
            specJ = sum([occ[ij[1]]*label for label,occ in zip(self.mobList, mobOcc)])
            del_lamb_vecs = np.zeros((len(self.FullClusterBasis), 3))
            initSite, initRvec = self.sup.ciR(ij[0])
            finSite, finRvec = self.sup.ciR(ij[1])
            BasisList_init = self.site2VclusBasis[initSite]  # get those basis groups which contain the initial site.
            # A "basis group" is defined by a an atomic configuration, and a symmetric list of (cluster, vector) pairs.
            # This is a dictionary - the keys are the basis indices, and the values are the (clusterInd, siteInd) tuples

            BasisList_fin = self.site2VclusBasis[finSite]

            for BasisGroupIndex, inGroupLocList in BasisList_init.items():

                # These are the groups of clusters that contain the initial site

                # Get the atomic arrangements
                # The update with be made to the row "BasisGroupIndex" of the matric del_lamb_vecs.
                atoms = list(self.FullClusterBasis[BasisGroupIndex][0])
                # Now we have to check if the atomic configurations match
                # get the translational vector
                # first get the actual cluster
                current_change = np.zeros(3)  # This is where we will add in the changes in the vectors
                for loc in inGroupLocList:
                    cl = self.VclusterList[self.FullClusterBasis[BasisGroupIndex][1]][loc[0]]
                    carrier = cl.sites[loc[1]]
                    trans = initRvec - carrier.R
                    newSites = [self.sup.index(site.R + trans, site.ci) for site in cl.sites]
                    # Get the occupancies after translation
                    newSpecies = [
                        sum([occ[site[0]] * label for occ, label in zip(self.mobList, mobOcc)]) if site[1] is True
                        else sum([occ[site[0]] * label for occ, label in zip(self.specList, specOcc)])
                        for site in newSites
                    ]

                    # check if both the initial and final sites are in the same cluster
                    bothSites = ij[1] in [site[0] for site in newSites if site[1] is True]

                    if not bothSites:
                        # 1. Turn off if carrier is at initial carrier site
                        if newSpecies == atoms:  # Then the cluster is on in the initial state
                            # and will be off in the final state
                            # A cluster that contains both the initial and final sites can be skipped and turned off during
                            # final state analysis (next for loop), since it switches off only once.
                            # So, get the mobile sites from newSites, and see if the final state is there
                            clVec = self.vecList[self.FullClusterBasis[BasisGroupIndex][1]][loc[0]]
                            # take it away from the change vector
                            current_change -= clVec

                        # 2. Turn on if specJ is at the initial carrier site
                        # First check by just changing the carrier site occupancy to final species and keeping all
                        # others same.
                        elif atoms == [atm if l != loc[1] else specJ for l, atm in enumerate(newSpecies)]:
                            clVec = self.vecList[self.FullClusterBasis[BasisGroupIndex][1]][loc[0]]
                            # take it away from the change vector
                            current_change += clVec

                del_lamb_vecs[BasisGroupIndex] += current_change

            # Now, let's work with the final site
            for BasisGroupIndex, inGroupLocList in BasisList_fin.items():
                atoms = self.FullClusterBasis[BasisGroupIndex][0]
                current_change = np.zeros(3)
                for loc in inGroupLocList:
                    cl = self.VclusterList[self.FullClusterBasis[BasisGroupIndex][1]][loc[0]]
                    finalLoc = cl.sites[loc[1]]
                    trans = finRvec - finalLoc.R
                    newSites = [self.sup.index(site.R + trans, site.ci) for site in cl.sites]
                    newSpecies = [
                        sum([occ[site[0]] * label for occ, label in zip(self.mobList, mobOcc)]) if site[1] is True
                        else sum([occ[site[0]] * label for occ, label in zip(self.specList, specOcc)])
                        for site in newSites
                    ]

                    # To test check that newSpecies[loc[1]] == specJ
                    clVec = self.vecList[self.FullClusterBasis[BasisGroupIndex][1]][loc[0]]
                    bothSites = ij[0] in [site[0] for site in newSites if site[1] is True]

                    if not bothSites:

                        if newSpecies == atoms and not bothSites:
                            # This also turns off clusters that contain both the carrier and the jumping species
                            # which we skipped when dealing with the carrier alone.
                            # take it away from the change vector
                            current_change -= clVec

                        # else check if there is the carrier at the final site J
                        elif newSpecies == [at if l != loc[1] else self.mobList[-1] for l, at in enumerate(atoms)]\
                                and not bothSites:
                            # take it away from the change vector
                            current_change += clVec
                    # Now, we need to deal with those clusters that contain BOTH specJ at the initial carrier site and
                    # the carrier at the final carrier site
                    if bothSites:
                        carrierSite = newSites.index((ij[0], True))
                        if newSpecies == atoms:
                            current_change -= clVec
                        # Check if we have the exchanged configuration - then this cluster will turn on
                        elif [specJ if l == carrierSite
                              else self.mobList[-1] if l == loc[1]
                              else atm
                              for l, atm in enumerate(newSpecies)]:

                            current_change += clVec