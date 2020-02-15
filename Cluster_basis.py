from onsager import crystal, cluster, supercell
import numpy as np
import collections
import itertools


class VectorClusterExpansion(object):
    """
    class to expand velocities and rates in vector cluster functions.
    """
    def __init__(self, sup, clusexp, specList, mobList):
        """
        param sup : clusterSupercell object
        clusexp: cluster expansion about a single unit cell.
        specList - list of labels for chemical species on spectator sites - in order as their occupancies are defined.
        mobList - list of labels for chemical species on mobile sites - in order as their occupancies are defined.
        """

        self.sup = sup
        self.clusexp = clusexp

        self.specList = specList
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
        specList = self.specList

        lenMob = [np.sum(occArray) for occArray in mobList]
        lenSpec = [np.sum(occArray) for occArray in specList]

        clusterGates = []
        for clistInd, clist in enumerate(clusexp):
            cl0 = clist[0]  # get the representative cluster
            # separate out the spectator and mobile sites
            # cluster.sites is a tuple, which maintains the order of the elements.
            Nmobile = len([1 for site in cl0.sites if site in sup.indexmobile])
            Nspec = len([1 for site in cl0.sites if site in sup.indexspectator])

            arrangespecs = itertools.product(specList, repeat=Nspec)  # arrange spectator species on spec sites.
            # Although spectator species never really move, a particular spectator configuration may be on in one
            # cluster while off in another.
            # Todo - Any way to "remember" this?

            arrangemobs = itertools.product(mobList, repeat=Nmobile)  # arrange mobile sites on mobile species.

            if len(list(arrangespecs)) == 0:  # if there are no spectators then just order the mobile sites.
                for tup1 in arrangemobs:
                    # TODO- Insert code for elimination of unwanted clusters
                    mobcount = collections.Counter(tup1)
                    if any(j > lenMob[i] for i, j in mobcount.items()):
                        continue
                    # Each cluster is associated with three vectors
                    clusterGates.append((tup1, clistInd*3))
                    clusterGates.append((tup1, clistInd*3 + 1))
                    clusterGates.append((tup1, clistInd*3 + 2))

                return clusterGates

            for tup1 in arrangemobs:
                # Now, check that the count of each species does not exceed the actual number of atoms that is present.
                # This will be helpful in case of clusters for example, when we have only one or two vacancies.
                # We can also identify those clusters separately as part of the initialization process.
                mobcount = collections.Counter(tup1)
                if any(j > lenMob[i] for i, j in mobcount.items()):
                    continue
                for tup2 in arrangespecs:
                    specCount = collections.Counter(tup1)
                    if any(j > lenSpec[i] for i, j in specCount.items()):
                        continue
                    # each cluster is associated to three vectors
                    nextmob = 0
                    nextspec = 0
                    total = []
                    for site in cl0.sites:
                        if site in sup.indexmobile:
                            total.append(tup1[nextmob])
                            nextmob += 1
                        else:
                            total.append(tup2[nextspec])
                            nextspec += 1
                    # each cluster is associated with 3 vectors in the vector function basis. Check this.
                    clusterGates.append((tuple(total), clistInd*3))
                    clusterGates.append((tuple(total), clistInd*3 + 1))
                    clusterGates.append((tuple(total), clistInd*3 + 2))

        self.FullClusterBasis = clusterGates

    def countGatechange(self, translist, specOcc, mobOcc):

        ijlist, ratelist, dxlist = translist

        # This stores the changed in the basis functions due to all the jumps out of the current state.

        W = np.zeros((len(self.FullClusterBasis), len(self.FullClusterBasis)))
        # W is the contribution to Wbar by the current state

        for ij, rate, dx in zip(ijlist, ratelist, dxlist):
            del_lamb_vecs = np.zeros((len(self.FullClusterBasis), 3))
            initSite, initRvec = self.sup.ciR(ij[0])
            finSite, finRvec = self.sup.ciR(ij[1])
            BasisList_init = self.site2VclusBasis[initSite]  # get those basis groups which contain the initial site.
            # A "basis group" is defined by a an atomic configuration, and a symmetric list of (cluster, vector) pairs.
            # This is a dictionary - the keys are the basis indices, and the values are the (clusterInd, siteInd) tuples

            for BasisGroupIndex, inGroupLocList in BasisList_init.items():
                # Get the atomic arrangements
                # The update with be made to the row "BasisGroupIndex" of the matric del_lamb_vecs.
                atoms = self.FullClusterBasis[BasisGroupIndex][0]
                # Now we have to check if the atomic configurations match
                # get the translational vector
                # first get the actual cluster
                current_change = np.zeros(3)  # This is where we will add in the changes in the vectors
                for loc in inGroupLocList:
                    cl = self.VclusterList[self.FullClusterBasis[BasisGroupIndex][1]][loc[0]]
                    carrier = cl.sites[loc[1]]
                    trans = initRvec - carrier.R
                    newSpecies = [
                        sum([occ[site[0]] * label for occ, label in zip(self.mobList, mobOcc)]) if site[1] is True
                        else sum([occ[site[0]] * label for occ, label in zip(self.specList, specOcc)])
                        for site in [self.sup.index(site.R + trans, site.ci) for site in cl.sites]
                    ]
                    # 1. Check changes due to initial sites
                    if tuple(newSpecies) == atoms:  # Then the cluster is on in the intial state, and will be off
                        # in the final state
                        clVec = self.vecList[self.FullClusterBasis[BasisGroupIndex][1]][loc[0]]
                        # take it away from the change vector
                        current_change -= clVec

                    # 2. Check changes due to after-jump clusters switching on



