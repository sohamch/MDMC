from onsager import crystal, cluster, supercell
import numpy as np
import collections
import itertools


class VectorClusterExpansion(object):
    """
    class to expand velocities and rates in vector cluster functions.
    """
    def __init__(self, sup, clusexp, specList, mobList):

        self.sup = sup
        self.clusexp = clusexp

        self.specList = specList
        self.mobList = mobList

        self.genVecs()
        self.index()
        self.createFullBasis()  # Generate the complete cluster basis including the arrangement of species

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
        siteToVclus = collections.defaultdict(list)
        for clListInd, clList in enumerate(self.clusexp):
            for clInd, cl in enumerate(clList):
                for siteInd, site in enumerate(cl.sites):
                    siteToVclus[site.ci].append([clListInd * 3, clInd, siteInd])
                    siteToVclus[site.ci].append([clListInd * 3 + 1, clInd, siteInd])
                    siteToVclus[site.ci].append([clListInd * 3 + 2, clInd, siteInd])
        self.site2Vclus = siteToVclus


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
                    clusterGates.append((tup1, [], clistInd*3))
                    clusterGates.append((tup1, [], clistInd*3 + 1))
                    clusterGates.append((tup1, [], clistInd*3 + 2))
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
                    clusterGates.append((tup1, tup2, clistInd*3))
                    clusterGates.append((tup1, tup2, clistInd*3 + 1))
                    clusterGates.append((tup1, tup2, clistInd*3 + 2))

        self.FullClusterBasis = clusterGates


def countGates(sup, mobOcc, clusexp, clusterGates, specOcc=None):
    """
    Takes in a set of occupancy vectors, and returns counts of open cluster gates.
    """
    # TODO - Need to verify that this worls when specOcc = None
    gateCounts = np.zeros(len(clusterGates), dtype=int)
    for gateIndex, gate in enumerate(clusterGates):
        MobSpecies = gate[0]
        SpecSpecies = gate[1]
        if specOcc is None and len(SpecSpecies) != 0:
            raise TypeError("spectator occupancy passed as None, but spectator states specified in gates.")
        clusterList = clusexp[gate[2]]

        # Now translate the clusters
        for Rtrans in sup.Rvectlist:
            for clust in clusterList:
                # The index function takes care of the modulus (periodicity).
                Mobsitesnew = [sup.index(site.R+Rtrans, site.ci)[0] for site in cluster.sites
                               if site.ci in sup.indexmobile]
                Specsitesnew = []
                if specOcc is not None:
                    Specsitesnew = [sup.index(site.R + Rtrans, site.ci)[0] for site in cluster.sites
                                    if site.ci in sup.indexspectator]
                # Now get the occupancies at each site
                species_mob = tuple([sum([i*occlist[j] for i, occlist in enumerate(mobOcc)]) for j in Mobsitesnew])
                species_spec = tuple([sum([i * occlist[j] for i, occlist in enumerate(specOcc)]) for j in Specsitesnew])
                if specOcc is None:
                    if species_mob == MobSpecies:
                        gateCounts[gateIndex] += 1
                else:
                    if species_mob == MobSpecies and species_spec == SpecSpecies:
                        gateCounts[gateIndex] += 1


