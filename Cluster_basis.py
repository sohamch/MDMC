from onsager import crystal, cluster, supercell
import numpy as np
import collections
import itertools

def createSpins(mobile_count, spec_count=0):

    """
    Intended use - mobile spins symmetric about zero - make spec spins positive after that
    """
    m = (mobile_count + spec_count) // 2
    # The number of species in 2m+1 or 2m.

    spins_mobile = np.array(list(range(-mobile_count, mobile_count + 1)))
    spins_spec = None
    if spec_count > 0:
        spins_spec = np.array(list(range(mobile_count+1, mobile_count+spec_count+1)))

    if mobile_count % 2 == 1:
        spins_spec.remove(0)  # zero not required here

    return spins_mobile, spins_spec


def createClusterBasis(sup, clusexp, specList, mobList):
    """
    Function to generate binary representer clusters.
    :param sup : ClusterSupercell object.
    :param clusexp - representative clusters grouped symmetrically.
    :param specList - chemical identifiers for spectator species.
    :param mobOcc - chemical identifiers for mobile species.
    :param vacSite - whether a certain site is specified as a vacancy.

    return gates - all the clusterr gates that determine state functions.
    """
    # count the number of each species in the spectator list and mobile list.
    # This will be used to eliminate unrealistic clusters that are always "off".
    lenMob = [np.sum(occArray) for occArray in mobList]
    lenSpec = [np.sum(occArray) for occArray in specList]

    clusterGates = []
    for clistInd, clist in enumerate(clusexp):
        cl0 = clist[0]  # get the representative cluster
        # separate out the spectator and mobile sites
        # cluster.sites is a tuple, which maintains the order of the elements.
        Nmobile = len([1 for site in cl0.sites if site in sup.indexmobile])
        Nspec = len([1 for site in cl0.sites if site in sup.indexspectator])

        arrangespecs = itertools.product(specList, repeat=Nspec)
        arrangemobs = itertools.product(mobList, repeat=Nmobile)

        if len(list(arrangespecs)) == 0:  # if there are no spectators then just order the mobile sites.
            for tup1 in arrangemobs:
                # TODO- Insert code for elimination of unwanted clusters
                mobcount = collections.Counter(tup1)
                if any(j > lenMob[i] for i, j in mobcount.items()):
                    continue
                clusterGates.append((tup1, clistInd))
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
                clusterGates.append((tup1, tup2, clistInd))
    return clusterGates
