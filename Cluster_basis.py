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


def createClusterBasis(sup, clusexp, specList, mobList, vacSite=None):
    """
    Function to generate binary representer clusters.
    :param sup : ClusterSupercell object.
    :param clusexp - representative clusters grouped symmetrically.
    :param specList - chemical identifiers for spectator species.
    :param mobOcc - chemical identifiers for mobile species.
    :param vacSite - whether a certain site is specified as a vacancy.

    return gates - all the clusterr gates that determine state functions.
    """
    if any(specChem==mobChem for specChem in specList for mobChem in mobList):
        return TypeError("Mobile species also identified as spectator?")

    # The no. species that can be present at each site is in sup.Nchem
    arrangements = []
    for clist in clusexp:
        cl0 = clist[0]  # get the representative cluster
        order = cl0.order
        # separate out the spectator and mobile sites
        # cluster.sites is a tuple, which maintains the order of the elements.
        Nmobile = len([1 for site in cl0.sites if site in sup.indexmobile])
        Nspec = len([1 for site in cl0.sites if site in sup.indexspectator])

        arrangespecs = itertools.product(specList, repeat=Nspec)
        arrangemobs = itertools.product(mobList, repeat=Nspec)

        for tup1 in arrangemobs:
            for tup2 in arrangespecs:
                arrangements.append((tup1, tup2))

    return arrangements
