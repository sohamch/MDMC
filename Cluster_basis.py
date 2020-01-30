from onsager import crystal, cluster, supercell
import numpy as np

def createSpins(mobile_count, spec_count=0):

    m = (len(mobile_occ) + spec_count) // 2
    # The number of species in 2m+1 or 2m.

    spins = list(range(-m, m + 1))

    if m % 2 == 1:
        spins.remove(0)  # zero not required here

    return np.array(spins)

def createLbam(sup, clusexp, mobile_occ, spins):
    """

    :param sup: supercell object, which is also the current state of the object
    :param clusexp: list of clusters around supercell
    :param mobile_occ: list of occupancies on mobile sites for each species that can occupy those sites
    :param spec_count : how many spectator species do we have. This doesn't matter except in deciding spin values.
    the lists should be so that sum_(site over all all sites) occupancy_(species)(site) = 1
    :return: one-d array : corresponding to the values of the cluster basis functions.

    """

    if not np.allclose(sum([arr for arr in mobile_occ]), 1):
        raise ValueError("The occupancies of sites do not add up to one")


