from onsager import crystal, cluster, supercell
import numpy as np

def Create_LBAM(sup, clusexp, mobile_occ=None, spec_count=1):
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

    # Now, we need to assign spins to each species - how to do this quickly
    # get the halfpoint
    m = (len(mobile_occ) + spec_count)//2
    spins = range(-m, m+1)