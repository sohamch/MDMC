from onsager import crystal, cluster, supercell
import numpy as np

def Create_LBAM(sup, clusexp, mobile_occ=None):
    """

    :param sup: supercell object, which is also the current state of the object
    :param clusexp: list of clusters around supercell
    :param mobile_occ: list of occupancies on mobile sites for each species that can occupy those sites
    the lists should be so that sum_(site over all all sites) occupancy_(species)(site) = 1
    :return: one-d array : corresponding to the values of the cluster basis functions.
    """

    if not np.allclose(sum([arr for arr in mobile_occ]), 1):
        raise ValueError("The occupancies of sites do not add up to one")
    pass