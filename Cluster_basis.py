from onsager import crystal, cluster, supercell
import numpy as np

def createSpins(mobile_count, spec_count=0):

    """
    Intended use - up to spec_count, the spins are for spectator species, after that for mobile species
    """
    m = (len(mobile_occ) + spec_count) // 2
    # The number of species in 2m+1 or 2m.

    spins = list(range(-m, m + 1))

    if m % 2 == 1:
        spins.remove(0)  # zero not required here

    return np.array(spins)

def DiffOcc(occlist1, occlist2):
    """
    Takes two occupancy lists and returns one that only contains 1 for sites that have exchanged.
    The returned array must contain only ones
    """
    diff = [np.abs(occ1 - occ2).astype(int) for occ1, occ2 in zip(occlist1, occlist2)]
    return diff

def ClusterProducts(sup, clusexp, mobilelist, spins):
    """

    :param sup: supercell object, which is also the current state of the object
    :param clusexp: list of clusters around supercell
    :param mobilelist: list of occupancies on mobile sites for each species that can occupy those sites
    "param spins : the spins of the chemical species - assume the spins of mobile species are stored first.
    :return: one-d array : corresponding to the values of the cluster basis functions for the different clusters.
    This needs to be used more than once, since the bias needs to be updated at every state.
    """

    if not np.allclose(sum([arr for arr in mobilelist]), 1):
        raise ValueError("The occupancies of sites do not add up to one")

    Nclusters = len(clist)
    PhiMobileArray = np.zeros((len(sup.Rvectlist), Nclusters))

    # Go through the position vectors
    for Rind, R in enumerate(sup.Rvectlist):
        # Go through the clusters in the representatives
        for clInd, cl in enumerate([cl for clist in clusexp for cl in clist]):
            # add the translation to all the sites in the cluster and get the indices of those sites
            # Verify this
            sitelist = [sup.index(R, site)[0] for site in cl.sites]
            # Next, get the spin products of the sites in sitelist
            Phi_mob = np.prod(np.array([sum([spin * mob[ind] for spin, spec in zip(spins, mobilelist)])
                                        for ind in sitelist]))
            # Now store the spin products in the array

            PhiMobileArray[Rind, clInd] = Phi_mob

    return PhiMobileArray

# Todo - Can we make the above computations more efficient. Imganine a case with a large nuumber of sites.
# As we go from one state to another, only a few clusters change. So the Phi array for the next state
# Can be evaluated from the Phi array of the current state easily. Check how.