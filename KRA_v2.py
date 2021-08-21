from onsager import crystal, cluster, supercell
import numpy as np
import itertools
import collections

class KRA_expansion():
    """
    This is to compute KRA cluster expansion involving sites up to a cutoff and transition sites
    to be taken from a jumpnetwork in mono-atomic lattices.
    """
    def __init__(self, sup, jnet, chem, cutoff):
        """
        :param sup: the supercell object based on which sites will be given indices
        :param jnet: the jumpnetwork from transition sites are taken
        :param chem : the sublattice on which the vacancy jumps
        :param cutoff: the maximum distance up to which sites will be considered
        (from either the vacancy (at 0,0,0) or the transition sites
        """
        self.sup = sup
        self.jnet = jnet
        self.cutoff = cutoff
        self.crys = sup.crys
        # first let's get the jump sites
        jList = []
        dxList = []
        for jList in jnet:
            newList = []
            for (i, j), dx in jList:
                dxR, (chem, cj) = sup.crys.cart2pos(dx)
                assert j == cj
                siteInd = sup.index()