from onsager import crystal, cluster, supercell
import numpy as np
import itertools
import collections

class KRA_expansion():
    """
    This is to compute KRA cluster expansion involving sites up to a cutoff and transition sites
    to be taken from a jumpnetwork in mono-atomic lattices.
    """
    def __init__(self, sup, jnet, cutoff):
        """
        :param sup: the supercell object based on which sites will be given indices
        :param jnet: the jumpnetwork from transition sites are taken
        :param cutoff: the maximum distance up to which sites will be considered
        (from either the vacancy (at 0,0,0) or the transition sites
        """