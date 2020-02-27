from onsager import crystal, cluster, supercell
import numpy as np

"""
This is to create KRA expansions of activation barriers. We assume for now that the expansion coefficients are given.
"""

class KRAExpand(object):
    """
    Object that contains all information regarding the KRA expansion of a jumpnetwork in a supercell.
    """
    def __init__(self, sup, chem, jumpnetwork, clusexp, InteractionCutoff):
        """
        :param sup: clusterSupercell Object
        :param chem: the sublattice index on which the jumpnetwork has been built.
        :param jumpnetwork: jumpnetwork to expand
        :param clusexp: representative set of clusters - out put of make clusters function.
        :param InteractionCutoff: the maximum distance of a cluster from a transition to consider
        """
        self.sup = sup
        self.chem = chem
        self.crys = self.sup.crys
        self.jumpnetwork = jumpnetwork
        self.clusexp = clusexp
        self.cutoff = InteractionCutoff

        # First, we reform the jumpnetwork
        self.clusterJumps = self.reDefineJumps()

    def reDefineJumps(self):
        """
        Redefining the jumps in terms of the initial and final states involved. Since KRA values are same for both
        forward and backward jumps, we will represent both with the same definition.
        :return - clusterJumps - all the jumps defined as two supercell sites (a,b) - the clusters associated with the KRA
        of the jumps, grouped together into lists so that all clusters in a list are related by transition-invariant
        symmetry operations (that is, those group operations that leave (a, b) unchanged.
        """
        # Go through the jumps in the jumpnetwork
        # Define Rvecs.
        clusterjumplist = {}
        for jlist in self.jumpnetwork:
            for ((i, j), dx) in jlist:
                siteA = self.sup.index((self.chem, i), np.zeros(3, dtype=int))
                Rj, (c, cj) = self.crys.cart2pos(dx + np.dot(self.crys.lattice, self.crys.basis[self.chem][j]))
                # check we have the correct site
                if not cj == j:
                    raise ValueError("improper coordinate transformation, did not get same site")
                siteB = self.sup.index((self.chem, j), Rj)


