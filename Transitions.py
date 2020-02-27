from onsager import crystal, cluster, supercell
import numpy as np

"""
This is to create KRA expansions of activation barriers. We assume for now that the expansion coefficients are given.
"""

class KRAExpand(object):
    def __init__(self, sup, jumpnetwork, clusexp, InteractionCutoff):
        """
        :param sup: clusterSupercell Object
        :param jumpnetwork: jumpnetwork to expand
        :param clusexp: representative set of clusters - out put of make clusters function.
        :param InteractionCutoff: the maximum distance of a cluster from a transition to consider
        """
        self.sup = sup
        self.jumpnetwork = jumpnetwork
        self.clusexp = clusexp
        self.cutoff = InteractionCutoff

        # First, we reform the jumpnetwork
        self.clusterJumps = self.reDefineJumps()

    def reDefineJumps(self):
        """
        Redefining the jumps in terms of the initial and final states involved. Since KRA values are same for both
        forward and backward jumps, we will represent both with the same definition.
        :return - clusterJumps - all the jumps defined as two sites (a,b) - the clusters associated with the KRA
        of the jumps, grouped together into lists so that all clusters in a list are related by transition-invariant
        symmetry operations (that is, those group operations that leave (a, b) unchanged.
        """
        pass