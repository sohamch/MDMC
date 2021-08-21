from onsager import crystal, cluster, supercell
import numpy as np
import itertools
import collections

# Test notebooks for this file:
# "Test_site_generation"

class KRA_expansion():
    """
    This is to compute KRA cluster expansion involving sites up to a cutoff and transition sites
    to be taken from a jumpnetwork in mono-atomic lattices.
    """
    def __init__(self, sup, jnet, chem, order, cutoff):
        """
        :param sup: the supercell object based on which sites will be given indices
        :param jnet: the jumpnetwork from transition sites are taken
        :param chem : the sublattice on which the vacancy jumps
        :param order : the max number of sites in the clusters (excluding the transition sites)
        :param cutoff: the maximum distance up to which sites will be considered
        (from either the vacancy (at 0,0,0) or the transition sites
        """
        self.sup = sup
        self.jnet = jnet
        self.chem = chem
        self.order = order
        self.cutoff = cutoff
        self.crys = sup.crys
        self.IndexJumps()

    def IndexJumps(self):
        jList = []
        dxList = []
        nJumpSets = len(self.jnet)

        for jumplist in self.jnet:
            newJList = []
            newdxList = []
            for (i, j), dx in jumplist:
                dxR, (ch, cj) = self.sup.crys.cart2pos(dx)
                assert j == cj
                assert ch == self.chem
                siteInd, mob = self.sup.index(dxR, (ch, cj))
                # the site must be mobile if a vacancy can jump to it
                assert mob, "Check supercell. Mobile site not detected."

                if nJumpSets == 1:  # if there is only a single set of symmetry-related jumps
                    jList.append(siteInd)
                    dxList.append(dx)
                else:
                    newJList.append(siteInd)
                    newdxList.append(dx)

            if nJumpSets > 1:
                jList.append(newJList)
                dxList.append(newdxList)

        self.jList = jList
        self.dxList = dxList

    # Now let's generate all the sites required

