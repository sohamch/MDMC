from onsager import crystal, cluster, supercell
import numpy as np
import itertools
import collections

# Test notebooks for this file:
# "Test_site_generation"

class KRA3bodyInteractions():
    """
    This is to compute KRA cluster expansion involving sites up to a cutoff and transition sites
    to be taken from a jumpnetwork in mono-atomic lattices, according to the specification in
    'doi.org/10.1016/j.msea.2018.11.064'.
    """
    def __init__(self, sup, jnet, chem, combinedShellRange, nnRange, cutoff):
        """
        :param sup: the supercell object based on which sites will be given indices
        :param jnet: the jumpnetwork from transition sites are taken
        :param chem : the sublattice on which the vacancy jumps
        :param combinedShellRange: which combined shell of the initial and final atom should the third atom be within
        :param nnRange: max nearest neighbor range to search within the sites that are in combinedShellRange
        :param cutoff: the maximum distance up to which sites will be considered
        (from either the vacancy (at 0,0,0) or the transition sites
        """
        self.sup = sup
        self.jnet = jnet
        self.chem = chem
        self.cutoff = cutoff
        self.crys = sup.crys
        self.IndexJumps()
        self.TransGroupsNN = self.GenerateInteractionSites(combinedShellRange, nnRange, cutoff)

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

    # Now let's generate all the sites required - we'll generate the sites from a jump network
    def GenerateInteractionSites(self, combinedShellRange, nnRange, cutoff):
        """
        Function to generation three-body transition state clusters grouped according to nearest neighborhoods
        within the combined first neighbor unit cell from the initial and final transition states.
        This is based on the paper 'doi.org/10.1016/j.msea.2018.11.064' and is applicable to ONLY monoatomic
        lattices.
        :param combinedShellRange: which combined shell of the initial and final atom should the third atom be within
        :param nnRange: max nearest neighbor range to search within the sites that are in combinedShellRange
        :param cutoff : the cutoff distance to get to nnrange
        :return: Trnas2NNGroups: a dictionary containing transitions as keys and interaction sites as outputs.
        """

        # First we generate a jumpnetwork containing nn translation vectors upto nn range
        nnJnet = self.crys.jumpnetwork(self.chem, cutoff)
        nnVecs = [[np.dot(np.linalg.inv(self.crys.lattice), dx).astype(int) for (i, j), dx in jList]
                              for jList in nnJnet]

        Trans2NNGroups = {}
        z = np.zeros(3, dtype=int)
        chem = self.chem
        for jList in self.jnet:
            for (i, j), dx in jList:
                siteA = cluster.ClusterSite(ci=(chem, i), R=z)
                dxR, _ = self.crys.cart2pos(dx)
                siteB = cluster.ClusterSite(ci=(chem, j), R=dxR)

                # First, make the combined first nearest shell of both siteA and siteB
                siteAshell1 = set()
                siteBshell1 = set()
                for nnT in nnVecs[:combinedShellRange]:
                    siteAshell1.add(siteA + nnT)
                    siteBshell1.add(siteB + nnT)
                combShell = siteAshell1.union(siteBshell1)

                # Now go through nn translations
                nnGroups = {}
                for nn in range(nnRange):
                    siteAnn = set()
                    siteBnn = set()
                    for vec in nnVecs[nn]:
                        siteAvec = siteA + vec
                        siteBvec = siteB + vec
                        # check that this is within 1nn of either A or B
                        if siteAvec in combShell:
                            siteAnn.add(siteAvec)
                        if siteBvec in combShell:
                            siteBnn.add(siteBvec)

                    if nn == 0:  # for first nearest neighbor we need intersection
                        nnGroups[nn + 1] = siteAnn.intersection(siteBnn)
                    else: # for the second nearest neighbor onwards, union
                        nnGroups[nn + 1] = siteAnn.union(siteBnn)

                Trans2NNGroups[(siteA, siteB)] = nnGroups

        return Trans2NNGroups

