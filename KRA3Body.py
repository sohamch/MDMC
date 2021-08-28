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
    def __init__(self, sup, jnet, chem, combinedShellRange, nnRange, cutoff, NSpec, Nvac, vacSite):
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
        self.NSpec = NSpec
        self.vacSpec = NSpec - 1
        self.Nvac = Nvac
        self.Nsites = len(self.sup.mobilepos)
        self.vacSite = vacSite
        self.maxOrderTrans = 3
        self.IndexJumps()
        self.TransGroupsNN = self.GenerateInteractionSites(combinedShellRange, nnRange, cutoff)
        self.clusterSpeciesJumps = self.defineTransSpecies()
        self.assignTransInd()

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
                assert siteA == self.vacSite
                # First, make the combined first nearest shell of both siteA and siteB
                siteAshell1 = set()
                siteBshell1 = set()
                for nnTList in nnVecs[:combinedShellRange]:
                    for nnT in nnTList:
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

    def defineTransSpecies(self):
        """
        Assign species to the transition sites only. In this form of KRA, we are only going to be seeing which type
        of atom occupies the exchange site, and whether a specified atom (Re in the reference paper) occupies the
        third site or not. Since we are only dealing with three-body clusters, we don't need to store occupancies of
        all kinds of atoms on the third site.
        """
        Nmobile = self.NSpec
        clusterJumpsSpecies = {}
        for AB, clusterSymLists in self.TransGroupsNN.items():
            # For this transition, first assign species to the Transition.
            siteA = AB[0]
            siteB = AB[1]
            assert siteA == self.vacSite
            indA = self.sup.index(siteA.R, siteA.ci)[0]
            indB = self.sup.index(siteB.R, siteB.ci)[0]
            assert indB in self.jList
            for specJ in range(Nmobile - 1):
                ABspecJ = (indA, indB, specJ)
                clusterJumpsSpecies[ABspecJ] = clusterSymLists
            # use itertools.product like in normal cluster expansion.
            # Then, assign species to the final site of the jumps.
        return clusterJumpsSpecies

    # assign IDs to the jumps in clusterSpeciesJumps for use in JIT calculations
    def assignTransInd(self):

        self.jump2Index = {}
        self.Index2Jump = {}

        for jumpInd, (Jumpkey, interactGroupList) in zip(itertools.count(),
                                                         self.clusterSpeciesJumps.items()):

            self.jump2Index[Jumpkey] = jumpInd
            self.Index2Jump[jumpInd] = Jumpkey

    # Then we make the arrays for Jit calculations
    def makeTransJitData(self, counterSpec, KRAEnergies):

        TsInteractIndexDict = {}
        Index2TSinteractDict = {}
        # 1 Get the maximum of cluster groups amongst all jumps
        maxInteractGroups = max([len(interactGroupList)
                                 for Jumpkey, interactGroupList in self.clusterSpeciesJumps.items()])

        # 2 get the maximum number of clusters in any given group
        # noinspection PyTypeChecker
        maxInteractsInGroups = max([len(interactList)
                                    for Jumpkey, interactGroupList in self.clusterSpeciesJumps.items()
                                    for interactGroupInd, interactList in interactGroupList.items()])

        # 3 create arrays to store
        # 3.1 initial and final sites in transitions
        jumpFinSites = np.full(len(self.clusterSpeciesJumps), -1, dtype=int)
        jumpFinSpec = np.full(len(self.clusterSpeciesJumps), -1, dtype=int)

        FinSiteFinSpecJumpInd = np.full((self.Nsites, self.NSpec), -1, dtype=int)

        # 3.2 To store the number of TSInteraction groups for each transition
        numJumpPointGroups = np.full(len(self.clusterSpeciesJumps), -1, dtype=int)

        # 3.3 To store the number of clusters in each TSInteraction group for each transition
        numTSInteractsInPtGroups = np.full((len(self.clusterSpeciesJumps), maxInteractGroups), -1,
                                           dtype=int)

        # 3.4
        # To store the main interaction index of each TSInteraction (will be used to check on or off status)
        JumpInteracts = np.full((len(self.clusterSpeciesJumps), maxInteractGroups, maxInteractsInGroups),
                                -1, dtype=int)

        # 3.5 To store the KRA energies for each transition state cluster
        Jump2KRAEng = np.zeros((len(self.clusterSpeciesJumps), maxInteractGroups, maxInteractsInGroups))
        # Fill up the arrays
        count = 0  # to keep track of the integer assigned to each TS interaction.
        for (Jumpkey, interactGroupList) in self.clusterSpeciesJumps.items():

            jumpInd = self.jump2Index[Jumpkey]

            jumpFinSites[jumpInd] = Jumpkey[1]
            jumpFinSpec[jumpInd] = Jumpkey[2]
            FinSiteFinSpecJumpInd[Jumpkey[1], Jumpkey[2]] = jumpInd
            numJumpPointGroups[jumpInd] = len(interactGroupList)

            for interactGroupInd, interactGroup in enumerate(interactGroupList):
                clusterList = interactGroupList[interactGroup]
                numTSInteractsInPtGroups[jumpInd, interactGroupInd] = len(clusterList)
                for interactInd, site3 in enumerate(clusterList):
                    TSInteract = tuple([(Jumpkey[0], self.vacSpec), (Jumpkey[1], Jumpkey[2]),
                                  (self.sup.index(site3.R, site3.ci)[0], counterSpec)])

                    # counterSpec is the species that will be considered for determining energies.
                    if TSInteract not in TsInteractIndexDict:
                        TsInteractIndexDict[TSInteract] = count
                        Index2TSinteractDict[count] = TSInteract
                        count += 1

                    JumpInteracts[jumpInd, interactGroupInd, interactInd] = TsInteractIndexDict[TSInteract]
                    Jump2KRAEng[jumpInd, interactGroupInd, interactInd] = KRAEnergies[jumpInd][interactGroupInd]

        # 4 Next, make arrays that store the sites and species in each TS interaction.
        TSInteractSites = np.full((len(TsInteractIndexDict), self.maxOrderTrans), -1, dtype=int)
        TSInteractSpecs = np.full((len(TsInteractIndexDict), self.maxOrderTrans), -1, dtype=int)
        numSitesTSInteracts = np.full(len(TsInteractIndexDict), -1, dtype=int)

        for index, TSInteract in Index2TSinteractDict.items():
            numSitesTSInteracts[index] = len(TSInteract)
            for siteIdx, (site, spec) in zip(itertools.count(), TSInteract):
                TSInteractSites[index, siteIdx] = site
                TSInteractSpecs[index, siteIdx] = spec

        return TsInteractIndexDict, Index2TSinteractDict, numSitesTSInteracts, TSInteractSites, TSInteractSpecs,\
               jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups,\
               JumpInteracts, Jump2KRAEng


