from onsager import crystal, supercell, cluster
import numpy as np
import itertools
import Transitions
import Cluster_Expansion
import MC_JIT
import unittest
import time
import warnings
import collections

warnings.filterwarnings('error', category=RuntimeWarning)

np.seterr(all='raise')

class Test_MC_Arrays(unittest.TestCase):

    def setUp(self):
        self.NSpec = 3
        self.MaxOrder = 3
        self.MaxOrderTrans = 3
        self.crys = crystal.Crystal.BCC(0.2836, chemistry="A")
        self.jnetBCC = self.crys.jumpnetwork(0, 0.26)
        self.superlatt = 8 * np.eye(3, dtype=int)
        self.superBCC = supercell.ClusterSupercell(self.crys, self.superlatt)
        # get the number of sites in the supercell - should be 8x8x8
        numSites = len(self.superBCC.mobilepos)
        self.vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
        self.vacsiteInd = self.superBCC.index(np.zeros(3, dtype=int), (0, 0))[0]
        self.mobOccs = np.zeros((self.NSpec, numSites), dtype=int)
        for site in range(1, numSites):
            spec = np.random.randint(0, self.NSpec-1)
            self.mobOccs[spec][site] = 1
        self.mobOccs[-1, self.vacsiteInd] = 1
        self.mobCountList = [np.sum(self.mobOccs[i]) for i in range(self.NSpec)]
        self.clusexp = cluster.makeclusters(self.crys, 0.284, self.MaxOrder)
        self.Tclusexp = cluster.makeclusters(self.crys, 0.29, self.MaxOrderTrans)
        self.KRAexpander = Transitions.KRAExpand(self.superBCC, 0, self.jnetBCC, self.Tclusexp, self.Tclusexp, self.mobCountList,
                                                 self.vacsite)
        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(self.superBCC, self.clusexp, self.Tclusexp, self.jnetBCC,
                                                                 self.mobCountList, self.vacsite, self.MaxOrder,
                                                                 self.MaxOrderTrans)

        self.Energies = np.random.rand(len(self.VclusExp.SpecClusters))
        KRAEn = np.random.rand()
        self.KRAEnergies = [np.ones(len(TSptGroups))*KRAEn for (key, TSptGroups) in self.VclusExp.KRAexpander.clusterSpeciesJumps.items()]

        print("Done setting up")

        self.MakeJITs()

    def MakeJITs(self):
        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts, VecsInteracts, \
        VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray, vacSiteInd, InteractionIndexDict, InteractionRepClusDict, \
        Index2InteractionDict, repClustCounter = \
            self.VclusExp.makeJitInteractionsData(self.Energies)

        self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites, self.Interaction2En, self.numVecsInteracts, self.VecsInteracts,\
        self.VecGroupInteracts, self.numInteractsSiteSpec, self.SiteSpecInterArray, self.vacSiteInd, self.InteractionIndexDict,\
        self.InteractionRepClusDict, self.Index2InteractionDict, self.repClustCounter = \
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts, VecsInteracts, \
            VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray, vacSiteInd, InteractionIndexDict, InteractionRepClusDict, \
            Index2InteractionDict, repClustCounter

        TsInteractIndexDict, Index2TSinteractDict, numSitesTSInteracts, TSInteractSites, TSInteractSpecs, \
        jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, \
        JumpInteracts, Jump2KRAEng = \
            self.VclusExp.KRAexpander.makeTransJitData(self.KRAEnergies)

        self.TsInteractIndexDict, self.Index2TSinteractDict, self.numSitesTSInteracts, self.TSInteractSites, self.TSInteractSpecs, \
        self.jumpFinSites, self.jumpFinSpec, self.FinSiteFinSpecJumpInd, self.numJumpPointGroups, self.numTSInteractsInPtGroups, \
        self.JumpInteracts, self.Jump2KRAEng = \
            TsInteractIndexDict, Index2TSinteractDict, numSitesTSInteracts, TSInteractSites, TSInteractSpecs, \
            jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, \
            JumpInteracts, Jump2KRAEng

        Nsites = self.VclusExp.Nsites
        N_units = self.VclusExp.sup.superlatt[0, 0]
        siteIndtoR = np.zeros((Nsites, 3), dtype=int)
        RtoSiteInd = np.zeros((N_units, N_units, N_units), dtype=int)

        for siteInd in range(Nsites):
            R = self.VclusExp.sup.ciR(siteInd)[1]
            siteIndtoR[siteInd, :] = R
            RtoSiteInd[R[0], R[1], R[2]] = siteInd

        self.RtoSiteInd = RtoSiteInd
        self.siteIndtoR = siteIndtoR

        self.KMC_Jit = MC_JIT.KMC_JIT(numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts,
                                 VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray,
                                 numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
                                 FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng,
                                 siteIndtoR, RtoSiteInd, N_units)

        initState = np.zeros(len(self.VclusExp.sup.mobilepos), dtype=int)
        # Now assign random species (excluding the vacancy)
        for i in range(len(self.VclusExp.sup.mobilepos)):
            initState[i] = np.random.randint(0, self.NSpec - 1)

        # Now put in the vacancy at the vacancy site
        initState[self.vacsiteInd] = self.NSpec - 1

        self.initState = initState

        self.MCSampler = Cluster_Expansion.MCSamplerClass(
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts,
            VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray,
            numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
            FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng,
            vacSiteInd, initState
        )

        OffSiteCount = np.zeros_like(self.MCSampler.OffSiteCount)

        self.MCSampler_Jit = MC_JIT.MCSamplerClass(
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts,
            VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray,
            numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
            FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng,
            vacSiteInd, initState, OffSiteCount)



    def test_arrays(self):

        start = time.time()

        # Check that each cluster has been translated as many times as there are sites in the supercell
        # Only then we have constructed every possible interaction
        print("Done creating arrays : {}".format(time.time() - start))
        # Now, we first test the interaction arrays - the ones to be used in the MC sweeps

        # check that every representative clusters has been translated as many times
        # as there are unit cells in the super cell.
        for repClust, count in self.repClustCounter.items():
            self.assertEqual(count, len(self.VclusExp.sup.mobilepos))
        # numSitesInteracts - the number of sites in an interaction
        for i in range(len(self.numSitesInteracts)):
            siteCountInArray = self.numSitesInteracts[i]
            # get the interaction
            interaction = self.Index2InteractionDict[i]
            # get the stored index for this interaction
            i_stored = self.InteractionIndexDict[interaction]
            self.assertEqual(i, i_stored)
            self.assertEqual(len(interaction), siteCountInArray)

        # Now, check the supercell sites and species stored for these interactions
        self.assertEqual(len(self.SupSitesInteracts), len(self.numSitesInteracts))
        self.assertEqual(len(self.SpecOnInteractSites), len(self.numSitesInteracts))
        for i in range(len(self.numSitesInteracts)):
            siteSpecSet = set([(self.SupSitesInteracts[i, j], self.SpecOnInteractSites[i, j])
                               for j in range(self.numSitesInteracts[i])])
            interaction = self.Index2InteractionDict[i]
            interactionSet = set(interaction)
            self.assertEqual(siteSpecSet, interactionSet)
        print("checked interactions")

        # Now, test the vector basis and energy information for the clusters
        for i in range(len(self.numSitesInteracts)):
            # get the interaction
            interaction = self.Index2InteractionDict[i]
            # Now, get the representative cluster
            repClus = self.InteractionRepClusDict[interaction]

            # test the energy index
            enIndex = self.VclusExp.clust2SpecClus[repClus][0]
            self.assertEqual(self.Interaction2En[i], self.Energies[enIndex])

            # if the vector basis is empty, continue
            if self.VclusExp.clus2LenVecClus[self.VclusExp.clust2SpecClus[repClus][0]] == 0:
                self.assertEqual(self.numVecsInteracts[i], -1)
                continue
            # get the vector basis info for this cluster
            vecList = self.VclusExp.clust2vecClus[repClus]
            # check the number of vectors
            self.assertEqual(self.numVecsInteracts[i], len(vecList))
            # check that the correct vector have been stored, in the same order as in vecList (not necessary but short testing)
            for vecind in range(len(vecList)):
                vec = self.VclusExp.vecVec[vecList[vecind][0]][vecList[vecind][1]]
                self.assertTrue(np.allclose(vec, self.VecsInteracts[i, vecind, :]))

        # Next, test the interactions each (site, spec) is a part of
        self.assertEqual(self.numInteractsSiteSpec.shape[0], len(self.superBCC.mobilepos))
        self.assertEqual(self.numInteractsSiteSpec.shape[1], self.NSpec)
        for site in range(len(self.superBCC.mobilepos)):
            for spec in range(self.NSpec):
                numInteractStored = self.numInteractsSiteSpec[site, spec]
                # get the actual count
                ci, R = self.VclusExp.sup.ciR(site)
                clsite = cluster.ClusterSite(ci=ci, R=R)
                self.assertEqual(len(self.VclusExp.SiteSpecInteractions[(clsite, spec)]), numInteractStored)
                for IdxOfInteract in range(numInteractStored):
                    interactMainIndex = self.SiteSpecInterArray[site, spec, IdxOfInteract]
                    self.assertEqual(self.Index2InteractionDict[interactMainIndex],
                                     self.VclusExp.SiteSpecInteractions[(clsite, spec)][IdxOfInteract][0])

    def testTransArrays(self):
        # Now, we start testing the jump arrays
        # jumpFinSites, jumpFinSpec, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng
        # TODO : Re write after figuring out detailed balance

        for (jumpkey, TSptGrps) in self.VclusExp.KRAexpander.clusterSpeciesJumps.items():
            jumpInd = self.VclusExp.KRAexpander.jump2Index[jumpkey]
            FinSite = jumpkey[1]
            FinSpec = jumpkey[2]
            # Check that the correct initial and final states have been stored
            self.assertEqual(self.jumpFinSites[jumpInd], FinSite)
            self.assertEqual(self.jumpFinSpec[jumpInd], FinSpec)

            # Check that the correct number of point groups are stored
            NptGrp = len(TSptGrps)
            self.assertEqual(self.numJumpPointGroups[jumpInd], NptGrp)

            # Check that in for each point group, the correct interactions are stored.
            for TsPtGpInd, (spectup, TSinteractList) in zip(itertools.count(), TSptGrps):
                self.assertEqual(self.numTSInteractsInPtGroups[jumpInd, TsPtGpInd], len(TSinteractList))
                specList = [self.NSpec - 1, FinSpec] + [spec for spec in spectup]
                for interactInd, TSClust in enumerate(TSinteractList):
                    interact = tuple([(self.VclusExp.sup.index(site.R, site.ci)[0], spec)
                                      for site, spec in zip(TSClust.sites, specList)])
                    interactStored = self.Index2TSinteractDict[self.JumpInteracts[jumpInd, TsPtGpInd, interactInd]]

                    self.assertEqual(set(interact), set(interactStored))
                    self.assertEqual(self.Jump2KRAEng[jumpInd, TsPtGpInd, interactInd], self.KRAEnergies[jumpInd][TsPtGpInd])

class Test_MC(Test_MC_Arrays):

    def test_MC_step(self):
        # First, create a random state
        initState = self.initState

        # Now put in the vacancy at the vacancy site
        initCopy = initState.copy()
        initJit = initState.copy()

        # Get the MC samplers
        MCSampler = self.MCSampler
        MCSampler_Jit = self.MCSampler_Jit

        # First check that the initial OffsiteCount is computed correctly
        for (interaction, interactionInd) in self.InteractionIndexDict.items():
            offsiteCount = 0
            for (site, spec) in interaction:
                if initState[site] != spec:
                    offsiteCount += 1
            self.assertEqual(MCSampler.OffSiteCount[interactionInd], offsiteCount, msg="{} {} \n{} \n{}".format(
                MCSampler.OffSiteCount[interactionInd], offsiteCount,
                interaction, [initState[site] for site in [site for site, spec in interaction]]
            ))

        # calculate Initial energy
        InitEn = 0.
        for i in range(len(MCSampler.OffSiteCount)):
            if MCSampler.OffSiteCount[i] == 0:
                InitEn += self.Interaction2En[i]

        # Do MC single metropolis step that updates the state
        offsc = MCSampler.OffSiteCount.copy()
        TransOffSiteCountNew = np.zeros(len(self.TSInteractSites), dtype=int)
        Nsites = len(self.VclusExp.sup.mobilepos)
        Nswaptrials = 1  # We are only testing a single step here
        swaptrials = np.zeros((Nswaptrials, 2), dtype=int)

        count = 0
        while count < Nswaptrials:
            # first select two random sites to swap - for now, let's just select naively.
            siteA = np.random.randint(0, Nsites)
            siteB = np.random.randint(0, Nsites)

            # make sure we are swapping different atoms because otherwise we are in the same state
            if initState[siteA] == initState[siteB] or siteA == self.vacSiteInd or siteB == self.vacSiteInd:
                continue

            swaptrials[count, 0] = siteA
            swaptrials[count, 1] = siteB
            count += 1
        randarr = np.random.rand(Nswaptrials)

        randarr = np.log(randarr)

        # Do MC sweep
        MCSampler.makeMCsweep(initState, offsc, TransOffSiteCountNew, swaptrials, 1.0, randarr, Nswaptrials)

        # Check that offsitecounts are updated correctly
        for interactIdx in range(self.numSitesInteracts.shape[0]):
            numSites = self.numSitesInteracts[interactIdx]
            offcount = 0
            for intSiteind in range(numSites):
                interSite = self.SupSitesInteracts[interactIdx, intSiteind]
                interSpec = self.SpecOnInteractSites[interactIdx, intSiteind]
                if initState[interSite] != interSpec:
                    offcount += 1
            self.assertTrue(offsc[interactIdx] == offcount)


        for TsInteractIdx in range(len(TransOffSiteCountNew)):
            offcount = 0
            for Siteind in range(self.numSitesTSInteracts[TsInteractIdx]):
                if initState[self.TSInteractSites[TsInteractIdx, Siteind]] != self.TSInteractSpecs[TsInteractIdx, Siteind]:
                    offcount += 1
            self.assertEqual(TransOffSiteCountNew[TsInteractIdx], offcount)

        offsc2 = MCSampler.OffSiteCount.copy()
        for intInd in range(self.numInteractsSiteSpec[siteA, initCopy[siteA]]):
            offsc2[self.SiteSpecInterArray[siteA, initCopy[siteA], intInd]] += 1

        for intInd in range(self.numInteractsSiteSpec[siteB, initCopy[siteB]]):
            offsc2[self.SiteSpecInterArray[siteB, initCopy[siteB], intInd]] += 1

        for intInd in range(self.numInteractsSiteSpec[siteA, initCopy[siteB]]):
            offsc2[self.SiteSpecInterArray[siteA, initCopy[siteB], intInd]] -= 1

        for intInd in range(self.numInteractsSiteSpec[siteB, initCopy[siteA]]):
            offsc2[self.SiteSpecInterArray[siteB, initCopy[siteA], intInd]] -= 1

        # Now calculate the final energy based on the new offsite counts
        FinEn = 0.
        for i in range(len(offsc2)):
            if offsc2[i] == 0:
                FinEn += self.Interaction2En[i]

        OffSiteCount = np.zeros_like(MCSampler.OffSiteCount, dtype=int)
        TSOffSiteCount2 = np.zeros_like(TransOffSiteCountNew)

        # make the offsite counts
        offscjit = MCSampler_Jit.OffSiteCount.copy()

        # get the array counting how many representative clusters are on from offscjit

        MCSampler_Jit.makeMCsweep(initJit, offscjit, TSOffSiteCount2, swaptrials, 1.0, randarr, Nswaptrials)

        # Check that the same results are found
        self.assertTrue(np.array_equal(initJit, initState))
        self.assertTrue(np.array_equal(offscjit, offsc))
        self.assertTrue(np.array_equal(TransOffSiteCountNew, TSOffSiteCount2))

        # test that energy calculation and site swaps are done correctly.
        if np.exp(-MCSampler.delE) > np.exp(randarr[0]):
            self.assertEqual(initJit[siteA], initCopy[siteB])
            self.assertEqual(initJit[siteB], initCopy[siteA])
            print("move was accepted {} {}".format(np.exp(-MCSampler.delE), np.exp(randarr[0])))
        else:
            self.assertTrue(np.array_equal(initJit, initCopy))
            print("move was not accepted {} {}".format(np.exp(-MCSampler.delE), np.exp(randarr[0])))

        self.assertTrue(np.allclose(MCSampler.delE, FinEn - InitEn), msg="{}, {}".format(MCSampler.delE, FinEn - InitEn))

    def test_exit_states(self):
        # First, create a random state
        initState = self.initState
        initCopy = initState.copy()
        initJit = initState.copy()

        MCSampler_Jit = self.MCSampler_Jit

        ijList = self.VclusExp.KRAexpander.ijList
        dxList = self.VclusExp.KRAexpander.dxList

        # get the offsite counts
        state = initCopy.copy()
        offsc = MCSampler_Jit.OffSiteCount.copy()
        Nsites = self.VclusExp.Nsites

        # make the TS offsite counts
        TSoffsc = np.zeros(len(MCSampler_Jit.TSInteractSites))
        for TsInteractIdx in range(len(MCSampler_Jit.TSInteractSites)):
            TSoffsc[TsInteractIdx] = 0
            for Siteind in range(MCSampler_Jit.numSitesTSInteracts[TsInteractIdx]):
                if state[MCSampler_Jit.TSInteractSites[TsInteractIdx, Siteind]] != MCSampler_Jit.TSInteractSpecs[TsInteractIdx, Siteind]:
                    TSoffsc[TsInteractIdx] += 1

        # Next, get the exit data
        statesTrans, ratelist, Specdisps = MCSampler_Jit.getExitData(state, ijList, dxList, offsc, TSoffsc, 1.0, Nsites)

        self.assertEqual(len(statesTrans), ijList.shape[0])

        # check that the species have been exchanged properly
        for jInd in range(ijList.shape[0]):
            stateFin = statesTrans[jInd, :]
            finSite = ijList[jInd]
            # Check that in the final state the correct species have been swapped
            self.assertEqual(stateFin[finSite], state[self.vacSiteInd])
            self.assertEqual(stateFin[self.vacSiteInd], state[finSite])

            specB = state[finSite]  # the species that has moved
            dx = Specdisps[jInd, specB, :]
            self.assertTrue(np.allclose(dx, -dxList[jInd]))

    def test_random_state(self):
        initState = self.initState
        initCopy = initState.copy()

        MCSampler_Jit = self.MCSampler_Jit

        # test random state generation
        state = initCopy.copy()
        offsc = MCSampler_Jit.OffSiteCount.copy()

        Nswaptrials = 100  # We are only testing a single step here
        swaptrials = np.zeros((Nswaptrials, 2), dtype=int)

        Nsites = len(self.VclusExp.sup.mobilepos)

        count = 0
        while count < Nswaptrials:
            # first select two random sites to swap - for now, let's just select naively.
            siteA = np.random.randint(0, Nsites)
            siteB = np.random.randint(0, Nsites)

            # make sure we are swapping different atoms because otherwise we are in the same state
            if state[siteA] == state[siteB] or siteA == self.vacSiteInd or siteB == self.vacSiteInd:
                continue

            swaptrials[count, 0] = siteA
            swaptrials[count, 1] = siteB
            count += 1

        InitEn = 0.
        for i in range(len(offsc)):
            if offsc[i] == 0:
                InitEn += self.Interaction2En[i]

        En_new = MCSampler_Jit.GetNewRandState(state, offsc, InitEn, swaptrials, Nswaptrials)

        FinEn = 0.
        for i in range(len(offsc)):
            if offsc[i] == 0:
                FinEn += self.Interaction2En[i]

        self.assertTrue(np.allclose(FinEn, En_new))

    def test_expansion(self):
        """
        To test if Wbar and Bbar are computed correctly
        """
        # Compute them with the dicts and match with what comes out from the code.
        # Wbar_test = np.zeros()

        initState = self.initState

        # initiate MC Sampler
        MCSampler_Jit = self.MCSampler_Jit

        for (interaction, interactionInd) in self.InteractionIndexDict.items():
            offsiteCount = 0
            for (site, spec) in interaction:
                if initState[site] != spec:
                    offsiteCount += 1
            self.assertEqual(MCSampler_Jit.OffSiteCount[interactionInd], offsiteCount)

        TransOffSiteCount = np.zeros(len(self.TSInteractSites), dtype=int)

        # Build TS offsites
        for TsInteractIdx in range(len(TransOffSiteCount)):
            for Siteind in range(self.numSitesTSInteracts[TsInteractIdx]):
                if initState[self.TSInteractSites[TsInteractIdx, Siteind]] != self.TSInteractSpecs[TsInteractIdx, Siteind]:
                    TransOffSiteCount[TsInteractIdx] += 1

        ijList, dxList = self.VclusExp.KRAexpander.ijList.copy(), self.VclusExp.KRAexpander.dxList.copy()
        lenVecClus = len(self.VclusExp.vecClus)

        # Now, do the expansion
        offscjit = MCSampler_Jit.OffSiteCount.copy()
        state = initState.copy()
        Wbar, Bbar = MCSampler_Jit.Expand(state, ijList, dxList, offscjit, TransOffSiteCount,
                                                                  lenVecClus, 1.0)

        # Check that the offsitecounts have been correctly reverted and state is unchanged.

        self.assertTrue(np.array_equal(MCSampler_Jit.OffSiteCount, offscjit))
        self.assertTrue(np.array_equal(state, initState))

        self.assertTrue(np.array_equal(self.SiteSpecInterArray, MCSampler_Jit.SiteSpecInterArray))
        self.assertTrue(np.array_equal(self.numInteractsSiteSpec, MCSampler_Jit.numInteractsSiteSpec))
        self.assertTrue(np.allclose(self.Interaction2En, MCSampler_Jit.Interaction2En))

        print("Starting TS tests")
        Wbar_test = np.zeros_like(Wbar, dtype=float)
        Bbar_test = np.zeros_like(Bbar, dtype=float)

        # Now test the rate expansion by explicitly constructing it
        for vs1 in range(len(self.VclusExp.vecVec)):
            for vs2 in range(len(self.VclusExp.vecVec)):
                # Go through all the jumps
                for TInd in range(len(ijList)):
                    # For every jump, reset the offsite count
                    offscjit = MCSampler_Jit.OffSiteCount.copy()
                    TSOffCount = TransOffSiteCount.copy()

                    delEKRA = 0.0
                    vec1 = np.zeros(3, dtype=float)
                    vec2 = np.zeros(3, dtype=float)

                    siteB = ijList[TInd]
                    siteA = MCSampler_Jit.vacSiteInd
                    # Check that the initial site is always the vacancy
                    specA = state[siteA]  # the vacancy
                    self.assertEqual(specA, self.NSpec - 1)
                    self.assertEqual(siteA, self.vacsiteInd)
                    specB = state[siteB]
                    # get the index of this transition
                    # self.assertEqual(specB, SpecTransArray[TInd])
                    # self.assertEqual(siteB, SiteTransArray[TInd])

                    jumpInd = self.FinSiteFinSpecJumpInd[siteB, specB]
                    # get the KRA energy for this jump in this state
                    for ptgrpInd in range(self.numJumpPointGroups[jumpInd]):
                        for ptGpInteractInd in range(self.numTSInteractsInPtGroups[jumpInd, ptgrpInd]):
                            # See if the interaction is on
                            offcount = TSOffCount[self.JumpInteracts[jumpInd, ptgrpInd, ptGpInteractInd]]
                            if offcount == 0:
                                delEKRA += self.Jump2KRAEng[jumpInd, ptgrpInd, ptGpInteractInd]

                    # Now do the site swaps and calculate the energy
                    delE = 0.0

                    for interactnun in range(self.numInteractsSiteSpec[siteA, specA]):
                        interactInd = self.SiteSpecInterArray[siteA, specA, interactnun]
                        repClus = self.InteractionRepClusDict[self.Index2InteractionDict[interactInd]]
                        vecList = self.VclusExp.clust2vecClus[repClus]

                        if len(vecList) > 0:
                            self.assertEqual(self.numVecsInteracts[interactInd], len(vecList))
                        else:
                            self.assertEqual(self.numVecsInteracts[interactInd], -1)

                        for i in range(self.numVecsInteracts[interactInd]):
                            self.assertTrue(np.allclose(self.VecsInteracts[interactInd, i, :],
                                                        self.VclusExp.vecVec[vecList[i][0]][vecList[i][1]]))
                        if offscjit[interactInd] == 0:
                            delE -= self.Interaction2En[interactInd]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs1:
                                    vec1 -= self.VecsInteracts[interactInd, tupInd, :]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs2:
                                    vec2 -= self.VecsInteracts[interactInd, tupInd, :]
                        offscjit[interactInd] += 1

                    for interactnun in range(self.numInteractsSiteSpec[siteB, specB]):
                        interactInd = self.SiteSpecInterArray[siteB, specB, interactnun]
                        repClus = self.InteractionRepClusDict[self.Index2InteractionDict[interactInd]]
                        vecList = self.VclusExp.clust2vecClus[repClus]

                        if len(vecList) > 0:
                            self.assertEqual(self.numVecsInteracts[interactInd], len(vecList))
                        else:
                            self.assertEqual(self.numVecsInteracts[interactInd], -1)

                        for i in range(self.numVecsInteracts[interactInd]):
                            self.assertTrue(np.allclose(self.VecsInteracts[interactInd, i, :],
                                                        self.VclusExp.vecVec[vecList[i][0]][vecList[i][1]]))
                        if offscjit[interactInd] == 0:
                            delE -= self.Interaction2En[interactInd]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs1:
                                    vec1 -= self.VecsInteracts[interactInd, tupInd, :]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs2:
                                    vec2 -= self.VecsInteracts[interactInd, tupInd, :]
                        offscjit[interactInd] += 1

                    for interactnun in range(self.numInteractsSiteSpec[siteA, specB]):
                        interactInd = self.SiteSpecInterArray[siteA, specB, interactnun]

                        repClus = self.InteractionRepClusDict[self.Index2InteractionDict[interactInd]]
                        vecList = self.VclusExp.clust2vecClus[repClus]

                        if len(vecList) > 0:
                            self.assertEqual(self.numVecsInteracts[interactInd], len(vecList))
                        else:
                            self.assertEqual(self.numVecsInteracts[interactInd], -1)

                        for i in range(self.numVecsInteracts[interactInd]):
                            self.assertTrue(np.allclose(self.VecsInteracts[interactInd, i, :],
                                                        self.VclusExp.vecVec[vecList[i][0]][vecList[i][1]]))

                        offscjit[interactInd] -= 1
                        if offscjit[interactInd] == 0:
                            delE += self.Interaction2En[interactInd]

                            repClus = self.InteractionRepClusDict[self.Index2InteractionDict[interactInd]]
                            vecList = self.VclusExp.clust2vecClus[repClus]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs1:
                                    vec1 += self.VecsInteracts[interactInd, tupInd, :]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs2:
                                    vec2 += self.VecsInteracts[interactInd, tupInd, :]

                    for interactnun in range(self.numInteractsSiteSpec[siteB, specA]):
                        interactInd = self.SiteSpecInterArray[siteB, specA, interactnun]

                        repClus = self.InteractionRepClusDict[self.Index2InteractionDict[interactInd]]
                        vecList = self.VclusExp.clust2vecClus[repClus]

                        if len(vecList) > 0:
                            self.assertEqual(self.numVecsInteracts[interactInd], len(vecList))
                        else:
                            self.assertEqual(self.numVecsInteracts[interactInd], -1)

                        for i in range(self.numVecsInteracts[interactInd]):
                            self.assertTrue(np.allclose(self.VecsInteracts[interactInd, i, :],
                                                        self.VclusExp.vecVec[vecList[i][0]][vecList[i][1]]))

                        offscjit[interactInd] -= 1
                        if offscjit[interactInd] == 0:
                            delE += self.Interaction2En[interactInd]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs1:
                                    self.assertEqual(self.VecGroupInteracts[interactInd, tupInd], vs1)
                                    vec1 += self.VecsInteracts[interactInd, tupInd, :]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs2:
                                    self.assertEqual(self.VecGroupInteracts[interactInd, tupInd], vs2)
                                    vec2 += self.VecsInteracts[interactInd, tupInd, :]
                    # get the rate
                    rate = np.exp(-(0.5 * delE + delEKRA))
                    # get the dot product
                    dot = np.dot(vec1, vec2)

                    Wbar_test[vs1, vs2] += rate*dot
                    if vs1 == 0:
                        Bbar_test[vs2] += rate*np.dot(dxList[TInd], vec2)

                self.assertAlmostEqual(Wbar[vs1, vs2], Wbar_test[vs1, vs2], 8,
                                       msg="\n{} {}".format(Wbar[vs1, vs2], Wbar_test[vs1, vs2]))

        self.assertTrue(np.allclose(Bbar, Bbar_test))


class Test_KMC(Test_MC_Arrays):

    def test_translation(self):
        initState = self.initState
        state = initState.copy()

        # Now make the RtoSiteInd and SiteIndtoR arrays
        Nsites = self.VclusExp.Nsites
        N_units = self.VclusExp.sup.superlatt[0, 0]
        siteIndtoR = self.siteIndtoR
        RtoSiteInd = self.RtoSiteInd

        KMC_Jit = self.KMC_Jit

        # Now produce a translated state
        # take two random site Indices
        siteA = np.random.randint(0, Nsites)
        siteB = np.random.randint(0, Nsites)

        print(siteA, siteB)

        stateTrans = KMC_Jit.TranslateState(state, siteB, siteA)

        Rf = self.VclusExp.sup.ciR(siteB)[1]
        Ri = self.VclusExp.sup.ciR(siteA)[1]

        # Now, test the translated state
        # since monoatomic BCC, ci = (0,0)
        ci = (0, 0)
        for site in range(Nsites):
            Rsite = self.VclusExp.sup.ciR(site)[1]

            self.assertTrue(np.array_equal(Rsite, siteIndtoR[site, :]))

            R2 = Rsite + Rf - Ri  # get the new location of the site
            siteTrans = self.VclusExp.sup.index(R2, ci)[0]
            R2_incell = self.VclusExp.sup.ciR(siteTrans)[1]

            RTrans2 = (siteIndtoR[site, :] + Rf - Ri) % N_units

            siteT2 = RtoSiteInd[RTrans2[0], RTrans2[1], RTrans2[2]]

            self.assertEqual(siteT2, siteTrans, msg="\n{} {} \n{} \n{} \n{}".format(siteT2, siteTrans, R2, R2_incell, RTrans2))
            self.assertEqual(state[site], stateTrans[siteTrans])
            self.assertEqual(state[site], stateTrans[siteT2])

        # Now check that no change occurs when no translation is needed
        stateTrans = KMC_Jit.TranslateState(state, siteB, siteB)
        self.assertTrue(np.array_equal(state, stateTrans))

    def test_Offsite(self):
        # here we check if the off site counts are being computed correctly for a state
        state = self.initState.copy()

        # First, the cluster off site counts
        OffSiteCount = self.KMC_Jit.GetOffSite(state)
        for (interaction, interactionInd) in self.InteractionIndexDict.items():
            offsiteCount = 0
            for (site, spec) in interaction:
                if state[site] != spec:
                    offsiteCount += 1
            self.assertEqual(OffSiteCount[interactionInd], offsiteCount)

        # Next, the TS cluster off site counts
        TransOffSiteCount = self.KMC_Jit.GetTSOffSite(state)
        for TsInteractIdx in range(self.numSitesTSInteracts.shape[0]):
            Interaction = self.Index2TSinteractDict[TsInteractIdx]
            offcount = 0
            for Siteind in range(self.numSitesTSInteracts[TsInteractIdx]):
                self.assertEqual(self.TSInteractSpecs[TsInteractIdx, Siteind], Interaction[Siteind][1])
                self.assertEqual(self.TSInteractSites[TsInteractIdx, Siteind], Interaction[Siteind][0])
                if state[self.TSInteractSites[TsInteractIdx, Siteind]] != self.TSInteractSpecs[TsInteractIdx, Siteind]:
                    offcount += 1
            self.assertEqual(offcount, TransOffSiteCount[TsInteractIdx])

    def test_swap_energy_change(self):
        state = self.initState.copy()
        OffSiteCount = self.KMC_Jit.GetOffSite(state)
        offscCopy = OffSiteCount.copy()
        N_units = self.VclusExp.sup.superlatt[0, 0]

        siteIndtoR = self.siteIndtoR
        RtoSiteInd = self.RtoSiteInd

        jmpFinSiteList = self.VclusExp.KRAexpander.ijList

        # Calculate the energy of the initial state
        EnState = 0.
        for interactInd in range(len(OffSiteCount)):
            if OffSiteCount[interactInd] == 0:
                EnState += self.Interaction2En[interactInd]

        # collect energy changes due to the jumps
        delEJumps = self.KMC_Jit.getEnergyChangeJumps(state, OffSiteCount, self.vacSiteInd, jmpFinSiteList)

        # check that the state and the offsite counts have been left unchanged
        self.assertTrue(np.array_equal(OffSiteCount, offscCopy))
        self.assertTrue(np.array_equal(state, self.initState))

        # Now go through each of the transitions and evaluate the energy changes explicitly
        for jumpInd, siteInd in enumerate(jmpFinSiteList):
            stateNew = state.copy()
            stateNew[siteInd] = self.NSpec - 1  # this site will contain the vacancy in the new site
            stateNew[self.vacSiteInd] = state[siteInd]  # this site will contain the vacancy in the new site

            OffScTrans = self.KMC_Jit.GetOffSite(stateNew)

            EnNew = 0.
            for interactInd in range(len(OffSiteCount)):
                if OffScTrans[interactInd] == 0:
                    EnNew += self.Interaction2En[interactInd]

            delE = EnNew - EnState

            # print("\n{:.4f} {:.4f}".format(delE, delEJumps[jumpInd]))

            self.assertAlmostEqual(delE, delEJumps[jumpInd])

        # now let's swap the sites so that the vacancy is not at the origin anymore, and then test it
        state = self.initState.copy()

        siteSwap = np.random.randint(0, len(state))

        temp = state[siteSwap]
        state[siteSwap] = state[self.vacSiteInd]
        state[self.vacSiteInd] = temp
        self.assertEqual(state[siteSwap], self.NSpec-1)
        statecpy = state.copy()

        OffSiteCount = self.KMC_Jit.GetOffSite(state)
        offscCopy = OffSiteCount.copy()

        jmpFinSiteList = self.VclusExp.KRAexpander.ijList

        jmpFinSiteListTrans = np.zeros_like(jmpFinSiteList)

        dR = self.siteIndtoR[siteSwap] - self.siteIndtoR[self.vacSiteInd]

        for jmp in range(jmpFinSiteList.shape[0]):
            RfinSiteNew = (dR + self.siteIndtoR[jmpFinSiteList[jmp]]) % N_units
            jmpFinSiteListTrans[jmp] = self.RtoSiteInd[RfinSiteNew[0], RfinSiteNew[1], RfinSiteNew[2]]

        # Now get the energy changes during the jumps
        delEJumps = self.KMC_Jit.getEnergyChangeJumps(state, OffSiteCount, siteSwap, jmpFinSiteListTrans)

        # Now evaluate the energy changes explicitly and see if the energies are the same

        self.assertTrue(np.array_equal(OffSiteCount, offscCopy))
        self.assertTrue(np.array_equal(state, statecpy))

        # Calculate the energy of the initial state
        EnState = 0.
        for interactInd in range(len(OffSiteCount)):
            if OffSiteCount[interactInd] == 0:
                EnState += self.Interaction2En[interactInd]

        # Now go through each of the transitions and evaluate the energy changes explicitly
        for jumpInd, siteInd in enumerate(jmpFinSiteListTrans):
            stateNew = state.copy()
            stateNew[siteInd] = self.NSpec - 1  # this site will contain the vacancy in the new site
            stateNew[siteSwap] = state[siteInd]  # this site will contain the vacancy in the new site

            OffScTrans = self.KMC_Jit.GetOffSite(stateNew)

            EnNew = 0.
            for interactInd in range(len(OffSiteCount)):
                if OffScTrans[interactInd] == 0:
                    EnNew += self.Interaction2En[interactInd]

            delE = EnNew - EnState

            print("\n{:.4f} {:.4f}".format(delE, delEJumps[jumpInd]))

            self.assertAlmostEqual(delE, delEJumps[jumpInd])

    def test_state_updating(self):
        state = self.initState.copy()
        OffSiteCount = self.KMC_Jit.GetOffSite(state)
        offscCopy = OffSiteCount.copy()

        Nsites = self.VclusExp.Nsites
        N_units = self.VclusExp.sup.superlatt[0, 0]

        # do some random swaps and check if updates are being done correctly
        for i in range(10):
            siteA = np.random.randint(0, Nsites)
            siteB = np.random.randint(0, Nsites)

            # produce a new state by swapping the two

            stateNew = state.copy()
            stateNew[siteA] = state[siteB]
            stateNew[siteB] = state[siteA]

            offscnew = self.KMC_Jit.GetOffSite(stateNew)

            self.KMC_Jit.updateState(state, OffSiteCount, siteA, siteB)

            self.assertTrue(np.array_equal(state, stateNew))
            self.assertTrue(np.array_equal(offscnew, OffSiteCount))

class test_shells(Test_MC_Arrays):

    def test_ShellBuild(self):
        state = self.initState.copy()
        offsc = self.KMC_Jit.GetOffSite(state)
        TSoffsc = self.KMC_Jit.GetTSOffSite(state)
        beta = 1.0
        ijList = self.VclusExp.KRAexpander.ijList
        dxList = self.VclusExp.KRAexpander.dxList
        Nsites = len(state)
        Nspec = len(self.VclusExp.mobCountList)

        state2Index, Index2State, TransitionRates, TransitionsZero, velocities = MC_JIT.makeShells(self.MCSampler_Jit, self.KMC_Jit, state, offsc,
                                                                                                   TSoffsc, ijList, dxList, beta, Nsites, Nspec,
                                                                                                   Nshells=2)

        # First verify the starting state

        state0Ind = state2Index[state.tobytes()]
        self.assertEqual(state0Ind, 0)
        self.assertEqual(len(TransitionsZero), len(ijList)+1, msg="\n{}".format(TransitionsZero))

        vel0 = velocities[0]
        exitstates, exitRates, Specdisps = self.MCSampler_Jit.getExitData(state, ijList, dxList, offsc, TSoffsc, beta, Nsites)
        vel_test = np.zeros(3)
        for jmpInd in range(ijList.shape[0]):
            vel_test += exitRates[jmpInd]*dxList[jmpInd]

        self.assertTrue(np.allclose(vel0, vel_test))

        # First, we test the velocities out of a state
        for stateInd, vel in velocities.items():
            state1 = Index2State[stateInd]
            offsc1 = self.KMC_Jit.GetOffSite(state1)
            TSoffsc1 = self.KMC_Jit.GetTSOffSite(state1)

            self.assertTrue(state1[self.vacSiteInd] == Nspec - 1)
            if stateInd == 0:
                self.assertTrue(np.array_equal(state1, state))
                self.assertEqual(state1.tobytes(), state.tobytes())
                self.assertTrue(np.array_equal(offsc1, offsc))
                self.assertTrue(np.array_equal(TSoffsc1, TSoffsc))

            exitstates, exitRates, Specdisps = self.MCSampler_Jit.getExitData(state1, ijList, dxList, offsc1, TSoffsc1, beta, Nsites)

            vel_test = np.zeros(3)
            for jmpInd in range(ijList.shape[0]):
                self.assertTrue(np.allclose(Specdisps[jmpInd, -1, :], dxList[jmpInd, :]))
                # check that the correct species has been exchanged.
                vel_test += exitRates[jmpInd]*dxList[jmpInd]

            self.assertTrue(np.allclose(vel, vel_test), msg="\n{}\n{}\n{}\n{}".format(stateInd, vel, vel_test, vel0))

        # Now test the transitions
        exitcounts = collections.defaultdict(int)
        for (state1Ind, state2Ind), (rate, jumpInd) in TransitionRates.items():
            state1 = Index2State[state1Ind]
            exitcounts[state1Ind] += 1

            state2 = Index2State[state2Ind]

            offsc1 = self.KMC_Jit.GetOffSite(state1)
            TSoffsc1 = self.KMC_Jit.GetTSOffSite(state1)

            exitstates, exitRates, Specdisps = self.MCSampler_Jit.getExitData(state1, ijList, dxList, offsc1, TSoffsc1, beta, Nsites)

            total = 0.
            for r in exitRates:
                total += r

            self.assertAlmostEqual(total, -TransitionRates[(state1Ind, state1Ind)][0])

            if not np.array_equal(state1, state2):
                offsc2 = self.KMC_Jit.GetOffSite(state2)

                self.assertAlmostEqual(exitRates[jumpInd], rate)

                # translate all exit state to the origin
                currentExitState = exitstates[jumpInd]

                origVacexit = self.KMC_Jit.TranslateState(currentExitState, self.vacSiteInd, ijList[jumpInd])
                offsc2Orig = self.KMC_Jit.GetOffSite(origVacexit)
                TSoffsc2Orig = self.KMC_Jit.GetTSOffSite(origVacexit)

                self.assertTrue(np.array_equal(origVacexit, state2))

                # let's check the energies explicitly

                # First, get the KRA energy change
                KRA = self.KMC_Jit.getKRAEnergies(state1, TSoffsc1, ijList)
                delEKra = KRA[jumpInd]

                En1 = 0.
                for Interaction in range(offsc1.shape[0]):
                    if offsc1[Interaction] == 0:
                        En1 += self.Interaction2En[Interaction]

                En2 = 0.
                for Interaction in range(offsc2.shape[0]):
                    if offsc2[Interaction] == 0:
                        En2 += self.Interaction2En[Interaction]

                En22 = 0.
                for Interaction in range(offsc2Orig.shape[0]):
                    if offsc2Orig[Interaction] == 0:
                        En22 += self.Interaction2En[Interaction]

                self.assertAlmostEqual(En22, En2)


                delE = -En1 + En2

                rateCheck = np.exp(-beta*(0.5*delE + delEKra))
                self.assertAlmostEqual(rateCheck, rate)

                # Check detailed balance

                # get exit rates out of state2
                exitstates2, exitRates2, Specdisps2 = self.MCSampler_Jit.getExitData(origVacexit, ijList, dxList, offsc2Orig,
                                                                                  TSoffsc2Orig, beta, Nsites)

                initIdx = 0
                initCount = 0
                for i in range(ijList.shape[0]):
                    stex2 = exitstates2[i]
                    stex2Orig = self.KMC_Jit.TranslateState(stex2, self.vacSiteInd, ijList[i])
                    if np.array_equal(stex2Orig, state1):
                        initIdx = i
                        initCount += 1

                self.assertEqual(initCount, 1)
                rateRev = exitRates2[initIdx]

                self.assertTrue(np.allclose(np.exp(-(En2 - En1)), rate/rateRev))


        # Check that all jumps have been accounted for including diagonal elements
        for key, item in exitcounts.items():
            self.assertEqual(item, ijList.shape[0]+1)
