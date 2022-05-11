from onsager import crystal, supercell, cluster
import numpy as np
import itertools
# import Transitions
from KRA3Body import KRA3bodyInteractions
import Cluster_Expansion
import MC_JIT
import unittest
import time
import warnings
import collections

warnings.filterwarnings('error', category=RuntimeWarning)

np.seterr(all='raise')

__FCC__ = True

class Test_Make_Arrays(unittest.TestCase):

    def setUp(self):
        self.NSpec = 3
        self.KRACounterSpec = 1
        self.Nvac = 1
        self.MaxOrder = 3
        self.MaxOrderTrans = 3
        a0 = 1
        self.a0 = a0
        self.crys = crystal.Crystal.BCC(a0, chemistry="A")
        jumpCutoff = 1.01 * np.sqrt(3) * a0 / 2
        self.jnetBCC = self.crys.jumpnetwork(0, jumpCutoff)
        self.N_units = 8
        self.superlatt = self.N_units * np.eye(3, dtype=int)
        self.superBCC = supercell.ClusterSupercell(self.crys, self.superlatt)
        # get the number of sites in the supercell - should be 8x8x8
        numSites = len(self.superBCC.mobilepos)
        self.vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
        self.vacsiteInd = self.superBCC.index(np.zeros(3, dtype=int), (0, 0))[0]
        self.clusexp = cluster.makeclusters(self.crys, 1.01 * a0, self.MaxOrder)

        TScombShellRange = 1  # upto 1nn combined shell
        TSnnRange = 4
        TScutoff = np.sqrt(3) * a0  # 5th nn cutoff

        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(
            self.superBCC, self.clusexp, self.NSpec, self.vacsite, self.MaxOrder,
                TScutoff=TScutoff, TScombShellRange=TScombShellRange,
                TSnnRange=TSnnRange, jumpnetwork=self.jnetBCC, TclusExp=True
        )

        self.TScombShellRange = TScombShellRange
        self.TSnnRange = TSnnRange
        self.TScutoff = TScutoff

        self.VclusExp.generateSiteSpecInteracts()
        self.VclusExp.genVecClustBasis(self.VclusExp.SpecClusters)
        self.VclusExp.indexVclus2Clus()  # Index vector cluster list to cluster symmetry groups
        self.VclusExp.indexClustertoVecClus()

        self.Energies = np.random.rand(len(self.VclusExp.SpecClusters))
        self.KRAEnergies = [np.random.rand(len(TSptGroups))
                            for (key, TSptGroups) in self.VclusExp.KRAexpander.clusterSpeciesJumps.items()]

        self.KRASpecConstants = np.random.rand(self.NSpec - 1)

        self.MakeJITs()
        print("Done setting up BCC data")

    def MakeJITs(self):
        # First, the chemical data
        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, \
        numInteractsSiteSpec, SiteSpecInterArray = self.VclusExp.makeJitInteractionsData(self.Energies)

        self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites, self.Interaction2En, \
        self.numInteractsSiteSpec, self.SiteSpecInterArray= \
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, \
            numInteractsSiteSpec, SiteSpecInterArray

        # Next, the vector basis data
        self.numVecsInteracts, self.VecsInteracts, self.VecGroupInteracts = self.VclusExp.makeJitVectorBasisData()

        TsInteractIndexDict, Index2TSinteractDict, numSitesTSInteracts, TSInteractSites, TSInteractSpecs, \
        jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, \
        JumpInteracts, Jump2KRAEng = \
            self.VclusExp.KRAexpander.makeTransJitData(self.KRACounterSpec, self.KRAEnergies)

        self.TsInteractIndexDict, self.Index2TSinteractDict, self.numSitesTSInteracts, self.TSInteractSites, self.TSInteractSpecs, \
        self.jumpFinSites, self.jumpFinSpec, self.FinSiteFinSpecJumpInd, self.numJumpPointGroups, self.numTSInteractsInPtGroups, \
        self.JumpInteracts, self.Jump2KRAEng = \
            TsInteractIndexDict, Index2TSinteractDict, numSitesTSInteracts, TSInteractSites, TSInteractSpecs, \
            jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, \
            JumpInteracts, Jump2KRAEng

        Nsites = self.VclusExp.Nsites
        N_units = self.VclusExp.sup.superlatt[0, 0]

        siteIndtoR, RtoSiteInd = self.VclusExp.makeSiteIndToSite()
        self.RtoSiteInd = RtoSiteInd
        self.siteIndtoR = siteIndtoR
        self.Nsites = Nsites
        self.N_units = N_units

        self.KMC_Jit = MC_JIT.KMC_JIT(numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En,
                                      numInteractsSiteSpec, SiteSpecInterArray, numSitesTSInteracts, TSInteractSites,
                                      TSInteractSpecs, jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd,
                                      numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng, self.KRASpecConstants,
                                      siteIndtoR, RtoSiteInd, N_units)

        initState = np.zeros(len(self.VclusExp.sup.mobilepos), dtype=int)
        # Now assign random species (excluding the vacancy)
        for i in range(len(self.VclusExp.sup.mobilepos)):
            initState[i] = np.random.randint(0, self.NSpec - 1)

        # Now put in the vacancy at the vacancy site
        initState[self.vacsiteInd] = self.NSpec - 1

        self.initState = initState

        self.MCSampler_Jit = MC_JIT.MCSamplerClass(
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numInteractsSiteSpec, SiteSpecInterArray,
            numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
            FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng, self.KRASpecConstants
        )

    def make3bodyKRAEnergies(self):
        KRASpecConstants = np.random.rand(self.NSpec-1)
        En0Jumps = np.random.rand(self.TSnnRange)
        En1Jumps = np.random.rand(self.TSnnRange)
        Energies = []
        for jumpInd in range(len(self.VclusExp.KRAexpander.jump2Index)):
            jumpkey = self.VclusExp.KRAexpander.Index2Jump[jumpInd]
            jumpSpec = jumpkey[2]
            if jumpSpec == 0:
                Energies.append(En0Jumps.copy())
            else:
                Energies.append(En1Jumps.copy())
        return KRASpecConstants, Energies

class Test_Make_Arrays_FCC(Test_Make_Arrays):

    def setUp(self):
        self.NSpec = 3
        self.KRACounterSpec = 1
        self.Nvac = 1
        self.MaxOrder = 3
        self.MaxOrderTrans = 3
        a0 = 1.0
        self.a0 = a0
        self.crys = crystal.Crystal.FCC(a0, chemistry="A")
        self.chem = 0
        self.jnetFCC = self.crys.jumpnetwork(0, 1.01 * a0 / np.sqrt(2))
        self.superlatt = 8 * np.eye(3, dtype=int)
        self.superFCC = supercell.ClusterSupercell(self.crys, self.superlatt)
        # get the number of sites in the supercell - should be 8x8x8
        numSites = len(self.superFCC.mobilepos)
        self.vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
        self.vacsiteInd = self.superFCC.index(np.zeros(3, dtype=int), (0, 0))[0]
        self.clusexp = cluster.makeclusters(self.crys, 1.01 * a0, self.MaxOrder)

        TScombShellRange = 1  # upto 1nn combined shell
        TSnnRange = 4
        TScutoff = np.sqrt(2) * a0  # 4th nn cutoff

        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(self.superFCC, self.clusexp, self.NSpec, self.vacsite,
                                                                 self.MaxOrder, TScutoff, TScombShellRange, TSnnRange,
                                                                 self.jnetFCC, TclusExp=True)

        self.TScombShellRange = TScombShellRange
        self.TSnnRange = TSnnRange
        self.TScutoff = TScutoff

        self.VclusExp.generateSiteSpecInteracts()

        self.VclusExp.genVecClustBasis(self.VclusExp.SpecClusters)
        self.VclusExp.indexVclus2Clus()  # Index vector cluster list to cluster symmetry groups
        self.VclusExp.indexClustertoVecClus()

        self.Energies = np.random.rand(len(self.VclusExp.SpecClusters))

        self.KRASpecConstants, self.KRAEnergies = self.make3bodyKRAEnergies()

        self.MakeJITs()
        print("Done setting up FCC data.")

DataClass = Test_Make_Arrays
if __FCC__:
    DataClass = Test_Make_Arrays_FCC

class Test_MC(DataClass):

    def test_MC_step(self):
        # First, create a random state
        initState = self.initState

        # Now put in the vacancy at the vacancy site
        initCopy = initState.copy()

        # Get the MC samplers
        MCSampler_Jit = self.MCSampler_Jit

        Nswaptrials = 1  # We are only testing a single step here
        # randLogarr = np.log(np.random.rand(Nswaptrials))

        # If we want to ensure acceptance, we keep it really small
        # randLogarr = np.ones(Nswaptrials)*-1000.0

        # If we want to ensure rejection, we keep it really big
        randLogarr = np.ones(Nswaptrials) * 1000.0

        # Put in tests for Jit calculations
        # make the offsite counts
        initJit = initCopy.copy()

        # get the Number of species of each type
        N_nonvacSpec = np.zeros(self.NSpec-1, dtype=int)
        Nsites = len(self.VclusExp.sup.mobilepos)
        assert Nsites == initJit.shape[0]
        for siteInd in range(Nsites):
            if siteInd == self.vacsiteInd:
                assert initJit[siteInd] == self.NSpec-1
                continue
            spec = initJit[siteInd]
            N_nonvacSpec[spec] += 1
        # print("In test:{}".format(N_nonvacSpec))
        specLocations = np.full((N_nonvacSpec.shape[0], np.max(N_nonvacSpec)), -1, dtype=int)
        spec_counter = np.zeros_like(N_nonvacSpec, dtype=int)
        for siteInd in range(Nsites):
            if siteInd == self.vacsiteInd:
                continue
            spec = initJit[siteInd]
            specLocations[spec, spec_counter[spec]] = siteInd
            spec_counter[spec] += 1

        # build the offsite count and get initial energy
        En1 = 0.
        offscjit = np.zeros_like(self.numSitesInteracts)
        for interactIdx in range(self.numSitesInteracts.shape[0]):
            numSites = self.numSitesInteracts[interactIdx]
            offcount = 0
            for intSiteind in range(numSites):
                interSite = self.SupSitesInteracts[interactIdx, intSiteind]
                interSpec = self.SpecOnInteractSites[interactIdx, intSiteind]
                if initJit[interSite] != interSpec:
                    offcount += 1
            offscjit[interactIdx] = offcount
            if offcount == 0:
                En1 += self.Interaction2En[interactIdx]

        offscCalc = MC_JIT.GetOffSite(initJit, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)

        self.assertTrue(np.array_equal(offscjit, offscCalc))

        TSOffSiteCount = np.full(len(self.numSitesTSInteracts), -1, dtype=int)
        beta = 1.0

        # args in order:
        # state, N_nonVacSpecs, OffSiteCount, TransOffSiteCount, beta, randLogarr, Nswaptrials, vacSiteInd
        print("Starting sweep")
        start = time.time()
        SpecsOnSites, acceptCount =\
            MCSampler_Jit.makeMCsweep(initJit, N_nonvacSpec, offscjit, TSOffSiteCount,
                                      beta, randLogarr, Nswaptrials, self.vacsiteInd)

        print("Sweep completed in :{} seconds".format(time.time() - start))

        # Get the energy for the swapped state - will determine if move is calculated correctly or not
        # swap the occupancies
        offscCalc = MC_JIT.GetOffSite(initJit, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)
        self.assertTrue(np.array_equal(offscjit, offscCalc))

        stateSwap = initJit.copy()
        EnSwap = 0.
        # Now get its energy
        for interactIdx in range(self.numSitesInteracts.shape[0]):
            numSites = self.numSitesInteracts[interactIdx]
            offcount = 0
            for intSiteind in range(numSites):
                interSite = self.SupSitesInteracts[interactIdx, intSiteind]
                interSpec = self.SpecOnInteractSites[interactIdx, intSiteind]
                if stateSwap[interSite] != interSpec:
                    offcount += 1
            self.assertEqual(offscjit[interactIdx], offcount)
            if offcount == 0:
                EnSwap += self.Interaction2En[interactIdx]

        # test that site swaps are done correctly.

        # if the move was selected
        if -beta*MCSampler_Jit.delEArray[0] > randLogarr[0]:
            print("move was accepted {} {}".format(-1.0*MCSampler_Jit.delEArray[0], randLogarr[0]))
            self.assertFalse(np.array_equal(initJit, initCopy))

            # Check that the swapping was done properly
            initCopycopy = initCopy.copy()
            A, B = np.where(initJit - initCopy != 0)[0]
            temp = initCopycopy[A]
            initCopycopy[A] = initCopycopy[B]
            initCopycopy[B] = temp
            self.assertTrue(np.array_equal(initJit, initCopycopy))

            # Check the update to the new site locations
            spLPrev = specLocations.copy()
            specLocations = np.full((N_nonvacSpec.shape[0], np.max(N_nonvacSpec)), -1, dtype=int)
            spec_counter = np.zeros_like(N_nonvacSpec, dtype=int)
            self.assertEqual(Nsites, initJit.shape[0])
            for siteInd in range(Nsites):
                if siteInd == self.vacsiteInd:
                    self.assertEqual(initJit[siteInd], self.NSpec-1)
                    continue
                spec = initJit[siteInd]
                specLocations[spec, spec_counter[spec]] = siteInd
                spec_counter[spec] += 1

            self.assertTrue(np.array_equal(spec_counter, N_nonvacSpec))
            # check that for same species, we have the same site locations in the explcit update
            # and also by just exchanging locations as in the code.
            for NVspec in range(N_nonvacSpec.shape[0]):
                specSitesExchange = np.sort(specLocations[NVspec, :N_nonvacSpec[NVspec]])
                specSitesUpdated = np.sort(SpecsOnSites[NVspec, :N_nonvacSpec[NVspec]])
                self.assertTrue(np.array_equal(specSitesUpdated, specSitesExchange))
            self.assertEqual(specLocations.shape, SpecsOnSites.shape)

            self.assertAlmostEqual(EnSwap - En1, MCSampler_Jit.delEArray[0], msg="{}".format(MCSampler_Jit.delEArray[0]))
            # Check that TS offsite counts were constructed correctly.
            for TsInteractIdx in range(len(TSOffSiteCount)):
                offcount = 0
                for Siteind in range(self.numSitesTSInteracts[TsInteractIdx]):
                    if initJit[self.TSInteractSites[TsInteractIdx, Siteind]] != self.TSInteractSpecs[TsInteractIdx, Siteind]:
                        offcount += 1
                self.assertEqual(TSOffSiteCount[TsInteractIdx], offcount)

            self.assertEqual(acceptCount, 1)

            # check that offsite counts were properly updated
            offscCalc = MC_JIT.GetOffSite(initJit, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)
            self.assertTrue(np.array_equal(offscjit, offscCalc))

        else:
            self.assertTrue(np.array_equal(initJit, initCopy))
            self.assertAlmostEqual(EnSwap, En1)
            print("move was not accepted {} {}".format(-1.0*MCSampler_Jit.delEArray[0], randLogarr[0]))
            self.assertTrue(np.array_equal(specLocations, SpecsOnSites))
            # check that offsite counts were properly reverted
            offscCalc = MC_JIT.GetOffSite(initJit, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)
            self.assertTrue(np.array_equal(offscjit, offscCalc))

        print("MC sweep tests done.")

        # Check the energies by translating the clusters around
        allSpCl = [SpCl for SpClList in self.VclusExp.SpecClusters for SpCl in SpClList]
        # First, the initial state
        EnInit = 0.
        # translate every cluster by all lattice translations and get the corresponding energy
        for SpCl in allSpCl:
            EnCl = self.Energies[self.VclusExp.clust2SpecClus[SpCl][0]]  # get the energy of the group this cluster belongs to
            siteSpecList = list(SpCl.SiteSpecs)
            siteList = [site for (site, spec) in siteSpecList]
            specList = [spec for (site, spec) in siteSpecList]
            for transInd in range(self.VclusExp.Nsites):
                # Get the off counts for this interactions
                offcount = 0
                R = self.VclusExp.sup.ciR(transInd)[1]  # get the translation
                # translate all sites in the cluster by that amount
                for clSiteInd, clSite in enumerate(siteList):
                    clSiteNew = clSite + R
                    supSiteNew = self.VclusExp.sup.index(clSiteNew.R, clSiteNew.ci)[0]
                    # Check if the required species is there
                    if initCopy[supSiteNew] != specList[clSiteInd]:
                        offcount += 1
                if offcount == 0:
                    EnInit += EnCl

        self.assertAlmostEqual(EnInit, En1)

        # Now do the same for the swapped state
        EnSwap = 0.
        # translate every cluster by all lattice translations and get the corresponding energy
        for SpCl in allSpCl:
            EnCl = self.Energies[self.VclusExp.clust2SpecClus[SpCl][0]]  # get the energy of the group this cluster belongs to
            siteSpecList = list(SpCl.SiteSpecs)
            siteList = [site for (site, spec) in siteSpecList]
            specList = [spec for (site, spec) in siteSpecList]
            for transInd in range(self.VclusExp.Nsites):
                # Get the off counts for this interactions
                offcount = 0
                R = self.VclusExp.sup.ciR(transInd)[1]  # get the translation
                # translate all sites in the cluster by that amount
                for clSiteInd, clSite in enumerate(siteList):
                    clSiteNew = clSite + R
                    supSiteNew = self.VclusExp.sup.index(clSiteNew.R, clSiteNew.ci)[0]
                    # Check if the required species is there
                    if stateSwap[supSiteNew] != specList[clSiteInd]:
                        offcount += 1
                if offcount == 0:
                    EnSwap += EnCl

        if -beta*MCSampler_Jit.delEArray[0] > randLogarr[0]:
            self.assertAlmostEqual(EnSwap, En1+MCSampler_Jit.delEArray[0])
        else:
            self.assertAlmostEqual(EnSwap, En1)

        print("Explicit energy tests done.")

    def test_multiple_swaps(self):

        # First, create a random state
        initState = self.initState

        # Now put in the vacancy at the vacancy site
        initCopy = initState.copy()

        # Get the MC samplers
        MCSampler_Jit = self.MCSampler_Jit

        Nswaptrials = 1000
        randLogarr = np.log(np.random.rand(Nswaptrials))

        # Put in tests for Jit calculations
        # make the offsite counts
        initJit = initCopy.copy()

        # get the Number of species of each type
        N_nonvacSpec = np.zeros(self.NSpec - 1, dtype=int)
        Nsites = len(self.VclusExp.sup.mobilepos)
        assert Nsites == initJit.shape[0]
        for siteInd in range(Nsites):
            if siteInd == self.vacsiteInd:
                assert initJit[siteInd] == self.NSpec - 1
                continue
            spec = initJit[siteInd]
            N_nonvacSpec[spec] += 1
        # print("In test:{}".format(N_nonvacSpec))
        specLocations = np.full((N_nonvacSpec.shape[0], np.max(N_nonvacSpec)), -1, dtype=int)
        spec_counter = np.zeros_like(N_nonvacSpec, dtype=int)
        for siteInd in range(Nsites):
            if siteInd == self.vacsiteInd:
                continue
            spec = initJit[siteInd]
            specLocations[spec, spec_counter[spec]] = siteInd
            spec_counter[spec] += 1

        # build the offsite count and get initial energy
        En1 = 0.
        offscjit = np.zeros_like(self.numSitesInteracts)
        for interactIdx in range(self.numSitesInteracts.shape[0]):
            numSites = self.numSitesInteracts[interactIdx]
            offcount = 0
            for intSiteind in range(numSites):
                interSite = self.SupSitesInteracts[interactIdx, intSiteind]
                interSpec = self.SpecOnInteractSites[interactIdx, intSiteind]
                if initJit[interSite] != interSpec:
                    offcount += 1
            offscjit[interactIdx] = offcount
            if offcount == 0:
                En1 += self.Interaction2En[interactIdx]

        offscCalc = MC_JIT.GetOffSite(initJit, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)

        self.assertTrue(np.array_equal(offscjit, offscCalc))

        TSOffSiteCount = np.full(len(self.numSitesTSInteracts), -1, dtype=int)
        beta = 1.0

        # args in order:
        # state, N_nonVacSpecs, OffSiteCount, TransOffSiteCount, beta, randLogarr, Nswaptrials, vacSiteInd
        print("Starting sweep")
        start = time.time()
        SpecsOnSites, acceptCount = \
            MCSampler_Jit.makeMCsweep(initJit, N_nonvacSpec, offscjit, TSOffSiteCount,
                                      beta, randLogarr, Nswaptrials, self.vacsiteInd)

        print("Sweep completed in :{} seconds".format(time.time() - start))

        # Get the energy for the swapped state - will determine if move is calculated correctly or not
        # swap the occupancies
        offscCalc = MC_JIT.GetOffSite(initJit, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)
        self.assertTrue(np.array_equal(offscjit, offscCalc))

        stateSwap = initJit.copy()
        EnSwap = 0.
        # Now get its energy
        for interactIdx in range(self.numSitesInteracts.shape[0]):
            numSites = self.numSitesInteracts[interactIdx]
            offcount = 0
            for intSiteind in range(numSites):
                interSite = self.SupSitesInteracts[interactIdx, intSiteind]
                interSpec = self.SpecOnInteractSites[interactIdx, intSiteind]
                if stateSwap[interSite] != interSpec:
                    offcount += 1
            self.assertEqual(offscjit[interactIdx], offcount)
            if offcount == 0:
                EnSwap += self.Interaction2En[interactIdx]

        print(MCSampler_Jit.delETotal, acceptCount)
        self.assertAlmostEqual(EnSwap - En1, MCSampler_Jit.delETotal)

        # Next, check the updated species locations
        specLocations = np.full((N_nonvacSpec.shape[0], np.max(N_nonvacSpec)), -1, dtype=int)
        spec_counter = np.zeros_like(N_nonvacSpec, dtype=int)
        self.assertEqual(Nsites, initJit.shape[0])
        for siteInd in range(Nsites):
            if siteInd == self.vacsiteInd:
                self.assertEqual(initJit[siteInd], self.NSpec - 1)
                continue
            spec = initJit[siteInd]
            specLocations[spec, spec_counter[spec]] = siteInd
            spec_counter[spec] += 1

        self.assertTrue(np.array_equal(spec_counter, N_nonvacSpec))
        # check that for same species, we have the same site locations in the explicit update
        # and also by just exchanging locations as in the code.
        for NVspec in range(N_nonvacSpec.shape[0]):
            specSitesExchange = np.sort(specLocations[NVspec, :N_nonvacSpec[NVspec]])
            specSitesUpdated = np.sort(SpecsOnSites[NVspec, :N_nonvacSpec[NVspec]])
            self.assertTrue(np.array_equal(specSitesUpdated, specSitesExchange))
        self.assertEqual(specLocations.shape, SpecsOnSites.shape)

        # check energies by explicit cluster translation
        allSpCl = [SpCl for SpClList in self.VclusExp.SpecClusters for SpCl in SpClList]
        # First, the initial state
        EnInit = 0.
        # translate every cluster by all lattice translations and get the corresponding energy
        for SpCl in allSpCl:
            EnCl = self.Energies[self.VclusExp.clust2SpecClus[SpCl][0]]  # get the energy of the group this cluster belongs to
            siteSpecList = list(SpCl.SiteSpecs)
            siteList = [site for (site, spec) in siteSpecList]
            specList = [spec for (site, spec) in siteSpecList]
            for transInd in range(self.VclusExp.Nsites):
                # Get the off counts for this interactions
                offcount = 0
                R = self.VclusExp.sup.ciR(transInd)[1]  # get the translation
                # translate all sites in the cluster by that amount
                for clSiteInd, clSite in enumerate(siteList):
                    clSiteNew = clSite + R
                    supSiteNew = self.VclusExp.sup.index(clSiteNew.R, clSiteNew.ci)[0]
                    # Check if the required species is there
                    if initCopy[supSiteNew] != specList[clSiteInd]:
                        offcount += 1
                if offcount == 0:
                    EnInit += EnCl

        self.assertAlmostEqual(EnInit, En1)

        # Now do the same for the swapped state
        EnFin = 0.
        # translate every cluster by all lattice translations and get the corresponding energy
        for SpCl in allSpCl:
            EnCl = self.Energies[self.VclusExp.clust2SpecClus[SpCl][0]]  # get the energy of the group this cluster belongs to
            siteSpecList = list(SpCl.SiteSpecs)
            siteList = [site for (site, spec) in siteSpecList]
            specList = [spec for (site, spec) in siteSpecList]
            for transInd in range(self.VclusExp.Nsites):
                # Get the off counts for this interactions
                offcount = 0
                R = self.VclusExp.sup.ciR(transInd)[1]  # get the translation
                # translate all sites in the cluster by that amount
                for clSiteInd, clSite in enumerate(siteList):
                    clSiteNew = clSite + R
                    supSiteNew = self.VclusExp.sup.index(clSiteNew.R, clSiteNew.ci)[0]
                    # Check if the required species is there
                    if stateSwap[supSiteNew] != specList[clSiteInd]:
                        offcount += 1
                if offcount == 0:
                    EnFin += EnCl

        self.assertAlmostEqual(EnFin, EnInit + MCSampler_Jit.delETotal)
        self.assertAlmostEqual(EnFin, EnSwap)

    def test_expansion(self):
        """
        To test if Wbar and Bbar are computed correctly
        """
        # Compute them with the dicts and match with what comes out from the code.
        # Wbar_test = np.zeros()

        initState = self.initState

        MCSampler_Jit = self.MCSampler_Jit

        TransOffSiteCount = np.zeros(len(self.TSInteractSites), dtype=int)

        # Build TS offsites
        for TsInteractIdx in range(len(TransOffSiteCount)):
            for Siteind in range(self.numSitesTSInteracts[TsInteractIdx]):
                if initState[self.TSInteractSites[TsInteractIdx, Siteind]] != self.TSInteractSpecs[TsInteractIdx, Siteind]:
                    TransOffSiteCount[TsInteractIdx] += 1

        ijList = np.array(self.VclusExp.KRAexpander.jList)
        dxList = np.array(self.VclusExp.KRAexpander.dxList)
        lenVecClus = len(self.VclusExp.vecClus)

        # Now, do the expansion
        offscjit = MC_JIT.GetOffSite(initState, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)
        state = initState.copy()
        spec = 2
        beta = 1.0
        Wbar, Bbar, _ = MCSampler_Jit.Expand(state, ijList, dxList, spec, offscjit, TransOffSiteCount,
               self.numVecsInteracts, self.VecGroupInteracts, self.VecsInteracts,
               lenVecClus, beta, self.vacsiteInd, None)

        offscjit2 = MC_JIT.GetOffSite(initState, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)

        # Check that the offsitecounts have been correctly reverted and state is unchanged.

        self.assertTrue(np.array_equal(offscjit2, offscjit))
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
                    TSOffCount = TransOffSiteCount.copy()

                    delEKRA = 0.0
                    vec1 = np.zeros(3, dtype=float)
                    vec2 = np.zeros(3, dtype=float)

                    siteB = ijList[TInd]
                    siteA = self.vacsiteInd
                    # Check that the initial site is always the vacancy
                    specA = state[siteA]  # the vacancy
                    self.assertEqual(specA, self.NSpec - 1)
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

                    for interactnum in range(self.numInteractsSiteSpec[siteA, specA]):
                        interactInd = self.SiteSpecInterArray[siteA, specA, interactnum]
                        repClusInd = self.VclusExp.InteractionId2ClusId[interactInd]
                        repClus = self.VclusExp.Num2Clus[repClusInd]
                        repClustSymListInd = self.VclusExp.clust2SpecClus[repClus][0]
                        if self.VclusExp.clus2LenVecClus[repClustSymListInd] == 0:
                            self.assertEqual(self.numVecsInteracts[interactInd], -1)
                            continue
                        vecList = self.VclusExp.clust2vecClus[repClus]
                        self.assertEqual(self.numVecsInteracts[interactInd], len(vecList))

                        for i in range(self.numVecsInteracts[interactInd]):
                            self.assertTrue(np.allclose(self.VecsInteracts[interactInd, i, :],
                                                        self.VclusExp.vecVec[vecList[i][0]][vecList[i][1]]))
                            self.assertEqual(self.VecGroupInteracts[interactInd, i], vecList[i][0])

                        if offscjit[interactInd] == 0:
                            delE -= self.Interaction2En[interactInd]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs1:
                                    vec1 -= self.VecsInteracts[interactInd, tupInd, :]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs2:
                                    vec2 -= self.VecsInteracts[interactInd, tupInd, :]
                        offscjit[interactInd] += 1

                    for interactnum in range(self.numInteractsSiteSpec[siteB, specB]):
                        interactInd = self.SiteSpecInterArray[siteB, specB, interactnum]
                        repClusInd = self.VclusExp.InteractionId2ClusId[interactInd]
                        repClus = self.VclusExp.Num2Clus[repClusInd]
                        repClustSymListInd = self.VclusExp.clust2SpecClus[repClus][0]
                        if self.VclusExp.clus2LenVecClus[repClustSymListInd] == 0:
                            self.assertEqual(self.numVecsInteracts[interactInd], -1)
                            continue
                        vecList = self.VclusExp.clust2vecClus[repClus]
                        self.assertEqual(self.numVecsInteracts[interactInd], len(vecList))

                        for i in range(self.numVecsInteracts[interactInd]):
                            self.assertTrue(np.allclose(self.VecsInteracts[interactInd, i, :],
                                                        self.VclusExp.vecVec[vecList[i][0]][vecList[i][1]]))
                            self.assertEqual(self.VecGroupInteracts[interactInd, i], vecList[i][0])

                        if offscjit[interactInd] == 0:
                            delE -= self.Interaction2En[interactInd]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs1:
                                    vec1 -= self.VecsInteracts[interactInd, tupInd, :]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs2:
                                    vec2 -= self.VecsInteracts[interactInd, tupInd, :]
                        offscjit[interactInd] += 1

                    for interactnum in range(self.numInteractsSiteSpec[siteA, specB]):
                        interactInd = self.SiteSpecInterArray[siteA, specB, interactnum]
                        repClusInd = self.VclusExp.InteractionId2ClusId[interactInd]
                        repClus = self.VclusExp.Num2Clus[repClusInd]
                        repClustSymListInd = self.VclusExp.clust2SpecClus[repClus][0]
                        if self.VclusExp.clus2LenVecClus[repClustSymListInd] == 0:
                            self.assertEqual(self.numVecsInteracts[interactInd], -1)
                            continue
                        vecList = self.VclusExp.clust2vecClus[repClus]
                        self.assertEqual(self.numVecsInteracts[interactInd], len(vecList))

                        for i in range(self.numVecsInteracts[interactInd]):
                            self.assertTrue(np.allclose(self.VecsInteracts[interactInd, i, :],
                                                        self.VclusExp.vecVec[vecList[i][0]][vecList[i][1]]))
                            self.assertEqual(self.VecGroupInteracts[interactInd, i], vecList[i][0])

                        offscjit[interactInd] -= 1
                        if offscjit[interactInd] == 0:
                            delE += self.Interaction2En[interactInd]
                            vecList = self.VclusExp.clust2vecClus[repClus]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs1:
                                    vec1 += self.VecsInteracts[interactInd, tupInd, :]
                            for tupInd, tup in enumerate(vecList):
                                if tup[0] == vs2:
                                    vec2 += self.VecsInteracts[interactInd, tupInd, :]

                    for interactnum in range(self.numInteractsSiteSpec[siteB, specA]):
                        interactInd = self.SiteSpecInterArray[siteB, specA, interactnum]
                        repClusInd = self.VclusExp.InteractionId2ClusId[interactInd]
                        repClus = self.VclusExp.Num2Clus[repClusInd]
                        repClustSymListInd = self.VclusExp.clust2SpecClus[repClus][0]
                        if self.VclusExp.clus2LenVecClus[repClustSymListInd] == 0:
                            self.assertEqual(self.numVecsInteracts[interactInd], -1)
                            continue
                        vecList = self.VclusExp.clust2vecClus[repClus]
                        self.assertEqual(self.numVecsInteracts[interactInd], len(vecList))

                        for i in range(self.numVecsInteracts[interactInd]):
                            self.assertTrue(np.allclose(self.VecsInteracts[interactInd, i, :],
                                                        self.VclusExp.vecVec[vecList[i][0]][vecList[i][1]]))
                            self.assertEqual(self.VecGroupInteracts[interactInd, i], vecList[i][0])

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

                    # get the delta lambda and check for the vector stars
                    jSite = ijList[TInd]

                    Wbar_test[vs1, vs2] += rate*dot
                    if vs1 == 0:
                        Bbar_test[vs2] += rate*np.dot(dxList[TInd], vec2)

                self.assertAlmostEqual(Wbar[vs1, vs2], Wbar_test[vs1, vs2], 8,
                                       msg="\n{} {}".format(Wbar[vs1, vs2], Wbar_test[vs1, vs2]))

        self.assertTrue(np.allclose(Bbar, Bbar_test))


class Test_KMC(DataClass):

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

        stateTrans = KMC_Jit.TranslateState(state, siteB, siteA)

        Rf = self.VclusExp.sup.ciR(siteB)[1]
        Ri = self.VclusExp.sup.ciR(siteA)[1]

        # Now, test the translated state
        # since monoatomic, ci = (0,0)
        ci = (0, 0)
        for site in range(Nsites):
            Rsite = self.VclusExp.sup.ciR(site)[1]

            self.assertTrue(np.array_equal(Rsite, siteIndtoR[site, :]))

            R2 = Rsite + Rf - Ri  # get the new location of the site
            siteTrans = self.VclusExp.sup.index(R2, ci)[0]
            R2_incell = self.VclusExp.sup.ciR(siteTrans)[1]

            RTrans2 = (siteIndtoR[site, :] + Rf - Ri) % N_units

            siteT2 = RtoSiteInd[RTrans2[0], RTrans2[1], RTrans2[2]]

            self.assertTrue(np.array_equal(RTrans2, R2_incell))
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
        OffSiteCount = MC_JIT.GetOffSite(state, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)
        for (interaction, interactionInd) in self.VclusExp.Interaction2IdDict.items():
            offsiteCount = 0
            for (site, spec) in interaction:
                if state[site] != spec:
                    offsiteCount += 1
            self.assertEqual(OffSiteCount[interactionInd], offsiteCount)

        # Next, the TS cluster off site counts
        TransOffSiteCount = MC_JIT.GetTSOffSite(state, self.numSitesTSInteracts, self.TSInteractSites, self.TSInteractSpecs)
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
        OffSiteCount = MC_JIT.GetOffSite(state, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)
        offscCopy = OffSiteCount.copy()
        N_units = self.VclusExp.sup.superlatt[0, 0]

        jmpFinSiteList = self.VclusExp.KRAexpander.jList

        # Calculate the energy of the initial state
        EnState = 0.
        for interactInd in range(len(OffSiteCount)):
            if OffSiteCount[interactInd] == 0:
                EnState += self.Interaction2En[interactInd]

        # collect energy changes due to the jumps
        delEJumps = self.KMC_Jit.getEnergyChangeJumps(state, OffSiteCount, self.vacsiteInd, np.array(jmpFinSiteList, dtype=int))

        # check that the state and the offsite counts have been left unchanged
        self.assertTrue(np.array_equal(OffSiteCount, offscCopy))
        self.assertTrue(np.array_equal(state, self.initState))

        # Now go through each of the transitions and evaluate the energy changes explicitly
        for jumpInd, siteInd in enumerate(jmpFinSiteList):
            stateNew = state.copy()
            stateNew[siteInd] = self.NSpec - 1  # this site will contain the vacancy in the new site
            stateNew[self.vacsiteInd] = state[siteInd]

            OffScTrans = MC_JIT.GetOffSite(stateNew, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)

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
        state[siteSwap] = state[self.vacsiteInd]
        state[self.vacsiteInd] = temp
        self.assertEqual(state[siteSwap], self.NSpec-1)
        statecpy = state.copy()

        OffSiteCount = MC_JIT.GetOffSite(state, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)
        offscCopy = OffSiteCount.copy()

        jmpFinSiteList = self.VclusExp.KRAexpander.jList
        if __FCC__:
            self.assertEqual(len(jmpFinSiteList), 12)

        jmpFinSiteListTrans = np.zeros_like(jmpFinSiteList)

        dR = self.siteIndtoR[siteSwap] - self.siteIndtoR[self.vacsiteInd]

        for jmp in range(len(jmpFinSiteList)):
            RfinSiteNew = (dR + self.siteIndtoR[jmpFinSiteList[jmp]]) % N_units
            jmpFinSiteListTrans[jmp] = self.RtoSiteInd[RfinSiteNew[0], RfinSiteNew[1], RfinSiteNew[2]]

        # Now get the energy changes during the jumps
        delEJumps = self.KMC_Jit.getEnergyChangeJumps(state, OffSiteCount, siteSwap, np.array(jmpFinSiteListTrans, dtype=int))

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

            OffScTrans = MC_JIT.GetOffSite(stateNew, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)

            EnNew = 0.
            for interactInd in range(len(OffSiteCount)):
                if OffScTrans[interactInd] == 0:
                    EnNew += self.Interaction2En[interactInd]

            delE = EnNew - EnState

            print("\n{:.4f} {:.4f}".format(delE, delEJumps[jumpInd]))

            self.assertAlmostEqual(delE, delEJumps[jumpInd])

    def test_state_updating(self):
        state = self.initState.copy()
        OffSiteCount = MC_JIT.GetOffSite(state, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)

        Nsites = self.VclusExp.Nsites

        # do some random swaps and check if updates are being done correctly
        for i in range(10):
            siteA = np.random.randint(0, Nsites)
            siteB = np.random.randint(0, Nsites)

            # produce a new state by swapping the two

            stateNew = state.copy()
            stateNew[siteA] = state[siteB]
            stateNew[siteB] = state[siteA]

            offscnew = MC_JIT.GetOffSite(stateNew, self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites)

            self.KMC_Jit.updateState(state, OffSiteCount, siteA, siteB)

            self.assertTrue(np.array_equal(state, stateNew))
            self.assertTrue(np.array_equal(offscnew, OffSiteCount))

    def test_KRA3body3Spec_values(self):
        warnings.warn("Make sure this test is being run for FCC 3 body KRA with the Ni-Re paper parameters.")
        if self.NSpec != 3:
            raise ValueError("This test is only valid for 3-body interactions")

        # Let's get the KRA energies of a state from the code
        jmpFinSiteList = np.array(self.VclusExp.KRAexpander.jList)
        state = self.initState.copy()
        self.assertEqual(state[self.vacsiteInd], self.NSpec-1)
        initTSoffsc = MC_JIT.GetTSOffSite(state, self.numSitesTSInteracts,
                                          self.TSInteractSites, self.TSInteractSpecs)

        InitE_KRA = self.KMC_Jit.getKRAEnergies(state, initTSoffsc, jmpFinSiteList)

        # Now for each jump, let's calculate the energies from the TS cluster expansion
        for jNum in range(jmpFinSiteList.shape[0]):
            # Get the final site and the final spec
            siteB, specB = jmpFinSiteList[jNum], state[jmpFinSiteList[jNum]]
            jumpKey = (self.vacsiteInd, siteB, specB)

            # Get the energies for the different group of this transition
            JumpPlaceInEnArray = self.VclusExp.KRAexpander.jump2Index[jumpKey]
            EnGroups = self.KRAEnergies[JumpPlaceInEnArray]

            # Now go through each of the third sites
            TsPtGrpDict = self.VclusExp.KRAexpander.clusterSpeciesJumps[jumpKey]
            delEKRA = self.KRASpecConstants[specB]
            for type, site3List in TsPtGrpDict.items():
                GroupIndex = type - 1
                for site3 in site3List:
                    site3Ind = self.VclusExp.sup.index(site3.R, site3.ci)[0]
                    if state[site3Ind] == self.KRACounterSpec:
                        delEKRA += EnGroups[GroupIndex]

            self.assertAlmostEqual(InitE_KRA[jNum], delEKRA)
            print(InitE_KRA[jNum], " ", delEKRA, " ", siteB)

        print(jmpFinSiteList)

    def test_KRA3body3SpecDetBal(self):
        # test that the KRA energies satisfy detailed balance
        warnings.warn("Make sure this test is being run for FCC 3 body KRA with the Ni-Re paper parameters.")
        if self.NSpec != 3:
            raise ValueError("This test is only valid for 3-body interactions")

        jmpFinSiteList = np.array(self.VclusExp.KRAexpander.jList)

        state = self.initState.copy()
        initTSoffsc = MC_JIT.GetTSOffSite(state, self.numSitesTSInteracts,
                                          self.TSInteractSites, self.TSInteractSpecs)

        InitE_KRA = self.KMC_Jit.getKRAEnergies(state, initTSoffsc, jmpFinSiteList)

        # we are going to test the KRA energies explicitly here

        vacSiteCi, vacSiteR = self.VclusExp.sup.ciR(self.vacsiteInd)
        self.assertEqual(vacSiteCi, (0, 0))
        self.assertTrue(np.array_equal(vacSiteR, np.zeros(3, dtype=int)))

        # Convert all the jump vectors to lattice vectors
        dxR = [np.dot(np.linalg.inv(self.VclusExp.crys.lattice), dx)
        for jList in self.jnetFCC for (i, j), dx in jList]

        sitesJump = []
        for R in dxR:
            Rnew = vacSiteR + R
            siteInd = self.VclusExp.sup.index(Rnew, vacSiteCi)[0]
            sitesJump.append(siteInd)

        self.assertTrue(set(sitesJump)==set(jmpFinSiteList), msg="{} {}".format(sitesJump, jmpFinSiteList))

        # Now make the exchanges, and see if detailed balance is satisfied
        for forwardJumpInd in range(0, len(jmpFinSiteList), 2):
            forwardSiteInd = jmpFinSiteList[forwardJumpInd]

            # Swap to make the jump
            stateNew = state.copy()
            stateNew[forwardSiteInd] = state[self.vacsiteInd]
            stateNew[self.vacsiteInd] = state[forwardSiteInd]

            # Now translate stateNew to origin and check detailed balance
            stateNewTrans = self.KMC_Jit.TranslateState(stateNew, self.vacsiteInd, forwardSiteInd)
            TSoffscNewTrans = MC_JIT.GetTSOffSite(stateNewTrans, self.numSitesTSInteracts,
                                                  self.TSInteractSites, self.TSInteractSpecs)

            # Now get KRA energies out of translated stateNew
            delE_KRA_newTrans = self.KMC_Jit.getKRAEnergies(stateNewTrans, TSoffscNewTrans, jmpFinSiteList)

            # Now check that for the forward and the reverse jumps, the KRA energies are the same.
            revJumpInd = forwardJumpInd + 1
            self.assertAlmostEqual(delE_KRA_newTrans[revJumpInd], InitE_KRA[forwardJumpInd])
            print(delE_KRA_newTrans[revJumpInd], InitE_KRA[forwardJumpInd])

