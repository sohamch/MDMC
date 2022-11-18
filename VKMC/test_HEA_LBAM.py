from onsager import crystal, supercell, cluster
import numpy as np
import scipy.linalg as spla
import Cluster_Expansion
import MC_JIT
import pickle
import h5py
import unittest

from HEA_LBAM import *

class Test_HEA_LBAM(unittest.TestCase):

    def setUp(self):
        self.DataPath = ("../../MD_KMC_single/Run_2/singleStep_Run2_1073_AllRates.h5")
        self.CrysDatPath = ("../../")

        self.state1List, self.dispList, self.rateList, self.AllJumpRates, self.jumpSelects = Load_Data(self.DataPath)
        self.jList, self.dxList, self.jumpNewIndices, self.superCell, self.jnet, self.vacsite, self.vacsiteInd =\
            Load_crys_Data(self.CrysDatPath, typ="FCC")

        self.AllSpecs = np.unique(self.state1List[0])
        self.NSpec = self.AllSpecs.shape[0]
        self.vacSpec = self.state1List[0, 0]
        self.SpecExpand = 5
        print("All Species: {}".format(self.AllSpecs))
        print("Vacancy Species: {}".format(self.vacSpec))
        print("Expanding Species: {}".format(self.SpecExpand))

        self.state1ListNew, self.dispListNew, self.SpecExpandNew = getNewSpecs(self.state1List, self.dispList, self.SpecExpand)
        print("Generating New cluster expansion with vacancy at {}, {}".format(self.vacsite.ci, self.vacsite.R))

        self.ClustCut = 1.01
        self.MaxOrder = 2
        self.VclusExp = makeVClusExp(self.superCell, self.jnet, self.jList, self.ClustCut, self.MaxOrder, self.NSpec, self.vacsite,
                                AllInteracts=False)

        self.MCJit, self.numVecsInteracts, self.VecsInteracts, self.VecGroupInteracts, NVclus = CreateJitCalculator(self.VclusExp, self.NSpec,
                                                                                                1073,
                                                                                                scratch=True,
                                                                                                save=True)

    def test_getNewSpecs(self):
        if self.vacSpec == 0:
            self.assertEqual(self.SpecExpandNew, 0)
            for stateInd in range(self.state1ListNew.shape[0]):
                for site in range(self.state1ListNew.shape[1]):
                    self.assertEqual(self.state1ListNew[stateInd, site], self.NSpec - self.state1List[stateInd, site] - 1)

                for spec in range(self.NSpec):
                    self.assertTrue(np.array_equal(self.dispListNew[stateInd, spec, :], self.dispList[stateInd, self.NSpec - 1 - spec, :]))

                self.assertTrue(np.array_equal(self.dispListNew[stateInd, self.NSpec - 1, :],
                                               self.dispList[stateInd, self.vacSpec, :]))

    def test_CreateJitCalculator(self):
        # This is to check whether the Jit arrays have been properly stored
        with h5py.File("JitArrays.h5", "r") as fl:
            numSitesInteracts = np.array(fl["numSitesInteracts"])
            self.assertTrue(np.array_equal(self.MCJit.numSitesInteracts, numSitesInteracts))

            SupSitesInteracts = np.array(fl["SupSitesInteracts"])
            self.assertTrue(np.array_equal(self.MCJit.SupSitesInteracts, SupSitesInteracts))

            SpecOnInteractSites = np.array(fl["SpecOnInteractSites"])
            self.assertTrue(np.array_equal(self.MCJit.SpecOnInteractSites, SpecOnInteractSites))

            numInteractsSiteSpec = np.array(fl["numInteractsSiteSpec"])
            self.assertTrue(np.array_equal(self.MCJit.numInteractsSiteSpec, numInteractsSiteSpec))

            SiteSpecInterArray = np.array(fl["SiteSpecInterArray"])
            self.assertTrue(np.array_equal(self.MCJit.SiteSpecInterArray, SiteSpecInterArray))

            numVecsInteracts = np.array(fl["numVecsInteracts"])
            self.assertTrue(np.array_equal(self.numVecsInteracts, numVecsInteracts))

            VecsInteracts = np.array(fl["VecsInteracts"])
            self.assertTrue(np.array_equal(self.VecsInteracts, VecsInteracts))

            VecGroupInteracts = np.array(fl["VecGroupInteracts"])
            self.assertTrue(np.array_equal(self.VecGroupInteracts, VecGroupInteracts))

            numSitesTSInteracts = np.array(fl["numSitesTSInteracts"])
            self.assertTrue(np.array_equal(self.MCJit.numSitesTSInteracts, numSitesTSInteracts))

            TSInteractSites = np.array(fl["TSInteractSites"])
            self.assertTrue(np.array_equal(self.MCJit.TSInteractSites, TSInteractSites))

            TSInteractSpecs = np.array(fl["TSInteractSpecs"])
            self.assertTrue(np.array_equal(self.MCJit.TSInteractSpecs, TSInteractSpecs))

            jumpFinSites = np.array(fl["jumpFinSites"])
            self.assertTrue(np.array_equal(self.MCJit.jumpFinSites, jumpFinSites))

            jumpFinSpec = np.array(fl["jumpFinSpec"])
            self.assertTrue(np.array_equal(self.MCJit.jumpFinSpec, jumpFinSpec))

            FinSiteFinSpecJumpInd = np.array(fl["FinSiteFinSpecJumpInd"])
            self.assertTrue(np.array_equal(self.MCJit.FinSiteFinSpecJumpInd, FinSiteFinSpecJumpInd))

            numJumpPointGroups = np.array(fl["numJumpPointGroups"])
            self.assertTrue(np.array_equal(self.MCJit.numJumpPointGroups, numJumpPointGroups))

            numTSInteractsInPtGroups = np.array(fl["numTSInteractsInPtGroups"])
            self.assertTrue(np.array_equal(self.MCJit.numTSInteractsInPtGroups, numTSInteractsInPtGroups))

            JumpInteracts = np.array(fl["JumpInteracts"])
            self.assertTrue(np.array_equal(self.MCJit.JumpInteracts, JumpInteracts))

            Jump2KRAEng = np.array(fl["Jump2KRAEng"])
            self.assertTrue(np.array_equal(self.MCJit.Jump2KRAEng, Jump2KRAEng))

            KRASpecConstants = np.array(fl["KRASpecConstants"])
            self.assertTrue(np.array_equal(self.MCJit.KRASpecConstants, KRASpecConstants))

            NVclus = np.array(fl["NVclus"])[0]
            self.assertTrue(len(self.VclusExp.vecClus), NVclus)

            self.assertEqual(np.max(VecGroupInteracts), NVclus - 1)