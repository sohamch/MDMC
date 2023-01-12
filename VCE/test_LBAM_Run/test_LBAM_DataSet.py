import sys

import scipy.linalg

sys.path.append("../")

from onsager import crystal, supercell, cluster
import numpy as np
import scipy.linalg as spla
import Cluster_Expansion
import MC_JIT
import pickle
import h5py
import unittest

from LBAM_DataSet import *

class Test_HEA_LBAM(unittest.TestCase):

    def setUp(self):
        self.DataPath = ("../../MD_KMC_single/Run_2/singleStep_Run2_1073_AllRates.h5")
        self.CrysDatPath = ("../../CrysDat_FCC/CrystData.h5")

        self.state1List, self.dispList, self.rateList, self.AllJumpRates, self.jumpSelects = Load_Data(self.DataPath)
        self.jList, self.dxList, self.jumpNewIndices, self.superCell, self.jnet, self.vacsite, self.vacsiteInd =\
            Load_crys_Data(self.CrysDatPath)

        self.AllSpecs = np.unique(self.state1List[0])
        self.NSpec = self.AllSpecs.shape[0]
        self.vacSpec = self.state1List[0, 0]
        print(self.vacSpec)
        self.SpecExpand = 5
        print("All Species: {}".format(self.AllSpecs))
        print("Vacancy Species: {}".format(self.vacSpec))
        print("Expanding Species: {}".format(self.SpecExpand))

        self.ClustCut = 1.01
        self.MaxOrder = 2

        self.VclusExp = makeVClusExp(self.superCell, self.jnet, self.jList, self.ClustCut, self.MaxOrder, self.NSpec, self.vacsite, self.vacSpec,
                                AllInteracts=False)

        self.MCJit, self.numVecsInteracts, self.VecsInteracts, self.VecGroupInteracts, self.NVclus = CreateJitCalculator(self.VclusExp, self.NSpec,
                                                                                                scratch=True,
                                                                                                save=True)

        self.VclusExp_all = makeVClusExp(self.superCell, self.jnet, self.jList, self.ClustCut, self.MaxOrder, self.NSpec, self.vacsite, self.vacSpec,
                                AllInteracts=True)

        self.MCJit_all, self.numVecsInteracts_all, self.VecsInteracts_all, self.VecGroupInteracts_all,\
        self.NVclus_all = CreateJitCalculator(self.VclusExp_all, self.NSpec, scratch=True, save=False)

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

            vacSpec = np.array(fl["vacSpec"])[0]
            self.assertEqual(self.vacSpec, vacSpec)

            self.assertEqual(np.max(VecGroupInteracts), NVclus - 1)

    def test_Expand(self):
        sampInd = 3
        stateList = self.state1List[sampInd:sampInd + 1]
        AllJumpRates = self.AllJumpRates[sampInd:sampInd + 1]
        jumpSelects = self.jumpSelects[sampInd:sampInd + 1]
        dispList = self.dispList[sampInd:sampInd + 1]
        rateList = self.rateList[sampInd:sampInd + 1]
        SpecExpand = self.SpecExpand
        NVclus = len(self.VclusExp.vecClus)
        assert NVclus == self.NVclus

        state = stateList[0]

        # small check for jump indexing
        state2 = state[self.jumpNewIndices[jumpSelects[0]]]

        state2_explict = state.copy()
        jmp = jumpSelects[0]
        jSite = self.jList[jmp]
        state2_explict[jSite] = state[0]
        state2_explict[0] = state[jSite]

        dxR, _ = self.superCell.crys.cart2pos(self.dxList[jmp])

        state2Trans = np.zeros_like(state)
        for siteInd in range(state.shape[0]):
            ci, Rsite = self.superCell.ciR(siteInd)
            RsiteNew = Rsite - dxR
            siteIndNew, _ = self.superCell.index(RsiteNew, ci)
            state2Trans[siteIndNew] = state2_explict[siteInd]
            self.assertEqual(state2[siteIndNew], state2_explict[siteInd], msg="{} {} {} {} {}".format(siteInd, siteIndNew,
                                                                                                      self.jList[jmp],
                                                                                                      state2[siteIndNew],
                                                                                                      state2_explict[siteInd]))
        self.assertTrue(np.array_equal(state2Trans, state2))

        # Next we'll check the change in the basis vectors with and without considering only the required sites
        off_sc = MC_JIT.GetOffSite(state, self.MCJit.numSitesInteracts, self.MCJit.SupSitesInteracts,
                                   self.MCJit.SpecOnInteractSites)
        self.assertTrue(off_sc.shape == self.MCJit.numSitesInteracts.shape == (self.MCJit.numSitesInteracts.shape[0],),
                        msg="{} {} {}".format(off_sc.shape, self.MCJit.numSitesTSInteracts.shape,
                                              (self.MCJit.numSitesInteracts.shape[0],)))
        # check off site count
        for interactID in range(self.MCJit.numSitesInteracts.shape[0]):
            offCount = 0
            for IntSiteInd in range(self.MCJit.numSitesInteracts[interactID]):
                supSite = self.MCJit.SupSitesInteracts[interactID, IntSiteInd]
                sp = self.MCJit.SpecOnInteractSites[interactID, IntSiteInd]
                if state[supSite] != sp:
                    offCount += 1
            self.assertEqual(offCount, off_sc[interactID])

        off_sc_all = MC_JIT.GetOffSite(state, self.MCJit_all.numSitesInteracts, self.MCJit_all.SupSitesInteracts,
                                   self.MCJit_all.SpecOnInteractSites)

        off_sc2 = MC_JIT.GetOffSite(state2, self.MCJit.numSitesInteracts, self.MCJit.SupSitesInteracts,
                                   self.MCJit.SpecOnInteractSites)

        off_sc2_all = MC_JIT.GetOffSite(state2, self.MCJit_all.numSitesInteracts, self.MCJit_all.SupSitesInteracts,
                                    self.MCJit_all.SpecOnInteractSites)

        off_sc2_explicit = MC_JIT.GetOffSite(state2_explict, self.MCJit.numSitesInteracts, self.MCJit.SupSitesInteracts,
                                    self.MCJit.SpecOnInteractSites)

        off_sc2_explicit_all = MC_JIT.GetOffSite(state2_explict, self.MCJit_all.numSitesInteracts, self.MCJit_all.SupSitesInteracts,
                                             self.MCJit_all.SpecOnInteractSites)

        # Now get the lambda vectors for each state
        lamb1 = np.zeros((NVclus, 3))
        lamb2 = np.zeros((NVclus, 3))
        lamb2_explicit = np.zeros((NVclus, 3))

        assert self.numVecsInteracts.shape[0] == self.MCJit.numSitesInteracts.shape[0]

        for interactID in range(self.numVecsInteracts.shape[0]):
            if off_sc[interactID] == 0:
                numVecs = self.numVecsInteracts[interactID]
                for vecInd in range(numVecs):
                    vec = self.VecsInteracts[interactID, vecInd]
                    vecGroup = self.VecGroupInteracts[interactID, vecInd]
                    lamb1[vecGroup] += vec

            if off_sc2[interactID] == 0:
                numVecs = self.numVecsInteracts[interactID]
                for vecInd in range(numVecs):
                    vec = self.VecsInteracts[interactID, vecInd]
                    vecGroup = self.VecGroupInteracts[interactID, vecInd]
                    lamb2[vecGroup] += vec

            if off_sc2_explicit[interactID] == 0:
                numVecs = self.numVecsInteracts[interactID]
                for vecInd in range(numVecs):
                    vec = self.VecsInteracts[interactID, vecInd]
                    vecGroup = self.VecGroupInteracts[interactID, vecInd]
                    lamb2_explicit[vecGroup] += vec

        # Now get the lambda vectors for each state
        assert self.NVclus_all == len(self.VclusExp_all.vecClus)
        lamb1_all = np.zeros((self.NVclus_all, 3))
        lamb2_all = np.zeros((self.NVclus_all, 3))
        lamb2_explicit_all = np.zeros((self.NVclus_all, 3))

        assert self.numVecsInteracts_all.shape[0] == self.MCJit_all.numSitesInteracts.shape[0]
        for interactID in range(self.numVecsInteracts_all.shape[0]):
            if off_sc_all[interactID] == 0:
                numVecs = self.numVecsInteracts_all[interactID]
                for vecInd in range(numVecs):
                    vec = self.VecsInteracts_all[interactID, vecInd]
                    vecGroup = self.VecGroupInteracts_all[interactID, vecInd]
                    lamb1_all[vecGroup] += vec

            if off_sc2_all[interactID] == 0:
                numVecs = self.numVecsInteracts_all[interactID]
                for vecInd in range(numVecs):
                    vec = self.VecsInteracts_all[interactID, vecInd]
                    vecGroup = self.VecGroupInteracts_all[interactID, vecInd]
                    lamb2_all[vecGroup] += vec

            if off_sc2_explicit_all[interactID] == 0:
                numVecs = self.numVecsInteracts_all[interactID]
                for vecInd in range(numVecs):
                    vec = self.VecsInteracts_all[interactID, vecInd]
                    vecGroup = self.VecGroupInteracts_all[interactID, vecInd]
                    lamb2_explicit_all[vecGroup] += vec

        # Translational symmetry will be broken if not all interactions are included.
        # However, all necessary interactions to compute CHANGE in lambda are still accounted for.
        self.assertTrue(np.allclose(lamb2_explicit_all, lamb2_all))
        del_lamb_all_interacts = lamb2_all - lamb1_all
        del_lamb = lamb2_explicit - lamb1
        self.assertTrue(np.allclose(del_lamb, del_lamb_all_interacts),
                        msg="\n{} \n {}".format(del_lamb[:5], del_lamb_all_interacts[:5]))

        lamb1_comp = self.MCJit.getLambda(off_sc, NVclus, self.numVecsInteracts, self.VecGroupInteracts,
                                             self.VecsInteracts)

        self.assertTrue(np.allclose(lamb1_comp, lamb1))

        lamb2_comp = self.MCJit.getLambda(off_sc2, NVclus, self.numVecsInteracts, self.VecGroupInteracts,
                                     self.VecsInteracts)

        self.assertTrue(np.allclose(lamb2_comp, lamb2))

        _, del_lamb_comp = self.MCJit.DoSwapUpdate(state, 0, self.jList[jumpSelects[0]], NVclus, off_sc,
                                        self.numVecsInteracts, self.VecGroupInteracts, self.VecsInteracts)

        self.assertTrue(np.array_equal(off_sc, off_sc2_explicit))

        self.MCJit.revert(off_sc, state, 0, self.jList[jumpSelects[0]])

        # check the del_lamb function
        del_lamb_comp_2 = self.MCJit.getDelLamb(state, off_sc, 0, self.jList[jumpSelects[0]], self.NVclus,
                   self.numVecsInteracts, self.VecGroupInteracts, self.VecsInteracts)

        self.assertTrue(np.allclose(del_lamb_comp, del_lamb), msg="\n{} \n {}".format(del_lamb_comp[:5], del_lamb[:5]))
        self.assertTrue(np.allclose(del_lamb_comp_2, del_lamb), msg="\n{} \n {}".format(del_lamb_comp[:5], del_lamb[:5]))

        # Try with single jump dataset first
        Wbar, Bbar, etaBar, offscTime, expandTime = Expand(1073, stateList, 0, 100, self.jList,
                                                                 self.dxList, AllJumpRates, jumpSelects, dispList, rateList,
                                                                 SpecExpand, self.MCJit, NVclus,
                                                                 self.numVecsInteracts, self.VecsInteracts, self.VecGroupInteracts,
                                                                 aj=False)

        Wbar_comp = np.zeros((NVclus, NVclus))
        Bbar_comp = np.zeros(NVclus)

        for l1 in range(NVclus):
            dx = dispList[0, SpecExpand]
            Bbar_comp[l1] = rateList[0] * np.dot(dx, del_lamb[l1])
            for l2 in range(NVclus):
                Wbar_comp[l1, l2] = rateList[0] * np.dot(del_lamb[l1], del_lamb[l2])

        self.assertTrue(np.allclose(Wbar, Wbar_comp))
        self.assertTrue(np.allclose(Bbar, Bbar_comp), msg="{} \n {} \n {}".format(SpecExpand, Bbar_comp[:10], Bbar[:10]))

        # Then try with all jumps
        a0 = np.linalg.norm(dispList[0, self.NSpec - 1, :]) / np.linalg.norm(self.dxList[0])
        print(a0)
        Wbar, Bbar, etaBar, offscTime, expandTime = Expand(1073, stateList, self.vacsiteInd, 1, self.jList,
                                                                 self.dxList * a0, AllJumpRates, jumpSelects, dispList, rateList,
                                                                 SpecExpand, self.MCJit, NVclus,
                                                                 self.numVecsInteracts, self.VecsInteracts, self.VecGroupInteracts,
                                                                 aj=True)

        Wbar_comp = np.zeros((NVclus, NVclus))
        Bbar_comp = np.zeros(NVclus)

        # check that off site counts have been correctly reverted in previous operations
        self.assertTrue(np.array_equal(off_sc, MC_JIT.GetOffSite(state, self.MCJit.numSitesInteracts, self.MCJit.SupSitesInteracts,
                                   self.MCJit.SpecOnInteractSites)))

        if SpecExpand == self.vacSpec:
            print("Testing vacancy")

        for jmp in range(self.dxList.shape[0]):
            del_lamb = self.MCJit.getDelLamb(state, off_sc, 0, self.jList[jmp], self.NVclus,
                                                    self.numVecsInteracts, self.VecGroupInteracts, self.VecsInteracts)

            for l1 in range(NVclus):
                if SpecExpand != self.vacSpec:
                    dx = -self.dxList[jmp] if state[self.jList[jmp]] == self.SpecExpand else np.zeros(3)
                else:
                    dx = self.dxList[jmp]
                Bbar_comp[l1] += AllJumpRates[0, jmp] * np.dot(dx * a0, del_lamb[l1])
                for l2 in range(NVclus):
                    Wbar_comp[l1, l2] += AllJumpRates[0, jmp] * np.dot(del_lamb[l1], del_lamb[l2])

        self.assertTrue(np.allclose(Wbar, Wbar_comp))
        self.assertTrue(np.allclose(Bbar, Bbar_comp))


    def test_symm(self):
        """
        test the symmetry of the basis vectors when only interactions that cantain
        the sites whose occupancies change during vacancy jumps are considered.
        """
        initState = self.state1List[0]

        for g in tqdm(self.VclusExp.sup.crys.G, ncols=65, position=0, leave=True):
            stateG = np.zeros_like(initState)

            for site in range(initState.shape[0]):
                ciSite, Rsite = self.VclusExp.sup.ciR(site)
                Rnew, ciNew = self.VclusExp.sup.crys.g_pos(g, Rsite, ciSite)

                assert ciNew == ciSite == (0, 0)
                siteIndNew, _ = self.VclusExp.sup.index(Rnew, (0, 0))
                stateG[siteIndNew] = initState[site]

            # Now get their off site counts and basis vectors
            offsc1 = MC_JIT.GetOffSite(initState, self.MCJit.numSitesInteracts, self.MCJit.SupSitesInteracts, self.MCJit.SpecOnInteractSites)
            offsc2 = MC_JIT.GetOffSite(stateG, self.MCJit.numSitesInteracts, self.MCJit.SupSitesInteracts, self.MCJit.SpecOnInteractSites)

            # Now get their basis vectors
            NVclus = len(self.VclusExp.vecClus)
            print(NVclus)
            lamb1 = self.MCJit.getLambda(offsc1, NVclus, self.numVecsInteracts, self.VecGroupInteracts,
                                                 self.VecsInteracts)
            lamb2 = self.MCJit.getLambda(offsc2, NVclus, self.numVecsInteracts, self.VecGroupInteracts,
                                                 self.VecsInteracts)

            # Rotate lamb1 by the group operation
            lamb1G = np.dot(g.cartrot, lamb1.T).T
            self.assertTrue(np.allclose(lamb1G, lamb2))


class Test_HEA_LBAM_vac(Test_HEA_LBAM):

    def setUp(self):
        self.DataPath = ("../../MD_KMC_single/Run_2/singleStep_Run2_1073_AllRates.h5")
        self.CrysDatPath = ("../../CrysDat_FCC/CrystData.h5")

        self.state1List, self.dispList, self.rateList, self.AllJumpRates, self.jumpSelects = Load_Data(self.DataPath)
        self.jList, self.dxList, self.jumpNewIndices, self.superCell, self.jnet, self.vacsite, self.vacsiteInd =\
            Load_crys_Data(self.CrysDatPath)

        self.AllSpecs = np.unique(self.state1List[0])
        self.NSpec = self.AllSpecs.shape[0]
        self.vacSpec = self.state1List[0, 0]
        self.SpecExpand = 0
        print("All Species: {}".format(self.AllSpecs))
        print("Vacancy Species: {}".format(self.vacSpec))
        print("Expanding Species: {}".format(self.SpecExpand))

        
        print("Generating New cluster expansion with vacancy at {}, {}".format(self.vacsite.ci, self.vacsite.R))

        self.ClustCut = 1.01
        self.MaxOrder = 2

        self.VclusExp = makeVClusExp(self.superCell, self.jnet, self.jList, self.ClustCut, self.MaxOrder, self.NSpec, self.vacsite, self.vacSpec,
                                AllInteracts=False)

        self.MCJit, self.numVecsInteracts, self.VecsInteracts, self.VecGroupInteracts, self.NVclus = CreateJitCalculator(self.VclusExp, self.NSpec,
                                                                                                scratch=True,
                                                                                                save=True)

        self.VclusExp_all = makeVClusExp(self.superCell, self.jnet, self.jList, self.ClustCut, self.MaxOrder, self.NSpec, self.vacsite, self.vacSpec,
                                AllInteracts=True)

        self.MCJit_all, self.numVecsInteracts_all, self.VecsInteracts_all, self.VecGroupInteracts_all,\
        self.NVclus_all = CreateJitCalculator(self.VclusExp_all, self.NSpec, save=False)

