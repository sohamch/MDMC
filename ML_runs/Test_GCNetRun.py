import unittest
import os
import sys
RunPath = os.getcwd() + "/"
CrysDatPath = "../CrysDat_FCC/"
DataPath = "../MD_KMC_single/"
ModulePath = "../Symm_Network/"

import numpy as np
import h5py
import torch as pt
from tqdm import tqdm
import pickle
from GCNetRun import Load_Data, makeComputeData, makeDataTensors, Load_crysDats, SpecBatchOuts, vacBatchOuts
from GCNetRun import Train

class TestGCNetRun(unittest.TestCase):
    def setUp(self):
        self.T = 1073

        self.state1List, self.state2List, self.dispList, self.rateList, self.AllJumpRates_st1,\
        self.AllJumpRates_st2, self.avgDisps_st1, self.avgDisps_st2 =\
            Load_Data(DataPath + "singleStep_{}_AllRates.h5".format(self.T))

        self.GpermNNIdx, self.NNsiteList, self.JumpNewSites, self.dxJumps = Load_crysDats(1, CrysDatPath)

        self.z = self.dxJumps.shape[0]

        with open(CrysDatPath + "supercellFCC.pkl", "rb") as fl:
            self.superFCC = pickle.load(fl)
        self.N_units = self.superFCC.superlatt[0, 0]

        self.specCheck = 5
        self.VacSpec = 0
        print("Setup complete.")

    def test_Data_constr_singleJump(self):
        specCheck = self.specCheck
        specsToTrain = [specCheck]
        VacSpec = self.VacSpec
        N_check = 200
        AllJumps = False

        State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.AllJumpRates_st1, self.AllJumpRates_st2, self.avgDisps_st1, self.avgDisps_st2,
                            self.JumpNewSites, self.dxJumps, self.NNsiteList, N_check,
                            AllJumps=AllJumps, mode="train")

        for samp in tqdm(range(N_check), position=0, leave=True):
            for site in range(self.state1List.shape[1]):
                if site == 0:
                    self.assertTrue(np.all(State1_occs[samp, :, site] == 0))
                    self.assertTrue(np.all(State2_occs[samp, :, site] == 0))
                else:
                    spec1 = self.state1List[samp, site]
                    self.assertEqual(State1_occs[samp, spec1 - 1, site], 1)
                    self.assertEqual(np.sum(State1_occs[samp, :, site]), 1)

                    spec2 = self.state2List[samp, site]
                    self.assertEqual(State2_occs[samp, spec2 - 1, site], 1)
                    self.assertEqual(np.sum(State2_occs[samp, :, site]), 1)

            # check the displacements
            jSelect = None
            count = 0
            for jInd in range(self.JumpNewSites.shape[0]):
                state2_try = self.state1List[samp][self.JumpNewSites[jInd]]
                if np.all(state2_try == self.state2List[samp]):
                    jSelect = jInd
                    count += 1
            self.assertEqual(count, 1) # there should be exactly one match

            # Check occupancies using supercell
            state1 = self.state1List[samp]
            state2 = state1.copy()
            NNsiteVac = self.NNsiteList[jSelect + 1, 0]
            _, RsiteVac = self.superFCC.ciR(NNsiteVac)
            state2[0] = state2[NNsiteVac]
            state2[NNsiteVac] = 0

            for site in range(self.state1List.shape[1]):
                _, Rsite = self.superFCC.ciR(site)
                RsiteNew = (Rsite + RsiteVac)
                siteNew, _ = self.superFCC.index(RsiteNew, (0, 0))
                spec1 = state1[site]
                spec2 = state2[siteNew]

                if site == 0:
                    assert spec2 == 0
                    assert np.all(State1_occs[samp, :, site] == 0)
                    assert np.all(State2_occs[samp, :, site] == 0)

                else:
                    assert spec2 != 0
                    assert State1_occs[samp, spec1 - 1, site] == 1
                    assert np.sum(State1_occs[samp, :, site]) == 1

                    assert State2_occs[samp, spec2 - 1, site] == 1
                    assert np.sum(State2_occs[samp, :, site]) == 1

            NNR = np.dot(np.linalg.inv(self.superFCC.crys.lattice), self.dxJumps[jSelect]).astype(int)
            # Check that the displacements for each jump are okay
            self.assertTrue(np.allclose(np.dot(self.superFCC.crys.lattice, NNR), self.dxJumps[jSelect]))
            NNRSite = self.superFCC.index(NNR, (0, 0))[0]

            NNsiteVac = self.NNsiteList[jSelect + 1, 0]
            # check that NN sequence matches
            self.assertTrue(np.all(NNRSite == NNsiteVac))

            spec = self.state1List[samp, NNsiteVac]
            assert np.allclose(disps[samp, 0], 3.59 * self.dxJumps[jSelect])
            # Check that displacements are okay
            if spec == specCheck:
                self.assertTrue(np.allclose(disps[samp, 1], -3.59 * self.dxJumps[jSelect]))
            else:
                self.assertTrue(np.allclose(disps[samp, 1], 0.0))

            # check the rate
            self.assertTrue(np.math.isclose(rates[samp], self.rateList[samp]))

        print("Done single jump data construction test")

    def testBatchCalc_SpNN(self):
        specCheck = self.specCheck
        specsToTrain = [specCheck]
        VacSpec = self.VacSpec
        N_check = 200
        AllJumps = False
        State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.AllJumpRates_st1, self.AllJumpRates_st2, self.avgDisps_st1, self.avgDisps_st2,
                            self.JumpNewSites, self.dxJumps, self.NNsiteList, N_check,
                            AllJumps=AllJumps, mode="train")

# Train(T, dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, rates, disps,
#       jProbs_st1, jProbs_st2, NNsites, SpecsToTrain, sp_ch, VacSpec, start_ep, end_ep, interval, N_train,
#       gNet, lRate=0.001, batch_size=128, scratch_if_no_init=True, DPr=False, Boundary_train=False,
#       jumpSort=True, jumpSwitch=True, scaleL0=False)



