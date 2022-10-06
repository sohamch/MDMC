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
from SymmLayers import GCNet

class TestGCNetRun(unittest.TestCase):
    def setUp(self):
        self.T = 1073

        self.state1List, self.state2List, self.dispList, self.rateList, self.AllJumpRates_st1,\
        self.AllJumpRates_st2, self.avgDisps_st1, self.avgDisps_st2 =\
            Load_Data(DataPath + "singleStep_{}_AllRates.h5".format(self.T))

        self.GpermNNIdx, self.NNsiteList, self.JumpNewSites, self.dxJumps = Load_crysDats(1, CrysDatPath)

        self.z = self.dxJumps.shape[0]
        self.N_ngb = self.NNsiteList.shape[0]

        self.GnnPerms = pt.tensor(self.GpermNNIdx).long()

        print("Filter neighbor range : {}nn. Filter neighborhood size: {}".format(1, self.N_ngb - 1))
        self.Nsites = self.NNsiteList.shape[1]

        self.assertEqual(self.Nsites, self.state1List.shape[1])
        self.assertEqual(self.Nsites, self.state2List.shape[1])

        self.a0 = 3.59
        self.NNsites = pt.tensor(self.NNsiteList).long()
        self.JumpVecs = pt.tensor(self.dxJumps.T * self.a0, dtype=pt.double)
        print("Jump Vectors: \n", self.JumpVecs.T, "\n")
        self.Ndim = self.dxJumps.shape[1]

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

    def testBatchCalc_SpNN_1Jump(self):
        specCheck = self.specCheck
        specsToTrain = [specCheck]
        VacSpec = self.VacSpec
        N_check = 50
        AllJumps = False
        State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.AllJumpRates_st1, self.AllJumpRates_st2, self.avgDisps_st1, self.avgDisps_st2,
                            self.JumpNewSites, self.dxJumps, self.NNsiteList, N_check,
                            AllJumps=AllJumps, mode="train")

        # Everything related to JPINN will be None
        jProbs_st1 = None
        jProbs_st2 = None
        start_ep = 0
        end_ep = 0
        interval = 100 # don't save networks
        dirPath="."

        # Make the network
        specs = np.unique(self.state1List[0])
        NSpec = specs.shape[0] - 1
        gNet = GCNet(self.GnnPerms.long(), self.NNsites, self.JumpVecs, N_ngb=self.N_ngb, NSpec=NSpec,
                     mean=0.02, std=0.2, nl=1, nch=8, nchLast=1).double()

        sd = gNet.state_dict()

        gNet2 = GCNet(self.GnnPerms.long(), self.NNsites, self.JumpVecs, N_ngb=self.N_ngb, NSpec=NSpec,
                     mean=0.02, std=0.2, nl=1, nch=8, nchLast=1).double()

        # Get the original network
        gNet2.load_state_dict(sd)

        y1, y2 = Train(self.T, dirPath, State1_occs, State2_occs, OnSites_state1, OnSites_state2, rates, disps,
                       jProbs_st1, jProbs_st2, self.NNsites, specsToTrain, sp_ch, VacSpec, start_ep, end_ep, interval,
                       N_check, gNet, lRate=0.001, batch_size=N_check, scratch_if_no_init=True, chkpt=False)

        # Now for each sample, compute the y explicitly and match
        for samp in tqdm(range(N_check)):
            state1Input = pt.tensor(State1_occs[samp:samp + 1], dtype=pt.double)
            state2Input = pt.tensor(State2_occs[samp:samp + 1], dtype=pt.double)

            with pt.no_grad():
                y1SampAllSites = gNet2(state1Input)
                y2SampAllSites = gNet2(state2Input)

            # Now sum explicitly
            y1_samp = pt.zeros(3).double()
            y2_samp = pt.zeros(3).double()
            for site in range(self.Nsites):
                occs1 = State1_occs[samp, :, site]
                occs2 = State2_occs[samp, :, site]
                if occs1[specCheck - 1] == 1:
                    y1_samp += y1SampAllSites[0, 0, :, site]
                if occs2[specCheck - 1] == 1:
                    y2_samp += y2SampAllSites[0, 0, :, site]

            self.assertTrue(np.allclose(y1[samp], y1_samp.detach().numpy()),
                            msg="{} \n {}".format(y1[samp], y1_samp.detach().numpy()))
            self.assertTrue(np.allclose(y2[samp], y2_samp.detach().numpy()))

    def testVacBatchCalc_SpNN_1Jump(self):
        VacSpec = self.VacSpec
        specsToTrain = [VacSpec]
        N_check = 50
        AllJumps = False
        State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.AllJumpRates_st1, self.AllJumpRates_st2, self.avgDisps_st1, self.avgDisps_st2,
                            self.JumpNewSites, self.dxJumps, self.NNsiteList, N_check,
                            AllJumps=AllJumps, mode="train")

        # Everything related to JPINN will be None
        jProbs_st1 = None
        jProbs_st2 = None
        start_ep = 0
        end_ep = 0
        interval = 100 # don't save networks
        dirPath="."

        # Make the network
        specs = np.unique(self.state1List[0])
        NSpec = specs.shape[0] - 1
        gNet = GCNet(self.GnnPerms.long(), self.NNsites, self.JumpVecs, N_ngb=self.N_ngb, NSpec=NSpec,
                     mean=0.02, std=0.2, nl=1, nch=8, nchLast=1).double()

        sd = gNet.state_dict()

        gNet2 = GCNet(self.GnnPerms.long(), self.NNsites, self.JumpVecs, N_ngb=self.N_ngb, NSpec=NSpec,
                     mean=0.02, std=0.2, nl=1, nch=8, nchLast=1).double()

        # Get the original network
        gNet2.load_state_dict(sd)

        y1, y2 = Train(self.T, dirPath, State1_occs, State2_occs, OnSites_state1, OnSites_state2, rates, disps,
                       jProbs_st1, jProbs_st2, self.NNsites, specsToTrain, sp_ch, VacSpec, start_ep, end_ep, interval,
                       N_check, gNet, lRate=0.001, batch_size=N_check, scratch_if_no_init=True, chkpt=False)

        # Now for each sample, compute the y explicitly and match
        for samp in tqdm(range(N_check)):
            state1Input = pt.tensor(State1_occs[samp:samp + 1], dtype=pt.double)
            state2Input = pt.tensor(State2_occs[samp:samp + 1], dtype=pt.double)

            with pt.no_grad():
                y1SampAllSites = gNet2(state1Input)
                y2SampAllSites = gNet2(state2Input)

            # Now sum explicitly
            y1_samp = pt.zeros(3).double()
            y2_samp = pt.zeros(3).double()

            # check that it is the negative of all non-vacancy sites
            for site in range(1, self.Nsites):
                y1_samp -= y1SampAllSites[0, 0, :, site]
                y2_samp -= y2SampAllSites[0, 0, :, site]

            self.assertTrue(np.allclose(y1[samp], y1_samp.detach().numpy()),
                            msg="{} \n {}".format(y1[samp], y1_samp.detach().numpy()))
            self.assertTrue(np.allclose(y2[samp], y2_samp.detach().numpy()))




