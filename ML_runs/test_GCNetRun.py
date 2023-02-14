import unittest
import os
import sys
RunPath = os.getcwd() + "/"
CrysDatPath = "../CrysDat_FCC/CrystData.h5"
DataPath = "../MD_KMC_single/Run_2/"
ModulePath = "../Symm_Network/"

import numpy as np
import h5py
import torch as pt
from tqdm import tqdm
import pickle
from onsager import crystal, supercell
from GCNetRun import Load_Data, makeComputeData, makeDataTensors, Load_crysDats
from GCNetRun import Train
from SymmLayers import GCNet


class TestGCNetRun(unittest.TestCase):
    def setUp(self):
        self.T = 1073

        self.state1List, self.state2List, self.dispList, self.rateList, self.AllJumpRates_st1,\
        self.AllJumpRates_st2=\
            Load_Data(DataPath + "singleStep_Run2_{}_AllRates.h5".format(self.T))

        self.GpermNNIdx, self.NNsiteList, self.JumpNewSites, self.dxJumps = Load_crysDats(CrysDatPath)

        with h5py.File(CrysDatPath, "r") as fl:
            lattice = np.array(fl["Lattice_basis_vectors"])
            superlatt = np.array(fl["SuperLatt"])
            NNList = np.array(fl["NNsiteList_sitewise"])

        jList = NNList[1:, 0]

        crys = crystal.Crystal(lattice=lattice, basis=[[np.array([0., 0., 0.])]], chemistry=["A"])
        self.superCell = supercell.ClusterSupercell(crys, superlatt)

        self.N_units = self.superCell.superlatt[0, 0]
        
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

        self.specCheck = 5
        self.VacSpec = 0
        self.sp_ch = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        print("Setup complete.")

    def test_makeComputeData_singleJump(self):
        specCheck = self.specCheck
        specsToTrain = [specCheck]
        VacSpec = self.VacSpec
        N_check = 200
        N_train = 1000
        AllJumps = False

        for m in ["train", "all"]:
            print("testing mode : {}".format(m))
            State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
                makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                                self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps, self.NNsiteList, N_train,
                                AllJumps=AllJumps, mode=m)

            if m == "all":
                self.assertTrue(State1_occs.shape[0] == self.state1List.shape[0] == State2_occs.shape[0])
                self.assertTrue(rates.shape[0] == self.state1List.shape[0] == disps.shape[0])
                self.assertTrue(OnSites_state1.shape[0] == self.state1List.shape[0] == OnSites_state2.shape[0])
                sampsCheck = np.random.randint(0, State1_occs.shape[0], N_check)

            else:
                self.assertTrue(State1_occs.shape[0] == N_train == State2_occs.shape[0])
                self.assertTrue(rates.shape[0] == N_train == disps.shape[0])
                self.assertTrue(OnSites_state1.shape[0] == N_train == OnSites_state2.shape[0])
                sampsCheck = np.random.randint(0, N_train, N_check)

            for samp in tqdm(sampsCheck, position=0, leave=True):
                for site in range(self.state1List.shape[1]):
                    if site == 0:
                        self.assertTrue(np.all(State1_occs[samp, :, site] == 0))
                        self.assertTrue(np.all(State2_occs[samp, :, site] == 0))
                    else:
                        spec1 = self.state1List[samp, site]
                        self.assertEqual(State1_occs[samp, self.sp_ch[spec1], site], 1)
                        self.assertEqual(np.sum(State1_occs[samp, :, site]), 1)

                        spec2 = self.state2List[samp, site]
                        self.assertEqual(State2_occs[samp, self.sp_ch[spec2], site], 1)
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
                _, RsiteVac = self.superCell.ciR(NNsiteVac)
                state2[0] = state2[NNsiteVac]
                state2[NNsiteVac] = 0

                for site in range(self.state1List.shape[1]):
                    _, Rsite = self.superCell.ciR(site)
                    RsiteNew = (Rsite + RsiteVac)
                    siteNew, _ = self.superCell.index(RsiteNew, (0, 0))
                    spec1 = state1[site]
                    spec2 = state2[siteNew]

                    if site == 0:
                        assert spec2 == 0
                        assert np.all(State1_occs[samp, :, site] == 0)
                        assert np.all(State2_occs[samp, :, site] == 0)

                    else:
                        assert spec2 != 0
                        assert State1_occs[samp, self.sp_ch[spec1], site] == 1
                        assert np.sum(State1_occs[samp, :, site]) == 1

                        assert State2_occs[samp, self.sp_ch[spec2], site] == 1
                        assert np.sum(State2_occs[samp, :, site]) == 1

                NNR = np.dot(np.linalg.inv(self.superCell.crys.lattice), self.dxJumps[jSelect]).astype(int)
                # Check that the displacements for each jump are okay
                self.assertTrue(np.allclose(np.dot(self.superCell.crys.lattice, NNR), self.dxJumps[jSelect]))
                NNRSite = self.superCell.index(NNR, (0, 0))[0]

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

    def test_makeDataTensor(self):
        specCheck = self.specCheck
        specsToTrain = [specCheck]
        VacSpec = self.VacSpec
        N_check = 200
        N_train = 1000
        AllJumps = False
        State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps, self.NNsiteList, N_train,
                            AllJumps=AllJumps, mode="train")

        state1Data, state2Data, dispData, rateData, On_st1, On_st2 = \
            makeDataTensors(State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2,
                            specsToTrain, VacSpec, sp_ch, Ndim=3)

        self.assertTrue(pt.equal(pt.tensor(State1_occs), state1Data))
        self.assertTrue(pt.equal(pt.tensor(State2_occs), state2Data))

        self.assertTrue(pt.allclose(rateData, pt.tensor(rates)))

        self.assertTrue(pt.allclose(dispData, pt.tensor(disps[:, 1, :])))

        specsToTrain=[VacSpec]

        State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps, self.NNsiteList, N_train,
                            AllJumps=AllJumps, mode="train")

        state1Data, state2Data, dispData, rateData, On_st1, On_st2 = \
            makeDataTensors(State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2,
                            specsToTrain, VacSpec, sp_ch, Ndim=3)

        self.assertTrue(pt.equal(pt.tensor(State1_occs), state1Data))
        self.assertTrue(pt.equal(pt.tensor(State2_occs), state2Data))

        self.assertTrue(pt.allclose(rateData, pt.tensor(rates)))

        self.assertTrue(pt.allclose(dispData, pt.tensor(disps[:, 0, :])))

    def test_Train_Batch_SpNN_1Jump(self):
        specCheck = self.specCheck
        specsToTrain = [specCheck]
        VacSpec = self.VacSpec
        N_check = 50
        AllJumps = False
        State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps, self.NNsiteList, N_check,
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
                       jProbs_st1, jProbs_st2, specsToTrain, sp_ch, VacSpec, start_ep, end_ep, interval,
                       N_check, gNet, lRate=0.001, batch_size=N_check, scratch_if_no_init=True, chkpt=False)

        print("Max, min and avg values")
        print(np.max(y1), np.max(y2))
        print(np.min(y1), np.min(y2))
        print(np.mean(y1), np.mean(y2))

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
                if occs1[self.sp_ch[specCheck]] == 1:
                    assert self.state1List[samp, site] == specCheck
                    y1_samp += y1SampAllSites[0, 0, :, site]
                if occs2[self.sp_ch[specCheck]] == 1:
                    assert self.state2List[samp, site] == specCheck
                    y2_samp += y2SampAllSites[0, 0, :, site]

            self.assertTrue(np.allclose(y1[samp], y1_samp.detach().numpy()),
                            msg="{} \n {}".format(y1[samp], y1_samp.detach().numpy()))
            self.assertTrue(np.allclose(y2[samp], y2_samp.detach().numpy()))

    def test_Train_Vac_Batch_SpNN_1Jump(self):
        VacSpec = self.VacSpec
        specsToTrain = [VacSpec]
        N_check = 50
        AllJumps = False
        State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps, self.NNsiteList, N_check,
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
                       jProbs_st1, jProbs_st2, specsToTrain, sp_ch, VacSpec, start_ep, end_ep, interval,
                       N_check, gNet, lRate=0.001, batch_size=N_check, scratch_if_no_init=True, chkpt=False)

        print("Max, min and avg values")
        print(np.max(y1), np.max(y2))
        print(np.min(y1), np.min(y2))
        print(np.mean(y1), np.mean(y2))

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

    def test_makeComputeData_AllJumps(self):
        specCheck = self.specCheck
        specsToTrain = [specCheck]
        VacSpec = self.VacSpec
        N_check = 200
        N_train = 1000
        AllJumps = True

        for m in ["train", "all"]:
            print("testing mode : {}".format(m))
            State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
                makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                                self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps, self.NNsiteList, N_train,
                                AllJumps=AllJumps, mode=m)

            if m == "all":
                self.assertTrue(State1_occs.shape[0] == self.state1List.shape[0] * self.z == State2_occs.shape[0])
                self.assertTrue(rates.shape[0] == self.state1List.shape[0] * self.z == disps.shape[0])
                self.assertTrue(OnSites_state1.shape[0] == self.state1List.shape[0] * self.z == OnSites_state2.shape[0])
                sampsCheck = np.random.randint(0, self.state1List.shape[0], N_check)

            else:
                self.assertTrue(State1_occs.shape[0] == N_train * self.z == State2_occs.shape[0])
                self.assertTrue(rates.shape[0] == N_train * self.z == disps.shape[0])
                self.assertTrue(OnSites_state1.shape[0] == N_train * self.z == OnSites_state2.shape[0])
                sampsCheck = np.random.randint(0, N_train, N_check)

            for stateInd in tqdm(sampsCheck, position=0, leave=True):
                state1 = self.state1List[stateInd]
                for jInd in range(self.z):
                    state2 = state1.copy()
                    NNsiteVac = self.NNsiteList[jInd + 1, 0]
                    _, RsiteVacNN = self.superCell.ciR(NNsiteVac)
                    state2[0] = state2[NNsiteVac]
                    state2[NNsiteVac] = 0

                    for site in range(self.state1List.shape[1]):
                        _, Rsite = self.superCell.ciR(site)
                        RsiteNew = (Rsite + RsiteVacNN) % self.N_units
                        siteNew, _ = self.superCell.index(RsiteNew, (0,0))
                        spec1 = state1[site]
                        spec2 = state2[siteNew]

                        if site == 0:
                            self.assertEqual(spec2, 0)
                            self.assertTrue(np.all(State1_occs[stateInd * self.dxJumps.shape[0] + jInd, :, site] == 0))
                            self.assertTrue(np.all(State2_occs[stateInd * self.dxJumps.shape[0] + jInd, :, site] == 0))

                        else:
                            self.assertNotEqual(spec2, 0)
                            self.assertEqual(State1_occs[stateInd * self.dxJumps.shape[0] + jInd, self.sp_ch[spec1], site], 1)
                            self.assertEqual(np.sum(State1_occs[stateInd * self.dxJumps.shape[0] + jInd, :, site]), 1)

                            self.assertEqual(State2_occs[stateInd * self.dxJumps.shape[0] + jInd, self.sp_ch[spec2], site], 1)
                            self.assertEqual(np.sum(State2_occs[stateInd * self.dxJumps.shape[0] + jInd, :, site]), 1)

                    # check the displacements
                    NNR = np.dot(np.linalg.inv(self.superCell.crys.lattice), self.dxJumps[jInd]).astype(int) % self.N_units
                    NNsiteVac = self.NNsiteList[jInd + 1, 0]
                    self.assertTrue(np.all(NNR == RsiteVacNN))
                    spec = state1[NNsiteVac]
                    self.assertTrue(np.allclose(disps[stateInd * self.dxJumps.shape[0] + jInd, 0],
                                                self.a0 * self.dxJumps[jInd]))
                    if spec == specCheck:
                        self.assertTrue(np.allclose(disps[stateInd * self.dxJumps.shape[0] + jInd, 1],
                                                    -self.a0 * self.dxJumps[jInd]))

                    # check the rate
                    assert np.allclose(rates[stateInd * self.dxJumps.shape[0] + jInd], self.AllJumpRates_st1[stateInd, jInd])

    def test_Train_Batch_SpNN_AllJumps(self):
        specCheck = self.specCheck
        specsToTrain = [specCheck]
        VacSpec = self.VacSpec
        N_check = 20
        AllJumps = True
        State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps, self.NNsiteList, N_check,
                            AllJumps=AllJumps, mode="train")

        # Everything related to JPINN will be None
        jProbs_st1 = None
        jProbs_st2 = None
        start_ep = 0
        end_ep = 0
        interval = 100  # don't save networks
        dirPath = "."

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
                              jProbs_st1, jProbs_st2, specsToTrain, sp_ch, VacSpec, start_ep, end_ep,
                              interval, N_check * self.z, gNet,
                              lRate=0.001, batch_size=N_check * self.z, scratch_if_no_init=True, chkpt=False)

        print("Max, min and avg values")
        print(np.max(y1), np.max(y2))
        print(np.min(y1), np.min(y2))
        print(np.mean(y1), np.mean(y2))

        # Now for each sample, compute the y explicitly and match
        for samp in tqdm(range(N_check * self.z)):
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
                if occs1[self.sp_ch[specCheck]] == 1:
                    y1_samp += y1SampAllSites[0, 0, :, site]
                if occs2[self.sp_ch[specCheck]] == 1:
                    y2_samp += y2SampAllSites[0, 0, :, site]

            self.assertTrue(np.allclose(y1[samp], y1_samp.detach().numpy()),
                            msg="{} \n {}".format(y1[samp], y1_samp.detach().numpy()))
            self.assertTrue(np.allclose(y2[samp], y2_samp.detach().numpy()))

    def test_Train_Vac_Batch_SpNN_AllJumps(self):
        VacSpec = self.VacSpec
        specsToTrain = [VacSpec]
        N_check = 20
        AllJumps = True
        State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps, self.NNsiteList, N_check,
                            AllJumps=AllJumps, mode="train")

        for samp in range(N_check):
            for j in range(self.z):
                self.assertTrue(np.allclose(disps[samp * self.z + j, 0], self.dxJumps[j] * self.a0))

        # Everything related to JPINN will be None
        jProbs_st1 = None
        jProbs_st2 = None
        start_ep = 0
        end_ep = 0
        interval = 100  # don't save networks
        dirPath = "."

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
                              jProbs_st1, jProbs_st2, specsToTrain, sp_ch, VacSpec, start_ep, end_ep,
                              interval, N_check * self.z, gNet,
                              lRate=0.001, batch_size=N_check * self.z, scratch_if_no_init=True, chkpt=False)

        print("Max, min and avg values")
        print(np.max(y1), np.max(y2))
        print(np.min(y1), np.min(y2))
        print(np.mean(y1), np.mean(y2))

        # Now for each sample, compute the y explicitly and match
        for samp in tqdm(range(N_check * self.z)):
            state1Input = pt.tensor(State1_occs[samp:samp + 1], dtype=pt.double)
            state2Input = pt.tensor(State2_occs[samp:samp + 1], dtype=pt.double)

            with pt.no_grad():
                y1SampAllSites = gNet2(state1Input)
                y2SampAllSites = gNet2(state2Input)

            # Now sum explicitly
            y1_samp = pt.zeros(3).double()
            y2_samp = pt.zeros(3).double()
            for site in range(1, self.Nsites):
                y1_samp -= y1SampAllSites[0, 0, :, site]
                y2_samp -= y2SampAllSites[0, 0, :, site]

            self.assertTrue(np.allclose(y1[samp], y1_samp.detach().numpy()),
                            msg="{} \n {}".format(y1[samp], y1_samp.detach().numpy()))
            self.assertTrue(np.allclose(y2[samp], y2_samp.detach().numpy()))

    def test_Train_JPINN_1Jump_FullSym_Vac(self):
        specCheck = self.specCheck
        specsToTrain = [specCheck]
        VacSpec = self.VacSpec
        N_check = 50
        AllJumps = False
        State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.AllJumpRates_st1,  self.JumpNewSites, self.dxJumps, self.NNsiteList, N_check,
                            AllJumps=AllJumps, mode="train")

        # Everything related to JPINN will be None
        jProbs_st1 = self.AllJumpRates_st1 / np.sum(self.AllJumpRates_st1, axis=1).reshape(-1, 1)
        jProbs_st2 = self.AllJumpRates_st1 / np.sum(self.AllJumpRates_st1, axis=1).reshape(-1, 1)

        start_ep = 0
        end_ep = 0
        interval = 100  # don't save networks
        dirPath = "."

        # Make the network
        specs = np.unique(self.state1List[0])
        NSpec = specs.shape[0] - 1
        gNet = GCNet(self.GnnPerms.long(), self.NNsites, self.JumpVecs, N_ngb=self.N_ngb, NSpec=NSpec,
                     mean=0.02, std=0.2, nl=1, nch=8, nchLast=self.z).double()

        sd = gNet.state_dict()

        gNet2 = GCNet(self.GnnPerms.long(), self.NNsites, self.JumpVecs, N_ngb=self.N_ngb, NSpec=NSpec,
                      mean=0.02, std=0.2, nl=1, nch=8, nchLast=self.z).double()

        # Get the original network
        gNet2.load_state_dict(sd)

        # Compute vectors with only vacancy sites
        y1, y2 = Train(self.T, dirPath, State1_occs, State2_occs, OnSites_state1, OnSites_state2, rates, disps,
                       jProbs_st1, jProbs_st2, specsToTrain, sp_ch, VacSpec, start_ep, end_ep, interval,
                       N_check, gNet, lRate=0.001, batch_size=N_check, scratch_if_no_init=True, chkpt=False,
                       Boundary_train=True, jumpSort=True, AddOnSites=False)
        print("Max, min and avg values")
        print(np.max(y1), np.max(y2))
        print(np.min(y1), np.min(y2))
        print(np.mean(y1), np.mean(y2))

        # Now for each sample, compute the y explicitly and match
        for samp in tqdm(range(N_check)):
            state1Input = pt.tensor(State1_occs[samp:samp + 1], dtype=pt.double)
            state2Input = pt.tensor(State2_occs[samp:samp + 1], dtype=pt.double)
            state1JumpProbs = np.sort(jProbs_st1[samp])
            state2JumpProbs = np.sort(jProbs_st2[samp])

            with pt.no_grad():
                y1SampAllSites = gNet2(state1Input).detach().numpy()
                y2SampAllSites = gNet2(state2Input).detach().numpy()

            # Now sum explicitly with jump probs
            y1_samp = -y1SampAllSites[0, :, :, 0]
            y2_samp = -y2SampAllSites[0, :, :, 0]

            y1_jumpSum = np.zeros(3)
            y2_jumpSum = np.zeros(3)

            for jump in range(self.z):
                y1_jumpSum += y1_samp[jump] * state1JumpProbs[jump]
                y2_jumpSum += y2_samp[jump] * state2JumpProbs[jump]

            self.assertTrue(np.allclose(y1[samp], y1_jumpSum),
                            msg="{} \n {}".format(y1[samp], y1_jumpSum))
            self.assertTrue(np.allclose(y2[samp], y2_jumpSum))

        # Now check for symmetry
        state1Input = pt.tensor(State1_occs[:1], dtype=pt.double)
        state1JumpProbs = np.sort(jProbs_st1[0])

        with pt.no_grad():
            y1SampAllSites = gNet2(state1Input).detach().numpy()

        for g in tqdm(self.superCell.crys.G, position=0, leave=True, ncols=65):
            state2Input = pt.zeros_like(state1Input)
            for site in range(1, self.Nsites):
                ci, R = self.superCell.ciR(site)
                Rnew, ciNew = self.superCell.crys.g_pos(g, R, ci)
                siteNew, _ = self.superCell.index(Rnew, ciNew)
                state2Input[:, :, siteNew] = state1Input[:, :, site]

            if np.allclose(g.cartrot, np.eye(3)):
                assert np.array_equal(state2Input, state1Input)

            with pt.no_grad():
                y2SampAllSites = gNet2(state2Input).detach().numpy()
            for site in range(self.Nsites):
                ci, R = self.superCell.ciR(site)
                Rnew, ciNew = self.superCell.crys.g_pos(g, R, ci)
                siteNew, _ = self.superCell.index(Rnew, ciNew)
                for jmp in range(self.z):
                    self.assertTrue(np.allclose(
                        y2SampAllSites[0, jmp, :, siteNew], np.dot(g.cartrot, y1SampAllSites[0, jmp, :, site])
                    )
                    )

            # Now sum explicitly with jump probs
            y1_samp = -y1SampAllSites[0, :, :, 0]
            y2_samp = -y2SampAllSites[0, :, :, 0]

            y1_jumpSum = np.zeros(3)
            y2_jumpSum = np.zeros(3)

            for jump in range(self.z):
                # Since states are symmetric, jump probs in sorted order will be the same
                y1_jumpSum += y1_samp[jump] * state1JumpProbs[jump]
                y2_jumpSum += y2_samp[jump] * state1JumpProbs[jump]

            assert np.allclose(y2_jumpSum, np.dot(g.cartrot, y1_jumpSum))

    def test_Train_JPINN_1Jump_FullSym_Vac_and_Others(self):
        specCheck = self.specCheck
        specsToTrain = [specCheck]
        VacSpec = self.VacSpec
        N_check = 50
        AllJumps = False
        State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.AllJumpRates_st1,  self.JumpNewSites, self.dxJumps, self.NNsiteList, N_check,
                            AllJumps=AllJumps, mode="train")

        # Everything related to JPINN will be None
        jProbs_st1 = self.AllJumpRates_st1 / np.sum(self.AllJumpRates_st1, axis=1).reshape(-1, 1)
        jProbs_st2 = self.AllJumpRates_st1 / np.sum(self.AllJumpRates_st1, axis=1).reshape(-1, 1)

        start_ep = 0
        end_ep = 0
        interval = 100  # don't save networks
        dirPath = "."

        # Make the network
        specs = np.unique(self.state1List[0])
        NSpec = specs.shape[0] - 1
        gNet = GCNet(self.GnnPerms.long(), self.NNsites, self.JumpVecs, N_ngb=self.N_ngb, NSpec=NSpec,
                     mean=0.02, std=0.2, nl=1, nch=8, nchLast=self.z).double()

        sd = gNet.state_dict()

        gNet2 = GCNet(self.GnnPerms.long(), self.NNsites, self.JumpVecs, N_ngb=self.N_ngb, NSpec=NSpec,
                      mean=0.02, std=0.2, nl=1, nch=8, nchLast=self.z).double()

        # Get the original network
        gNet2.load_state_dict(sd)

        # Compute vectors with only vacancy sites
        y1, y2 = Train(self.T, dirPath, State1_occs, State2_occs, OnSites_state1, OnSites_state2, rates, disps,
                       jProbs_st1, jProbs_st2, specsToTrain, sp_ch, VacSpec, start_ep, end_ep, interval,
                       N_check, gNet, lRate=0.001, batch_size=N_check, scratch_if_no_init=True, chkpt=False,
                       Boundary_train=True, jumpSort=True, AddOnSites=True)
        print("Max, min and avg values")
        print(np.max(y1), np.max(y2))
        print(np.min(y1), np.min(y2))
        print(np.mean(y1), np.mean(y2))

        # Now for each sample, compute the y explicitly and match
        for samp in tqdm(range(N_check)):
            state1Input = pt.tensor(State1_occs[samp:samp + 1], dtype=pt.double)
            state2Input = pt.tensor(State2_occs[samp:samp + 1], dtype=pt.double)
            state1JumpProbs = np.sort(jProbs_st1[samp])
            state2JumpProbs = np.sort(jProbs_st2[samp])

            with pt.no_grad():
                y1SampAllSites = gNet2(state1Input).detach().numpy()
                y2SampAllSites = gNet2(state2Input).detach().numpy()

            # Now sum explicitly with jump probs
            y1_samp = -y1SampAllSites[0, :, :, 0]
            y2_samp = -y2SampAllSites[0, :, :, 0]

            for site in range(self.Nsites):
                occs1 = State1_occs[samp, :, site]
                occs2 = State2_occs[samp, :, site]
                if occs1[self.sp_ch[specCheck]] == 1:
                    y1_samp += y1SampAllSites[0, :, :, site]
                if occs2[self.sp_ch[specCheck]] == 1:
                    y2_samp += y2SampAllSites[0, :, :, site]

            y1_jumpSum = np.zeros(3)
            y2_jumpSum = np.zeros(3)

            for jump in range(self.z):
                y1_jumpSum += y1_samp[jump] * state1JumpProbs[jump]
                y2_jumpSum += y2_samp[jump] * state2JumpProbs[jump]

            self.assertTrue(np.allclose(y1[samp], y1_jumpSum),
                            msg="{} \n {}".format(y1[samp], y1_jumpSum))
            self.assertTrue(np.allclose(y2[samp], y2_jumpSum))

        # Now check for symmetry
        state1Input = pt.tensor(State1_occs[:1], dtype=pt.double)
        state1JumpProbs = np.sort(jProbs_st1[0])

        with pt.no_grad():
            y1SampAllSites = gNet2(state1Input).detach().numpy()

        for g in tqdm(self.superCell.crys.G, position=0, leave=True, ncols=65):
            state2Input = pt.zeros_like(state1Input)
            for site in range(1, self.Nsites):
                ci, R = self.superCell.ciR(site)
                Rnew, ciNew = self.superCell.crys.g_pos(g, R, ci)
                siteNew, _ = self.superCell.index(Rnew, ciNew)
                state2Input[:, :, siteNew] = state1Input[:, :, site]

            if np.allclose(g.cartrot, np.eye(3)):
                assert np.array_equal(state2Input, state1Input)

            with pt.no_grad():
                y2SampAllSites = gNet2(state2Input).detach().numpy()
            for site in range(self.Nsites):
                ci, R = self.superCell.ciR(site)
                Rnew, ciNew = self.superCell.crys.g_pos(g, R, ci)
                siteNew, _ = self.superCell.index(Rnew, ciNew)
                for jmp in range(self.z):
                    self.assertTrue(np.allclose(
                        y2SampAllSites[0, jmp, :, siteNew], np.dot(g.cartrot, y1SampAllSites[0, jmp, :, site])
                    )
                    )

            # Now sum explicitly with jump probs
            y1_samp = -y1SampAllSites[0, :, :, 0]
            y2_samp = -y2SampAllSites[0, :, :, 0]
            for site in range(self.Nsites):
                occs1 = State1_occs[0, :, site]
                occs2 = state2Input[0, :, site]
                if occs1[self.sp_ch[specCheck]] == 1:
                    y1_samp += y1SampAllSites[0, :, :, site]
                if occs2[self.sp_ch[specCheck]] == 1:
                    y2_samp += y2SampAllSites[0, :, :, site]
            y1_jumpSum = np.zeros(3)
            y2_jumpSum = np.zeros(3)

            for jump in range(self.z):
                # Since states are symmetric, jump probs in sorted order will be the same
                y1_jumpSum += y1_samp[jump] * state1JumpProbs[jump]
                y2_jumpSum += y2_samp[jump] * state1JumpProbs[jump]

            assert np.allclose(y2_jumpSum, np.dot(g.cartrot, y1_jumpSum))



