import unittest
import os
import sys
RunPath = os.getcwd() + "/"
CrysDatPath = "../CrysDat_FCC/CrystData.h5"
Data1 = "Test_Data/testData_HEA.h5" # test data set of HEA at 1073 K.
Data2 = "Test_Data/testData_SR2.h5" # test data set of SR2 at 60% c0.

import numpy as np
import h5py
import torch as pt
from tqdm import tqdm
import pickle
from onsager import crystal, supercell
from GCNetRun import Load_Data, makeComputeData, makeDataTensors, Load_crysDats
from GCNetRun import Train
from SymmLayers import GCNet


class TestGCNetRun_HEA_collective(unittest.TestCase):
    def setUp(self):
        self.T = 1073

        self.state1List, self.state2List, self.dispList, self.rateList, self.AllJumpRates_st1,\
        self.AllJumpRates_st2, self.JumpSelects =\
            Load_Data(Data1)

        self.GpermNNIdx, self.NNsiteList, self.JumpNewSites, self.dxJumps = Load_crysDats(CrysDatPath)

        with h5py.File(CrysDatPath, "r") as fl:
            lattice = np.array(fl["Lattice_basis_vectors"])
            superlatt = np.array(fl["SuperLatt"])

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
            State1_occs, State2_occs, rates, disps, GatherTensor_tracers, OnSites_state1, OnSites_state2, sp_ch = \
                makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                                self.JumpSelects, self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps,
                                self.NNsiteList, N_train, tracers=False, AllJumps=AllJumps, mode=m)

            self.assertTrue(GatherTensor_tracers is None) # check that no gathering tracer is built for collective training.

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
                        self.assertTrue(OnSites_state1[samp, site] == 0)
                        self.assertTrue(OnSites_state2[samp, site] == 0)
                    else:
                        spec1 = self.state1List[samp, site]
                        self.assertEqual(State1_occs[samp, self.sp_ch[spec1], site], 1)
                        self.assertEqual(np.sum(State1_occs[samp, :, site]), 1)
                        if spec1 == self.specCheck:
                            self.assertTrue(OnSites_state1[samp, site] == 1)
                        else:
                            self.assertTrue(OnSites_state1[samp, site] == 0)

                        spec2 = self.state2List[samp, site]
                        self.assertEqual(State2_occs[samp, self.sp_ch[spec2], site], 1)
                        self.assertEqual(np.sum(State2_occs[samp, :, site]), 1)
                        if spec2 == self.specCheck:
                            self.assertTrue(OnSites_state2[samp, site] == 1)
                        else:
                            self.assertTrue(OnSites_state2[samp, site] == 0)

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
                xsiteVac = self.dxJumps[jSelect]
                RsiteVac, ciSiteVac = self.superCell.crys.cart2pos(xsiteVac)
                NN_sup, _ = self.superCell.index(RsiteVac, ciSiteVac)
                self.assertEqual(NN_sup, NNsiteVac)
                self.assertEqual(state2[0], self.VacSpec)
                state2[0] = state2[NNsiteVac]
                state2[NNsiteVac] = self.VacSpec

                for site in range(self.state1List.shape[1]):
                    ciSite, Rsite = self.superCell.ciR(site)
                    x_site = self.superCell.crys.pos2cart(Rsite, ciSite)
                    x_site_new = x_site + xsiteVac
                    RsiteNew, ciSiteNew = self.superCell.crys.cart2pos(x_site_new)
                    siteNew, _ = self.superCell.index(RsiteNew, ciSiteNew)
                    spec1 = state1[site]
                    spec2 = state2[siteNew]

                    if site == 0:
                        self.assertEqual(spec2, self.VacSpec)
                        self.assertTrue(np.all(State1_occs[samp, :, site] == 0))
                        self.assertTrue(np.all(State2_occs[samp, :, site] == 0))

                    else:
                        self.assertNotEqual(spec2, self.VacSpec)
                        self.assertEqual(State1_occs[samp, self.sp_ch[spec1], site], 1)
                        self.assertEqual(np.sum(State1_occs[samp, :, site]), 1)

                        self.assertEqual(State2_occs[samp, self.sp_ch[spec2], site], 1)
                        self.assertEqual(np.sum(State2_occs[samp, :, site]), 1)

                # check displacement recording
                spec = self.state1List[samp, NNsiteVac]
                assert np.allclose(disps[samp, 0], self.a0 * self.dxJumps[jSelect])
                # Check that displacements are okay
                if spec == specCheck:
                    self.assertTrue(np.allclose(disps[samp, 1], -self.a0 * self.dxJumps[jSelect]))
                else:
                    self.assertTrue(np.allclose(disps[samp, 1], 0.0))

                # check the rate
                self.assertTrue(np.math.isclose(rates[samp], self.rateList[samp]))

        print("Done single jump data construction test")

    def test_makeDataTensor(self):
        specCheck = self.specCheck
        specsToTrain = [specCheck]
        VacSpec = self.VacSpec
        N_train = 1000
        AllJumps = False

        State1_occs, State2_occs, rates, disps, GatherTensor_tracers, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.JumpSelects, self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps,
                            self.NNsiteList, N_train, tracers=False, AllJumps=AllJumps, mode="train")

        self.assertTrue(GatherTensor_tracers is None)

        state1Data, state2Data, dispData, rateData, On_st1, On_st2 = \
            makeDataTensors(State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2,
                            specsToTrain, VacSpec, sp_ch, Ndim=3)

        self.assertTrue(pt.equal(pt.tensor(State1_occs), state1Data))
        self.assertTrue(pt.equal(pt.tensor(State2_occs), state2Data))

        self.assertTrue(pt.allclose(rateData, pt.tensor(rates)))

        self.assertTrue(pt.allclose(dispData, pt.tensor(disps[:, 1, :])))

        specsToTrain=[VacSpec]

        State1_occs, State2_occs, rates, disps, GatherTensor_tracers, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.JumpSelects, self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps,
                            self.NNsiteList, N_train, tracers=False, AllJumps=AllJumps, mode="train")

        state1Data, state2Data, dispData, rateData, On_st1, On_st2 = \
            makeDataTensors(State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2,
                            specsToTrain, VacSpec, sp_ch, Ndim=3)

        self.assertTrue(pt.equal(pt.tensor(State1_occs), state1Data))
        self.assertTrue(pt.equal(pt.tensor(State2_occs), state2Data))

        self.assertTrue(pt.allclose(rateData, pt.tensor(rates)))

        print(dispData.shape, disps.shape)
        self.assertTrue(pt.allclose(dispData, pt.tensor(disps[:, 0, :])))

    def test_Train_Batch_SpNN_1Jump(self):
        specCheck = self.specCheck
        specsToTrain = [specCheck]
        VacSpec = self.VacSpec
        N_check = 50
        AllJumps = False
        State1_occs, State2_occs, rates, disps, GatherTensor_tracers, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.JumpSelects, self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps,
                            self.NNsiteList, N_check, tracers=False, AllJumps=AllJumps, mode="train")

        self.assertTrue(GatherTensor_tracers is None)

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

        diff, y1, y2 = Train(self.T, dirPath, State1_occs, State2_occs, OnSites_state1, OnSites_state2, rates, disps,
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
        State1_occs, State2_occs, rates, disps, GatherTensor_tracers, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.JumpSelects, self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps,
                            self.NNsiteList, N_check, tracers=False, AllJumps=AllJumps, mode="train")

        self.assertTrue(GatherTensor_tracers is None)

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

        diff, y1, y2 = Train(self.T, dirPath, State1_occs, State2_occs, OnSites_state1, OnSites_state2, rates, disps,
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
        N_train = 500
        AllJumps = True

        for m in ["train", "all"]:
            print("testing mode : {}".format(m))

            State1_occs, State2_occs, rates, disps, GatherTensor_tracers, OnSites_state1, OnSites_state2, sp_ch = \
                makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                                self.JumpSelects, self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps,
                                self.NNsiteList, N_train, tracers=False, AllJumps=AllJumps, mode=m)

            self.assertTrue(GatherTensor_tracers is None)  # check that no gathering tracer is built for collective training.

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
                    xsiteVac = self.dxJumps[jInd]

                    RsiteVac, ciSiteVac = self.superCell.crys.cart2pos(xsiteVac)
                    NN_sup, _ = self.superCell.index(RsiteVac, ciSiteVac)
                    self.assertEqual(NN_sup, NNsiteVac)

                    self.assertEqual(state2[0], self.VacSpec)
                    state2[0] = state2[NNsiteVac]
                    state2[NNsiteVac] = self.VacSpec

                    for site in range(self.state1List.shape[1]):
                        ciSite, Rsite = self.superCell.ciR(site)
                        xSite = self.superCell.crys.pos2cart(Rsite, ciSite)
                        xSiteNew = xSite + self.dxJumps[jInd]
                        RsiteNew, ciSiteNew = self.superCell.crys.cart2pos(xSiteNew)
                        siteNew, _ = self.superCell.index(RsiteNew, ciSiteNew)
                        spec1 = state1[site]
                        spec2 = state2[siteNew]

                        if site == 0:
                            self.assertEqual(spec2, self.VacSpec)
                            self.assertTrue(np.all(State1_occs[stateInd * self.dxJumps.shape[0] + jInd, :, site] == 0))
                            self.assertTrue(np.all(State2_occs[stateInd * self.dxJumps.shape[0] + jInd, :, site] == 0))
                            self.assertEqual(OnSites_state1[stateInd * self.dxJumps.shape[0] + jInd, site], 0)
                            self.assertEqual(OnSites_state2[stateInd * self.dxJumps.shape[0] + jInd, site], 0)

                        else:
                            self.assertNotEqual(spec2, self.VacSpec)
                            self.assertEqual(State1_occs[stateInd * self.dxJumps.shape[0] + jInd, self.sp_ch[spec1], site], 1)
                            self.assertEqual(np.sum(State1_occs[stateInd * self.dxJumps.shape[0] + jInd, :, site]), 1)
                            if spec1 == self.specCheck:
                                self.assertEqual(OnSites_state1[stateInd * self.dxJumps.shape[0] + jInd, site], 1)
                            else:
                                self.assertEqual(OnSites_state1[stateInd * self.dxJumps.shape[0] + jInd, site], 0)

                            self.assertEqual(State2_occs[stateInd * self.dxJumps.shape[0] + jInd, self.sp_ch[spec2], site], 1)
                            self.assertEqual(np.sum(State2_occs[stateInd * self.dxJumps.shape[0] + jInd, :, site]), 1)
                            if spec2 == self.specCheck:
                                self.assertEqual(OnSites_state2[stateInd * self.dxJumps.shape[0] + jInd, site], 1)
                            else:
                                self.assertEqual(OnSites_state2[stateInd * self.dxJumps.shape[0] + jInd, site], 0)

                    # check the displacements
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

        State1_occs, State2_occs, rates, disps, GatherTensor_tracers, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.JumpSelects, self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps,
                            self.NNsiteList, N_check, tracers=False, AllJumps=AllJumps, mode="train")

        self.assertTrue(GatherTensor_tracers is None)  # check that no gathering tracer is built for collective training.

        # State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
        #     makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
        #                     self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps, self.NNsiteList, N_check,
        #                     AllJumps=AllJumps, mode="train")

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

        diff, y1, y2 = Train(self.T, dirPath, State1_occs, State2_occs, OnSites_state1, OnSites_state2, rates, disps,
                              jProbs_st1, jProbs_st2, specsToTrain, sp_ch, VacSpec, start_ep, end_ep,
                              interval, N_check * self.z, gNet,
                              lRate=0.001, batch_size=N_check * self.z, scratch_if_no_init=True, chkpt=False)

        print("Max, min and avg values")
        print(np.max(y1), np.max(y2))
        print(np.min(y1), np.min(y2))
        print(np.mean(y1), np.mean(y2))

        # Now for each sample, compute the y explicitly and match
        for samp in tqdm(range(N_check * self.z)):
            # The occupancies were checked in test_makeComputeData_AllJumps
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

        State1_occs, State2_occs, rates, disps, GatherTensor_tracers, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.JumpSelects, self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps,
                            self.NNsiteList, N_check, tracers=False, AllJumps=AllJumps, mode="train")

        self.assertTrue(GatherTensor_tracers is None)  # check that no gathering tracer is built for collective training.
        # State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
        #     makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
        #                     self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps, self.NNsiteList, N_check,
        #                     AllJumps=AllJumps, mode="train")

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

        diff, y1, y2 = Train(self.T, dirPath, State1_occs, State2_occs, OnSites_state1, OnSites_state2, rates, disps,
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

        State1_occs, State2_occs, rates, disps, GatherTensor_tracers, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.JumpSelects, self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps,
                            self.NNsiteList, N_check, tracers=False, AllJumps=AllJumps, mode="train")

        self.assertTrue(GatherTensor_tracers is None)

        # State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = \
        #     makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
        #                     self.AllJumpRates_st1,  self.JumpNewSites, self.dxJumps, self.NNsiteList, N_check,
        #                     AllJumps=AllJumps, mode="train")

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

        # Compute vectors with only vacancy sites (AddOnSites=False)
        diff, y1, y2 = Train(self.T, dirPath, State1_occs, State2_occs, OnSites_state1, OnSites_state2, rates, disps,
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

        State1_occs, State2_occs, rates, disps, GatherTensor_tracers, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.JumpSelects, self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps,
                            self.NNsiteList, N_check, tracers=False, AllJumps=AllJumps, mode="train")

        self.assertTrue(GatherTensor_tracers is None)

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

        # Compute vectors with vacancy sites and other occupied sites (AddOnsites = True).
        diff, y1, y2 = Train(self.T, dirPath, State1_occs, State2_occs, OnSites_state1, OnSites_state2, rates, disps,
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


class TestGCNetRun_HEA_collective_orthogonal(TestGCNetRun_HEA_collective):
    def setUp(self):
        self.T = 1173

        self.state1List, self.state2List, self.dispList, self.rateList, self.AllJumpRates_st1, \
        self.AllJumpRates_st2, self.JumpSelects = \
            Load_Data("Test_Data/testData_HEA_MEAM_orthogonal.h5")

        self.GpermNNIdx, self.NNsiteList, self.JumpNewSites, self.dxJumps =\
            Load_crysDats("../CrysDat_FCC/CrystData_ortho_5_cube.h5")

        with h5py.File("../CrysDat_FCC/CrystData_ortho_5_cube.h5", "r") as fl:
            lattice = np.array(fl["Lattice_basis_vectors"])
            superlatt = np.array(fl["SuperLatt"])
            basis_cubic = np.array(fl["basis_sites"])

        crys = crystal.Crystal(lattice=lattice, basis=[[b for b in basis_cubic]], chemistry=["A"], noreduce=True)
        self.superCell = supercell.ClusterSupercell(crys, superlatt)

        self.N_units = self.superCell.superlatt[0, 0]

        self.z = self.dxJumps.shape[0]
        self.N_ngb = self.NNsiteList.shape[0]

        self.GnnPerms = pt.tensor(self.GpermNNIdx).long()

        print("Filter neighbor range : {}nn. Filter neighborhood size: {}".format(1, self.N_ngb - 1))
        self.Nsites = self.NNsiteList.shape[1]

        self.assertEqual(self.Nsites, self.state1List.shape[1])
        self.assertEqual(self.Nsites, self.state2List.shape[1])
        self.assertEqual(self.Nsites, 500)

        self.a0 = 3.595
        self.NNsites = pt.tensor(self.NNsiteList).long()
        self.JumpVecs = pt.tensor(self.dxJumps.T * self.a0, dtype=pt.double)
        print("Jump Vectors: \n", self.JumpVecs.T, "\n")
        self.Ndim = self.dxJumps.shape[1]

        self.specCheck = 5
        self.VacSpec = 0
        self.sp_ch = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        print("Setup complete.")


# In the previous test, the vacancy had the lowest species. Now we make it the largest like in our lattice gas data
class TestGCNetRun_binary_collective(TestGCNetRun_HEA_collective):
    def setUp(self):
        self.T = 60
        self.state1List, self.state2List, self.dispList, self.rateList, self.AllJumpRates_st1, \
        self.AllJumpRates_st2, self.JumpSelects = \
            Load_Data(Data2)

        self.GpermNNIdx, self.NNsiteList, self.JumpNewSites, self.dxJumps = Load_crysDats(CrysDatPath)

        with h5py.File(CrysDatPath, "r") as fl:
            lattice = np.array(fl["Lattice_basis_vectors"])
            superlatt = np.array(fl["SuperLatt"])

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

        self.a0 = 1.0
        self.NNsites = pt.tensor(self.NNsiteList).long()
        self.JumpVecs = pt.tensor(self.dxJumps.T * self.a0, dtype=pt.double)
        print("Jump Vectors: \n", self.JumpVecs.T, "\n")
        self.Ndim = self.dxJumps.shape[1]

        self.specCheck = 1
        self.VacSpec = 2
        self.sp_ch = {0: 0, 1: 1}
        print("Setup complete.")


class TestGCNetRun_HEA_tracers(unittest.TestCase):
    def setUp(self):
        self.T = 1073

        self.state1List, self.state2List, self.dispList, self.rateList, self.AllJumpRates_st1, \
        self.AllJumpRates_st2, self.JumpSelects = \
            Load_Data(Data1)

        self.GpermNNIdx, self.NNsiteList, self.JumpNewSites, self.dxJumps = Load_crysDats(CrysDatPath)

        with h5py.File(CrysDatPath, "r") as fl:
            lattice = np.array(fl["Lattice_basis_vectors"])
            superlatt = np.array(fl["SuperLatt"])

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
        N_train = 500
        AllJumps = False

        for m in ["train", "all"]:
            print("testing mode : {}".format(m))
            State1_occs, State2_occs, rates, disps, GatherTensor_tracers, OnSites_state1, OnSites_state2, sp_ch = \
                makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                                self.JumpSelects, self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps,
                                self.NNsiteList, N_train, tracers=True, AllJumps=AllJumps, mode=m)

            self.assertTrue(GatherTensor_tracers is not None)  # check that gathering tracer is built for tracers.

            if m == "all":
                self.assertTrue(State1_occs.shape[0] == self.state1List.shape[0] == State2_occs.shape[0])
                self.assertTrue(rates.shape[0] == self.state1List.shape[0] == disps.shape[0])
                self.assertTrue(OnSites_state1.shape[0] == self.state1List.shape[0] == OnSites_state2.shape[0])
                self.assertTrue(GatherTensor_tracers.shape[0] == self.state1List.shape[0])
                sampsCheck = np.random.randint(0, State1_occs.shape[0], N_check)

            else:
                self.assertTrue(State1_occs.shape[0] == N_train == State2_occs.shape[0])
                self.assertTrue(rates.shape[0] == N_train == disps.shape[0])
                self.assertTrue(OnSites_state1.shape[0] == N_train == OnSites_state2.shape[0])
                self.assertTrue(GatherTensor_tracers.shape[0] == N_train)
                sampsCheck = np.random.randint(0, N_train, N_check)

            for samp in tqdm(sampsCheck, position=0, leave=True):
                for site in range(self.state1List.shape[1]):
                    if site == 0:
                        self.assertTrue(np.all(State1_occs[samp, :, site] == 0))
                        self.assertTrue(np.all(State2_occs[samp, :, site] == 0))
                        self.assertTrue(OnSites_state1[samp, site] == 0)
                        self.assertTrue(np.all(GatherTensor_tracers[samp, :, site] == 0))
                    else:
                        spec1 = self.state1List[samp, site]
                        self.assertEqual(State1_occs[samp, self.sp_ch[spec1], site], 1)
                        self.assertEqual(np.sum(State1_occs[samp, :, site]), 1)
                        if spec1 == self.specCheck:
                            self.assertTrue(OnSites_state1[samp, site] == 1)

                        spec2 = self.state2List[samp, site]
                        self.assertEqual(State2_occs[samp, self.sp_ch[spec2], site], 1)
                        self.assertEqual(np.sum(State2_occs[samp, :, site]), 1)
                        if spec2 == self.specCheck:
                            self.assertTrue(np.all(OnSites_state2[samp, site] == 1))

                        # check the Gather tensor
                        jSelect = self.JumpSelects[samp]
                        sourceSite = self.JumpNewSites[jSelect, site]
                        self.assertTrue(np.all(GatherTensor_tracers[samp, :, sourceSite] == site),
                                        msg="\n{} {} {}".format(site, sourceSite, GatherTensor_tracers[samp, :, site]))

                # check the displacements
                jSelect = None
                count = 0
                for jInd in range(self.JumpNewSites.shape[0]):
                    state2_try = self.state1List[samp][self.JumpNewSites[jInd]]
                    if np.all(state2_try == self.state2List[samp]):
                        jSelect = jInd
                        count += 1
                self.assertEqual(count, 1)  # there should be exactly one match

                # Check occupancies using supercell
                state1 = self.state1List[samp]
                state2 = state1.copy()
                NNsiteVac = self.NNsiteList[jSelect + 1, 0]
                ciSiteVacNN, RsiteVacNN = self.superCell.ciR(NNsiteVac)
                xVacNN = self.superCell.crys.pos2cart(RsiteVacNN, ciSiteVacNN)
                state2[0] = state2[NNsiteVac]
                state2[NNsiteVac] = self.VacSpec

                for site in range(self.state1List.shape[1]):
                    ciSite, Rsite = self.superCell.ciR(site)
                    xSite = self.superCell.crys.pos2cart(Rsite, ciSite)
                    xSiteNew = xSite + xVacNN
                    RsiteNew, ciSiteNew = self.superCell.crys.cart2pos(xSiteNew)
                    siteNew, _ = self.superCell.index(RsiteNew, ciSiteNew)
                    spec1 = state1[site]
                    spec2 = state2[siteNew]

                    if site == 0:
                        self.assertEqual(spec2, self.VacSpec)
                        self.assertTrue(np.all(State1_occs[samp, :, site] == 0))
                        self.assertTrue(np.all(State2_occs[samp, :, site] == 0))

                    else:
                        self.assertNotEqual(spec2, self.VacSpec)
                        self.assertEqual(State1_occs[samp, self.sp_ch[spec1], site], 1)
                        self.assertEqual(np.sum(State1_occs[samp, :, site]), 1)

                        self.assertEqual(State2_occs[samp, self.sp_ch[spec2], site], 1)
                        self.assertEqual(np.sum(State2_occs[samp, :, site]), 1)

                # check the rate
                self.assertTrue(np.math.isclose(rates[samp], self.rateList[samp]))

                # check the displacement - only the jumping site will move
                for site in range(self.state1List.shape[1]):
                    if site == NNsiteVac:
                        self.assertTrue(np.allclose(disps[samp, :, site], -self.dxJumps[jSelect] * self.a0))
                    else:
                        self.assertTrue(np.allclose(disps[samp, :, site], 0))

        print("Done single jump data construction test")

    def test_makeComputeData_AllJumps(self):
        specCheck = self.specCheck
        specsToTrain = [specCheck]
        VacSpec = self.VacSpec
        N_check = 200
        N_train = 500
        AllJumps = True

        for m in ["train", "all"]:
            print("testing mode : {}".format(m))
            State1_occs, State2_occs, rates, disps, GatherTensor_tracers, OnSites_state1, OnSites_state2, sp_ch = \
                makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                                self.JumpSelects, self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps,
                                self.NNsiteList, N_train, tracers=True, AllJumps=AllJumps, mode=m)

            self.assertTrue(GatherTensor_tracers is not None)  # check that no gathering tracer is built for collective training.

            if m == "all":
                self.assertTrue(State1_occs.shape[0] == self.state1List.shape[0] * self.z == State2_occs.shape[0])
                self.assertTrue(rates.shape[0] == self.state1List.shape[0] * self.z == disps.shape[0])
                self.assertTrue(OnSites_state1.shape[0] == self.state1List.shape[0] * self.z == OnSites_state2.shape[0])
                self.assertTrue(GatherTensor_tracers.shape[0] == self.state1List.shape[0] * self.z)
                sampsCheck = np.random.randint(0, self.state1List.shape[0], N_check)

            else:
                self.assertTrue(State1_occs.shape[0] == N_train * self.z == State2_occs.shape[0])
                self.assertTrue(rates.shape[0] == N_train * self.z == disps.shape[0])
                self.assertTrue(OnSites_state1.shape[0] == N_train * self.z == OnSites_state2.shape[0])
                self.assertTrue(GatherTensor_tracers.shape[0] == N_train * self.z)
                sampsCheck = np.random.randint(0, N_train, N_check)

            for stateInd in tqdm(sampsCheck, position=0, leave=True):
                state1 = self.state1List[stateInd]
                for jInd in range(self.z):
                    state2 = state1.copy()
                    NNsiteVac = self.NNsiteList[jInd + 1, 0]
                    ciSiteVacNN, RsiteVacNN = self.superCell.ciR(NNsiteVac)
                    xVacNN = self.superCell.crys.pos2cart(RsiteVacNN, ciSiteVacNN)
                    state2[0] = state2[NNsiteVac]
                    state2[NNsiteVac] = self.VacSpec

                    for site in range(self.state1List.shape[1]):
                        ciSite, Rsite = self.superCell.ciR(site)
                        xSite = self.superCell.crys.pos2cart(Rsite, ciSite)
                        xSiteNew = xSite + xVacNN
                        RsiteNew, ciSiteNew = self.superCell.crys.cart2pos(xSiteNew)
                        siteNew, _ = self.superCell.index(RsiteNew, ciSiteNew)
                        spec1 = state1[site]
                        spec2 = state2[siteNew]

                        if site == 0:
                            self.assertEqual(spec2, self.VacSpec)
                            self.assertTrue(np.all(State1_occs[stateInd * self.dxJumps.shape[0] + jInd, :, site] == 0))
                            self.assertTrue(np.all(State2_occs[stateInd * self.dxJumps.shape[0] + jInd, :, site] == 0))
                            self.assertEqual(OnSites_state1[stateInd * self.dxJumps.shape[0] + jInd, site], 0)
                            self.assertEqual(OnSites_state2[stateInd * self.dxJumps.shape[0] + jInd, site], 0)
                            self.assertTrue(np.all(GatherTensor_tracers[stateInd * self.dxJumps.shape[0] + jInd, :, site] == 0))

                        else:
                            self.assertNotEqual(spec2, self.VacSpec)
                            self.assertEqual(State1_occs[stateInd * self.dxJumps.shape[0] + jInd, self.sp_ch[spec1], site], 1)
                            self.assertEqual(np.sum(State1_occs[stateInd * self.dxJumps.shape[0] + jInd, :, site]), 1)
                            if spec1 == self.specCheck:
                                self.assertEqual(OnSites_state1[stateInd * self.dxJumps.shape[0] + jInd, site], 1)
                            else:
                                self.assertEqual(OnSites_state1[stateInd * self.dxJumps.shape[0] + jInd, site], 0)

                            self.assertEqual(State2_occs[stateInd * self.dxJumps.shape[0] + jInd, self.sp_ch[spec2], site], 1)
                            self.assertEqual(np.sum(State2_occs[stateInd * self.dxJumps.shape[0] + jInd, :, site]), 1)
                            if spec2 == self.specCheck:
                                self.assertEqual(OnSites_state2[stateInd * self.dxJumps.shape[0] + jInd, site], 1)
                            else:
                                self.assertEqual(OnSites_state2[stateInd * self.dxJumps.shape[0] + jInd, site], 0)

                            # check the Gather tensor
                            sourceSite = self.JumpNewSites[jInd, site]
                            self.assertTrue(np.all(GatherTensor_tracers[stateInd * self.dxJumps.shape[0] + jInd, :, sourceSite] == site))

                    # check the rate
                    self.assertTrue(np.math.isclose(rates[stateInd * self.dxJumps.shape[0] + jInd],
                                                    self.AllJumpRates_st1[stateInd, jInd]))

                    # check the displacement - only the jumping site will move
                    for site in range(self.state1List.shape[1]):
                        if site == NNsiteVac:
                            self.assertTrue(np.allclose(disps[stateInd * self.dxJumps.shape[0] + jInd, :, site],
                                                        -self.dxJumps[jInd] * self.a0))

                        else:
                            self.assertTrue(np.allclose(disps[stateInd * self.dxJumps.shape[0] + jInd, :, site],
                                                        0))

    def test_makeDataTensor(self):
        specCheck = self.specCheck
        specsToTrain = [specCheck]
        VacSpec = self.VacSpec
        N_train = 1000
        AllJumps = False

        State1_occs, State2_occs, rates, disps, GatherTensor_tracers, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.JumpSelects, self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps,
                            self.NNsiteList, N_train, tracers=True, AllJumps=AllJumps, mode="train")

        self.assertTrue(GatherTensor_tracers is not None)

        state1Data, state2Data, dispData, rateData, On_st1, On_st2 = \
            makeDataTensors(State1_occs, State2_occs, rates, disps, OnSites_state1, OnSites_state2,
                            specsToTrain, VacSpec, sp_ch, Ndim=3, tracers=True)

        self.assertTrue(pt.equal(pt.tensor(State1_occs), state1Data))
        self.assertTrue(pt.equal(pt.tensor(State2_occs), state2Data))

        self.assertTrue(pt.allclose(rateData, pt.tensor(rates)))

        print("{} {}".format(dispData.shape, disps.shape))

        self.assertTrue(pt.allclose(dispData, pt.tensor(disps).double()))

    def test_Train_Batch_SpNN_1Jump(self):
        specCheck = self.specCheck
        specsToTrain = [specCheck]
        VacSpec = self.VacSpec
        N_check = 300
        AllJumps = False
        State1_occs, State2_occs, rates, disps, GatherTensor_tracers, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.JumpSelects, self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps,
                            self.NNsiteList, N_check, tracers=True, AllJumps=AllJumps, mode="train")

        self.assertTrue(GatherTensor_tracers is not None)

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

        diff, y1, y2 = Train(self.T, dirPath, State1_occs, State2_occs, OnSites_state1, OnSites_state2, rates, disps,
                       jProbs_st1, jProbs_st2, specsToTrain, sp_ch, VacSpec, start_ep, end_ep, interval,
                       N_check, gNet, lRate=0.001, batch_size=N_check, scratch_if_no_init=True, chkpt=False,
                       tracers=True, GatherTensor=GatherTensor_tracers)

        print("Max, min and avg values")
        print(np.max(y1), np.max(y2))
        print(np.min(y1), np.min(y2))
        print(np.mean(y1), np.mean(y2))
        #
        # Now for each sample, compute the y explicitly and match
        NNs_vac = self.NNsiteList[1:, 0]
        z = np.zeros(3)
        diff_total = 0
        for samp in tqdm(range(N_check)):
            state1Input = pt.tensor(State1_occs[samp:samp + 1], dtype=pt.double)
            state2Input = pt.tensor(State2_occs[samp:samp + 1], dtype=pt.double)

            with pt.no_grad():
                y1SampAllSites = gNet2(state1Input)
                y2SampAllSites = gNet2(state2Input)

            self.assertTrue(np.allclose(y1SampAllSites[0, 0, :, :].detach().numpy(), y1[samp], rtol=0, atol=1e-8))
            self.assertTrue(np.allclose(y2SampAllSites[0, 0, :, :].detach().numpy(), y2[samp], rtol=0, atol=1e-8))

            # Now compute tracer coefficients explicitly
            diff_tracers = 0.
            rate_samp = rates[samp]
            count = 0
            for site in range(1, self.state1List.shape[1]):
                if self.state2List[samp, site] == specCheck:
                    count += 1
                    # get the source site
                    source = self.JumpNewSites[self.JumpSelects[samp], site]
                    self.assertEqual(self.state1List[samp, source], specCheck)
                    y1_site = y1[samp, :, source]
                    y2_site = y2[samp, :, site]

                    if source == NNs_vac[self.JumpSelects[samp]]: # check if this is the jumping site
                        dx = -self.dxJumps[self.JumpSelects[samp]] * self.a0
                    else:
                        dx = z

                    dxMod_sq = np.linalg.norm(dx + y2_site - y1_site)**2
                    diff_tracers += rate_samp * dxMod_sq / 6.

            self.assertEqual(np.sum(OnSites_state1[samp]), count)
            diff_total += diff_tracers #/ (1.0 * count)

        self.assertTrue(np.math.isclose(diff_total, diff, rel_tol=0, abs_tol=1e-8), msg="{} {} {}")

    def test_Train_Batch_SpNN_AllJump(self):
        specCheck = self.specCheck
        specsToTrain = [specCheck]
        VacSpec = self.VacSpec
        N_check = 30
        AllJumps = True
        State1_occs, State2_occs, rates, disps, GatherTensor_tracers, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(self.state1List, self.state2List, self.dispList, specsToTrain, VacSpec, self.rateList,
                            self.JumpSelects, self.AllJumpRates_st1, self.JumpNewSites, self.dxJumps,
                            self.NNsiteList, N_check, tracers=True, AllJumps=AllJumps, mode="train")

        self.assertTrue(GatherTensor_tracers is not None)

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

        # Get the diffusivities and y vectors
        diff, y1, y2 = Train(self.T, dirPath, State1_occs, State2_occs, OnSites_state1, OnSites_state2,
                             rates, disps, jProbs_st1, jProbs_st2, specsToTrain, sp_ch, VacSpec, start_ep,
                             end_ep, interval, N_check * self.z, gNet, lRate=0.001, batch_size=N_check * self.z,
                             scratch_if_no_init=True, chkpt=False, tracers=True, GatherTensor=GatherTensor_tracers)

        print(y1.shape, y2.shape)

        print("Max, min and avg values")
        print(np.max(y1), np.max(y2))
        print(np.min(y1), np.min(y2))
        print(np.mean(y1), np.mean(y2))
        #
        # Now for each sample, compute the y explicitly and match
        NNs_vac = self.NNsiteList[1:, 0]
        zero = np.zeros(3)
        diff_total = 0
        for stateInd in tqdm(range(N_check)):
            state1 = self.state1List[stateInd]
            for jInd in range(self.dxJumps.shape[0]):
                samp = stateInd * self.dxJumps.shape[0] + jInd
                state1Input = pt.tensor(State1_occs[samp:samp + 1], dtype=pt.double)
                state2Input = pt.tensor(State2_occs[samp:samp + 1], dtype=pt.double)

                with pt.no_grad():
                    y1SampAllSites = gNet2(state1Input)
                    y2SampAllSites = gNet2(state2Input)

                self.assertTrue(np.allclose(y1SampAllSites[0, 0, :, :].detach().numpy(), y1[samp], rtol=0, atol=1e-8))
                self.assertTrue(np.allclose(y2SampAllSites[0, 0, :, :].detach().numpy(), y2[samp], rtol=0, atol=1e-8))

                # Now compute tracer coefficients explicitly
                diff_tracers = 0.
                rate_samp = self.AllJumpRates_st1[stateInd, jInd]
                count = 0
                state2 = state1[self.JumpNewSites[jInd]]
                for site in range(1, self.state1List.shape[1]):
                    if state2[site] == specCheck:
                        count += 1
                        # get the source site
                        source = self.JumpNewSites[jInd, site]
                        self.assertEqual(state1[source], specCheck)
                        y1_site = y1[samp, :, source]
                        y2_site = y2[samp, :, site]

                        if source == NNs_vac[jInd]:  # check if this is the jumping site
                            dx = -self.dxJumps[jInd] * self.a0
                        else:
                            dx = zero

                        dxMod_sq = np.linalg.norm(dx + y2_site - y1_site)**2
                        diff_tracers += rate_samp * dxMod_sq / 6.

                self.assertEqual(np.sum(OnSites_state1[samp]), count)
                diff_total += diff_tracers #/ (1.0 * count)

        self.assertTrue(np.math.isclose(diff_total, diff, rel_tol=0, abs_tol=1e-8))


class TestGCNetRun_HEA_tracers_orthogonal(TestGCNetRun_HEA_tracers):
    def setUp(self):
        self.T = 1173

        self.state1List, self.state2List, self.dispList, self.rateList, self.AllJumpRates_st1, \
        self.AllJumpRates_st2, self.JumpSelects = \
            Load_Data("Test_Data/testData_HEA_MEAM_orthogonal.h5")

        self.GpermNNIdx, self.NNsiteList, self.JumpNewSites, self.dxJumps =\
            Load_crysDats("../CrysDat_FCC/CrystData_ortho_5_cube.h5")

        with h5py.File("../CrysDat_FCC/CrystData_ortho_5_cube.h5", "r") as fl:
            lattice = np.array(fl["Lattice_basis_vectors"])
            superlatt = np.array(fl["SuperLatt"])
            basis_cubic = np.array(fl["basis_sites"])

        crys = crystal.Crystal(lattice=lattice, basis=[[b for b in basis_cubic]], chemistry=["A"], noreduce=True)
        self.superCell = supercell.ClusterSupercell(crys, superlatt)

        self.N_units = self.superCell.superlatt[0, 0]

        self.z = self.dxJumps.shape[0]
        self.N_ngb = self.NNsiteList.shape[0]

        self.GnnPerms = pt.tensor(self.GpermNNIdx).long()

        print("Filter neighbor range : {}nn. Filter neighborhood size: {}".format(1, self.N_ngb - 1))
        self.Nsites = self.NNsiteList.shape[1]

        self.assertEqual(self.Nsites, self.state1List.shape[1])
        self.assertEqual(self.Nsites, self.state2List.shape[1])

        self.a0 = 3.595
        self.NNsites = pt.tensor(self.NNsiteList).long()
        self.JumpVecs = pt.tensor(self.dxJumps.T * self.a0, dtype=pt.double)
        print("Jump Vectors: \n", self.JumpVecs.T, "\n")
        self.Ndim = self.dxJumps.shape[1]

        self.specCheck = 5
        self.VacSpec = 0
        self.sp_ch = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        print("Setup complete.")

class TestGCNetRun_binary_tracers(TestGCNetRun_HEA_tracers):
    def setUp(self):
        self.T = 60
        self.state1List, self.state2List, self.dispList, self.rateList, self.AllJumpRates_st1, \
        self.AllJumpRates_st2, self.JumpSelects = \
            Load_Data(Data2)

        self.GpermNNIdx, self.NNsiteList, self.JumpNewSites, self.dxJumps = Load_crysDats(CrysDatPath)

        with h5py.File(CrysDatPath, "r") as fl:
            lattice = np.array(fl["Lattice_basis_vectors"])
            superlatt = np.array(fl["SuperLatt"])

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

        self.a0 = 1.0
        self.NNsites = pt.tensor(self.NNsiteList).long()
        self.JumpVecs = pt.tensor(self.dxJumps.T * self.a0, dtype=pt.double)
        print("Jump Vectors: \n", self.JumpVecs.T, "\n")
        self.Ndim = self.dxJumps.shape[1]

        self.specCheck = 1
        self.VacSpec = 2
        self.sp_ch = {0: 0, 1: 1}
        print("Setup complete.")
