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
from Tracer_corr_fac import Load_crysDats, make_SiteJumpDests, make_JumpGatherTensor, get_InputTensors, Train
from SymmLayers import GCNet

class Test_TCF(unittest.TestCase):
    def setUp(self):
        self.BatchSize = 2

        self.GpermNNIdx, self.NNsiteList, self.dxList, self.superCell = Load_crysDats(CrysDatPath)

        with h5py.File(CrysDatPath, "r") as fl:
            self.JumpNewSites = np.array(fl["JumpSiteIndexPermutation"])

        N_ngb = self.NNsiteList.shape[0]
        z = N_ngb - 1
        assert z == self.dxList.shape[0]
        self.Nsites = self.NNsiteList.shape[1]
        self.z = z
        self.N_ngb = N_ngb

        # Convert to tensors
        self.GnnPerms = pt.tensor(self.GpermNNIdx).long()
        self.NNsites = pt.tensor(self.NNsiteList).long()
        self.JumpVecs = pt.tensor(self.dxList.T)
        self.Ng = self.GnnPerms.shape[0]
        self.Ndim = self.dxList.shape[1]

        # Get the destination of each site after the jumps
        self.SitesJumpDests = make_SiteJumpDests(self.dxList, self.superCell, self.NNsiteList)
        # Now convert the jump indexing into a form suitable for gathering batches of vectors
        self.jumpGather = make_JumpGatherTensor(self.SitesJumpDests, self.BatchSize, z, self.Nsites, self.Ndim)
        self.GatherTensor = pt.tensor(self.jumpGather, dtype=pt.long)

        self.dispTensor, self.hostState = get_InputTensors(self.BatchSize, z, self.Ndim, self.Nsites, self.NNsiteList, self.dxList)

        self.gNet = GCNet(self.GnnPerms, self.NNsites, self.JumpVecs, N_ngb=N_ngb, NSpec=1,
                     mean=0.05, std=0.05, nl=3, nch=8, nchLast=1).double()

        self.tcf_epoch, self.y1, self.y2 =\
            Train(self.hostState, self.dispTensor, self.GatherTensor, self.dxList, self.gNet, self.Nsites,
                  0, 0, self.BatchSize, 1, chkpt=False)

        print(self.tcf_epoch)

    def test_make_SiteJumpDests(self):
        # Check that the sites are taken to the correct destination after a jump has occured

        # Check that the vacancy does not move
        self.assertTrue(np.all(self.SitesJumpDests[:, 0] == 0))

        source2Dest = np.zeros_like(self.JumpNewSites)
        for jmp in range(self.dxList.shape[0]):
            for siteInd in range(self.Nsites):
                source = self.JumpNewSites[jmp, siteInd]
                source2Dest[jmp, source] = siteInd

        self.assertTrue(np.array_equal(self.SitesJumpDests, source2Dest))

        for jmp in range(self.dxList.shape[0]):
            # get the negative jump index
            if jmp % 2 == 0:
                jmpNeg = jmp + 1
            else:
                jmpNeg = jmp - 1

            # Check that the exchange site goes to the negative jump site
            self.assertEqual(self.SitesJumpDests[jmp, self.NNsiteList[1 + jmp, 0]], self.NNsiteList[1 + jmpNeg, 0])

            # Now check the rest of the sites
            dxR, _ = self.superCell.crys.cart2pos(self.dxList[jmp])
            for siteInd in range(1, self.Nsites):
                if siteInd == self.NNsiteList[1 + jmp, 0]:
                    continue
                ciSite, Rsite = self.superCell.ciR(siteInd)
                assert ciSite == (0, 0) # monoatomic
                RsiteNew = Rsite - dxR
                siteIndNew , _= self.superCell.index(RsiteNew, (0, 0))
                self.assertEqual(self.SitesJumpDests[jmp, siteInd], siteIndNew)

    def test_make_JumpGatherTensor(self):
        # Make a random batch of y vectors
        self.assertEqual(len(self.GatherTensor.shape), 3)
        self.assertEqual(self.GatherTensor.shape[0], self.BatchSize * self.dxList.shape[0])
        self.assertEqual(self.GatherTensor.shape[1], 3)
        self.assertEqual(self.GatherTensor.shape[2], self.Nsites)

        y_test = pt.tensor(np.random.rand(self.BatchSize * self.dxList.shape[0], 3, self.Nsites)).double()
        y_test_fin = pt.gather(y_test, 2, self.GatherTensor)

        dy_test = y_test_fin - y_test

        # Now go through each jump and check that the y vectors are correctly shuffled to the final state one
        for samp in range(y_test.shape[0]):
            jmpIndex = samp % self.dxList.shape[0]
            for siteInd in range(self.Nsites):
                siteAfterJump = self.SitesJumpDests[jmpIndex, siteInd]
                self.assertTrue(pt.equal(y_test[samp, :, siteAfterJump], y_test_fin[samp, :, siteInd]))
                self.assertTrue(pt.equal(y_test[samp, :, siteAfterJump] - y_test[samp, :, siteInd],
                                         dy_test[samp, :, siteInd]))

    def test_Train(self):
        # Make a random batch of y vectors
        y_test = self.y1
        y_test_fin = pt.gather(y_test, 2, self.GatherTensor)
        self.assertTrue(pt.equal(y_test_fin, self.y2))

        dy_test = y_test_fin - y_test

        # Now go through each jump and check that the y vectors are correctly shuffled to the final state one
        for samp in range(y_test.shape[0]):
            jmpIndex = samp % self.dxList.shape[0]
            self.assertTrue(pt.allclose(y_test[samp, :, 0], pt.zeros(3, dtype=pt.double)))
            self.assertTrue(pt.allclose(y_test_fin[samp, :, 0], pt.zeros(3, dtype=pt.double)))
            for siteInd in range(1, self.Nsites):
                siteAfterJump = self.SitesJumpDests[jmpIndex, siteInd]
                self.assertFalse(siteAfterJump == 0)
                self.assertTrue(pt.equal(y_test[samp, :, siteAfterJump], y_test_fin[samp, :, siteInd]))
                self.assertTrue(pt.equal(y_test[samp, :, siteAfterJump] - y_test[samp, :, siteInd],
                                         dy_test[samp, :, siteInd]))
        print ("min, max dy component : {}, {}".format(pt.min(dy_test), pt.max(dy_test)))
        # Now check the diffusivity
        # First the uncorrelated part
        l0 = 0
        # Go through it sample by sample
        for samp in range(y_test.shape[0]):
            jmpIndex = samp % self.dxList.shape[0]
            dx_tr = -self.dxList[jmpIndex]
            # Only one tracer moves, all the others have zero displacement
            l0 += (np.linalg.norm(dx_tr) ** 2) / (1.0 * self.Nsites - 1.0) # the rates are assumed unity

        # Now the correlated part
        l = 0
        for samp in range(y_test.shape[0]):
            jmpIndex = samp % self.dxList.shape[0]
            moveSite = self.NNsiteList[1 + jmpIndex, 0]
            dx_tr = -self.dxList[jmpIndex]
            # Only one tracer moves, all the others have zero displacement
            l_samp = 0
            # The 0th site is not a tracer site - skip it
            for siteInd in range(1, self.Nsites):
                dxSite = dx_tr if siteInd == moveSite else np.zeros(3)
                dy = dy_test[samp, :, siteInd].detach().numpy()
                l_samp += (np.linalg.norm(dxSite + dy) ** 2) / (1.0 * self.Nsites - 1.0)

            l += l_samp

        tcf = l / l0
        assert np.math.isclose(tcf, self.tcf_epoch[0], rel_tol=0, abs_tol=1e-8)