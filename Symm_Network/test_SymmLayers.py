import unittest

import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm

from onsager import crystal, cluster, supercell
from SymmLayers import GCNet

class TestGConv(unittest.TestCase):
    def setUp(self):
        crysType="FCC"
        CrysDatPath = "../CrysDat_" + crysType + "/"
        self.GpermNNIdx = np.load(CrysDatPath + "GroupNNpermutations.npy")
        self.siteShellIndices = np.load(CrysDatPath + "SitesToShells.npy")

        self.NNsiteList = np.load(CrysDatPath + "NNsites_sitewise.npy")
        self.dxJumps = np.load(CrysDatPath + "dxList.npy")

        self.N_ngb = self.NNsiteList.shape[0]
        self.Nsites = self.NNsiteList.shape[1]

        self.z = self.dxJumps.shape[0]
        self.GnnPerms = pt.tensor(self.GpermNNIdx).long()

        self.NNsites = pt.tensor(self.NNsiteList).long()
        self.JumpVecs = pt.tensor(self.dxJumps.T, dtype=pt.double)
        print("Jump Vectors: \n", self.JumpVecs.T, "\n")
        self.Ndim = self.dxJumps.shape[1]

        with open(CrysDatPath + "supercellFCC.pkl", "rb") as fl:
            self.superFCC = pickle.load(fl)

        assert len(self.superFCC.mobilepos) == self.Nsites
        self.N_units = self.superFCC.superlatt[0, 0]

        # Now make a symmetry-related batch of states
        self.NspCh = 2
        self.TestStates = np.random.randint(0, 2, (len(self.GpermNNIdx), self.NspCh, self.Nsites))
        self.state0 = self.TestStates[0, :, :].copy()
        self.GIndToGDict = {}
        self.IdentityIndex = None
        for gInd, g in enumerate(list(self.superFCC.crys.G)):
            self.GIndToGDict[gInd] = g # Index the group operation
            for siteInd in range(self.Nsites):
                _, RSite = self.superFCC.ciR(siteInd)
                Rnew, _ = self.superFCC.crys.g_pos(g, RSite, (0, 0))
                siteIndNew, _ = self.superFCC.index(Rnew, (0, 0))
                self.TestStates[gInd, :, siteIndNew] = self.state0[:, siteInd]

            # Check the identity operation
            if np.allclose(g.cartrot, np.eye(3)):
                assert np.array_equal(self.TestStates[gInd, :, :], self.state0[:, :])
                self.IdentityIndex = gInd

        self.StateTensors = pt.tensor(self.TestStates, dtype=pt.double)

        print("Setup complete.")

    def test_GConv(self):
        gNet = GCNet(self.GnnPerms.long(), self.NNsites, self.JumpVecs, N_ngb=self.N_ngb, NSpec=self.NspCh,
                     mean=0.02, std=0.2, nl=1, nch=8, nchLast=5).double()

        # Do the first Gconv explicitly
        with pt.no_grad():
            l = 0
            outl_Gconv = gNet.net[l].forward(self.StateTensors[:1])
            for stateInd in range(1):
                for chOut in range(gNet.net[l].Psi.shape[0]):
                    b = gNet.net[l].bias[chOut]
                    for site in tqdm(range(self.Nsites), position=0, leave=True, ncols=65):
                        sitesNgb = self.NNsites[:, site]
                        for gInd in range(self.GnnPerms.shape[0]):
                            sm = 0.
                            for chIn in range(gNet.net[l].Psi.shape[1]):
                                Psi_g = gNet.net[l].Psi[chOut, chIn][self.GnnPerms[gInd]]
                                siteConv = pt.sum(Psi_g * self.StateTensors[stateInd, chIn, sitesNgb])
                                sm += siteConv
                            self.assertTrue(pt.allclose(outl_Gconv[stateInd, chOut, gInd, site], sm + b))
            print("Gconv explicit symmetry test passed")

    def test_GConv_noSym(self):
        GnnPerms = self.GnnPerms[:1].long()
        print(GnnPerms)
        self.assertTrue(pt.equal(GnnPerms, pt.arange(GnnPerms.shape[1]).unsqueeze(0)))
        gNet = GCNet(GnnPerms, self.NNsites, self.JumpVecs, N_ngb=self.N_ngb, NSpec=self.NspCh,
                     mean=0.02, std=0.2, nl=1, nch=8, nchLast=5).double()

        # Do the first Gconv explicitly
        with pt.no_grad():
            l = 0
            outl_Gconv = gNet.net[l].forward(self.StateTensors[:1])
            for stateInd in range(1):
                for chOut in range(gNet.net[l].Psi.shape[0]):
                    b = gNet.net[l].bias[chOut]
                    for site in tqdm(range(self.Nsites), position=0, leave=True, ncols=65):
                        sitesNgb = self.NNsites[:, site]
                        sm = 0.
                        for chIn in range(gNet.net[l].Psi.shape[1]):
                            Psi_g = gNet.net[l].Psi[chOut, chIn]
                            siteConv = pt.sum(Psi_g * self.StateTensors[stateInd, chIn, sitesNgb])
                            sm += siteConv
                        self.assertTrue(pt.allclose(outl_Gconv[stateInd, chOut, 0, site], sm + b))

            out_non_lin = gNet.net[l+1].forward(outl_Gconv)
            out_Gav = gNet.net[l+2].forward(out_non_lin)

            self.assertTrue(pt.allclose(out_Gav, out_non_lin[:, :, 0, :]), msg="{} {}".format(out_Gav, out_non_lin))

            print("Non-symmetry explicit symmetry test passed")

    def test_Symmetry(self):
        # here, we'll check the full symmetry of the network with the batch of symmetry-related states
        gNet = GCNet(self.GnnPerms.long(), self.NNsites, self.JumpVecs, N_ngb=self.N_ngb, NSpec=self.NspCh,
                     mean=0.02, std=0.2, nl=2, nch=8, nchLast=5).double()
        out = pt.clone(self.StateTensors)
        with pt.no_grad():
            for l in range(0, len(gNet.net), 3):
                self.assertEqual(out.shape[1], gNet.net[l].Psi.shape[1])
                out = gNet.net[l].forward(out)
                out = gNet.net[l + 1].forward(out) # apply non-linearity

                # Do the group averaging explicitly
                outGAv = pt.zeros(out.shape[0], out.shape[1], out.shape[3], dtype=pt.double)
                for gInd in range(self.GnnPerms.shape[0]):
                    outGAv[:, :, :] += out[:, :, gInd, :]/self.GnnPerms.shape[0]

                out = gNet.net[l + 2].forward(out) # do group average from network

                # Match the two
                self.assertTrue(pt.allclose(out, outGAv))

                Nch = out.shape[1]
                self.assertEqual(Nch, gNet.net[l].Psi.shape[0])

                print("layer: {}. max value: {}, min value: {}, mean vslue: {}".format((l + 3) // 3, pt.max(out), pt.min(out), pt.mean(out) ) )
                # Test the symmetry for each output channel
                for ch in range(Nch):
                    # Get the identity sample
                    outsamp0 = out[self.IdentityIndex, ch, :]
                    for gInd, g in self.GIndToGDict.items():
                        outsamp = pt.zeros_like(outsamp0)
                        for siteInd in range(self.Nsites):
                            _, RSite = self.superFCC.ciR(siteInd)
                            Rnew, _ = self.superFCC.crys.g_pos(g, RSite, (0, 0))
                            siteIndNew, _ = self.superFCC.index(Rnew, (0,0))
                            outsamp[siteIndNew] = outsamp0[siteInd]
                        self.assertTrue(pt.allclose(outsamp, out[gInd, ch, :], atol=1e-12))
                print("Layer {} symmetry assertion passed".format((l + 3) // 3))

    def test_yVecs(self):
        # Check the y vectors explicitly
        gNet = GCNet(self.GnnPerms.long(), self.NNsites, self.JumpVecs, N_ngb=self.N_ngb, NSpec=self.NspCh,
                     mean=0.02, std=0.2, nl=2, nch=8, nchLast=5).double()
        with pt.no_grad():
            y = gNet(self.StateTensors)

        y_np = y.data.numpy().copy()
        mn = np.min(y_np.flatten())
        mx = np.max(y_np.flatten())
        av = np.mean(y_np.flatten())
        print("min : {}, max : {}, mean: {}".format(mn, mx, av))

        # First, check the y vectors explicitly
        with pt.no_grad():
            out = gNet.net(self.StateTensors)
            for stateInd in tqdm(range(out.shape[0]), position=0, leave=True, ncols=65):
                for channel in range(out.shape[1]):
                    for site in range(self.Nsites):
                        _, Rsite = self.superFCC.ciR(site)
                        a_site = pt.zeros(self.z, dtype=pt.double)
                        for jmp in range(self.z):
                            nnJump, ci = self.superFCC.crys.cart2pos(self.dxJumps[jmp])
                            self.assertEqual(ci, (0, 0))
                            RsiteNew = Rsite + nnJump
                            siteNew, _ = self.superFCC.index(RsiteNew, (0, 0))
                            self.assertEqual(siteNew, self.NNsites[jmp + 1, site])
                            a_site[jmp] = out[stateInd, channel, siteNew]

                        siteVec = sum(a_site[i] * self.JumpVecs[:, i] for i in range(self.z)).detach().numpy()
                        assert np.allclose(siteVec, y_np[stateInd, channel, :, site])

        # Then check their symmetry
        y0 = y_np[self.IdentityIndex].copy()
        for gInd, g in tqdm(self.GIndToGDict.items(), position=0, leave=True, ncols=65):
            for site in range(512):
                _, Rsite = self.superFCC.ciR(site)
                RsiteNew, _ = self.superFCC.crys.g_pos(g, Rsite, (0, 0))
                RsiteNew = RsiteNew
                siteNew, _ = self.superFCC.index(RsiteNew, (0, 0))
                for ch in range(y0.shape[0]):
                    assert np.allclose(np.dot(g.cartrot, y0[ch, :, site]), y_np[gInd, ch, :, siteNew])