import unittest

import numpy as np
import torch as pt
import h5py
from tqdm import tqdm
from onsager import crystal, supercell
from SymmLayers import GCNet, msgPassLayer, msgPassNet
import torch.nn.functional as F


class TestGConv(unittest.TestCase):
    def setUp(self):
        crysType="FCC"
        CrysDatPath = "../CrysDat_" + crysType + "/"

        with h5py.File(CrysDatPath + "CrystData.h5", "r") as fl:
            self.lattice = np.array(fl["Lattice_basis_vectors"])
            self.superlatt = np.array(fl["SuperLatt"])
            self.dxJumps = np.array(fl["dxList_1nn"])
            self.JumpNewSites = np.array(fl["JumpSiteIndexPermutation"])
            self.GpermNNIdx = np.array(fl["GroupNNPermutation"])
            self.NNsiteList = np.array(fl["NNsiteList_sitewise"])

        self.N_ngb = self.NNsiteList.shape[0]
        self.Nsites = self.NNsiteList.shape[1]

        self.z = self.dxJumps.shape[0]
        self.GnnPerms = pt.tensor(self.GpermNNIdx).long()

        self.NNsites = pt.tensor(self.NNsiteList).long()
        self.JumpVecs = pt.tensor(self.dxJumps.T, dtype=pt.double)
        print("Jump Vectors: \n", self.JumpVecs.T, "\n")
        self.Ndim = self.dxJumps.shape[1]

        crys = crystal.Crystal(lattice=self.lattice, basis=[[np.array([0., 0., 0.])]], chemistry=["A"])
        self.superCell = supercell.ClusterSupercell(crys, self.superlatt)
        
        assert len(self.superCell.mobilepos) == self.Nsites
        self.N_units = self.superCell.superlatt[0, 0]

        # Now make a symmetry-related batch of states
        self.NspCh = 2
        self.TestStates = np.random.randint(0, 5, (len(self.GpermNNIdx), self.NspCh, self.Nsites))
        self.state0 = self.TestStates[0, :, :].copy()
        self.GIndToGDict = {}
        self.IdentityIndex = None
        for gInd, g in enumerate(list(self.superCell.crys.G)):
            self.GIndToGDict[gInd] = g  # Index the group operation
            for siteInd in range(self.Nsites):
                _, RSite = self.superCell.ciR(siteInd)
                Rnew, _ = self.superCell.crys.g_pos(g, RSite, (0, 0))
                siteIndNew, _ = self.superCell.index(Rnew, (0, 0))
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
                     mean=0, std=0.1, nl=2, nch=8, nchLast=5).double()
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
                            _, RSite = self.superCell.ciR(siteInd)
                            Rnew, _ = self.superCell.crys.g_pos(g, RSite, (0, 0))
                            siteIndNew, _ = self.superCell.index(Rnew, (0,0))
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
                        _, Rsite = self.superCell.ciR(site)
                        a_site = pt.zeros(self.z, dtype=pt.double)
                        for jmp in range(self.z):
                            nnJump, ci = self.superCell.crys.cart2pos(self.dxJumps[jmp])
                            self.assertEqual(ci, (0, 0))
                            RsiteNew = Rsite + nnJump
                            siteNew, _ = self.superCell.index(RsiteNew, (0, 0))
                            self.assertEqual(siteNew, self.NNsites[jmp + 1, site])
                            a_site[jmp] = out[stateInd, channel, siteNew]

                        siteVec = sum(a_site[i] * self.JumpVecs[:, i] for i in range(self.z)).detach().numpy()
                        assert np.allclose(siteVec, y_np[stateInd, channel, :, site])

        # Then check their symmetry
        y0 = y_np[self.IdentityIndex].copy()
        for gInd, g in tqdm(self.GIndToGDict.items(), position=0, leave=True, ncols=65):
            for site in range(512):
                _, Rsite = self.superCell.ciR(site)
                RsiteNew, _ = self.superCell.crys.g_pos(g, Rsite, (0, 0))
                RsiteNew = RsiteNew
                siteNew, _ = self.superCell.index(RsiteNew, (0, 0))
                for ch in range(y0.shape[0]):
                    assert np.allclose(np.dot(g.cartrot, y0[ch, :, site]), y_np[gInd, ch, :, siteNew])


class TestMsgPassing(unittest.TestCase):

    def setUp(self):
        crysType = "FCC"
        CrysDatPath = "../CrysDat_" + crysType + "/"

        with h5py.File(CrysDatPath + "CrystData.h5", "r") as fl:
            self.lattice = np.array(fl["Lattice_basis_vectors"])
            self.superlatt = np.array(fl["SuperLatt"])
            self.dxJumps = np.array(fl["dxList_1nn"])
            self.NNsiteList = np.array(fl["NNsiteList_sitewise"])
            self.JumpNewSites = np.array(fl["JumpSiteIndexPermutation"])

        self.N_ngb = self.NNsiteList.shape[0]
        self.Nsites = self.NNsiteList.shape[1]

        self.z = self.dxJumps.shape[0]

        self.NNsites = pt.tensor(self.NNsiteList).long()
        self.JumpVecs = pt.tensor(self.dxJumps.T, dtype=pt.double)
        print("Jump Vectors: \n", self.JumpVecs.T, "\n")
        self.Ndim = self.dxJumps.shape[1]

        crys = crystal.Crystal(lattice=self.lattice, basis=[[np.array([0., 0., 0.])]], chemistry=["A"])
        self.superCell = supercell.ClusterSupercell(crys, self.superlatt)
        self.crys = crys

        assert len(self.superCell.mobilepos) == self.Nsites
        self.N_units = self.superCell.superlatt[0, 0]

        # Now make a symmetry-related batch of "states"
        # The occupancies here are random vectors - they don't necessarily need to be 0/1 valued
        # we just want to check if the input and outputs are symmetry-preserving.
        self.NspCh = 2
        self.TestStates = np.random.randint(0, 5, (len(self.crys.G), self.NspCh, self.Nsites))
        self.state0 = self.TestStates[0, :, :].copy()
        self.GIndToGDict = {}
        self.IdentityIndex = None
        self.GList = list(self.crys.G)
        for gInd, g in enumerate(self.GList):
            self.GIndToGDict[gInd] = g  # Index the group operation
            for siteInd in range(self.Nsites):
                _, RSite = self.superCell.ciR(siteInd)
                Rnew, _ = self.superCell.crys.g_pos(g, RSite, (0, 0))
                siteIndNew, _ = self.superCell.index(Rnew, (0, 0))
                self.TestStates[gInd, :, siteIndNew] = self.state0[:, siteInd]

            # Check the identity operation
            if np.allclose(g.cartrot, np.eye(3)):
                assert np.array_equal(self.TestStates[gInd, :, :], self.state0[:, :])
                self.IdentityIndex = gInd

        self.StateTensors = pt.tensor(self.TestStates, dtype=pt.double)

        print("Setup complete.")

    def test_msgPassLayer(self):

        # step 1 : make a message passing layer which produces vectors for each site then same size as
        # the number of species (output=False)
        NChannels = 8
        CompsPerSiteIn = self.StateTensors.shape[1]

        for CompsPerSiteOut in [1, CompsPerSiteIn]:  # check for single and multi-component outputs for each site.
            msgL = msgPassLayer(NChannels=NChannels, CompsPerSiteIn=CompsPerSiteIn, CompsPerSiteOut=CompsPerSiteOut,
                                NNsites=self.NNsites, mean=1.0, std=0.1).double()

            self.assertEqual(msgL.Weights.shape[0], NChannels)
            self.assertEqual(msgL.Weights.shape[1], CompsPerSiteOut)
            self.assertEqual(msgL.Weights.shape[2], CompsPerSiteIn * 2)

            self.assertEqual(msgL.bias.shape[0], NChannels)
            self.assertEqual(msgL.bias.shape[1], CompsPerSiteOut)

            with pt.no_grad():
                # step 2: pass our input states
                out = msgL(self.StateTensors)
                self.assertEqual(len(out.shape), 3)
                self.assertEqual(out.shape[0], self.StateTensors.shape[0])
                self.assertEqual(out.shape[1], CompsPerSiteOut)
                self.assertEqual(out.shape[2], self.Nsites)

                # print(msgL.Weights.shape, "\n", self.StateTensors.shape, "\n", out.shape)
                # step 3: Compute the message passing convolution explicitly for each site
                for samp in tqdm(range(self.StateTensors.shape[0]), position=0, leave=True, ncols=65):
                    for siteInd in range(self.Nsites):
                        zSum = pt.zeros(self.NspCh).double()
                        for z in range(self.dxJumps.shape[0]):
                            zSite = self.NNsiteList[1 + z, siteInd]

                            # concatenate the neighboring site's vector to the site's own
                            multTens = pt.zeros(2 * self.NspCh).double()
                            multTens[:self.NspCh] = self.StateTensors[samp, :, siteInd]
                            multTens[self.NspCh : 2*self.NspCh] = self.StateTensors[samp, :, zSite]

                            # Linearly sum across channels
                            chSum = pt.zeros(self.NspCh).double()
                            for channel in range(NChannels):
                                o = pt.matmul(msgL.Weights[channel], multTens) + msgL.bias[channel]
                                chSum += o

                            zSum += F.softplus(chSum)
                        # print(out[samp, :, siteInd])
                        self.assertTrue(pt.allclose(out[samp, :, siteInd], zSum, rtol=0, atol=1e-8),
                                        msg="\nout: {}\nzsum: {}".format(out[samp, :, siteInd], zSum))

                # step 4 - check symmetry
                out0 = out[self.IdentityIndex, :, :]
                for gInd in tqdm(range(len(self.GList)), position=0, leave=True, ncols=65):
                    g = self.GList[gInd]
                    for site in range(self.Nsites):
                        _, RSite = self.superCell.ciR(siteInd)
                        Rnew, _ = self.superCell.crys.g_pos(g, RSite, (0, 0))
                        siteIndNew, _ = self.superCell.index(Rnew, (0, 0))

                        self.assertTrue(pt.allclose(out0[:, siteInd], out[gInd, :, siteIndNew], rtol=0, atol=1e-8))

    def test_msgPassNet(self):
        # step 1 : make a message passing network to produce R3 vectors for each site
        NChannels = 8
        NLayers = 3
        VecsPerSite = 1
        # NLayers, NChannels, NSpec, VecsPerSite, NNsites, JumpVecs, mean=1.0, std=0.1
        JumpVecs = pt.tensor(self.dxJumps.T, dtype=pt.float64)
        msgNet = msgPassNet(NLayers=NLayers, NChannels=NChannels, NSpec=self.NspCh, VecsPerSite=VecsPerSite,
                            NNsites=self.NNsites, JumpVecs=JumpVecs, mean=0.0, std=0.01).double()

        with pt.no_grad():
            # step 2: Do the forward pass on our input states
            out = msgNet(self.StateTensors)

            self.assertEqual(len(out.shape), 4)
            self.assertEqual(out.shape[0], len(self.GList))
            self.assertEqual(out.shape[1], VecsPerSite)
            self.assertEqual(out.shape[2], self.dxJumps.shape[1])
            self.assertEqual(out.shape[3], self.Nsites)

            print(pt.mean(out).item(), pt.max(out).item(), pt.min(out).item())

            # Step 3 : Now let's check the correctness of the vectors
            outSeq = msgNet.net(self.StateTensors).detach().numpy()  # shape (Nbatch, 1, Nsites)
            for samp in tqdm(range(outSeq.shape[0]), position=0, leave=True, ncols=65):
                for siteInd in range(self.Nsites):
                    ngbSites = self.NNsiteList[1:, siteInd]
                    ngbSum = np.zeros(3)
                    for z in range(ngbSites.shape[0]):
                        ngbSum += outSeq[samp, 0, ngbSites[z]] * self.dxJumps[z]

                    self.assertTrue(np.allclose(ngbSum, out[samp, 0, :, siteInd]))

            # Next check rotational symmetries of the vectors
            outVecs = out.detach().numpy()
            self.assertTrue(outVecs.shape[1] == 1)
            outState0 = outVecs[self.IdentityIndex, 0, :, :]

            for gInd in tqdm(range(len(self.GList)), position=0, leave=True, ncols=65):
                g = self.GList[gInd]
                yvecsG = outVecs[gInd, 0, :, :]
                for siteInd in range(self.Nsites):

                    # get the transformed site
                    _, RSite = self.superCell.ciR(siteInd)
                    Rnew, _ = self.superCell.crys.g_pos(g, RSite, (0, 0))
                    siteIndNew, _ = self.superCell.index(Rnew, (0, 0))

                    # check the rotational transformation of the output vectors
                    yRot = np.dot(g.cartrot, outState0[:, siteInd])
                    self.assertTrue(np.allclose(yRot, yvecsG[:, siteIndNew]))


