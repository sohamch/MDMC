import unittest

import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm

from onsager import crystal, cluster, supercell
from SymmLayers import GCNet

class TestSymLayer(unittest.TestCase):
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
        NspCh = 2
        self.TestStates = np.random.randint(0, 2, (len(self.GpermNNIdx), NspCh, self.Nsites))
        self.state0 = self.TestStates[0, :, :].copy()
        self.GIndToGDict = {}
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

        print("Setup complete.")
