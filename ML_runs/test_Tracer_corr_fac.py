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
                     mean=0.1, std=0.1, nl=3, nch=8, nchLast=1).double()

        tcf_epoch = Train(self.hostState, self.dispTensor, self.GatherTensor, self.dxList, self.gNet, self.Nsites, 0, 1, 1)
        