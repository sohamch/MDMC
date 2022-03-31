CrysDatPath = "/home/sohamc2/HEA_FCC/CrysDat/"
DataPath = "/home/sohamc2/HEA_FCC/MDMC/ML_runs/DataSets/"
ModulePath = "/home/sohamc2/VKMC/SymNetworkRuns/CE_Symmetry/Symm_Network/"

import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import h5py
import pickle
import matplotlib.pyplot as plt
from SymmLayers import GConv, R3Conv, R3ConvSites, GAvg
import os
import sys
sys.path.append(ModulePath)
RunPath = os.getcwd() + "/"

device=None
if pt.cuda.is_available():
    device = pt.device("cuda:0")
else:
    device = pt.device("cpu")

def Load_crysDats():
"""## load the crystal data files"""
    GpermNNIdx = np.load(CrysDatPath + "GroupNNpermutations.npy")
    NNsiteList = np.load(CrysDatPath + "NNsites_sitewise.npy")
    siteShellIndices = np.load(CrysDatPath + "SitesToShells.npy")
    JumpNewSites = np.load(CrysDatPath + "JumpNewSiteIndices.npy")
    dxJumps = np.load(CrysDatPath + "dxList.npy")
    with open(CrysDatPath + "GroupCartIndices.pkl", "rb") as fl:
        GIndtoGDict = pickle.load(fl)
    return GpermNNIdx, NNsiteList, siteShellIndices, GIndtoGDict, JumpNewSites

def Load_Data():
    with h5py.File(DataPath + "singleStep_{}.h5".format(T), "r") as fl:
        state1List = np.array(fl["InitStates"])
        state2List = np.array(fl["FinStates"])
        dispList = np.array(fl["SpecDisps"])
        rateList = np.array(fl["rates"])
        AllJumpRates = np.array(fl["AllJumpRates"])
        jmpSelects = np.array(fl["JumpSelects"]).astype(np.int8)
    return state1List, state2List, dispList, rateList, AllJumpRates, jmpSelects, AtomMarkers

def MakeComputeData(state1List, state2List, dispList, specsToTrain, rateList,
        AllJumpRates, JumpNewSites, dxJumps, NNsiteList, N_train, AllJumps=False):

    # make the input tensors
    if AllJumps:
        Nsamples = N_train*2*AllJumpRates.shape[1]
    else:
        Nsamples = N_train*2
    a = np.linalg.norm(dispList[0, 0, :])/np.linalg.norm(dxJumps[0]) 
    specs = np.unique(state1List[0])
    NSpec = specs.shape[0] - 1
    Nsites = state1List.shape[1]
    NNsvac = NNsiteList[1:, 0]

    State1_occs = np.zeros((Nsamples, NSpec, Nsites), dtype=np.int8)
    State2_occs = np.zeros((Nsamples, NSpec, Nsites), dtype=np.int8)
    dispData = np.zeros((Nsamples, 2, 3)) # store 2 displacements - one for displacements, and one for all other species to be trained
    rateData = np.zeros(Nsamples)
    
    # Make the multichannel occupancies
    for samp in range(Nsamples):
        state1 = state1List[samp]
        if AllJumps:
            for jInd in range(AllJumpRates.shape[1]):
                JumpSpec = state1[NNsvac[jInd]]
                state2 = state1[JumpNewSites[jInd]]
                Idx = samp*AllJumpRates.shape[1]  + jInd
                dispData[Idx, 0, :] =  dxJumps[jInd]*a
                if JumpSpec in specsToTrain:
                    dispData[Idx, 1, :] -= dxJumps[jInd]*a

                rateData[Idx] = AllJumpRates[samp, jInd]
                
                for site in range(1, Nsites): # exclude the vacancy site
                    spec1 = state1[site]
                    spec2 = state2[site]
                    State1_occs[Idx, spec1-1, site] = 1
                    State2_occs[Idx, spec2-1, site] = 1

        else:
            state2 = state2List[samp]
            for site in range(1, Nsites): # exclude the vacancy site
                spec1 = state1[site]
                spec2 = state2[site]
                State1_occs[samp, spec1-1, site] = 1
                State2_occs[samp, spec2-1, site] = 1
            
            dispData[samp, 0, :] = dispList[samp, 0, :]
            dispData[samp, 1, :] sum(dispList[samp, spec, :] for spec in specsToTrain)

            rateData[samp] = rateList[samp]
    
    # Make the numpy tensor to indicate "on" sites (i.e, those whose y vectors will be collected)
    OnSites_state1 = None
    OnSites_state2 = None
    if specsToTrain != [NSpec + 1]:
        OnSites_state1 = np.zeros((Nsamples, Nsites), dtype=np.int8)
        OnSites_state2 = np.zeros((Nsamples, Nsites), dtype=np.int8)
    
        for spec in specsToTrain:
            OnSites_state1 += State1_occs[:, spec-1, :]
            OnSites_state2 += State2_occs[:, spec-1, :]

    return State1_occs, State2_occs, rateData, dispData, OnSites_state1, OnSites_state2


class GCNet(nn.Module):
    def __init__(self, GnnPerms, NNsites, SitesToShells,
                dim, N_ngb, mean, std, b=1.0, nl=3):
        
        super().__init__()
        modules = []
        modules += [GConv(NSpec, 8, GnnPerms, NNsites, N_ngb, mean=mean, std=std), 
                nn.Softplus(beta=b), Gavg()]

        for i in range(nl):
            modules += [GConv(8, 8, GnnPerms, NNsites, N_ngb, mean=mean, std=std), 
                    nn.Softplus(beta=b), Gavg()]

        modules += [GConv(8, 1, GnnPerms, NNsites, N_ngb, mean=mean, std=std), 
                nn.Softplus(beta=b), Gavg()]

        modules += [R3ConvSites(SitesToShells, GnnPerms, gdiags, NNsites, N_ngb, 
            dim, mean=mean, std=std)]
        
        self.net = nn.Sequential(*modules)
    
    def forward(self, InState):
        y = self.net(InState)
        return y


# test for correctness of indexing
gNet = GCNet(GnnPerms, NNsites, SitesToShells,
             dim=Ndim, N_ngb=N_ngb,
             mean=0.03, std=0.02).double().to(device)

"""## Write the training loop"""
N_train = Nsamples//2

def Train(T, dirPath, State1_Occs, State2_Occs, rateData, dispData, SpecsToTrain,
        start_ep, end_ep, interval, N_train, gNet,
        lRate=0.001, scratch_if_no_init=True):
     
    N_batch = 128
    try:
        gNet.load_state_dict(pt.load(dirPath + "/{1}ep.pt".format(T, start_ep)))
        print("Starting from epoch {}".format(start_ep), flush=True)
    except:
        if scratch_if_no_init = True:
            print("No Network found. Starting from scratch", flush=True)
        else:
            raise ValueError("No saved network found in {} at epoch {}".format(dirPath, start_ep))

    optimizer = pt.optim.Adam(gNet.parameters(), lr=lRate, weight_decay=0.0005)


    for epoch in range(start_ep, end_ep + 1):
        
        ## checkpoint
        if epoch%interval==0:
            pt.save(gNet.state_dict(), dirPath + "/{1}ep.pt".format(T, epoch))
            
        for batch in range(0, N_train, N_batch):
            optimizer.zero_grad()
            
            end = min(batch + N_batch, N_train)

            state1Batch = state1Data[batch : end]
            state2Batch = state2Data[batch : end]
            
            rateBatch = rateData[batch : end]
            dispBatch = dispData[batch : end]
            
            y1 = gNet(state1Batch)
            y2 = gNet(state2Batch)
            
            # sum up everything except the vacancy site

            y1 = pt.sum(y1[:, :, 1:], dim=2)
            y2 = pt.sum(y2[:, :, 1:], dim=2)

            dy = y1 - y2

            loss = pt.sum(rateBatch * pt.norm((dispBatch + dy), dim=1)**2)/6.
            
            loss.backward()
            optimizer.step()


def main(args):
    print("Running at : "+RunPath+"\n")

    # Get run parameters
    # Replace all of this with argparse after learning about it
    T = int(args[1])
    start_ep = int(args[2])
    end_ep = int(args[3])
    scratch_if_no_init = str(args(4))
    nLayers = int(args[5])
    specTrain = int(args[6])
    Mode = args[8] # train mode or eval mode
    AllJumps = args[9] # whether to train all jumps out of the samples or just stochastically selected one
    N_train = int(args[10]) # How many INITIAL STATES to consider for training
    interval = args[9] # for train mode, interval to save and for eval mode, interval to load

    learning_Rate = float(args[10]) if len(args)==11 else 0.001
    
    # Load data
    state1List, state2List, dispList, rateList, jmpSelects = Load_Data()
    
    specs = np.unique(state1List[0])
    NSpec = specs.shape[0] - 1
    Nsites = state1List.shape[1]
     
    specsToTrain = []
    while specTrain > 0:
        specsToTrain.append(specTrain%10)
        specTrain = specTrain//10
    
    specsToTrain = sorted(specsToTrain)
    
    direcString=""
    if specsToTrain == [NSpec + 1]:
        direcString = "vac"
    else:
        for spec in specsToTrain:
            direcString += "{}".format(spec)

    dirNameNets = "epochs_T_{0}_{1}_n{2}c8".format(T, direcString, nLayers)

    print("Training species: {}".format(specsToTrain), flush=True)
    
    # check if a run directory exists
    dirPath = RunPath + dirNameNets
    exists = os.path.isdir(dirPath)
    
    if not exists:
        if start_ep == 0:
            os.mkdir(dirPath)
        elif start_ep > 0:
            raise ValueError("Training directory does not exist but start epoch greater than zero: {}".format(start_ep))

    print(pt.__version__)
    print(pt.cuda.get_device_name())
    
    # Load crystal parameters
    GpermNNIdx, NNsiteList, siteShellIndices, GIndtoGDict, JumpNewSites = Load_crysDats()
    N_ngb = NNsiteList.shape[0]
    Nsites = NNsiteList.shape[1]
    SitesToShells = pt.tensor(siteShellIndices).long().to(device)
    GnnPerms = pt.tensor(GpermNNIdx).long().to(device)
    NNsites = pt.tensor(NNsiteList).long().to(device)

    Ng = GnnPerms.shape[0]
    Ndim = 3
    gdiagsCpu = pt.zeros(Ng*Ndim, Ng*Ndim).double()
    for gInd, gCart in GIndtoGDict.items():
        rowStart = gInd * Ndim
        rowEnd = (gInd + 1) * Ndim
        gdiagsCpu[rowStart : rowEnd, rowStart : rowEnd] = pt.tensor(gCart)
    gdiags = gdiagsCpu.to(device)
    
    # Make a network to either train from scratch or load saved state into
    gNet = GCNet(GnnPerms, NNsites, SitesToShells,
            dim=Ndim, N_ngb=N_ngb,
            mean=0.03, std=0.02, b=1.0, nl=nl).double().to(device)

    # Make the sample tensors here

    # Write the training loop here


if __name__ == "main":
    main(list(sys.argv))
