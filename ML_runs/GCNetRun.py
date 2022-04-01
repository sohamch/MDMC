import os
import sys
RunPath = os.getcwd() + "/"
CrysDatPath = "/home/sohamc2/HEA_FCC/CrysDat/"
DataPath = "/home/sohamc2/HEA_FCC/MDMC/ML_runs/DataSets/"
ModulePath = "/home/sohamc2/VKMC/SymNetworkRuns/CE_Symmetry/Symm_Network/"

#CrysDatPath = "CrysDat/"
#DataPath = "MD_KMC_single/"
#ModulePath = "/mnt/FCDEB3C5DEB3768C/UIUC/Research/KMC_ML/VKMC/SymNetworkRuns/CE_Symmetry/Symm_Network"

sys.path.append(ModulePath)

import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import h5py
import pickle
import matplotlib.pyplot as plt
from SymmLayers import GConv, R3Conv, R3ConvSites, GAvg

device=None
if pt.cuda.is_available():
    device = pt.device("cuda:0")
else:
    device = pt.device("cpu")


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


def Load_crysDats():
    ## load the crystal data files
    GpermNNIdx = np.load(CrysDatPath + "GroupNNpermutations.npy")
    NNsiteList = np.load(CrysDatPath + "NNsites_sitewise.npy")
    siteShellIndices = np.load(CrysDatPath + "SitesToShells.npy")
    JumpNewSites = np.load(CrysDatPath + "JumpNewSiteIndices.npy")
    dxJumps = np.load(CrysDatPath + "dxList.npy")
    with open(CrysDatPath + "GroupCartIndices.pkl", "rb") as fl:
        GIndtoGDict = pickle.load(fl)
    return GpermNNIdx, NNsiteList, siteShellIndices, GIndtoGDict, JumpNewSites, dxJumps

def Load_Data(T):
    with h5py.File(DataPath + "singleStep_{}.h5".format(T), "r") as fl:
        state1List = np.array(fl["InitStates"])
        state2List = np.array(fl["FinStates"])
        dispList = np.array(fl["SpecDisps"])
        rateList = np.array(fl["rates"])
        AllJumpRates = np.array(fl["AllJumpRates"])
        jmpSelects = np.array(fl["JumpSelects"]).astype(np.int8)
    
    return state1List, state2List, dispList, rateList, AllJumpRates, jmpSelects

def makeComputeData(state1List, state2List, dispList, specsToTrain, VacSpec, rateList,
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
    for samp in range(2*N_train):
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
            
            dispData[samp, 0, :] = dispList[samp, VacSpec, :]
            dispData[samp, 1, :] = sum(dispList[samp, spec, :] for spec in specsToTrain)

            rateData[samp] = rateList[samp]
    
    # Make the numpy tensor to indicate "on" sites (i.e, those whose y vectors will be collected)
    OnSites_state1 = None
    OnSites_state2 = None
    if specsToTrain != [VacSpec]:
        OnSites_state1 = np.zeros((Nsamples, Nsites), dtype=np.int8)
        OnSites_state2 = np.zeros((Nsamples, Nsites), dtype=np.int8)
    
        for spec in specsToTrain:
            OnSites_state1 += State1_occs[:, spec-1, :]
            OnSites_state2 += State2_occs[:, spec-1, :]

    return State1_occs, State2_occs, rateData, dispData, OnSites_state1, OnSites_state2


def makeProdTensor(OnSites, Ndim):
    Onst = pt.tensor(OnSites_st1)
    Onst = Onst.repeat_interleave(Ndim, dim=0)
    Onst = Onst.view(-1, Ndim, Nsites)
    return Onst


"""## Write the training loop"""
def Train(T, dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, 
        rates, disps, SpecsToTrain, VacSpec, start_ep, end_ep, interval, N_train,
        gNet, lRate=0.001, scratch_if_no_init=True):
    
    Ndim = disps.shape[2]
    N_batch = 128
    # Convert compute data to pytorch tensors
    state1Data = pt.tensor(State1_Occs[:N_train]).double().to(device)
    state2Data = pt.tensor(state2_Occs[:N_train]).double().to(device)
    rateData = pt.tensor(rates[:N_train]).double().to(device)
    On_st1 = None 
    On_st2 = None    
    
    if SpecsToTrain == [VacSpec]:
        assert OnSites_st1 == OnSites_st2 == None
        dispData = pt.tensor(disps[:N_train, 0, :]).double().to(device)
    else:
        dispData = pt.tensor(disps[:N_train, 1, :]).double().to(device) 
        On_st1 = makeProdTensor(OnSites_st1[:N_train], Ndim).long().to(device)
        On_st2 = makeProdTensor(OnSites_st2[:N_train], Ndim).long().to(device)

    try:
        gNet.load_state_dict(pt.load(dirPath + "/{1}ep.pt".format(T, start_ep)))
        print("Starting from epoch {}".format(start_ep), flush=True)
    except:
        if scratch_if_no_init:
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
            if SpecsToTrain==[VacSpec]:
                y1 = -pt.sum(y1[:, :, 1:], dim=2)
                y2 = -pt.sum(y2[:, :, 1:], dim=2)
            
            else:
                On_st1Batch = On_st1[batch : end]
                On_st2Batch = On_st2[batch : end]
                y1 = pt.sum(y1*On_st1Batch, dim=2)
                y2 = pt.sum(y2*On_st2Batch, dim=2)

            dy = y2 - y1
            diff = pt.sum(rateBatch * pt.norm((dispBatch + dy), dim=1)**2)/6. 
            diff.backward()
            optimizer.step()


def Evaluate(T, dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, 
        rates, disps, SpecsToTrain, VacSpec, start_ep, end_ep, interval, N_train,
        gNet):
    
    Ndim = disps.shape[2]
    N_batch = 512
    # Convert compute data to pytorch tensors
    state1Data = pt.tensor(State1_Occs).double().to(device)
    state2Data = pt.tensor(state2_Occs).double().to(device)
    rateData = pt.tensor(rates).double().to(device)
    On_st1 = None
    On_st2 = None
    
    if SpecsToTrain == [VacSpec]:
        assert OnSites_st1 == OnSites_st2 == None
        dispData = pt.tensor(disps[:, 0, :]).double().to(device)
    else:
        dispData = pt.tensor(disps[:, 1, :]).double().to(device) 
        On_st1 = makeProdTensor(OnSites_st1, Ndim).long().to(device)
        On_st2 = makeProdTensor(OnSites_st2, Ndim).long().to(device)

    def compute(startSample, endSample):
        diff_epochs = []
        with pt.no_grad():
            for epoch in range(start_ep, end_ep + 1, interval):
                ## load checkpoint
                gNet.load_state_dict(pt.load(dirPath + "/{1}ep.pt".format(T, epoch), map_location=device))
                    
                diff_ep = 0 
                for batch in range(startSample, endSample, N_batch):
                    end = min(batch + N_batch, endSample)

                    state1Batch = state1Data[batch : end]
                    state2Batch = state2Data[batch : end]
                    
                    rateBatch = rateData[batch : end]
                    dispBatch = dispData[batch : end]
                    
                    y1 = gNet(state1Batch)
                    y2 = gNet(state2Batch)
                    
                    # sum up everything except the vacancy site if vacancy is indicated
                    if SpecsToTrain==[VacSpec]:
                        y1 = -pt.sum(y1[:, :, 1:], dim=2)
                        y2 = -pt.sum(y2[:, :, 1:], dim=2)
                    
                    else:
                        On_st1Batch = On_st1[batch : end]
                        On_st2Batch = On_st2[batch : end]
                        y1 = pt.sum(y1*On_st1Batch, dim=2)
                        y2 = pt.sum(y2*On_st2Batch, dim=2)

                    dy = y2 - y1
                    loss = pt.sum(rateBatch * pt.norm((dispBatch + dy), dim=1)**2)/6.
                    diff_ep += loss.item()

                diff_epochs.append(diff_ep)

        return np.array(diff_ep)
    
    Nsamples = state1Data.shape[0]
    train_diff = compute(0, N_train)/(1.0*N_train)
    test_diff = compute(N_train, Nsamples - N_train)/(1.0*(Nsamples - N_train))

    return train_diff, test_diff


def Gather_Y(T, dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, SpecsToTrain, VacSpec, epoch, gNet):
    
    Ndim = disps.shape[2]
    N_batch = 512
    # Convert compute data to pytorch tensors
    state1Data = pt.tensor(State1_Occs).double().to(device)
    Nsamples = state1Data.shape[0]
    state2Data = pt.tensor(state2_Occs).double().to(device)
    On_st1 = None
    On_st2 = None 
    
    if SpecsToTrain == [VacSpec]:
        assert OnSites_st1 == OnSites_st2 == None
    else:
        On_st1 = makeProdTensor(OnSites_st1, Ndim).long().to(device)
        On_st2 = makeProdTensor(OnSites_st2, Ndim).long().to(device)

    y1Vecs = np.zeros((Nsamples, 3))
    y2Vecs = np.zeros((Nsamples, 3))

    with pt.no_grad():
        ## load checkpoint
        gNet.load_state_dict(pt.load(dirPath + "/{1}ep.pt".format(T, epoch), map_location=device))
                
        for batch in range(0, Nsamples, N_batch):
            end = min(batch + N_batch, Nsamples)

            state1Batch = state1Data[batch : end]
            state2Batch = state2Data[batch : end]
            
            rateBatch = rateData[batch : end]
            dispBatch = dispData[batch : end]
            
            y1 = gNet(state1Batch)
            y2 = gNet(state2Batch)
            
            # sum up everything except the vacancy site if vacancy is indicated
            if SpecsToTrain == [VacSpec]:
                y1 = -pt.sum(y1[:, :, 1:], dim=2)
                y2 = -pt.sum(y2[:, :, 1:], dim=2)
            
            else:
                On_st1Batch = On_st1[batch : end]
                On_st2Batch = On_st2[batch : end]
                y1 = pt.sum(y1*On_st1Batch, dim=2)
                y2 = pt.sum(y2*On_st2Batch, dim=2)
            
            y1Vecs[batch : end] = y1.cpu().numpy()[:, :]
            y2Vecs[batch : end] = y2.cpu().numpy()[:, :]

    return y1Vecs, y2Vecs


def main(args):
    print("Running at : "+RunPath+"\n")

    # Get run parameters
    # Replace all of this with argparse after learning about it
    count=1
     
    Mode = args[count] # "train" mode or "eval" mode or "getY" mode
    count += 1

    nLayers = int(args[count])
    count += 1
    
    scratch_if_no_init = bool(int(args[count]))
    count += 1
    
    T_data = int(args[count]) # temperature to load data from
    # Note : for binary random alloys, this should is the training composition instead of temperature
    count += 1
    
    T_net = int(args[count]) # must be same as T_data if "train", can be different if "getY" or "eval"
    count += 1
    
    start_ep = int(args[count])
    count += 1
    
    end_ep = int(args[count])
    count += 1
    
    specTrain = int(args[count]) # which species to train collectively: eg - 123 for species 1, 2 and 3
    # The entry is order independent as it is sorted later
    count += 1
    
    VacSpec = int(args[count]) # integer label for vacancy species
    count += 1
    
    AllJumps = bool(int(args[count])) # whether to train all jumps out of the samples or just stochastically selected one
    # False if 0, otherwise True
    count += 1
    
    N_train = int(args[count]) # How many INITIAL STATES to consider for training
    count += 1
    
    interval = args[count] # for train mode, interval to save and for eval mode, interval to load
    count += 1
    
    learning_Rate = float(args[count]) if len(args)==count+1 else 0.001
    
    if not (Mode == "train" or Mode == "eval" or Mode == "getY"):
        raise ValueError("Mode needs to be train, eval or getY and not : {}".format(Mode))

    if Mode == "train" or Mode == "eval":
        if T_data != T_net:
            raise ValueError("Training and Testing condition must be the same")

    # Load data
    state1List, state2List, dispList, rateList, AllJumpRates, jmpSelects = Load_Data(T_data)
    
    specs = np.unique(state1List[0])
    NSpec = specs.shape[0] - 1
    Nsites = state1List.shape[1]
     
    specsToTrain = []
    while specTrain > 0:
        specsToTrain.append(specTrain%10)
        specTrain = specTrain//10
    
    specsToTrain = sorted(specsToTrain)
    
    direcString=""
    if specsToTrain == [VacSpec]:
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
    GpermNNIdx, NNsiteList, siteShellIndices, GIndtoGDict, JumpNewSites, dxJumps = Load_crysDats()
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
            mean=0.03, std=0.02, b=1.0, nl=nLayers).double().to(device)

    # Call MakeComputeData here
    State1_occs, State2_occs, rateData, dispData, OnSites_state1, OnSites_state2 = makeComputeData(state1List, state2List, dispList,
            specsToTrain, VacSpec, rateList, AllJumpRates, JumpNewSites, dxJumps, NNsiteList, N_train, AllJumps=AllJumps)
    # Call Training or evaluating or y-evaluating function here
    if Mode == "train":
        Train(T_data, dirPath, State1_Occs, State2_Occs, OnSites_state1, OnSites_state2,
                rateData, dispData, specsToTrain, VacSpec, start_ep, end_ep, interval, N_train,
                gNet, lRate=learning_rate, scratch_if_no_init=scratch_if_no_init)

    elif Mode == "eval":
        train_diff, valid_diff = Evaluate(T_net, dirPath, State1_Occs, State2_Occs,
                OnSites_st1, OnSites_st2, rates, disps,
                SpecsToTrain, VacSpec, start_ep, end_ep,
                interval, N_train, gNet)
        np.save("training_{0}_{1}_n{2}c8.npy".format(T_data, T_net, nLayers), train_diff)
        np.save("validation_{0}_{1}_n{2}c8.npy".format(T_data, T_net, nLayers), valid_diff)

    elif Mode == "getY":
        y1Vecs, y2Vecs = Gather_Y(T_net, dirPath, State1_Occs, State2_Occs,
                OnSites_st1, OnSites_st2, SpecsToTrain, VacSpec, epoch, gNet)


if __name__ == "main":
    main(list(sys.argv))
