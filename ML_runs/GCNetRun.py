import os
import sys
import argparse
RunPath = os.getcwd() + "/"
#CrysPath = "/home/sohamc2/HEA_FCC/MDMC/CrysDat_FCC/"
#DataPath = "/home/sohamc2/HEA_FCC/MDMC/ML_runs/DataSets/"
ModulePath = "/home/sohamc2/HEA_FCC/MDMC/Symm_Network/"

sys.path.append(ModulePath)

import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import h5py
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from SymmLayers import GConv, R3Conv, R3ConvSites, GAvg, GCNet
import copy

device=None
if pt.cuda.is_available():
    print(pt.cuda.get_device_name())
    device = pt.device("cuda:0")
    DeviceIDList = list(range(pt.cuda.device_count()))
else:
    device = pt.device("cpu")


class WeightNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.L1 = nn.Linear(1, width)
        self.L2 = nn.Linear(width, 1)

    def forward(self, x):
        return F.softplus(self.L2(F.softplus(self.L1(x))))



class GCNetRes(GCNet):
    
    def forward(self, InState):
        
        layerInd = 0
        y = self.net[layerInd + 2](self.net[layerInd + 1](self.net[layerInd](InState)))

        for layerInd in range(3, len(self.net) - 4, 3):
            y = y + self.net[layerInd + 2](self.net[layerInd + 1](self.net[layerInd](y)))
        
        layerInd = len(self.net) - 4
        y = self.net[layerInd + 2](self.net[layerInd + 1](self.net[layerInd](y)))

        y = self.net[-1](y)

        return y

class GCSubNet(nn.Module):
    def __init__(self, GnnPerms, gdiags, NNsites, SitesToShells,
                dim, N_ngb, NSpec=5, specsToTrain=[5], mean=0.0, std=0.1, b=1.0, nl=3, nch=8):

        super().__init__()
        
        NspecsTrain = len(specsToTrain)
        NsubNets = NSpec - NspecsTrain
        NSpecChannels = NspecsTrain + 1
        
        self.subNets = nn.ModuleList([])
        for subNetInd in range(NsubNets):
            # Create a subNetwork
            subNet = GCNet(GnnPerms, gdiags, NNsites, SitesToShells, dim, N_ngb,
                    NSpecChannels, mean=mean, std=std, b=b, nl=nl, nch=nch)
            # store it in the module list
            self.subNets.append(subNet)
    
    def Distribute_subNets(self):
        for subNetInd, subNet in enumerate(self.subNets):
            subNet.to("cuda:{}".format(DeviceIDList[subNetInd % len(DeviceIDList)]))
    
    def forward(self, InState, specsToTrain_chIdx, BackgroundSpecs):
        
        Input = InState[:, specsToTrain_chIdx + BackgroundSpecs[0], :].to(device)
        y = self.subNets[0](Input)
        for bkgSpecInd in range(1, len(BackgroundSpecs)):
            Input = InState[:, specsToTrain_chIdx + BackgroundSpecs[bkgSpecInd], :].to("cuda:{}".format(DeviceIDList[bkgSpecInd % len(DeviceIDList)]))
            y += self.subNets[bkgSpecInd](Input).to(device)
            
        return y

class GCSubNetRes(nn.Module):
    def __init__(self, GnnPerms, gdiags, NNsites, SitesToShells,
                dim, N_ngb, NSpec=5, specsToTrain=[5], mean=0.0, std=0.1, b=1.0, nl=3, nch=8):
        
        super().__init__()

        NspecsTrain = len(specsToTrain)
        NsubNets = NSpec - NspecsTrain
        NSpecChannels = NspecsTrain + 1
        
        self.subNets = nn.ModuleList([])
        for subNetInd in range(NsubNets):
            # Create a residual subNetwork
            subNet = GCNetRes(GnnPerms, gdiags, NNsites, SitesToShells, dim, N_ngb,
                    NSpecChannels, mean=mean, std=std, b=b, nl=nl, nch=nch)
            # store it in the module list
            self.subNets.append(subNet)

    def Distribute_subNets(self):
        for subNetInd, subNet in enumerate(self.subNets):
            subNet.to("cuda:{}".format(DeviceIDList[subNetInd % len(DeviceIDList)]))
    
    def forward(self, InState, specsToTrain_chIdx, BackgroundSpecs):
        
        Input = InState[:, specsToTrain_chIdx + BackgroundSpecs[0], :].to(device)
        y = self.subNets[0](Input)
        for bkgSpecInd in range(1, len(BackgroundSpecs)):
            Input = InState[:, specsToTrain_chIdx + BackgroundSpecs[bkgSpecInd], :].to("cuda:{}".format(DeviceIDList[bkgSpecInd % len(DeviceIDList)]))
            y += self.subNets[bkgSpecInd](Input).to(device)
            
        return y

def Load_crysDats(nn, CrysDatPath):
    ## load the crystal data files
    if nn == 1:
        GpermNNIdx = np.load(CrysDatPath + "GroupNNpermutations.npy")
        NNsiteList = np.load(CrysDatPath + "NNsites_sitewise.npy")
    elif nn == 2:
        GpermNNIdx = np.load(CrysDatPath + "GroupNNpermutations_2nn.npy")
        NNsiteList = np.load(CrysDatPath + "NNsites_sitewise_2nn.npy")
    
    else:
        raise ValueError("Filter range should be 1 or 2 nn. Entered: {}".format(nn))

    siteShellIndices = np.load(CrysDatPath + "SitesToShells.npy")
    JumpNewSites = np.load(CrysDatPath + "JumpNewSiteIndices.npy")
    dxJumps = np.load(CrysDatPath + "dxList.npy")
    with open(CrysDatPath + "GroupCartIndices.pkl", "rb") as fl:
        GIndtoGDict = pickle.load(fl)
    return GpermNNIdx, NNsiteList, siteShellIndices, GIndtoGDict, JumpNewSites, dxJumps

def Load_Data(DataPath):
    with h5py.File(DataPath, "r") as fl:
        try:
            perm = fl["Permutation"]
            print("found permuation")
        except:        
            perm = np.arange(len(fl["InitStates"]))

        state1List = np.array(fl["InitStates"])[perm]
        state2List = np.array(fl["FinStates"])[perm]
        dispList = np.array(fl["SpecDisps"])[perm]
        rateList = np.array(fl["rates"])[perm]
        
        try:
            AllJumpRates = np.array(fl["AllJumpRates"])[perm]
        except:
            AllJumpRates = None
            print("All Jump Rates not provided in data set. Make sure AllJumps is not set to True with train or eval mode active.")
    
    return state1List, state2List, dispList, rateList, AllJumpRates

def makeComputeData(state1List, state2List, dispList, specsToTrain, VacSpec, rateList,
        AllJumpRates, JumpNewSites, dxJumps, NNsiteList, N_train, AllJumps=False, mode="train"):
    
    if not isinstance(AllJumpRates, np.ndarray):
        if AllJumps and (mode=="train" or mode=="eval"):
            raise ValueError("All Rates not provided. Cannot do training or evaluation.")

    # make the input tensors
    Nsamples = min(state1List.shape[0], 2*N_train)
    if AllJumps:
        NJumps = Nsamples*dxJumps.shape[0]
    else:
        NJumps = Nsamples
    a = np.linalg.norm(dispList[0, VacSpec, :])/np.linalg.norm(dxJumps[0]) 
    specs = np.unique(state1List[0])
    NSpec = specs.shape[0] - 1
    Nsites = state1List.shape[1]
    NNsvac = NNsiteList[1:, 0]
    
    sp_ch = {}
    for sp in specs:
        if sp == VacSpec:
            continue
        
        if sp - VacSpec < 0:
            sp_ch[sp] = sp
        else:
            sp_ch[sp] = sp-1

    State1_occs = np.zeros((NJumps, NSpec, Nsites), dtype=np.int8)
    State2_occs = np.zeros((NJumps, NSpec, Nsites), dtype=np.int8)
    dispData = np.zeros((NJumps, 2, 3))
    

    rateData = np.zeros(NJumps)
    
    # Make the multichannel occupancies
    print("Building Occupancy Tensors for species : {}".format(specsToTrain))
    print("No. of jumps : {}".format(NJumps))
    for samp in tqdm(range(Nsamples), position=0, leave=True):
        state1 = state1List[samp]
        if AllJumps:
            for jInd in range(dxJumps.shape[0]):
                JumpSpec = state1[NNsvac[jInd]]
                state2 = state1[JumpNewSites[jInd]]
                Idx = samp*dxJumps.shape[0]  + jInd
                dispData[Idx, 0, :] =  dxJumps[jInd]*a
                if JumpSpec in specsToTrain:
                    dispData[Idx, 1, :] -= dxJumps[jInd]*a
                
                if isinstance(AllJumpRates, np.ndarray):
                    rateData[Idx] = AllJumpRates[samp, jInd]
                
                for site in range(1, Nsites): # exclude the vacancy site
                    spec1 = state1[site]
                    spec2 = state2[site]
                    State1_occs[Idx, sp_ch[spec1], site] = 1
                    State2_occs[Idx, sp_ch[spec2], site] = 1

        else:
            state2 = state2List[samp]
            for site in range(1, Nsites): # exclude the vacancy site
                spec1 = state1[site]
                spec2 = state2[site]
                State1_occs[samp, sp_ch[spec1], site] = 1
                State2_occs[samp, sp_ch[spec2], site] = 1
            
            dispData[samp, 0, :] = dispList[samp, VacSpec, :]
            dispData[samp, 1, :] = sum(dispList[samp, spec, :] for spec in specsToTrain)

            rateData[samp] = rateList[samp]
    
    # Make the numpy tensor to indicate "on" sites (i.e, those whose y vectors will be collected)
    OnSites_state1 = None
    OnSites_state2 = None
    if specsToTrain != [VacSpec]:
        OnSites_state1 = np.zeros((NJumps, Nsites), dtype=np.int8)
        OnSites_state2 = np.zeros((NJumps, Nsites), dtype=np.int8)
    
        for spec in specsToTrain:
            OnSites_state1 += State1_occs[:, sp_ch[spec], :]
            OnSites_state2 += State2_occs[:, sp_ch[spec], :]
    
    return State1_occs, State2_occs, rateData, dispData, OnSites_state1, OnSites_state2, sp_ch


def makeProdTensor(OnSites, Ndim):
    Onst = pt.tensor(OnSites)
    Onst = Onst.repeat_interleave(Ndim, dim=0)
    Onst = Onst.view(-1, Ndim, OnSites.shape[1])
    return Onst


def makeDataTensors(State1_Occs, State2_Occs, rates, disps, OnSites_st1, OnSites_st2, SpecsToTrain, VacSpec, sp_ch, Nsamps, Ndim=3):
    # Do a small check that species channels were assigned correctly
    if sp_ch is not None:
        for key, item in sp_ch.items():
            if key > VacSpec:
                assert item == key - 1
            else:
                assert key < VacSpec
                assert item == key

    # Convert compute data to pytorch tensors
    state1Data = pt.tensor(State1_Occs[:Nsamps])
    state2Data = pt.tensor(State2_Occs[:Nsamps])
    rateData=None
    if rates is not None:
        rateData = pt.tensor(rates[:Nsamps]).double().to(device)
    On_st1 = None 
    On_st2 = None
    dispData=None
    if SpecsToTrain == [VacSpec]:
        assert OnSites_st1 == OnSites_st2 == None
        print("Training on Vacancy".format(SpecsToTrain)) 
        if disps is not None:
            dispData = pt.tensor(disps[:Nsamps, 0, :]).double().to(device)
    else:
        print("Training Species : {}".format(SpecsToTrain))
        if disps is not None:
            dispData = pt.tensor(disps[:Nsamps, 1, :]).double().to(device)
        
        On_st1 = makeProdTensor(OnSites_st1[:Nsamps], Ndim).long()
        On_st2 = makeProdTensor(OnSites_st2[:Nsamps], Ndim).long()

    return state1Data, state2Data, dispData, rateData, On_st1, On_st2

"""## Write the training loop"""
def Train(T, dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, 
        rates, disps, SpecsToTrain, sp_ch, VacSpec, start_ep, end_ep, interval, N_train,
        gNet, lRate=0.001, scratch_if_no_init=True, batch_size=128, Learn_wt=False, WeightSLP=None, DPr=False):
    Ndim = disps.shape[2] 
    state1Data, state2Data, dispData, rateData, On_st1, On_st2 = makeDataTensors(State1_Occs, State2_Occs, rates, disps,
            OnSites_st1, OnSites_st2, SpecsToTrain, VacSpec, sp_ch, N_train, Ndim=Ndim)

    N_batch = batch_size

    if isinstance(gNet, GCNet):
        if pt.cuda.device_count() > 1 and DPr:
            print("Running on Devices : {}".format(DeviceIDList))
            gNet = nn.DataParallel(gNet, device_ids=DeviceIDList)
            gNet.to(device)

    try:
        gNet.load_state_dict(pt.load(dirPath + "/ep_{1}.pt".format(T, start_ep), map_location=device))
        if Learn_wt:
            WeightSLP.load_state_dict(pt.load(dirPath + "/wt_ep_{1}.pt".format(T, start_ep), map_location="cpu"))
        print("Starting from epoch {}".format(start_ep), flush=True)

    except:
        if scratch_if_no_init:
            print("No Network found. Starting from scratch", flush=True)
        else:
            raise ValueError("Required saved networks not found in {} at epoch {}".format(dirPath, start_ep))

    print("Batch size : {}".format(N_batch)) 
    specTrainCh = [sp_ch[spec] for spec in SpecsToTrain]
    BackgroundSpecs = [[spec] for spec in range(state1Data.shape[1]) if spec not in specTrainCh] 
    
    if isinstance(gNet, GCSubNet) or isinstance(gNet, GCSubNetRes): 
        gNet.Distribute_subNets()
        print("Species Channels to train: {}".format(specTrainCh))
        print("Background species channels: {}".format(BackgroundSpecs))
    

    optims = [pt.optim.Adam(gNet.parameters(), lr=lRate, weight_decay=0.0005)]
    if Learn_wt:
        print("Learning sample reweighting with SLP.") 
        WeightSLP.to(device)
        optims.append(pt.optim.Adam(WeightSLP.parameters(), lr=0.0001))

    print("Starting Training loop") 

    for epoch in tqdm(range(start_ep, end_ep + 1), position=0, leave=True):
        
        ## checkpoint
        if epoch%interval==0:
            pt.save(gNet.state_dict(), dirPath + "/ep_{0}.pt".format(epoch))
            if Learn_wt:
                pt.save(WeightSLP.state_dict(), dirPath + "/wt_ep_{0}.pt".format(epoch))

            
        for batch in range(0, N_train, N_batch):
            
            for opt in optims:
                opt.zero_grad()
            
            end = min(batch + N_batch, N_train)

            state1Batch = state1Data[batch : end].double().to(device)
            state2Batch = state2Data[batch : end].double().to(device)
            
            rateBatch = rateData[batch : end]
            dispBatch = dispData[batch : end]
            
            if isinstance(gNet, GCSubNet) or isinstance(gNet, GCSubNetRes):
                y1 = gNet.forward(state1Batch, specTrainCh, BackgroundSpecs)
                y2 = gNet.forward(state2Batch, specTrainCh, BackgroundSpecs)
            
            else:
                y1 = gNet(state1Batch)
                y2 = gNet(state2Batch)
            
            # sum up everything except the vacancy site
            if SpecsToTrain==[VacSpec]:
                y1 = -pt.sum(y1[:, :, 1:], dim=2)
                y2 = -pt.sum(y2[:, :, 1:], dim=2)
            
            else:
                On_st1Batch = On_st1[batch : end].to(device)
                On_st2Batch = On_st2[batch : end].to(device)
                y1 = pt.sum(y1*On_st1Batch, dim=2)
                y2 = pt.sum(y2*On_st2Batch, dim=2)

            dy = y2 - y1
            sample_Losses = rateBatch * pt.norm((dispBatch + dy), dim=1)**2/6.
            
            if Learn_wt:
                sampleLossesInput = sample_Losses.data.clone().view(-1, 1).to(device)
                SampleReWt = WeightSLP(sampleLossesInput).view(-1)
                diff = pt.sum(SampleReWt * sample_Losses) # multiply sample losses with weight, and sum
            else:
                diff = pt.sum(sample_Losses) # just sum sample losses

            diff.backward()
            # Propagate all optimizers
            for opt in optims:
                opt.step()


def Evaluate(T, dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, 
        rates, disps, SpecsToTrain, sp_ch, VacSpec, start_ep, end_ep, interval, N_train,
        gNet, batch_size=512, DPr=False):
    
    for key, item in sp_ch.items():
        if key > VacSpec:
            assert item == key - 1
        else:
            assert key < VacSpec
            assert item == key

    N_batch = batch_size
    # Convert compute data to pytorch tensors
    Nsamples = State1_Occs.shape[0]

    print("Evaluating species: {}, Vacancy label: {}".format(SpecsToTrain, VacSpec))
    print("Sample Jumps: {}, Training: {}, Validation: {}, Batch size: {}".format(Nsamples, N_train, Nsamples-N_train, N_batch))
    print("Evaluating with networks at: {}".format(dirPath))
    
    Ndim = disps.shape[2] 
    state1Data, state2Data, dispData, rateData, On_st1, On_st2 = makeDataTensors(State1_Occs, State2_Occs, rates, disps,
            OnSites_st1, OnSites_st2, SpecsToTrain, VacSpec, sp_ch, Nsamples, Ndim=Ndim)
    
    specTrainCh = [sp_ch[spec] for spec in SpecsToTrain]
    BackgroundSpecs = [[spec] for spec in range(state1Data.shape[1]) if spec not in specTrainCh] 
    
    if isinstance(gNet, GCSubNet) or isinstance(gNet, GCSubNetRes): 
        print("Species Channels to train: {}".format(specTrainCh))
        print("Background species channels: {}".format(BackgroundSpecs))
    
    # pre-convert to data parallel if required

    elif isinstance(gNet, GCNet):
        if pt.cuda.device_count() > 1 and DPr:
            print("Using data parallel")
            print("Running on Devices : {}".format(DeviceIDList))
            gNet = nn.DataParallel(gNet, device_ids=DeviceIDList)

    def compute(startSample, endSample):
        diff_epochs = []
        with pt.no_grad():
            for epoch in tqdm(range(start_ep, end_ep + 1, interval), position=0, leave=True):
                ## load checkpoint
                if isinstance(gNet, GCSubNet) or isinstance(gNet, GCSubNetRes):
                    gNet.load_state_dict(pt.load(dirPath + "/ep_{0}.pt".format(epoch), map_location=device)) 
                    gNet.Distribute_subNets()
                
                else:
                    gNet.load_state_dict(pt.load(dirPath + "/ep_{0}.pt".format(epoch), map_location=device))
                    gNet.to(device)

 
                diff = 0 
                for batch in range(startSample, endSample, N_batch):
                    end = min(batch + N_batch, endSample)

                    state1Batch = state1Data[batch : end].double().to(device)
                    state2Batch = state2Data[batch : end].double().to(device)
                    
                    rateBatch = rateData[batch : end].to(device)
                    dispBatch = dispData[batch : end].to(device)
                     
                    if isinstance(gNet, GCSubNet) or isinstance(gNet, GCSubNetRes):
                        y1 = gNet.forward(state1Batch, specTrainCh, BackgroundSpecs)
                        y2 = gNet.forward(state2Batch, specTrainCh, BackgroundSpecs)
            
                    else:
                        y1 = gNet(state1Batch)
                        y2 = gNet(state2Batch)
            
                    # sum up everything except the vacancy site if vacancy is indicated
                    if SpecsToTrain==[VacSpec]:
                        y1 = -pt.sum(y1[:, :, 1:], dim=2)
                        y2 = -pt.sum(y2[:, :, 1:], dim=2)
                    
                    else:
                        On_st1Batch = On_st1[batch : end].to(device)
                        On_st2Batch = On_st2[batch : end].to(device)
                        y1 = pt.sum(y1*On_st1Batch, dim=2)
                        y2 = pt.sum(y2*On_st2Batch, dim=2)

                    dy = y2 - y1
                    loss = pt.sum(rateBatch * pt.norm((dispBatch + dy), dim=1)**2)/6.
                    diff += loss.item()

                diff_epochs.append(diff)

        return np.array(diff_epochs)
    
    train_diff = compute(0, N_train)#/(1.0*N_train)
    test_diff = compute(N_train, Nsamples)#/(1.0*(Nsamples - N_train))

    return train_diff, test_diff


def Gather_Y(T, dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, sp_ch, SpecsToTrain, VacSpec, gNet, Ndim, epoch=None, batch_size=256):
    
    for key, item in sp_ch.items():
        if key > VacSpec:
            assert item == key - 1
        else:
            assert key < VacSpec
            assert item == key

    N_batch = batch_size
    # Convert compute data to pytorch tensors
    Nsamples = State1_Occs.shape[0]
    rates=None
    disps=None

    state1Data, state2Data, dispData, rateData, On_st1, On_st2 = makeDataTensors(State1_Occs, State2_Occs, rates, disps,
            OnSites_st1, OnSites_st2, SpecsToTrain, VacSpec, sp_ch, Nsamples, Ndim=Ndim)

    y1Vecs = np.zeros((Nsamples, 3))
    y2Vecs = np.zeros((Nsamples, 3))

    specTrainCh = [sp_ch[spec] for spec in SpecsToTrain]
    BackgroundSpecs = [[spec] for spec in range(state1Data.shape[1]) if spec not in specTrainCh] 
    
    print("Evaluating network on species: {}, Vacancy label: {}".format(SpecsToTrain, VacSpec))
    if epoch is not None:
        print("Network: {}".format(dirPath))
    with pt.no_grad():
        ## load checkpoint
        if epoch is not None:
            print("Loading epoch: {}".format(epoch))
            gNet.load_state_dict(pt.load(dirPath + "/ep_{0}.pt".format(epoch), map_location=device))
             
        if isinstance(gNet, GCSubNet) or isinstance(gNet, GCSubNetRes): 
            gNet.Distribute_subNets()
        
        for batch in tqdm(range(0, Nsamples, N_batch), position=0, leave=True):
            end = min(batch + N_batch, Nsamples)

            state1Batch = state1Data[batch : end].double().to(device)
            state2Batch = state2Data[batch : end].double().to(device)
            
            if isinstance(gNet, GCNet) or isinstance(gNet, GCNetRes):
                y1 = gNet(state1Batch)
                y2 = gNet(state2Batch)

            else:
                y1 = gNet.forward(state1Batch, specTrainCh, BackgroundSpecs)
                y2 = gNet.forward(state2Batch, specTrainCh, BackgroundSpecs)
            
            # sum up everything except the vacancy site if vacancy is indicated
            if SpecsToTrain == [VacSpec]:
                y1 = -pt.sum(y1[:, :, 1:], dim=2)
                y2 = -pt.sum(y2[:, :, 1:], dim=2)
            
            else:
                On_st1Batch = On_st1[batch : end].to(device)
                On_st2Batch = On_st2[batch : end].to(device)
                y1 = pt.sum(y1*On_st1Batch, dim=2)
                y2 = pt.sum(y2*On_st2Batch, dim=2)
            
            y1Vecs[batch : end] = y1.cpu().numpy()
            y2Vecs[batch : end] = y2.cpu().numpy()

    return y1Vecs, y2Vecs

def GetRep(T_net, T_data, dirPath, State1_Occs, State2_Occs, epoch, gNet, LayerIndList, N_train, batch_size=1000,
           avg=True, AllJumps=False):
    
    N_batch = batch_size
    # Convert compute data to pytorch tensors
    state1Data = pt.tensor(State1_Occs)
    Nsamples = state1Data.shape[0]
    Nsites = state1Data.shape[2]
    state2Data = pt.tensor(State2_Occs)
    
    if avg:
        storeDir = RunPath + "StateReps_avg_{}".format(T_net)
    else:
        storeDir = RunPath + "StateReps_{}".format(T_net)

    exists = os.path.isdir(storeDir)
    if not exists:
        os.mkdir(storeDir)
    
    print("computing Representations after layers: {}".format(LayerIndList))
    
    glob_Nch = gNet.net[0].Psi.shape[0]
    with pt.no_grad():
        ## load checkpoint
        gNet.load_state_dict(pt.load(dirPath + "/ep_{0}.pt".format(epoch), map_location=device))
        nLayers = (len(gNet.net)-7)//3
        for LayerInd in LayerIndList:
            ch = gNet.net[LayerInd - 2].Psi.shape[0]
            y1Reps = np.zeros((Nsamples, ch, Nsites))
            y2Reps = np.zeros((Nsamples, ch, Nsites))
            for batch in tqdm(range(0, Nsamples, N_batch), position=0, leave=True):
                end = min(batch + N_batch, Nsamples)

                state1Batch = state1Data[batch : end].double().to(device)
                state2Batch = state2Data[batch : end].double().to(device)

                y1 = gNet.getRep(state1Batch, LayerInd)
                y2 = gNet.getRep(state2Batch, LayerInd)

                y1Reps[batch : end] = y1.cpu().numpy()
                y2Reps[batch : end] = y2.cpu().numpy()
            
            if avg:
                y1RepsTrain = np.mean(y1Reps[:N_train], axis = 0)
                y2RepsTrain = np.mean(y2Reps[:N_train], axis = 0)
                y1RepsVal = np.mean(y1Reps[N_train:], axis = 0)
                y2RepsVal = np.mean(y2Reps[N_train:], axis = 0)
                
                y1RepsTrain_err = np.std(y1Reps[:N_train], axis = 0)/np.sqrt(N_train)
                y2RepsTrain_err = np.std(y2Reps[:N_train], axis = 0)/np.sqrt(N_train)
                y1RepsVal_err = np.std(y1Reps[N_train:], axis = 0)/np.sqrt((Nsamples - N_train))
                y2RepsVal_err = np.std(y2Reps[N_train:], axis = 0)/np.sqrt((Nsamples - N_train))
                
                np.save(storeDir + "/Rep1_trAvg_l{6}_{0}_{1}_n{2}c{5}_all_{3}_{4}.npy".format(T_data, T_net, nLayers, int(AllJumps), epoch, glob_Nch, LayerInd), y1RepsTrain)
                np.save(storeDir + "/Rep2_trAvg_l{6}_{0}_{1}_n{2}c{5}_all_{3}_{4}.npy".format(T_data, T_net, nLayers, int(AllJumps), epoch, glob_Nch, LayerInd), y2RepsTrain)
                np.save(storeDir + "/Rep1_valAvg_l{6}_{0}_{1}_n{2}c{5}_all_{3}_{4}.npy".format(T_data, T_net, nLayers, int(AllJumps), epoch, glob_Nch, LayerInd), y1RepsVal)
                np.save(storeDir + "/Rep2_valAvg_l{6}_{0}_{1}_n{2}c{5}_all_{3}_{4}.npy".format(T_data, T_net, nLayers, int(AllJumps), epoch, glob_Nch, LayerInd), y2RepsVal)

                np.save(storeDir + "/Rep1_tr_stderr_l{6}_{0}_{1}_n{2}c{5}_all_{3}_{4}.npy".format(T_data, T_net, nLayers, int(AllJumps), epoch, glob_Nch, LayerInd), y1RepsTrain_err)
                np.save(storeDir + "/Rep2_tr_stderr_l{6}_{0}_{1}_n{2}c{5}_all_{3}_{4}.npy".format(T_data, T_net, nLayers, int(AllJumps), epoch, glob_Nch, LayerInd), y2RepsTrain_err)
                np.save(storeDir + "/Rep1_val_stderr_l{6}_{0}_{1}_n{2}c{5}_all_{3}_{4}.npy".format(T_data, T_net, nLayers, int(AllJumps), epoch, glob_Nch, LayerInd), y1RepsVal_err)
                np.save(storeDir + "/Rep2_val_stderr_l{6}_{0}_{1}_n{2}c{5}_all_{3}_{4}.npy".format(T_data, T_net, nLayers, int(AllJumps), epoch, glob_Nch, LayerInd), y2RepsVal_err)
            
            else:
                np.save(storeDir + "/Rep1_l{6}_{0}_{1}_n{2}c{5}_all_{3}_{4}.npy".format(T_data, T_net, nLayers, int(AllJumps), epoch, glob_Nch, LayerInd), y1Reps)
                np.save(storeDir + "/Rep2_l{6}_{0}_{1}_n{2}c{5}_all_{3}_{4}.npy".format(T_data, T_net, nLayers, int(AllJumps), epoch, glob_Nch, LayerInd), y2Reps)


def main(args):
    print("Running at : "+ RunPath)

    # Get run parameters
    FileName = args.DataPath # Name of data file to train on
    
    #CrystalType = args.Crys
    CrysPath = args.CrysDatPath
    
    Mode = args.Mode # "train" mode or "eval" mode or "getY" mode
    
    nLayers = args.Nlayers
    
    ch = args.Nchannels
    
    filter_nn = args.ConvNgbRange
    
    Residual_training = args.Residual
    subNetwork_training = args.SubNet
    
    if Mode=="getRep" and (Residual_training or subNetwork_training):
        raise NotImplementedError("getRep is not currently supported in residual and subnetwork training mode.")

    scratch_if_no_init = args.Scratch

    DPr = args.DatPar

    T_data = args.Tdata
    # Note : for binary random alloys, this should is the training composition instead of temperature

    T_net = args.TNet # must be same as T_data if "train", can be different if "getY" or "eval"

    if Mode=="train" and T_data != T_net:
        raise ValueError("Different temperatures in training mode not allowed")
    
    start_ep = args.Start_epoch
    end_ep = args.End_epoch

    if not (Mode == "train" or Mode == "eval"):
        print("Mode : {}, setting end epoch to start epoch".format(Mode))
        end_ep = start_ep
    
    specTrain = args.SpecTrain
    
    VacSpec = args.VacSpec
    
    AllJumps = args.AllJumps 
    
    AllJumps_net_type = args.AllJumpsNetType
    
    N_train = args.N_train

    batch_size = args.Batch_size
    interval = args.Interval # for train mode, interval to save and for eval mode, interval to load
    learning_Rate = args.Learning_rate

    wt_means = args.Mean_wt
    wt_std = args.Std_wt

    Learn_wt = args.Learn_weights
    
    wtNet = None
    if Learn_wt:
        wtNet = WeightNet(width=128).double().to(device) 

    if not (Mode == "train" or Mode == "eval" or Mode == "getY" or Mode == "getRep"):
        raise ValueError("Mode needs to be train, eval, getY or getRep but given : {}".format(Mode))

    if Mode == "train":
        if T_data != T_net:
            raise ValueError("Training and Testing condition must be the same")

    # Load data
    state1List, state2List, dispList, rateList, AllJumpRates = Load_Data(FileName)
    
    specs = np.unique(state1List[0])
    NSpec = specs.shape[0] - 1
    Nsites = state1List.shape[1]
     
    specsToTrain = [int(specTrain[i]) for i in range(len(specTrain))]
    specsToTrain = sorted(specsToTrain)
    
    direcString=""
    if specsToTrain == [VacSpec]:
        direcString = "vac"
    else:
        for spec in specsToTrain:
            direcString += "{}".format(spec)

    # This is where networks will be saved to and loaded from
    dirNameNets = "ep_T_{0}_{1}_n{2}c{4}_all_{3}".format(T_net, direcString, nLayers, int(AllJumps), ch)
    if Mode == "eval" or Mode == "getY" or Mode=="getRep":
        prepo = "saved at"
        dirNameNets = "ep_T_{0}_{1}_n{2}c{4}_all_{3}".format(T_net, direcString, nLayers, int(AllJumps_net_type), ch)
    
    if Mode == "train":
        prepo = "saving in"

    # check if a run directory exists
    dirPath = RunPath + dirNameNets
    exists = os.path.isdir(dirPath)
    
    if not exists:
        if start_ep == 0:
            os.mkdir(dirPath)
        elif start_ep > 0:
            raise ValueError("Training directory does not exist but start epoch greater than zero: {}\ndirectory given: {}".format(start_ep, dirPath))

    print("Running in Mode {} with networks {} {}".format(Mode, prepo, dirPath))

    print(pt.__version__)
    
    # Load crystal parameters
    GpermNNIdx, NNsiteList, siteShellIndices, GIndtoGDict, JumpNewSites, dxJumps = Load_crysDats(filter_nn, CrysPath)
    N_ngb = NNsiteList.shape[0]
    print("Filter neighbor range : {}nn. Filter neighborhood size: {}".format(filter_nn, N_ngb - 1))
    Nsites = NNsiteList.shape[1]
    SitesToShells = pt.tensor(siteShellIndices).long().to(device)
    GnnPerms = pt.tensor(GpermNNIdx).long().to(device)
    NNsites = pt.tensor(NNsiteList).long().to(device)

    Ng = GnnPerms.shape[0]
    Ndim = dispList.shape[2]
    gdiagsCpu = pt.zeros(Ng*Ndim, Ng*Ndim).double()
    for gInd, gCart in GIndtoGDict.items():
        rowStart = gInd * Ndim
        rowEnd = (gInd + 1) * Ndim
        gdiagsCpu[rowStart : rowEnd, rowStart : rowEnd] = pt.tensor(gCart)
    gdiags = gdiagsCpu#.to(device)


    # Call MakeComputeData here
    State1_Occs, State2_Occs, rateData, dispData, OnSites_state1, OnSites_state2, sp_ch = makeComputeData(state1List, state2List, dispList,
            specsToTrain, VacSpec, rateList, AllJumpRates, JumpNewSites, dxJumps, NNsiteList, N_train, AllJumps=AllJumps, mode=Mode)
    print("Done Creating occupancy tensors. Species channels: {}".format(sp_ch))

    # Make a network to either train from scratch or load saved state into
    if not Residual_training:
        if not subNetwork_training:
            print("Running in Non-Residual Non-Subnet Convolution mode")
            gNet = GCNet(GnnPerms, gdiags, NNsites, SitesToShells, Ndim, N_ngb, NSpec,
                    mean=wt_means, std=wt_std, b=1.0, nl=nLayers, nch=ch).double().to(device)

        else:
            print("Running in Non-Residual Binary SubNetwork Convolution mode")
            gNet = GCSubNet(GnnPerms, gdiags, NNsites, SitesToShells, Ndim, N_ngb, NSpec,
                    specsToTrain, mean=wt_means, std=wt_std, b=1.0, nl=nLayers, nch=ch).double().to(device)

    else:
        if not subNetwork_training:
            print("Running in Residual Convolution mode")
            gNet = GCNetRes(GnnPerms, gdiags, NNsites, SitesToShells, Ndim, N_ngb, NSpec,
                    mean=wt_means, std=wt_std, b=1.0, nl=nLayers, nch=ch).double().to(device)

        else:
            print("Running in Residual Binary SubNetwork Convolution mode")
            gNet = GCSubNetRes(GnnPerms, gdiags, NNsites, SitesToShells, Ndim, N_ngb, NSpec,
                    specsToTrain, mean=wt_means, std=wt_std, b=1.0, nl=nLayers, nch=ch).double().to(device)

    # Call Training or evaluating or y-evaluating function here
    N_train_jumps = (N_ngb - 1)*N_train if AllJumps else N_train
    if Mode == "train":
        Train(T_data, dirPath, State1_Occs, State2_Occs, OnSites_state1, OnSites_state2,
                rateData, dispData, specsToTrain, sp_ch, VacSpec, start_ep, end_ep, interval, N_train_jumps,
                gNet, lRate=learning_Rate, scratch_if_no_init=scratch_if_no_init, batch_size=batch_size,
                Learn_wt=Learn_wt, WeightSLP=wtNet, DPr=DPr)

    elif Mode == "eval":
        train_diff, valid_diff = Evaluate(T_net, dirPath, State1_Occs, State2_Occs,
                OnSites_state1, OnSites_state2, rateData, dispData,
                specsToTrain, sp_ch, VacSpec, start_ep, end_ep,
                interval, N_train_jumps, gNet, batch_size=batch_size, DPr=DPr)
        np.save("tr_{4}_{0}_{1}_n{2}c{5}_all_{3}.npy".format(T_data, T_net, nLayers, int(AllJumps), direcString, ch), train_diff/(1.0*N_train))
        np.save("val_{4}_{0}_{1}_n{2}c{5}_all_{3}.npy".format(T_data, T_net, nLayers, int(AllJumps), direcString, ch), valid_diff/(1.0*N_train))

    elif Mode == "getY":
        y1Vecs, y2Vecs = Gather_Y(T_net, dirPath, State1_Occs, State2_Occs,
                OnSites_state1, OnSites_state2, sp_ch, specsToTrain, VacSpec, gNet, Ndim, epoch=start_ep, batch_size=batch_size)
        np.save("y1_{4}_{0}_{1}_n{2}c{6}_all_{3}_{5}.npy".format(T_data, T_net, nLayers, int(AllJumps), direcString, start_ep, ch), y1Vecs)
        np.save("y2_{4}_{0}_{1}_n{2}c{6}_all_{3}_{5}.npy".format(T_data, T_net, nLayers, int(AllJumps), direcString, start_ep, ch), y2Vecs)
    
    elif Mode == "getRep":
        GetRep(T_net, T_data, dirPath, State1_Occs, State2_Occs, start_ep, gNet, args.RepLayer, N_train_jumps, batch_size=batch_size,
           avg=args.RepLayerAvg, AllJumps=AllJumps)

    print("All done\n\n")

# Add argument parser
parser = argparse.ArgumentParser(description="Input parameters for using GCnets", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-DP", "--DataPath", metavar="/path/to/data", type=str, help="Path to Data file.")
parser.add_argument("-cr", "--CrysDatPath", metavar="/path/to/crys/dat", type=str, help="Path to crystal Data.")

parser.add_argument("-m", "--Mode", metavar="M", type=str, help="Running mode (one of train, eval, getY, getRep). If getRep, then layer must specified with -RepLayer.")
parser.add_argument("-rl","--RepLayer", metavar="[L1, L2,..]", type=int, nargs="+", help="Layers to extract representation from (count starts from 0)")
parser.add_argument("-rlavg","--RepLayerAvg", action="store_true", help="Whether to average Representations across samples (training and validation will be made separate)")

parser.add_argument("-nl", "--Nlayers",  metavar="L", type=int, help="No. of layers of the neural network.")
parser.add_argument("-nch", "--Nchannels", metavar="Ch", type=int, help="No. of representation channels in non-input layers.")
parser.add_argument("-cngb", "--ConvNgbRange", type=int, default=1, metavar="NN", help="Nearest neighbor range of convolutional filters.")


parser.add_argument("-rn", "--Residual", action="store_true", help="Whether to do residual training.")
parser.add_argument("-sn", "--SubNet", action="store_true", help="Whether to train pairwise subnetworks.")
parser.add_argument("-scr", "--Scratch", action="store_true", help="Whether to create new network and start from scratch")
parser.add_argument("-DPr", "--DatPar", action="store_true", help="Whether to use data parallelism. Note - does not work for residual or subnet models. Used only in Train and eval modes.")

parser.add_argument("-td", "--Tdata", metavar="T", type=int, help="Temperature to read data from")
parser.add_argument("-tn", "--TNet", metavar="T", type=int, help="Temperature to use networks from\n For example one can evaluate a network trained on 1073 K data, on the 1173 K data, to see what it does.")
parser.add_argument("-sep", "--Start_epoch", metavar="Ep", type=int, help="Starting epoch (for training, this network will be read in.)")
parser.add_argument("-eep", "--End_epoch", metavar="Ep", type=int, help="Ending epoch (for training, this will be the last epoch.)")

parser.add_argument("-sp", "--SpecTrain", metavar="s1s2s3", type=str, help="species to consider, order independent (Eg, 123 or 213 etc for species 1, 2 and 3")
parser.add_argument("-vSp", "--VacSpec", metavar="SpV", type=int, default=0, help="species index of vacancy, must match dataset, default 0")


parser.add_argument("-aj", "--AllJumps", action="store_true", help="Whether to train on all jumps, or single selected jumps out of a state.")
parser.add_argument("-ajn", "--AllJumpsNetType", action="store_true", help="Whether to use network trained on all jumps, or single selected jumps out of a state.")

parser.add_argument("-nt", "--N_train", type=int, default=10000, help="No. of training samples.")
parser.add_argument("-i", "--Interval", type=int, default=1, help="Epoch intervals in which to save or load networks.")
parser.add_argument("-lr", "--Learning_rate", type=float, default=0.001, help="Learning rate for Adam algorithm.")
parser.add_argument("-bs", "--Batch_size", type=int, default=128, help="size of a single batch of samples.")
parser.add_argument("-wm", "--Mean_wt", type=float, default=0.02, help="Initialization mean value of weights.")
parser.add_argument("-ws", "--Std_wt", type=float, default=0.2, help="Initialization standard dev of weights.")
parser.add_argument("-lw", "--Learn_weights", action="store_true", help="Whether to learn reweighting of samples.")

parser.add_argument("-d", "--DumpArgs", action="store_true", help="Whether to dump arguments in a file")
parser.add_argument("-dpf", "--DumpFile", metavar="F", type=str, help="Name of file to dump arguments to (can be the jobID in a cluster for example).")

if __name__ == "__main__":
    # main(list(sys.argv))

    args = parser.parse_args()    
    if args.DumpArgs:
        print("Dumping arguments to: {}".format(args.DumpFile))
        opts = vars(args)
        with open(RunPath + args.DumpFile, "w") as fl:
            for key, val in opts.items():
                fl.write("{}: {}\n".format(key, val))

    main(args)
