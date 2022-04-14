#!/usr/bin/env python
# coding: utf-8
CrysDatPath = "/home/sohamc2/HEA_FCC/CrysDat/"
DataPath = "/home/sohamc2/HEA_FCC/MDMC/ML_runs/DataSets/"
import os
RunPath = os.getcwd() + "/"
import sys
from GCNetRun import Load_Data, Load_crysDats, makeComputeData, makeProdTensor

import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import h5py
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from SymmLayers import GConv, R3Conv, R3ConvSites, GAvg


class GCNet(nn.Module):
    def __init__(self, GnnPerms, gdiags, NNsites, SitesToShells,
                dim, N_ngb, NSpec, mean=0.0, std=0.1, b=1.0, nl=3):
        
        super().__init__()
        modules = []
        modules += [GConv(NSpec, 8, GnnPerms, NNsites, N_ngb, mean=mean, std=std), 
                nn.Softplus(beta=b), GAvg()]

        for i in range(nl):
            modules += [GConv(8, 8, GnnPerms, NNsites, N_ngb, mean=mean, std=std), 
                    nn.Softplus(beta=b), GAvg()]

        modules += [GConv(8, 1, GnnPerms, NNsites, N_ngb, mean=mean, std=std), 
                nn.Softplus(beta=b), GAvg()]

        modules += [R3ConvSites(SitesToShells, GnnPerms, gdiags, NNsites, N_ngb, 
            dim, mean=mean, std=std)]
        
        self.net = nn.Sequential(*modules)
    
    def forward(self, InState):
        y = self.net(InState)
        return y

"""## Write the training function"""
def Train(dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2,
          y1Target, y2Target, SpecsToTrain, VacSpec, start_ep, end_ep,
          interval, N_train, gNet, lRate=0.001, scratch_if_no_init=True):
    
    Ndim = disps.shape[2]
    N_batch = 128
    # Convert compute data to pytorch tensors
    state1Data = pt.tensor(State1_Occs[:N_train]).double()
    state2Data = pt.tensor(State2_Occs[:N_train]).double()
    y1TargetData = pt.tensor(y1Target).double().to(device)
    y2TargetData = pt.tensor(y2Target).double().to(device)

    On_st1 = None
    On_st2 = None   
    if SpecsToTrain == [VacSpec]:
        assert OnSites_st1 == OnSites_st2 == None
        print("Training on Vacancy".format(SpecsToTrain))
    else:
        print("Training Species : {}".format(SpecsToTrain)) 
        On_st1 = makeProdTensor(OnSites_st1[:N_train], Ndim).long()
        On_st2 = makeProdTensor(OnSites_st2[:N_train], Ndim).long()
    
    try:
        gNet.load_state_dict(pt.load(dirPath + "/ep_{1}.pt".format(T, start_ep)))
        print("Starting from epoch {}".format(start_ep), flush=True)
    except:
        if scratch_if_no_init:
            print("No Network found. Starting from scratch", flush=True)
        else:
            raise ValueError("No saved network found in {} at epoch {}".format(dirPath, start_ep))
    
    optimizer = pt.optim.Adam(gNet.parameters(), lr=lRate, weight_decay=0.001)
    print("Starting Training loop") 
    for epoch in tqdm(range(start_ep, end_ep + 1), position=0, leave=True):
        
        ## checkpoint
        if epoch%interval==0:
            pt.save(gNet.state_dict(), dirPath + "/ep_{1}.pt".format(T, epoch))
            
        for batch in range(0, N_train, N_batch):
            optimizer.zero_grad()
            
            end = min(batch + N_batch, N_train)

            state1Batch = state1Data[batch : end].to(device)
            state2Batch = state2Data[batch : end].to(device)
            
            y1_target_Batch = y1TargetData[batch : end]
            y2_target_Batch = y2TargetData[batch : end]
            
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

            loss1 = pt.sum(pt.norm((y1_target_Batch - y1), dim=1)**2)
            loss2 = pt.sum(pt.norm((y2_target_Batch - y2), dim=1)**2)
            loss = (loss1 + loss2)/(2.0*(end - batch))
            loss.backward()
            optimizer.step()


def Evaluate(dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2,
             y1Target, y2Target, SpecsToTrain, VacSpec,
             start_ep, end_ep, interval, N_train, gNet):
    
    Ndim = disps.shape[2]
    N_batch = 512
    # Convert compute data to pytorch tensors
    state1Data = pt.tensor(State1_Occs).double()
    state2Data = pt.tensor(State2_Occs).double()
    y1TargetData = pt.tensor(y1Target).double().to(device)
    y2TargetData = pt.tensor(y2Target).double().to(device)

    Nsamples = state1Data.shape[0]

    print("Evaluating species: {}, Vacancy label: {}".format(SpecsToTrain, VacSpec))
    print("Sample Jumps: {}, Training: {}, Validation: {}".format(Nsamples, N_train, Nsamples-N_train))
    print("Evaluating with networks at: {}".format(dirPath))
    On_st1 = None
    On_st2 = None
    
    if SpecsToTrain == [VacSpec]:
        assert OnSites_st1 == OnSites_st2 == None
    else:
        On_st1 = makeProdTensor(OnSites_st1, Ndim).long()
        On_st2 = makeProdTensor(OnSites_st2, Ndim).long()

    def compute(startSample, endSample):
        diff_epochs = []
        with pt.no_grad():
            for epoch in tqdm(range(start_ep, end_ep + 1, interval), position=0, leave=True):
                ## load checkpoint
                gNet.load_state_dict(pt.load(dirPath + "/ep_{1}.pt".format(epoch), map_location=device))
                loss = 0 
                for batch in range(startSample, endSample, N_batch):
                    end = min(batch + N_batch, endSample)

                    state1Batch = state1Data[batch : end].to(device)
                    state2Batch = state2Data[batch : end].to(device)
                    
                    y1_target_Batch = y1TargetData[batch : end]
                    y2_target_Batch = y2TargetData[batch : end]
                    
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

                    loss1 = pt.sum(pt.norm((y1_target_Batch - y1), dim=1)**2)
                    loss2 = pt.sum(pt.norm((y2_target_Batch - y2), dim=1)**2)
                    loss += (loss1 + loss2).item()
                    
                diff_epochs.append(loss)

        return np.array(diff_epochs)
    
    train_diff = compute(0, N_train)
    test_diff = compute(N_train, Nsamples)

    return train_diff/(2.0*N_train), test_diff/(2.0*(Nsamples - N_train))


def Gather_Y(dirPath, State1_Occs, State2_Occs,
                OnSites_st1, OnSites_st2, SpecsToTrain,
                VacSpec, epoch, gNet, Ndim):
    
    N_batch = 256
    # Convert compute data to pytorch tensors
    state1Data = pt.tensor(State1_Occs).double()
    Nsamples = state1Data.shape[0]
    state2Data = pt.tensor(State2_Occs).double()
    On_st1 = None
    On_st2 = None 
    
    if SpecsToTrain == [VacSpec]:
        assert OnSites_st1 == OnSites_st2 == None
    else:
        On_st1 = makeProdTensor(OnSites_st1, Ndim).long()
        On_st2 = makeProdTensor(OnSites_st2, Ndim).long()

    y1Vecs = np.zeros((Nsamples, 3))
    y2Vecs = np.zeros((Nsamples, 3))

    print("Evaluating network on species: {}, Vacancy label: {}".format(SpecsToTrain, VacSpec))
    print("Network: {}".format(dirPath) + "/ep_{1}.pt".format(T, epoch))
    with pt.no_grad():
        ## load checkpoint
        gNet.load_state_dict(pt.load(dirPath + "/ep_{1}.pt".format(epoch), map_location=device))
                
        for batch in tqdm(range(0, Nsamples, N_batch), position=0, leave=True):
            end = min(batch + N_batch, Nsamples)

            state1Batch = state1Data[batch : end].to(device)
            state2Batch = state2Data[batch : end].to(device)
            
            y1 = gNet(state1Batch)
            y2 = gNet(state2Batch)
            
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


def main(args):
    print("Running at : "+ RunPath)

    # Get run parameters
    count=1
    FileName = args[count] # Name of data file to train on
    count += 1
    
    Jac_iter = int(args[count]) # target jacobian iteration to learn vectors from
    count += 1
    
    Mode = args[count] # "train" mode or "eval" mode or "getY" mode
    count += 1

    nLayers = int(args[count])
    count += 1
    
    scratch_if_no_init = bool(int(args[count]))
    count += 1
    
    T_data = int(args[count]) # temperature to load data from
    # Note : for binary random alloys, this should is the training composition instead of temperature
    count += 1
    
    start_ep = int(args[count])
    count += 1
    
    end_ep = int(args[count])
    count += 1
    
    specTrain = args[count] # which species to train collectively: eg - 123 for species 1, 2 and 3
    # The entry is order independent as it is sorted later
    count += 1
    
    VacSpec = int(args[count]) # integer label for vacancy species
    count += 1
    
    N_train = int(args[count]) # How many INITIAL STATES to consider for training
    count += 1
    
    interval = int(args[count]) # for train mode, interval to save and for eval mode, interval to load
    count += 1
    
    learning_Rate = float(args[count]) if len(args)==count+1 else 0.001
        
    if Mode == "train" or Mode == "eval":
        AllJumps = False # whether to consider all jumps out of the samples or just stochastically selected one
    elif Mode == "getY":
        AllJumps = True
        
    # Load data
    state1List, state2List, dispList, rateList, AllJumpRates, jmpSelects = Load_Data(FileName)
    
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

    # Load target jacobian data
    y1Target = np.load("y1_{}".format(T)+"_"+direcString+"jac_{}.npy".format(Jac_iter))
    y2Target = np.load("y2_{}".format(T)+"_"+direcString+"jac_{}.npy".format(Jac_iter))
    
    dirNameNets = "ep_T_{0}_jac_{1}_n{2}c8".format(T_data, direcString, nLayers)
    if Mode == "eval" or Mode == "getY":
        prepo = "saved at"
    
    elif Mode == "train":
        prepo = "saving in"

    # check if a run directory exists
    dirPath = RunPath + dirNameNets
    exists = os.path.isdir(dirPath)
    
    if not exists:
        if scratch_if_no_init == False:
            raise FileNotFoundError("Target directory does not exist but looking for existing networks.")
        else:
            if start_ep == 0:
                os.mkdir(dirPath)
            elif start_ep > 0:
                raise FileNotFoundError("Training directory does not exist but start epoch greater than zero: {}\ndirectory given: {}".format(start_ep, dirPath))

    print("Running in Mode {} with networks {} {}".format(Mode, prepo, dirPath))
    
    # Load crystal parameters
    GpermNNIdx, NNsiteList, siteShellIndices, GIndtoGDict, JumpNewSites, dxJumps = Load_crysDats()
    N_ngb = NNsiteList.shape[0]
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
    gdiags = gdiagsCpu
    
    # Make a network to either train from scratch or load saved state into
    gNet = GCNet(GnnPerms, gdiags, NNsites, SitesToShells, Ndim, N_ngb, NSpec,
            mean=0.02, std=0.2, b=1.0, nl=nLayers).double().to(device)
        
    # Call MakeComputeData here
    State1_Occs, State2_Occs, _, _, OnSites_state1, OnSites_state2 =\
            makeComputeData(state1List, state2List, dispList, specsToTrain, VacSpec, 
                    rateList, AllJumpRates, JumpNewSites, dxJumps, NNsiteList,
                    N_train, AllJumps=AllJumps)
    print("Done Creating occupancy tensors")
    
    # Call Training or evaluating or y-gathering function here
    if Mode == "train":
        Train(dirPath, State1_Occs, State2_Occs, OnSites_state1, OnSites_state2,
              y1Target, y2Target, specsToTrain, VacSpec, start_ep, end_ep, interval, N_train, 
              gNet, lRate=learning_Rate, scratch_if_no_init=scratch_if_no_init)

    elif Mode == "eval":
        
        train_diff, valid_diff = Evaluate(dirPath, State1_Occs, State2_Occs,
                OnSites_state1, OnSites_state2, y1Target, y2Target,
                specsToTrain, VacSpec, start_ep, end_ep,
                interval, N_train, gNet)
        
        np.save("tr_{3}_{0}_fity_jac{1}_n{2}c8.npy".format(T_data, Jac_iter, nLayers, direcString),
                train_diff)
        
        np.save("val_{3}_{0}_fity_jac{1}_n{2}c8.npy".format(T_data, Jac_iter, nLayers, direcString),
                valid_diff)
        
    elif Mode == "getY":
        assert AllJumps
        y1Vecs, y2Vecs = Gather_Y(dirPath, State1_Occs, State2_Occs,
                                  OnSites_state1, OnSites_state2, specsToTrain,
                                  VacSpec, start_ep, gNet, Ndim)
        
        np.save("y1_{3}_{0}_fit_jac{1}_n{2}c8.npy".format(T_data, Jac_iter, nLayers,
                                                      direcString), y1Vecs)
        np.save("y2_{3}_{0}_fit_jac{1}_n{2}c8.npy".format(T_data, Jac_iter, nLayers,
                                                      direcString), y2Vecs)


if __name__ == "__main__":
    main(list(sys.argv))
