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


device=None
if pt.cuda.is_available():
    print(pt.cuda.get_device_name())
    device = pt.device("cuda:0")
else:
    device = pt.device("cpu")

class GCNet(nn.Module):
    def __init__(self, GnnPerms, gdiags, NNsites, SitesToShells,
                dim, N_ngb, NSpec, mean=0.0, std=0.1, b=1.0, nl=3, nch=8):
        
        super().__init__()
        modules = []
        modules += [GConv(NSpec, nch, GnnPerms, NNsites, N_ngb, mean=mean, std=std), 
                nn.Softplus(beta=b), GAvg()]

        for i in range(nl):
            modules += [GConv(nch, nch, GnnPerms, NNsites, N_ngb, mean=mean, std=std), 
                    nn.Softplus(beta=b), GAvg()]

        modules += [GConv(nch, 1, GnnPerms, NNsites, N_ngb, mean=mean, std=std), 
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
          interval, N_train, gNet, lRate=0.001, scratch_if_no_init=True, Train_mode=None, Freeze=False, wt_norms=False):
    
    Ndim = y1Target.shape[1]
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
        gNet.load_state_dict(pt.load(dirPath + "/ep_{0}.pt".format(start_ep)))
        print("Starting from epoch {}".format(start_ep), flush=True)

    except:
        if scratch_if_no_init:
            print("No Network found. Starting from scratch", flush=True)
        else:
            raise FileNotFoundError("No saved network found in {} at epoch {}".format(dirPath, start_ep))
    
    # freeze layers
    if Freeze:
        for layerInd in range(0, len(gNet.net)-1-6, 3):
            gNet.net[layerInd].Psi.requires_grad = False
            gNet.net[layerInd].bias.requires_grad = False
        
        print("Everything except last three layers frozen.")
    
    if wt_norms:
        print("weighting target y vectors by inverse exponential norm")

    # filter out with the method shown in the pytorch forum
    optimizer = pt.optim.Adam(filter(lambda p : p.requires_grad, gNet.parameters()), lr=lRate)
    print("Starting Training loop in {} mode".format(Train_mode)) 
    for epoch in tqdm(range(start_ep, end_ep + 1), position=0, leave=True):
        
        ## checkpoint
        if epoch%interval==0:
            pt.save(gNet.state_dict(), dirPath + "/ep_{0}.pt".format(epoch))
            
        for batch in range(0, N_train, N_batch):
            optimizer.zero_grad()
            
            end = min(batch + N_batch, N_train)

            state1Batch = state1Data[batch : end].to(device)
            state2Batch = state2Data[batch : end].to(device)
            
            
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
            
            # Then select the appropriate mode
            if Train_mode=="norms":
                y1_target_Batch = pt.exp(-pt.norm(y1TargetData[batch : end], dim=1))
                y2_target_Batch = pt.exp(-pt.norm(y2TargetData[batch : end], dim=1))

                y1 = pt.exp(-pt.norm(y1, dim = 1))
                y2 = pt.exp(-pt.norm(y2, dim = 1))
            
                loss1 = pt.sum((y1_target_Batch - y1)**2)
                loss2 = pt.sum((y2_target_Batch - y2)**2)

            elif Train_mode=="units":
                y1_target_Batch = F.normalize(y1TargetData[batch : end], p=2.0, dim=1)
                y2_target_Batch = F.normalize(y2TargetData[batch : end], p=2.0, dim=1)

                y1 = F.normalize(y1, p=2.0, dim=1, eps=1e-8)
                y2 = F.normalize(y2, p=2.0, dim=1, eps=1e-8)

                loss1 = pt.sum(pt.norm((y1_target_Batch - y1), dim=1)**2)
                loss2 = pt.sum(pt.norm((y2_target_Batch - y2), dim=1)**2)
            
            elif Train_mode=="direct":
                y1_target_Batch = y1TargetData[batch : end]
                y2_target_Batch = y2TargetData[batch : end]
                if not wt_norms:
                    loss1 = pt.sum(pt.norm((y1_target_Batch - y1), dim=1)**2)
                    loss2 = pt.sum(pt.norm((y2_target_Batch - y2), dim=1)**2)
                else:
                    y1_inv_exp_norms = pt.exp(-pt.norm(y1_target_Batch, dim=1))
                    y2_inv_exp_norms = pt.exp(-pt.norm(y2_target_Batch, dim=1))

                    loss1 = pt.sum(y1_inv_exp_norms * pt.norm((y1_target_Batch - y1), dim=1)**2)
                    loss2 = pt.sum(y2_inv_exp_norms * pt.norm((y2_target_Batch - y2), dim=1)**2)

            else:
                raise ValueError("Training Mode {} not recognized. Should be one of {}, {} or {}".format(Train_mode, "direct", "norms", "units"))

            loss = (loss1 + loss2)/(2.0*(end - batch))
            loss.backward()
            optimizer.step()


def Evaluate(dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2,
             rates, disps, y1Target, y2Target, SpecsToTrain, VacSpec,
             start_ep, end_ep, interval, N_train, gNet, Train_mode):
    
    Ndim = y1Target.shape[1]
    N_batch = 512
    # Convert compute data to pytorch tensors
    state1Data = pt.tensor(State1_Occs).double()
    state2Data = pt.tensor(State2_Occs).double()
    rateData = pt.tensor(rates).double()
    y1TargetData = pt.tensor(y1Target).double().to(device)
    y2TargetData = pt.tensor(y2Target).double().to(device)

    Nsamples = state1Data.shape[0]

    print("Evaluating species: {}, Vacancy label: {}".format(SpecsToTrain, VacSpec))
    print("Sample Jumps: {}, Training: {}, Validation: {}".format(Nsamples, N_train, Nsamples-N_train))
    print("Evaluating in mode {} with networks at: {}".format(Train_mode, dirPath))
    On_st1 = None
    On_st2 = None
    
    if SpecsToTrain == [VacSpec]:
        assert OnSites_st1 == OnSites_st2 == None
        dispData = pt.tensor(disps[:, 0, :]).double()
    else:
        dispData = pt.tensor(disps[:, 1, :]).double() 
        On_st1 = makeProdTensor(OnSites_st1, Ndim).long()
        On_st2 = makeProdTensor(OnSites_st2, Ndim).long()

    def compute(startSample, endSample):
        loss_epochs = []
        diff_epochs = []
        with pt.no_grad():
            for epoch in tqdm(range(start_ep, end_ep + 1, interval), position=0, leave=True):
                ## load checkpoint
                gNet.load_state_dict(pt.load(dirPath + "/ep_{0}.pt".format(epoch), map_location=device))
                loss = 0
                diff = 0
                for batch in range(startSample, endSample, N_batch):
                    end = min(batch + N_batch, endSample)

                    state1Batch = state1Data[batch : end].to(device)
                    state2Batch = state2Data[batch : end].to(device)
                    
                    rateBatch = rateData[batch : end].to(device)
                    dispBatch = dispData[batch : end].to(device)

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

                    dy = y2 - y1
                    
                    # compute transport coefficient
                    loss_diff = pt.sum(rateBatch * pt.norm((dispBatch + dy), dim=1)**2)/6.
                    diff += loss_diff.item()
                    
                    # Then select the appropriate mode to calculate fitting loss
                    if Train_mode=="norms":
                        y1_target_Batch = pt.exp(-pt.norm(y1TargetData[batch : end], dim=1))
                        y2_target_Batch = pt.exp(-pt.norm(y2TargetData[batch : end], dim=1))

                        y1 = pt.exp(-pt.norm(y1, dim = 1))
                        y2 = pt.exp(-pt.norm(y2, dim = 1))
                    
                        loss1 = pt.sum((y1_target_Batch - y1)**2)
                        loss2 = pt.sum((y2_target_Batch - y2)**2)

                    elif Train_mode=="units":
                        y1_target_Batch = F.normalize(y1TargetData[batch : end], p=2.0, dim=1)
                        y2_target_Batch = F.normalize(y2TargetData[batch : end], p=2.0, dim=1)

                        y1 = F.normalize(y1, p=2.0, dim=1, eps=1e-8)
                        y2 = F.normalize(y2, p=2.0, dim=1, eps=1e-8)

                        loss1 = pt.sum(pt.norm((y1_target_Batch - y1), dim=1)**2)
                        loss2 = pt.sum(pt.norm((y2_target_Batch - y2), dim=1)**2)
                    
                    elif Train_mode=="direct":
                        y1_target_Batch = y1TargetData[batch : end]
                        y2_target_Batch = y2TargetData[batch : end]

                        loss1 = pt.sum(pt.norm((y1_target_Batch - y1), dim=1)**2)
                        loss2 = pt.sum(pt.norm((y2_target_Batch - y2), dim=1)**2)

                    else:
                        raise ValueError("Training Mode {} not recognized. Should be one of {}, {} or {}".format(Train_mode, "direct", "norms", "units"))

                    
                    loss += (loss1 + loss2).item()    
                
                loss_epochs.append(loss)
                diff_epochs.append(diff)

        return np.array(loss_epochs), np.array(diff_epochs)
    
    train_loss, train_diff = compute(0, N_train)
    valid_loss, valid_diff = compute(N_train, Nsamples)

    return train_loss/(2.0*N_train), valid_loss/(2.0*(Nsamples - N_train)), train_diff/(N_train), valid_diff/(Nsamples-N_train)


def Gather_Y(dirPath, State1_Occs, State2_Occs,
                OnSites_st1, OnSites_st2, SpecsToTrain,
                VacSpec, epoch, gNet, Ndim, Train_mode):
    
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
    
    if Train_mode == "direct" or Train_mode == "units":
        y1Vecs = np.zeros((Nsamples, 3))
        y2Vecs = np.zeros((Nsamples, 3))

    else:
        y1Vecs = np.zeros(Nsamples)
        y2Vecs = np.zeros(Nsamples)

    print("Gathering y in mode {} on species: {}. Vacancy label: {}".format(Train_mode, SpecsToTrain, VacSpec))
    print("Using Network: {}".format(dirPath) + "/ep_{0}.pt".format(epoch))
    with pt.no_grad():
        ## load checkpoint
        gNet.load_state_dict(pt.load(dirPath + "/ep_{0}.pt".format(epoch), map_location=device))
                
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

            # Then select the appropriate mode
            if Train_mode=="norms":
                y1 = pt.exp(-pt.norm(y1, dim = 1))
                y2 = pt.exp(-pt.norm(y2, dim = 1))
            
            elif Train_mode=="units":
                y1 = F.normalize(y1, p=2.0, dim=1, eps=1e-8)
                y2 = F.normalize(y2, p=2.0, dim=1, eps=1e-8)

            elif Train_mode=="direct":
                1*1

            else:
                raise ValueError("Training Mode {} not recognized. Should be one of {}, {} or {}".format(Train_mode, "direct", "norms", "units"))

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
    
    nchann = int(args[count])
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
    
    learning_Rate = float(args[count])
    count += 1

    Train_mode = args[count]
    count += 1
    
    weight_norms = bool(int(args[count]))
    count += 1

    Freeze_inner_layers = bool(int(args[count]))
    
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
    y1Target = np.load("y1_{}".format(T_data)+"_"+direcString+"_jac_{}.npy".format(Jac_iter))
    y2Target = np.load("y2_{}".format(T_data)+"_"+direcString+"_jac_{}.npy".format(Jac_iter))
    
    dirNameNets = "ep_T_{0}_{1}_jac_{3}_n{2}c8_{4}".format(T_data, direcString, nLayers, Jac_iter, Train_mode)
    if Mode == "eval" or Mode == "getY":
        prepo = "saved at"
    
    elif Mode == "train":
        prepo = "saving in"

    # check if a run directory exists
    dirPath = RunPath + dirNameNets
    exists = os.path.isdir(dirPath)
    
    if not exists:
        if scratch_if_no_init == False:
            raise FileNotFoundError("Target directory {} does not exist but looking for existing networks.".format(dirPath))
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
            mean=0.02, std=0.02, b=1.0, nl=nLayers, nch = nchann).double().to(device)
        
    # Call MakeComputeData here
    State1_Occs, State2_Occs, rateData, dispData, OnSites_state1, OnSites_state2 =\
            makeComputeData(state1List, state2List, dispList, specsToTrain, VacSpec, 
                    rateList, AllJumpRates, JumpNewSites, dxJumps, NNsiteList,
                    N_train, AllJumps=AllJumps)
    print("Done Creating occupancy tensors")
    
    # Call Training or evaluating or y-gathering function here
    if Mode == "train":
        Train(dirPath, State1_Occs, State2_Occs, OnSites_state1, OnSites_state2,
              y1Target, y2Target, specsToTrain, VacSpec, start_ep, end_ep, interval, N_train, 
              gNet, lRate=learning_Rate, scratch_if_no_init=scratch_if_no_init, Train_mode=Train_mode,
              Freeze=Freeze_inner_layers, wt_norms=weight_norms)

    elif Mode == "eval":
        
        train_loss, valid_loss, train_diff, valid_diff = Evaluate(dirPath, State1_Occs, State2_Occs,
                OnSites_state1, OnSites_state2, rateData, dispData, y1Target, y2Target,
                specsToTrain, VacSpec, start_ep, end_ep,
                interval, N_train, gNet, Train_mode=Train_mode)
        
        if Train_mode=="direct":
            
            if weight_norms:
                np.save("tr_{3}_{0}_fity_jac{1}_n{2}c8_dir_wt_norm.npy".format(T_data, Jac_iter, nLayers, direcString),
                        train_loss)
            
                np.save("val_{3}_{0}_fity_jac{1}_n{2}c8_dir_wt_norm.npy".format(T_data, Jac_iter, nLayers, direcString),
                        valid_loss)
                
                np.save("tr_L{3}{3}_{0}_fity_jac{1}_n{2}c8_dir_wt_norm.npy".format(T_data, Jac_iter, nLayers, direcString),
                        train_diff)
            
                np.save("val_L_{3}{3}_{0}_fity_jac{1}_n{2}c8_dir_wt_norm.npy".format(T_data, Jac_iter, nLayers, direcString),
                        valid_diff)
            
            else:
                np.save("tr_{3}_{0}_fity_jac{1}_n{2}c8_dir.npy".format(T_data, Jac_iter, nLayers, direcString),
                        train_loss)
            
                np.save("val_{3}_{0}_fity_jac{1}_n{2}c8_dir.npy".format(T_data, Jac_iter, nLayers, direcString),
                        valid_loss)
                
                np.save("tr_L{3}{3}_{0}_fity_jac{1}_n{2}c8_dir.npy".format(T_data, Jac_iter, nLayers, direcString),
                        train_diff)
            
                np.save("val_L{3}{3}_{0}_fity_jac{1}_n{2}c8_dir.npy".format(T_data, Jac_iter, nLayers, direcString),
                        valid_diff)
        
        elif Train_mode=="units":
            np.save("tr_{3}_{0}_fity_jac{1}_n{2}c8_units.npy".format(T_data, Jac_iter, nLayers, direcString),
                train_loss)
            
            np.save("val_{3}_{0}_fity_jac{1}_n{2}c8_units.npy".format(T_data, Jac_iter, nLayers, direcString),
                valid_loss)

            np.save("tr_L{3}{3}_{0}_fity_jac{1}_n{2}c8_units.npy".format(T_data, Jac_iter, nLayers, direcString),
                train_diff)
            
            np.save("val_L{3}{3}_{0}_fity_jac{1}_n{2}c8_units.npy".format(T_data, Jac_iter, nLayers, direcString),
                valid_diff)

        elif Train_mode=="norms":
            np.save("tr_{3}_{0}_fity_jac{1}_n{2}c8_norms.npy".format(T_data, Jac_iter, nLayers, direcString),
                train_loss)
            
            np.save("val_{3}_{0}_fity_jac{1}_n{2}c8_norms.npy".format(T_data, Jac_iter, nLayers, direcString),
                valid_loss)
            
            np.save("tr_L{3}{3}_{0}_fity_jac{1}_n{2}c8_norms.npy".format(T_data, Jac_iter, nLayers, direcString),
                train_diff)
            
            np.save("val_L{3}{3}_{0}_fity_jac{1}_n{2}c8_norms.npy".format(T_data, Jac_iter, nLayers, direcString),
                valid_diff)

    elif Mode == "getY":
        assert AllJumps
        y1Vecs, y2Vecs = Gather_Y(dirPath, State1_Occs, State2_Occs,
                                  OnSites_state1, OnSites_state2, specsToTrain,
                                  VacSpec, start_ep, gNet, Ndim, Train_mode=Train_mode)

        if Train_mode=="norms":
            np.save("y1_{3}_{0}_fit_jac{1}_n{2}c8_norms.npy".format(T_data, Jac_iter, nLayers,
                                                      direcString), y1Vecs)
            np.save("y2_{3}_{0}_fit_jac{1}_n{2}c8_norms.npy".format(T_data, Jac_iter, nLayers,
                                                      direcString), y2Vecs)
        
        elif Train_mode=="direct":
            np.save("y1_{3}_{0}_fit_jac{1}_n{2}c8.npy".format(T_data, Jac_iter, nLayers,
                                                      direcString), y1Vecs)
            np.save("y2_{3}_{0}_fit_jac{1}_n{2}c8.npy".format(T_data, Jac_iter, nLayers,
                                                      direcString), y2Vecs)

        elif Train_mode=="units":
            np.save("y1_{3}_{0}_fit_jac{1}_n{2}c8_units.npy".format(T_data, Jac_iter, nLayers,
                                                      direcString), y1Vecs)
            np.save("y2_{3}_{0}_fit_jac{1}_n{2}c8_units.npy".format(T_data, Jac_iter, nLayers,
                                                      direcString), y2Vecs)


if __name__ == "__main__":
    main(list(sys.argv))
