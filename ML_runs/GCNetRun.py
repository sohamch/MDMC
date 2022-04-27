import os
import sys
RunPath = os.getcwd() + "/"
CrysPath = "/home/sohamc2/HEA_FCC/MDMC/"
DataPath = "/home/sohamc2/HEA_FCC/MDMC/ML_runs/DataSets/"
ModulePath = "/home/sohamc2/VKMC/SymNetworkRuns/CE_Symmetry/Symm_Network/"

#DataPath = "MD_KMC_single/"
#ModulePath = "/mnt/FCDEB3C5DEB3768C/UIUC/Research/KMC_ML/VKMC/SymNetworkRuns/CE_Symmetry/Symm_Network"

sys.path.append(ModulePath)

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


def Load_crysDats(nn=1, type="FCC"):
    ## load the crystal data files
    if type == "FCC":
        CrysDatPath = CrysPath + "CrysDat_FCC/"
    elif type == "BCC":
        CrysDatPath = CrysPath + "CrysDat_BCC/"

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

def Load_Data(FileName):
    with h5py.File(DataPath + FileName, "r") as fl:
        try:
            perm = fl["Permutation"]
            print("found permuation")
        except:        
            perm = np.arange(len(fl["InitStates"]))

        state1List = np.array(fl["InitStates"])[perm]
        state2List = np.array(fl["FinStates"])[perm]
        dispList = np.array(fl["SpecDisps"])[perm]
        rateList = np.array(fl["rates"])[perm]
        AllJumpRates = np.array(fl["AllJumpRates"])[perm]
        try:
            jmpSelects = np.array(fl["JumpSelects"])[perm]
        except:
            jmpSelects = np.array(fl["JumpSelection"])[perm]
    
    return state1List, state2List, dispList, rateList, AllJumpRates, jmpSelects

def makeComputeData(state1List, state2List, dispList, specsToTrain, VacSpec, rateList,
        AllJumpRates, JumpNewSites, dxJumps, NNsiteList, N_train, AllJumps=False):

    # make the input tensors
    Nsamples = min(state1List.shape[0], 2*N_train)
    if AllJumps:
        NJumps = Nsamples*AllJumpRates.shape[1]
    else:
        NJumps = Nsamples
    a = np.linalg.norm(dispList[0, 0, :])/np.linalg.norm(dxJumps[0]) 
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
    for samp in tqdm(range(Nsamples), position=0, leave=True):
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
    
    return State1_occs, State2_occs, rateData, dispData, OnSites_state1, OnSites_state2


def makeProdTensor(OnSites, Ndim):
    Onst = pt.tensor(OnSites)
    Onst = Onst.repeat_interleave(Ndim, dim=0)
    Onst = Onst.view(-1, Ndim, OnSites.shape[1])
    return Onst


"""## Write the training loop"""
def Train(T, dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, 
        rates, disps, SpecsToTrain, VacSpec, start_ep, end_ep, interval, N_train,
        gNet, lRate=0.001, scratch_if_no_init=True):
    
    Ndim = disps.shape[2]
    N_batch = 128
    # Convert compute data to pytorch tensors
    state1Data = pt.tensor(State1_Occs[:N_train]).double()
    state2Data = pt.tensor(State2_Occs[:N_train]).double()
    rateData = pt.tensor(rates[:N_train]).double().to(device)
    On_st1 = None 
    On_st2 = None    
    if SpecsToTrain == [VacSpec]:
        assert OnSites_st1 == OnSites_st2 == None
        print("Training on Vacancy".format(SpecsToTrain)) 
        dispData = pt.tensor(disps[:N_train, 0, :]).double().to(device)
    else:
        print("Training Species : {}".format(SpecsToTrain)) 
        dispData = pt.tensor(disps[:N_train, 1, :]).double().to(device) 
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

    optimizer = pt.optim.Adam(gNet.parameters(), lr=lRate, weight_decay=0.0005)
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
            
            rateBatch = rateData[batch : end]
            dispBatch = dispData[batch : end]
            
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
            diff = pt.sum(rateBatch * pt.norm((dispBatch + dy), dim=1)**2)/6. 
            diff.backward()
            optimizer.step()


def Evaluate(T, dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, 
        rates, disps, SpecsToTrain, VacSpec, start_ep, end_ep, interval, N_train,
        gNet):
    
    Ndim = disps.shape[2]
    N_batch = 512
    # Convert compute data to pytorch tensors
    state1Data = pt.tensor(State1_Occs).double()
    Nsamples = state1Data.shape[0]

    print("Evaluating species: {}, Vacancy label: {}".format(SpecsToTrain, VacSpec))
    print("Sample Jumps: {}, Training: {}, Validation: {}".format(Nsamples, N_train, Nsamples-N_train))
    print("Evaluating with networks at: {}".format(dirPath))
    
    state2Data = pt.tensor(State2_Occs).double()
    rateData = pt.tensor(rates).double()
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
        diff_epochs = []
        with pt.no_grad():
            for epoch in tqdm(range(start_ep, end_ep + 1, interval), position=0, leave=True):
                ## load checkpoint
                gNet.load_state_dict(pt.load(dirPath + "/ep_{1}.pt".format(T, epoch), map_location=device))
                    
                diff = 0 
                for batch in range(startSample, endSample, N_batch):
                    end = min(batch + N_batch, endSample)

                    state1Batch = state1Data[batch : end].to(device)
                    state2Batch = state2Data[batch : end].to(device)
                    
                    rateBatch = rateData[batch : end].to(device)
                    dispBatch = dispData[batch : end].to(device)
                    
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


def Gather_Y(T, dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, SpecsToTrain, VacSpec, epoch, gNet, Ndim):
    
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
    print("Network: {}".format(dirPath))
    with pt.no_grad():
        ## load checkpoint
        gNet.load_state_dict(pt.load(dirPath + "/ep_{1}.pt".format(T, epoch), map_location=device))
                
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
    # Replace all of this with argparse after learning about it
    count=1
    FileName = args[count] # Name of data file to train on
    count += 1
    
    CrystalType = args[count]
    count += 1

    Mode = args[count] # "train" mode or "eval" mode or "getY" mode
    count += 1

    nLayers = int(args[count])
    count += 1
    
    ch = int(args[count])
    count += 1
    
    filter_nn = int(args[count])
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
    
    specTrain = args[count] # which species to train collectively: eg - 123 for species 1, 2 and 3
    # The entry is order independent as it is sorted later
    count += 1
    
    VacSpec = int(args[count]) # integer label for vacancy species
    count += 1
    
    AllJumps = bool(int(args[count])) # whether to consider all jumps out of the samples or just stochastically selected one
    # False if 0, otherwise True
    count += 1
    
    AllJumps_net_type = bool(int(args[count])) # whether to use network trained on all jumps out of the samples or just stochastically selected one
    # This is the directory to search for in "eval" or "getY" modes
    # False if 0, otherwise True
    count += 1
    
    N_train = int(args[count]) # How many INITIAL STATES to consider for training
    count += 1
    
    interval = int(args[count]) # for train mode, interval to save and for eval mode, interval to load
    count += 1
    
    learning_Rate = float(args[count]) if len(args)==count+1 else 0.001
    
    if not (Mode == "train" or Mode == "eval" or Mode == "getY"):
        raise ValueError("Mode needs to be train, eval or getY and not : {}".format(Mode))

    if Mode == "train":
        if T_data != T_net:
            raise ValueError("Training and Testing condition must be the same")

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

    # This is where networks will be saved to and loaded from
    dirNameNets = "ep_T_{0}_{1}_n{2}c{4}_all_{3}".format(T_net, direcString, nLayers, int(AllJumps), ch)
    if Mode == "eval" or Mode == "getY":
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
    GpermNNIdx, NNsiteList, siteShellIndices, GIndtoGDict, JumpNewSites, dxJumps = Load_crysDats(nn=filter_nn)
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
    
    # Make a network to either train from scratch or load saved state into
    gNet = GCNet(GnnPerms, gdiags, NNsites, SitesToShells, Ndim, N_ngb, NSpec,
            mean=0.02, std=0.2, b=1.0, nl=nLayers, nch=ch).double().to(device)

    # Call MakeComputeData here
    State1_Occs, State2_Occs, rateData, dispData, OnSites_state1, OnSites_state2 = makeComputeData(state1List, state2List, dispList,
            specsToTrain, VacSpec, rateList, AllJumpRates, JumpNewSites, dxJumps, NNsiteList, N_train, AllJumps=AllJumps)
    print("Done Creating occupancy tensors")

    # Call Training or evaluating or y-evaluating function here
    N_train_jumps = 12*N_train if AllJumps else N_train
    if Mode == "train":
        Train(T_data, dirPath, State1_Occs, State2_Occs, OnSites_state1, OnSites_state2,
                rateData, dispData, specsToTrain, VacSpec, start_ep, end_ep, interval, N_train_jumps,
                gNet, lRate=learning_Rate, scratch_if_no_init=scratch_if_no_init)

    elif Mode == "eval":
        train_diff, valid_diff = Evaluate(T_net, dirPath, State1_Occs, State2_Occs,
                OnSites_state1, OnSites_state2, rateData, dispData,
                specsToTrain, VacSpec, start_ep, end_ep,
                interval, N_train_jumps, gNet)
        np.save("tr_{4}_{0}_{1}_n{2}c{5}_all_{3}.npy".format(T_data, T_net, nLayers, int(AllJumps), direcString, ch), train_diff/(1.0*N_train))
        np.save("val_{4}_{0}_{1}_n{2}c{5}_all_{3}.npy".format(T_data, T_net, nLayers, int(AllJumps), direcString, ch), valid_diff/(1.0*N_train))

    elif Mode == "getY":
        y1Vecs, y2Vecs = Gather_Y(T_net, dirPath, State1_Occs, State2_Occs,
                OnSites_state1, OnSites_state2, specsToTrain, VacSpec, start_ep, gNet, Ndim)
        np.save("y1_{4}_{0}_{1}_n{2}c{6}_all_{3}_{5}.npy".format(T_data, T_net, nLayers, int(AllJumps), direcString, start_ep, ch), y1Vecs)
        np.save("y2_{4}_{0}_{1}_n{2}c{6}_all_{3}_{5}.npy".format(T_data, T_net, nLayers, int(AllJumps), direcString, start_ep, ch), y2Vecs)


if __name__ == "__main__":
    main(list(sys.argv))
