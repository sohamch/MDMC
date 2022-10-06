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
from tqdm import tqdm
from SymmLayers import GCNet

device=None
if pt.cuda.is_available():
    print(pt.cuda.get_device_name())
    device = pt.device("cuda:0")
    DeviceIDList = list(range(pt.cuda.device_count()))
else:
    device = pt.device("cpu")


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

    JumpNewSites = np.load(CrysDatPath + "JumpNewSiteIndices.npy")
    dxJumps = np.load(CrysDatPath + "dxList.npy")
    return GpermNNIdx, NNsiteList, JumpNewSites, dxJumps

def Load_Data(DataPath):
    with h5py.File(DataPath, "r") as fl:
        try:
            perm = np.array(fl["Permutation"])
            print("found permuation")
        except:        
            perm = np.arange(len(fl["InitStates"]))

        state1List = np.array(fl["InitStates"])[perm]
        state2List = np.array(fl["FinStates"])[perm]
        dispList = np.array(fl["SpecDisps"])[perm]
        rateList = np.array(fl["rates"])[perm]
        try: 
            AllJumpRates_st1 = np.array(fl["AllJumpRates_Init"])[perm]
            AllJumpRates_st2 = np.array(fl["AllJumpRates_Fin"])[perm]
        
            avgDisps_st1 = np.array(fl["AvgDisps_Init"])[perm]
            avgDisps_st2 = np.array(fl["AvgDisps_Fin"])[perm]

        except:
            AllJumpRates_st1 = np.array(fl["AllJumpRates"])[perm]
            AllJumpRates_st2 = None
            avgDisps_st1 = None
            avgDisps_st2 = None
 
    return state1List, state2List, dispList, rateList, AllJumpRates_st1, AllJumpRates_st2, avgDisps_st1, avgDisps_st2

def makeComputeData(state1List, state2List, dispList, specsToTrain, VacSpec, rateList,
        AllJumpRates_st1, AllJumpRates_st2, avgDisps_st1, avgDisps_st2, JumpNewSites, 
        dxJumps, NNsiteList, N_train, AllJumps=False, mode="train"):
    
    # make the input tensors
    if mode=="train":
        Nsamples = N_train
    else:
        Nsamples = state1List.shape[0]

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
                
                rateData[Idx] = AllJumpRates_st1[samp, jInd]
                
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


def makeDataTensors(State1_Occs, State2_Occs, rates, disps, OnSites_st1, OnSites_st2, SpecsToTrain, VacSpec, sp_ch, Ndim=3):
    # Do a small check that species channels were assigned correctly
    if sp_ch is not None:
        for key, item in sp_ch.items():
            if key > VacSpec:
                assert item == key - 1
            else:
                assert key < VacSpec
                assert item == key

    # Convert compute data to pytorch tensors
    state1Data = pt.tensor(State1_Occs)
    state2Data = pt.tensor(State2_Occs)
    rateData=None
    On_st1 = None 
    On_st2 = None
    dispData=None

    if rates is not None:
        rateData = pt.tensor(rates).double().to(device)
    if SpecsToTrain == [VacSpec]:
        assert OnSites_st1 == OnSites_st2 == None
        print("Training on Vacancy".format(SpecsToTrain)) 
        if disps is not None:
            dispData = pt.tensor(disps[:, 0, :]).double().to(device)
    else:
        print("Training Species : {}".format(SpecsToTrain))
        if disps is not None:
            dispData = pt.tensor(disps[:, 1, :]).double().to(device)
        
        # Convert on-site tensor to boolean mask
        On_st1 = pt.tensor(OnSites_st1, dtype=pt.bool)
        On_st2 = pt.tensor(OnSites_st2, dtype=pt.bool)

    return state1Data, state2Data, dispData, rateData, On_st1, On_st2


# All vacancy batch output calculations to be done here
def vacBatchOuts(y1, y2, jProbs_st1, jProbs_st2, Boundary_Train):
    if not Boundary_Train:
        assert y1.shape[1] == 1
        y1 = -pt.sum(y1[:, 0, :, 1:], dim=2)
        y2 = -pt.sum(y2[:, 0, :, 1:], dim=2)

    else:
        assert y1.shape[1] == jProbs_st1.shape[1]
        # send the jump probabilities to the device
        jPr_1_batch = jProbs_st1.unsqueeze(2).to(device)
        jPr_2_batch = jProbs_st2.unsqueeze(2).to(device)

        y1 = -pt.sum(y1[:, :, :, 1:], dim=3) # take negative sum of all sites
        y2 = -pt.sum(y2[:, :, :, 1:], dim=3)

        # y have dimension (Nbatch, Njumps, 3)
        # jPr have dimension (Nbatch, Njumps, 1)
        y1 = pt.sum(y1*jPr_1_batch, dim = 1) # average across the jumps
        y2 = pt.sum(y2*jPr_2_batch, dim = 1)
    
    return y1, y2

# All non-vacancy batch calculations to be done here
def SpecBatchOuts(y1, y2, On_st1Batch, On_st2Batch, jProbs_st1, jProbs_st2, NNsvac_st1, NNsvac_st2, Boundary_train, jumpSwitch):
    if not Boundary_train:
        On_st1Batch = On_st1Batch.unsqueeze(1).to(device)
        On_st2Batch = On_st2Batch.unsqueeze(1).to(device)
        y1 = pt.sum(y1[:, 0, :, :]*On_st1Batch, dim=2)
        y2 = pt.sum(y2[:, 0, :, :]*On_st2Batch, dim=2)
               
    else:
        assert y1.shape[1] == jProbs_st1.shape[1]
        assert y2.shape[1] == jProbs_st2.shape[1]

        if jumpSwitch:
            # First, select jump probs from those jumps which actually move the atom we want
            JumpOnSites_st1Batch = On_st1Batch.gather(1, NNsvac_st1)
            JumpOnSites_st2Batch = On_st2Batch.gather(1, NNsvac_st2)
            # select jumps with a masked multiply
            jPr_1_batch = (jProbs_st1 * JumpOnSites_st1Batch).unsqueeze(2).to(device)
            jPr_2_batch = (jProbs_st2 * JumpOnSites_st2Batch).unsqueeze(2).to(device)

        else:
            jPr_1_batch = jProbs_st1.unsqueeze(2).to(device)
            jPr_2_batch = jProbs_st2.unsqueeze(2).to(device)
        
        # y have dimension (Nbacth, Njumps, Ndim, Nsites)
        # On_stBatch have dimensions (Nbatch, Nsites)
        # unsqueeze dim 1 twice to broadcast along jump channels and
        # cartesian components
        On_st1Batch = On_st1Batch.unsqueeze(1).unsqueeze(1).to(device)
        On_st2Batch = On_st2Batch.unsqueeze(1).unsqueeze(1).to(device)
        
        y1 = pt.sum(y1*On_st1Batch, dim=3) # sum across the occupied sites
        y2 = pt.sum(y2*On_st2Batch, dim=3)

        # Now average with the jump Probs
        # y have dimensions (Nbatch, Njumps, 3)
        # jPr have dimensions (Nbatch, Njumps, 1)
        # Do a broadcasted multiply, followed by sum along jumps
        y1 = pt.sum(y1 * jPr_1_batch, dim = 1)
        y2 = pt.sum(y2 * jPr_2_batch, dim = 1)
    
    return y1, y2
    

def sort_jp(NNsvac_st1, NNsvac_st2, jProbs_st1, jProbs_st2, jumpSort):
    if jumpSort:
        print("Sorting Jump Rates.")
        jProbs_st1_args = pt.tensor(np.argsort(jProbs_st1, axis=1), dtype=pt.long)
        jProbs_st2_args = pt.tensor(np.argsort(jProbs_st2, axis=1), dtype=pt.long)
        
        NNsvac_st1 = NNsvac_st1.gather(1, jProbs_st1_args)
        NNsvac_st2 = NNsvac_st2.gather(1, jProbs_st2_args)

        jProbs_st1 = np.sort(jProbs_st1, axis=1)
        jProbs_st2 = np.sort(jProbs_st2, axis=1)
    else:
        print("Jump Rates unsorted. Values will be non-symmetric if boundary-training is active.")
    
    jProbs_st1 = pt.tensor(jProbs_st1, dtype=pt.double)
    jProbs_st2 = pt.tensor(jProbs_st2, dtype=pt.double)

    return jProbs_st1, jProbs_st2, NNsvac_st1, NNsvac_st2


"""## Write the training loop"""
def Train(T, dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, rates, disps,
          jProbs_st1, jProbs_st2, NNsites, SpecsToTrain, sp_ch, VacSpec, start_ep, end_ep, interval, N_train,
          gNet, lRate=0.001, batch_size=128, scratch_if_no_init=True, DPr=False, Boundary_train=False, jumpSort=True,
          jumpSwitch=True, scaleL0=False, chkpt=True):
    
    print("Training conditions:")
    print("scratch: {}, DPr: {}, Boundary_train: {}, jumpSort: {}, jumpSwitch: {}, scaleL0: {}".format(scratch_if_no_init, DPr, Boundary_train, jumpSort, jumpSwitch, scaleL0))

    Ndim = disps.shape[2]
    state1Data, state2Data, dispData, rateData, On_st1, On_st2 = makeDataTensors(State1_Occs, State2_Occs, rates, disps,
            OnSites_st1, OnSites_st2, SpecsToTrain, VacSpec, sp_ch, Ndim=Ndim)


    # scale with L0 if indicated
    if scaleL0:
        L0 = pt.dot(rateData, pt.norm(dispData, dim=1)**2)/(6.0 * dispData.shape[0])
        L0 = L0.item()
    else:
        L0 = 1.0
    
    print("L0 : {}".format(L0))

    NNsvac_st1 = None
    NNsvac_st2 = None
    if Boundary_train:
        assert gNet.net[-3].Psi.shape[0] == jProbs_st1.shape[1] == jProbs_st2.shape[1] 
        print("Boundary training indicated. Using jump probabilities.")
        NNsvac_st1 = NNsites[1:, 0].repeat(N_train, 1)
        NNsvac_st2 = NNsites[1:, 0].repeat(N_train, 1)
        jProbs_st1, jProbs_st2, NNsvac_st1, NNsvac_st2 =\
            sort_jp(NNsvac_st1, NNsvac_st2, jProbs_st1[:N_train], jProbs_st2[:N_train], jumpSort)

    N_batch = batch_size

    if pt.cuda.device_count() > 1 and DPr:
        print("Running on Devices : {}".format(DeviceIDList))
        gNet = nn.DataParallel(gNet, device_ids=DeviceIDList)

    try:
        gNet.load_state_dict(pt.load(dirPath + "/ep_{1}.pt".format(T, start_ep), map_location="cpu"))
        print("Starting from epoch {}".format(start_ep), flush=True)

    except:
        if scratch_if_no_init:
            print("No Network found. Starting from scratch", flush=True)
        else:
            raise ValueError("Required saved networks not found in {} at epoch {}".format(dirPath, start_ep))

    print("Batch size : {}".format(N_batch)) 
    specTrainCh = [sp_ch[spec] for spec in SpecsToTrain]

    gNet.to(device)
    opt = pt.optim.Adam(gNet.parameters(), lr=lRate, weight_decay=0.0005)
    print("Starting Training loop")

    y1BatchTest = np.zeros((N_batch, 3))
    y2BatchTest = np.zeros((N_batch, 3))

    for epoch in tqdm(range(start_ep, end_ep + 1), position=0, leave=True):
        
        ## checkpoint
        if epoch%interval==0 and chkpt:
            pt.save(gNet.state_dict(), dirPath + "/ep_{0}.pt".format(epoch))
            
        for batch in range(0, N_train, N_batch):
            
            opt.zero_grad() 
            end = min(batch + N_batch, N_train)

            state1Batch = state1Data[batch : end].double().to(device)
            state2Batch = state2Data[batch : end].double().to(device)
            
            rateBatch = rateData[batch : end]
            dispBatch = dispData[batch : end]
            
            if Boundary_train:
                jProbs_st1_batch = jProbs_st1[batch : end]
                jProbs_st2_batch = jProbs_st2[batch : end]
                NNsvac_st1_batch = NNsvac_st1[batch : end]
                NNsvac_st2_batch = NNsvac_st2[batch : end]
            else:
                jProbs_st1_batch = None
                jProbs_st2_batch = None
                NNsvac_st1_batch = None
                NNsvac_st2_batch = None

            y1 = gNet(state1Batch)
            y2 = gNet(state2Batch)
                
            if SpecsToTrain==[VacSpec]: 
                y1, y2 = vacBatchOuts(y1, y2, jProbs_st1_batch, jProbs_st2_batch, Boundary_train)

            else:
                On_st1Batch = On_st1[batch : end]
                On_st2Batch = On_st2[batch : end]

                y1, y2 = SpecBatchOuts(y1, y2, On_st1Batch, On_st2Batch, jProbs_st1_batch, jProbs_st2_batch, NNsvac_st1_batch, NNsvac_st2_batch, Boundary_train, jumpSwitch)

            dy = y2 - y1
            diff = pt.sum(rateBatch * pt.norm((dispBatch + dy), dim=1)**2)/(6. * L0)

            diff.backward()
            opt.step()

            if epoch == 0 and batch == 0:
                y1BatchTest[:, :] = y1.cpu().detach().numpy()
                y2BatchTest[:, :] = y2.cpu().detach().numpy()

    # For testing return y1 and y2 - we'll test on a single epoch, single batch sample.
    return y1BatchTest, y2BatchTest


def Evaluate(T, dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, 
        rates, disps, SpecsToTrain, jProbs_st1, jProbs_st2, NNsites, sp_ch, VacSpec,
        start_ep, end_ep, interval, N_train, gNet, batch_size=512, Boundary_train=False,
        DPr=False, jumpSort=True, jumpSwitch=True):
    
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
            OnSites_st1, OnSites_st2, SpecsToTrain, VacSpec, sp_ch, Ndim=Ndim)
    
    NNsvac_st1 = None
    NNsvac_st2 = None
    
    if Boundary_train:
        assert gNet.net[-3].Psi.shape[0] == jProbs_st1.shape[1] == jProbs_st2.shape[1] 
        print("Boundary training indicated. Using jump probabilities.")
        NNsvac_st1 = NNsites[1:, 0].repeat(state1Data.shape[0], 1)
        NNsvac_st2 = NNsites[1:, 0].repeat(state2Data.shape[0], 1)
        jProbs_st1, jProbs_st2, NNsvac_st1, NNsvac_st2 = sort_jp(NNsvac_st1, NNsvac_st2, jProbs_st1, jProbs_st2, jumpSort)
    
    # pre-convert to data parallel if required
    if pt.cuda.device_count() > 1 and DPr:
        print("Using data parallel")
        print("Running on Devices : {}".format(DeviceIDList))
        gNet = nn.DataParallel(gNet, device_ids=DeviceIDList)

    gNet.to(device)
    def compute(startSample, endSample):
        diff_epochs = []
        with pt.no_grad():
            for epoch in tqdm(range(start_ep, end_ep + 1, interval), position=0, leave=True):
                ## load checkpoint
                gNet.load_state_dict(pt.load(dirPath + "/ep_{0}.pt".format(epoch), map_location=device))
 
                diff = 0 
                for batch in range(startSample, endSample, N_batch):
                    end = min(batch + N_batch, endSample)

                    state1Batch = state1Data[batch : end].double().to(device)
                    state2Batch = state2Data[batch : end].double().to(device)
                    
                    rateBatch = rateData[batch : end].to(device)
                    dispBatch = dispData[batch : end].to(device)
                    
                    if Boundary_train:
                        jProbs_st1_batch = jProbs_st1[batch : end]
                        jProbs_st2_batch = jProbs_st2[batch : end]
                        NNsvac_st1_batch = NNsvac_st1[batch : end]
                        NNsvac_st2_batch = NNsvac_st2[batch : end]
                    else:
                        jProbs_st1_batch = None
                        jProbs_st2_batch = None
                        NNsvac_st1_batch = None
                        NNsvac_st2_batch = None

                    y1 = gNet(state1Batch)
                    y2 = gNet(state2Batch)
            
                    if SpecsToTrain==[VacSpec]: 
                        y1, y2 = vacBatchOuts(y1, y2, jProbs_st1_batch, jProbs_st2_batch, Boundary_train)

                    else:
                        On_st1Batch = On_st1[batch : end]
                        On_st2Batch = On_st2[batch : end]
                        y1, y2 = SpecBatchOuts(y1, y2, On_st1Batch, On_st2Batch, jProbs_st1_batch, jProbs_st2_batch,
                                NNsvac_st1_batch, NNsvac_st2_batch, Boundary_train, jumpSwitch)

                    dy = y2 - y1
                    loss = pt.sum(rateBatch * pt.norm((dispBatch + dy), dim=1)**2)/6.
                    diff += loss.item()

                diff_epochs.append(diff)

        return np.array(diff_epochs)
    
    train_diff = compute(0, N_train)#/(1.0*N_train)
    test_diff = compute(N_train, Nsamples)#/(1.0*(Nsamples - N_train))

    return train_diff, test_diff


def Gather_Y(T, dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, jProbs_st1, jProbs_st2,
        NNsites, sp_ch, SpecsToTrain, VacSpec, gNet, Ndim, epoch=None, Boundary_train=False, batch_size=256,
        jumpSort=True, jumpSwitch=True):
    
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
            OnSites_st1, OnSites_st2, SpecsToTrain, VacSpec, sp_ch, Ndim=Ndim)
    
    NNsvac_st1 = None
    NNsvac_st2 = None
    
    if Boundary_train:
        assert gNet.net[-3].Psi.shape[0] == jProbs_st1.shape[1] == jProbs_st2.shape[1] 
        print("Boundary training indicated. Using jump probabilities.")
        NNsvac_st1 = NNsites[1:, 0].repeat(state1Data.shape[0], 1)
        NNsvac_st2 = NNsites[1:, 0].repeat(state1Data.shape[0], 1)
        jProbs_st1, jProbs_st2, NNsvac_st1, NNsvac_st2 = sort_jp(NNsvac_st1, NNsvac_st2, jProbs_st1, jProbs_st2, jumpSort)
    
    y1Vecs = np.zeros((Nsamples, 3))
    y2Vecs = np.zeros((Nsamples, 3))

    gNet.to(device)
    if epoch is not None:
        print("Network: {}".format(dirPath))
    with pt.no_grad():
        ## load checkpoint
        if epoch is not None:
            print("Loading epoch: {}".format(epoch))
            gNet.load_state_dict(pt.load(dirPath + "/ep_{0}.pt".format(epoch), map_location=device))
             
        for batch in tqdm(range(0, Nsamples, N_batch), position=0, leave=True):
            end = min(batch + N_batch, Nsamples)

            state1Batch = state1Data[batch : end].double().to(device)
            state2Batch = state2Data[batch : end].double().to(device)
            
            if Boundary_train:
                jProbs_st1_batch = jProbs_st1[batch : end]
                jProbs_st2_batch = jProbs_st2[batch : end]
                NNsvac_st1_batch = NNsvac_st1[batch : end]
                NNsvac_st2_batch = NNsvac_st2[batch : end]
            else:
                jProbs_st1_batch = None
                jProbs_st2_batch = None
                NNsvac_st1_batch = None
                NNsvac_st2_batch = None

            y1 = gNet(state1Batch)
            y2 = gNet(state2Batch)
                
            if SpecsToTrain==[VacSpec]: 
                y1, y2 = vacBatchOuts(y1, y2, jProbs_st1_batch, jProbs_st2_batch, Boundary_train)

            else:
                On_st1Batch = On_st1[batch : end]
                On_st2Batch = On_st2[batch : end]

                y1, y2 = SpecBatchOuts(y1, y2, On_st1Batch, On_st2Batch, jProbs_st1_batch, jProbs_st2_batch,
                        NNsvac_st1_batch, NNsvac_st2_batch, Boundary_train, jumpSwitch)

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
    
    gNet.to(device)
    glob_Nch = gNet.net[0].Psi.shape[0]
    with pt.no_grad():
        ## load checkpoint
        gNet.load_state_dict(pt.load(dirPath + "/ep_{0}.pt".format(epoch), map_location=device))
        nLayers = (len(gNet.net)-6)//3
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
    # 0. Get run parameters
    FileName = args.DataPath # Name of data file to train on
    CrysPath = args.CrysDatPath
    Mode = args.Mode # "train" mode or "eval" mode or "getY" mode
    nLayers = args.Nlayers
    ch = args.Nchannels
    chLast = args.NchLast
    filter_nn = args.ConvNgbRange
    scratch_if_no_init = args.Scratch
    DPr = args.DatPar
    T_data = args.Tdata # Note : for binary random alloys, this should is the training composition instead of temperature
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
    a0 = args.LatParam # Lattice parameter

    if not (Mode == "train" or Mode == "eval" or Mode == "getY" or Mode == "getRep"):
        raise ValueError("Mode needs to be train, eval, getY or getRep but given : {}".format(Mode))

    if Mode == "train":
        if T_data != T_net:
            raise ValueError("Network and data temperature (arguments \"TNet\"/\"tn\" and \"Tdata\"/\"td\") must be the same in train mode")
    
    if AllJumps and args.BoundTrain:
        raise NotImplementedError("Cannot do all-jump training with boundary states.")
    
    # 1. Load crystal parameters
    GpermNNIdx, NNsiteList, JumpNewSites, dxJumps = Load_crysDats(filter_nn, CrysPath)
    N_ngb = NNsiteList.shape[0]
    print("Filter neighbor range : {}nn. Filter neighborhood size: {}".format(filter_nn, N_ngb - 1))
    Nsites = NNsiteList.shape[1]
    if args.NoSymmetry:
        print("Switching off convolution symmetries (considering only identity operator).")
        GnnPerms = pt.tensor(GpermNNIdx[:1]).long()
        assert pt.equal(GnnPerms, pt.arange(N_ngb).unsqueeze(0))

    else:
        print("Considering full symmetry group.")
        GnnPerms = pt.tensor(GpermNNIdx).long()

    NNsites = pt.tensor(NNsiteList).long()
    JumpVecs = pt.tensor(dxJumps.T * a0, dtype=pt.double)
    print("Jump Vectors: \n", JumpVecs.T, "\n")
    Ndim = dxJumps.shape[1]

    # 2. Load data
    state1List, state2List, dispList, rateList, AllJumpRates_st1, AllJumpRates_st2, avgDisps_st1, avgDisps_st2 = Load_Data(FileName)
    
    if args.BoundTrain and (AllJumpRates_st2 is None or avgDisps_st1 is None or avgDisps_st2 is None):
        raise ValueError("Insufficient data to do boundary training. Need jump rates and average displacements from both initial and final states.")

    # 2.1 Convert jump rates to probabilities
    if args.BoundTrain:
        jProbs_st1 = AllJumpRates_st1 / np.sum(AllJumpRates_st1, axis=1).reshape(-1, 1)
        jProbs_st2 = AllJumpRates_st2 / np.sum(AllJumpRates_st2, axis=1).reshape(-1, 1)
        assert np.allclose(np.sum(jProbs_st1, axis=1), 1.0)
        assert np.allclose(np.sum(jProbs_st2, axis=1), 1.0)
    
    else:
        jProbs_st1 = None
        jProbs_st2 = None
    
    # 2.2 shift displacements if boundary training
    if args.BoundTrain and args.DispShift:
        print("Shifting displacements by differences of initial and final state averages.")
        dispList += avgDisps_st2 - avgDisps_st1
    else:
        print("Displacements not shifted.")

    # 2.3 Make numpy arrays to feed into training/evaluation functions
    specsToTrain = [int(specTrain[i]) for i in range(len(specTrain))]
    specsToTrain = sorted(specsToTrain)

    State1_occs, State2_occs, rateData, dispData, OnSites_state1, OnSites_state2, sp_ch =\
    makeComputeData(state1List, state2List, dispList, specsToTrain, VacSpec, rateList,
                AllJumpRates_st1, AllJumpRates_st2, avgDisps_st1, avgDisps_st2, JumpNewSites,
                dxJumps, NNsiteList, N_train, AllJumps=AllJumps, mode=Mode)
    print("Done Creating numpy occupancy tensors. Species channels: {}".format(sp_ch))

    # 3. Next, make directories
    specs = np.unique(state1List[0])
    NSpec = specs.shape[0] - 1
    Nsites = state1List.shape[1]
     
    direcString=""
    if specsToTrain == [VacSpec]:
        direcString = "vac"
    else:
        for spec in specsToTrain:
            direcString += "{}".format(spec)

    # 3.1 This is where networks will be saved to and loaded from
    dirNameNets = "ep_T_{0}_{1}_n{2}c{4}_all_{3}".format(T_net, direcString, nLayers, int(AllJumps), ch)
    if Mode == "eval" or Mode == "getY" or Mode=="getRep":
        prepo = "saved at"
        dirNameNets = "ep_T_{0}_{1}_n{2}c{4}_all_{3}".format(T_net, direcString, nLayers, int(AllJumps_net_type), ch)
    
    if Mode == "train":
        prepo = "saving in"

    # 3.2 check if a run directory exists
    dirPath = RunPath + dirNameNets
    exists = os.path.isdir(dirPath)
    
    if not exists:
        if start_ep == 0:
            os.mkdir(dirPath)
        elif start_ep > 0:
            raise ValueError("Training directory does not exist but start epoch greater than zero: {}\ndirectory given: {}".format(start_ep, dirPath))

    print("Running in Mode {} with networks {} {}".format(Mode, prepo, dirPath))
    
    if args.BoundTrain:
        assert chLast == N_ngb - 1

    gNet = GCNet(GnnPerms.long(), NNsites, JumpVecs, N_ngb=N_ngb, NSpec=NSpec,
            mean=wt_means, std=wt_std, nl=nLayers, nch=ch, nchLast=chLast).double()

    print("No. of channels in last layer: {}".format(gNet.net[-3].Psi.shape[0]))

    # 4. Call Training or evaluating or y-evaluating function here
    N_train_jumps = (N_ngb - 1)*N_train if AllJumps else N_train
    if Mode == "train":
        Train(T_data, dirPath, State1_occs, State2_occs, OnSites_state1, OnSites_state2,
                rateData, dispData, jProbs_st1, jProbs_st2, NNsites, specsToTrain, sp_ch, VacSpec,
                start_ep, end_ep, interval, N_train_jumps, gNet,
                lRate=learning_Rate, scratch_if_no_init=scratch_if_no_init, batch_size=batch_size,
                DPr=DPr, Boundary_train=args.BoundTrain, jumpSort=args.JumpSort, jumpSwitch=args.JumpSwitch, scaleL0=args.ScaleL0)

    elif Mode == "eval":
        train_diff, valid_diff = Evaluate(T_net, dirPath, State1_occs, State2_occs,
                OnSites_state1, OnSites_state2, rateData, dispData,
                specsToTrain, jProbs_st1, jProbs_st2, NNsites, sp_ch, VacSpec, start_ep, end_ep,
                interval, N_train_jumps, gNet, batch_size=batch_size, Boundary_train=args.BoundTrain,
                DPr=DPr, jumpSort=args.JumpSort, jumpSwitch=args.JumpSwitch)
        np.save("tr_{4}_{0}_{1}_n{2}c{5}_all_{3}.npy".format(T_data, T_net, nLayers, int(AllJumps), direcString, ch), train_diff/(1.0*N_train))
        np.save("val_{4}_{0}_{1}_n{2}c{5}_all_{3}.npy".format(T_data, T_net, nLayers, int(AllJumps), direcString, ch), valid_diff/(1.0*N_train))

    elif Mode == "getY":
        y1Vecs, y2Vecs = Gather_Y(T_net, dirPath, State1_occs, State2_occs,
                OnSites_state1, OnSites_state2, jProbs_st1, jProbs_st2, NNsites, sp_ch,
                specsToTrain, VacSpec, gNet, Ndim, batch_size=batch_size, epoch=start_ep,
                Boundary_train=args.BoundTrain,jumpSwitch=args.JumpSwitch)

        np.save("y1_{4}_{0}_{1}_n{2}c{6}_all_{3}_{5}.npy".format(T_data, T_net, nLayers, int(AllJumps), direcString, start_ep, ch), y1Vecs)
        np.save("y2_{4}_{0}_{1}_n{2}c{6}_all_{3}_{5}.npy".format(T_data, T_net, nLayers, int(AllJumps), direcString, start_ep, ch), y2Vecs)
    
    elif Mode == "getRep":
        GetRep(T_net, T_data, dirPath, State1_occs, State2_occs, start_ep, gNet, args.RepLayer, N_train_jumps, batch_size=batch_size,
           avg=args.RepLayerAvg, AllJumps=AllJumps)

    print("All done\n\n")


if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description="Input parameters for using GCnets", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-DP", "--DataPath", metavar="/path/to/data", type=str, help="Path to Data file.")
    parser.add_argument("-cr", "--CrysDatPath", metavar="/path/to/crys/dat", type=str, help="Path to crystal Data.")
    parser.add_argument("-a0", "--LatParam",  metavar="L", type=float, help="Lattice parameter.")

    parser.add_argument("-m", "--Mode", metavar="M", type=str, help="Running mode (one of train, eval, getY, getRep). If getRep, then layer must specified with -RepLayer.")
    parser.add_argument("-rl","--RepLayer", metavar="[L1, L2,..]", type=int, nargs="+", help="Layers to extract representation from (count starts from 0)")
    parser.add_argument("-rlavg","--RepLayerAvg", action="store_true", help="Whether to average Representations across samples (training and validation will be made separate)")
    parser.add_argument("-bt","--BoundTrain", action="store_true", help="Whether to train using boundary state averages.")
    parser.add_argument("-jsr","--JumpSort", action="store_false", help="Whether to switch on/off sort jumps by rates. Not doing it will cause symmetry to break.")
    parser.add_argument("-jsw","--JumpSwitch", action="store_true", help="Whether to switch on/off jump channels in boundary mode depending on occupancy.")
    parser.add_argument("-xsh","--DispShift", action="store_true", help="Whether to shift displacements with state averages.")
    parser.add_argument("-nosym","--NoSymmetry", action="store_true", help="Whether to switch off all symmetry operations except identity.")
    parser.add_argument("-l0","--ScaleL0", action="store_true", help="Whether to scale transport coefficients during training with uncorrelated value.")

    parser.add_argument("-nl", "--Nlayers",  metavar="L", type=int, help="No. of layers of the neural network.")
    parser.add_argument("-nch", "--Nchannels", metavar="Ch", type=int, help="No. of representation channels in non-input layers.")
    parser.add_argument("-ncL", "--NchLast", metavar="1", type=int, help="No. channels of the last layers - how many vectors to produce per site.")
    parser.add_argument("-cngb", "--ConvNgbRange", type=int, default=1, metavar="NN", help="Nearest neighbor range of convolutional filters.")


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

    parser.add_argument("-d", "--DumpArgs", action="store_true", help="Whether to dump arguments in a file")
    parser.add_argument("-dpf", "--DumpFile", metavar="F", type=str, help="Name of file to dump arguments to (can be the jobID in a cluster for example).")

    args = parser.parse_args()    
    if args.DumpArgs:
        print("Dumping arguments to: {}".format(args.DumpFile))
        opts = vars(args)
        with open(RunPath + args.DumpFile, "w") as fl:
            for key, val in opts.items():
                fl.write("{}: {}\n".format(key, val))

    main(args)
