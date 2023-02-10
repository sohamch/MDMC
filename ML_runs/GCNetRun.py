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


def Load_crysDats(CrysDatPath):
    ## load the crystal data files
    with h5py.File(CrysDatPath, "r") as fl:
        dxJumps = np.array(fl["dxList_1nn"])
        JumpNewSites = np.array(fl["JumpSiteIndexPermutation"])
        GpermNNIdx = np.array(fl["GroupNNPermutation"])
        NNsiteList = np.array(fl["NNsiteList_sitewise"])

    return GpermNNIdx, NNsiteList, JumpNewSites, dxJumps

def Load_Data(DataPath, NoPerm=False):
    with h5py.File(DataPath, "r") as fl:
        if not NoPerm:
            try:
                perm = np.array(fl["Permutation"])
                print("found permuation to mix data set.")
            except:        
                print("No permuation array found in data set to mix data set.")
                perm = np.arange(len(fl["InitStates"]))

        else:
                print("Permutation not enabled.")
                perm = np.arange(len(fl["InitStates"]))

        state1List = np.array(fl["InitStates"])[perm]
        state2List = np.array(fl["FinStates"])[perm]
        dispList = np.array(fl["SpecDisps"])[perm]
        rateList = np.array(fl["rates"])[perm]
        AllJumpRates_st1 = np.array(fl["AllJumpRates_Init"])[perm]
        try: 
            AllJumpRates_st2 = np.array(fl["AllJumpRates_Fin"])[perm]
        except:
            print("Second step rates not found.")
            AllJumpRates_st2 = None
        
        try:
            avgDisps_st1 = np.array(fl["AvgDisps_Init"])[perm]
            avgDisps_st2 = np.array(fl["AvgDisps_Fin"])[perm]

        except:
            print("Average disps not found.")
            avgDisps_st1 = None
            avgDisps_st2 = None
 
    return state1List, state2List, dispList, rateList, AllJumpRates_st1, AllJumpRates_st2, avgDisps_st1, avgDisps_st2

def makeOnSites(stateOccs, specsToTrain, VacSpec, sp_ch):
    OnSites = None
    NJumps = stateOccs.shape[0]
    Nsites = stateOccs.shape[2]
    if specsToTrain != [VacSpec]:
        OnSites = np.zeros((NJumps, Nsites), dtype=np.int8)

        for spec in specsToTrain:
            OnSites += stateOccs[:, sp_ch[spec], :]

    return OnSites

def makeStateTensors(stateList, specsToTrain, VacSpec, JumpNewSites, AllJumps=False):
    Nsamples = stateList.shape[0]
    Nj = JumpNewSites.shape[0]
    specs = np.unique(stateList[0])
    NSpec = specs.shape[0] - 1
    Nsites = stateList.shape[1]

    sp_ch = {}
    for sp in specs:
        if sp == VacSpec:
            continue

        if sp - VacSpec < 0:
            sp_ch[sp] = sp
        else:
            sp_ch[sp] = sp - 1

    if AllJumps:
        stateOccs = np.zeros((Nsamples * Nj, NSpec, Nsites), dtype=np.int8)
        stateExits = np.zeros((Nsamples * Nj, NSpec, Nsites), dtype=np.int8)
        for stateInd in tqdm(range(Nsamples), position=0, leave=True):
            state1 = stateList[stateInd]
            for jmp in range(Nj):
                state2 = state1[JumpNewSites[jmp]]
                for site in range(1, Nsites):
                    sp1 = state1[site]
                    sp2 = state2[site]
                    stateOccs[stateInd * Nj + jmp, sp_ch[sp1], site] = 1
                    stateExits[stateInd * Nj + jmp, sp_ch[sp2], site] = 1

        Onsites_st1 = makeOnSites(stateOccs, specsToTrain, VacSpec, sp_ch)
        Onsites_st2 = makeOnSites(stateExits, specsToTrain, VacSpec, sp_ch)

        return stateOccs, stateExits, Onsites_st1, Onsites_st2, sp_ch

    else:
        stateOccs = np.zeros((Nsamples, NSpec, Nsites), dtype=np.int8)
        for stateInd in tqdm(range(Nsamples), position=0, leave=True):
            state1 = stateList[stateInd]
            for site in range(1, Nsites):
                sp1 = state1[site]
                stateOccs[stateInd, sp_ch[sp1], site] = 1

        Onsites = makeOnSites(stateOccs, specsToTrain, VacSpec, sp_ch)
        return stateOccs, Onsites, sp_ch

def makeComputeData(state1List, state2List, dispList, specsToTrain, VacSpec, rateList,
        AllJumpRates_st1, JumpNewSites, dxJumps, NNsiteList, N_train, AllJumps=False, mode="train"):
    
    # make the input tensors
    if mode=="train":
        Nsamples = N_train
    else:
        Nsamples = state1List.shape[0]

    a = np.linalg.norm(dispList[0, VacSpec, :])/np.linalg.norm(dxJumps[0]) 
    specs = np.unique(state1List[0])
    NSpec = specs.shape[0] - 1
    Nsites = state1List.shape[1]
    NNsvac = NNsiteList[1:, 0]

    NData = Nsamples
    if AllJumps:
        NData = Nsamples*dxJumps.shape[0]

    dispData = np.zeros((NData, 2, 3))
    rateData = np.zeros(NData)
    
    # Make the multichannel occupancies
    print("Building Occupancy Tensors for species : {}".format(specsToTrain))
    print("No. of jumps : {}".format(NData))

    if AllJumps:
        State1_occs, State2_occs, OnSites_state1, OnSites_state2, sp_ch =\
            makeStateTensors(state1List[:Nsamples], specsToTrain, VacSpec, JumpNewSites, AllJumps=AllJumps)

        # Next, Build the rates and displacements
        for samp in tqdm(range(Nsamples), position=0, leave=True):
            state1 = state1List[samp]
            for jInd in range(dxJumps.shape[0]):
                JumpSpec = state1[NNsvac[jInd]]
                Idx = samp*dxJumps.shape[0] + jInd
                dispData[Idx, 0, :] = dxJumps[jInd]*a
                if JumpSpec in specsToTrain:
                    dispData[Idx, 1, :] -= dxJumps[jInd]*a

                rateData[Idx] = AllJumpRates_st1[samp, jInd]

    else:
        # Next, Build the rates and displacements
        State1_occs, OnSites_state1, sp_ch = \
            makeStateTensors(state1List[:Nsamples], specsToTrain, VacSpec, JumpNewSites, AllJumps=AllJumps)

        State2_occs, OnSites_state2, _ = \
            makeStateTensors(state2List[:Nsamples], specsToTrain, VacSpec, JumpNewSites, AllJumps=AllJumps)

        for samp in tqdm(range(Nsamples), position=0, leave=True):
            dispData[samp, 0, :] = dispList[samp, VacSpec, :]
            dispData[samp, 1, :] = sum(dispList[samp, spec, :] for spec in specsToTrain)
            rateData[samp] = rateList[samp]

    
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
        y1 = pt.sum(y1*jPr_1_batch, dim=1) # average across the jumps
        y2 = pt.sum(y2*jPr_2_batch, dim=1)
    
    return y1, y2

# All non-vacancy batch calculations to be done here
def SpecBatchOuts(y1, y2, On_st1Batch, On_st2Batch, jProbs_st1, jProbs_st2, Boundary_train, AddOnSites):
    if not Boundary_train:
        On_st1Batch = On_st1Batch.unsqueeze(1).to(device)
        On_st2Batch = On_st2Batch.unsqueeze(1).to(device)
        y1 = pt.sum(y1[:, 0, :, :]*On_st1Batch, dim=2)
        y2 = pt.sum(y2[:, 0, :, :]*On_st2Batch, dim=2)
               
    else:
        assert y1.shape[1] == jProbs_st1.shape[1]
        assert y2.shape[1] == jProbs_st2.shape[1]

        jPr_1_batch = jProbs_st1.unsqueeze(2).to(device)
        jPr_2_batch = jProbs_st2.unsqueeze(2).to(device)
        
        if AddOnSites:
            # y have dimension (Nbacth, Njumps, Ndim, Nsites)
            # On_stBatch have dimensions (Nbatch, Nsites)
            # unsqueeze dim 1 twice to broadcast along jump channels and
            # cartesian components
            On_st1Batch = On_st1Batch.unsqueeze(1).unsqueeze(1).to(device)
            On_st2Batch = On_st2Batch.unsqueeze(1).unsqueeze(1).to(device)
            
            y1 = -y1[:, :, :, 0] + pt.sum(y1*On_st1Batch, dim=3)
            y2 = -y2[:, :, :, 0] + pt.sum(y2*On_st2Batch, dim=3)
        
        else:
            y1 = -y1[:, :, :, 0]
            y2 = -y2[:, :, :, 0]

        # Now average with the jump Probs
        # y have dimensions (Nbatch, Njumps, 3)
        # jPr have dimensions (Nbatch, Njumps, 1)
        # Do a broadcasted multiply, followed by sum along jumps
        y1 = pt.sum(y1 * jPr_1_batch, dim=1)
        y2 = pt.sum(y2 * jPr_2_batch, dim=1)
    
    return y1, y2
    

def sort_jp(jProbs_st1, jProbs_st2, jumpSort):
    if jumpSort:
        print("Sorting Jump Rates.")
        jProbs_st1 = np.sort(jProbs_st1, axis=1)
        jProbs_st2 = np.sort(jProbs_st2, axis=1)
    else:
        print("Jump Rates unsorted. Values will be non-symmetric if boundary-training is active.")
    
    jProbs_st1 = pt.tensor(jProbs_st1, dtype=pt.double)
    jProbs_st2 = pt.tensor(jProbs_st2, dtype=pt.double)

    return jProbs_st1, jProbs_st2


"""## Write the training loop"""
def Train(T, dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, rates, disps,
          jProbs_st1, jProbs_st2, SpecsToTrain, sp_ch, VacSpec, start_ep, end_ep, interval, N_train,
          gNet, lRate=0.001, batch_size=128, scratch_if_no_init=True, DPr=False, Boundary_train=False, jumpSort=True,
          AddOnSites=False, scaleL0=False, chkpt=True, decay=0.0005):
    
    print("Training conditions:")
    print("scratch: {}, DPr: {}, Boundary_train: {}, jumpSort: {}, AddOnSites: {}, scaleL0: {}".format(scratch_if_no_init, DPr, Boundary_train, jumpSort, AddOnSites, scaleL0))

    Ndim = disps.shape[2]
    state1Data, state2Data, dispData, rateData, On_st1, On_st2 = makeDataTensors(State1_Occs, State2_Occs, rates, disps,
            OnSites_st1, OnSites_st2, SpecsToTrain, VacSpec, sp_ch, Ndim=Ndim)

    if SpecsToTrain == [VacSpec]:
        assert pt.allclose(dispData.cpu(), pt.tensor(disps[:, 0, :], dtype=pt.double))
    else:
        assert pt.allclose(dispData.cpu(), pt.tensor(disps[:, 1, :], dtype=pt.double))

    # scale with L0 if indicated
    if scaleL0:
        L0 = pt.dot(rateData, pt.norm(dispData, dim=1)**2)/(6.0 * dispData.shape[0])
        L0 = L0.item()
    else:
        L0 = 1.0
    
    print("L0 : {}".format(L0))

    if Boundary_train:
        assert gNet.net[-3].Psi.shape[0] == jProbs_st1.shape[1] == jProbs_st2.shape[1] 
        print("Boundary training indicated. Using jump probabilities.")
        jProbs_st1, jProbs_st2 = sort_jp(jProbs_st1[:N_train], jProbs_st2[:N_train], jumpSort)

    N_batch = batch_size

    if pt.cuda.device_count() > 1 and DPr:
        print("Running on Devices : {}".format(DeviceIDList))
        gNet = nn.DataParallel(gNet, device_ids=DeviceIDList)

    try:
        # strict=False to ignore the renamed buffer "JumpVecs" (renamed from JumpUnitVecs)
        # As long as the same crysdats are used, this will not change
        gNet.load_state_dict(pt.load(dirPath + "/ep_{1}.pt".format(T, start_ep), map_location="cpu"), strict=False)

        if scratch_if_no_init and start_ep == 0:
            print("Training from scratch indicated (check option --Scratch), but saved initial network for epoch 0 found at", flush=True)
            print("save/load directory : {}".format(dirPath), flush=True)
            print("Terminating so as not to replace existing this pre-existing initial network.".format(dirPath), flush=True)
            raise RuntimeError("Terminating out of caution to not replace existing epoch 0 network. Please check printed message for more details.")

        print("Starting from epoch {}".format(start_ep), flush=True)

    except:
        if scratch_if_no_init:
            print("No Network found. Starting from scratch", flush=True)
        else:
            raise FileNotFoundError("Required saved networks not found in {} at epoch {}".format(dirPath, start_ep))

    print("Batch size : {}".format(N_batch))

    gNet.to(device)
    opt = pt.optim.Adam(gNet.parameters(), lr=lRate, weight_decay=decay)
    print("Starting Training loop")

    y1BatchTest = np.zeros((N_batch, 3))
    y2BatchTest = np.zeros((N_batch, 3))

    for epoch in tqdm(range(start_ep, end_ep + 1), position=0, leave=True):
        
        ## checkpoint
        if epoch % interval == 0 and chkpt:
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
            else:
                jProbs_st1_batch = None
                jProbs_st2_batch = None

            y1 = gNet(state1Batch)
            y2 = gNet(state2Batch)
                
            if SpecsToTrain==[VacSpec]: 
                y1, y2 = vacBatchOuts(y1, y2, jProbs_st1_batch, jProbs_st2_batch, Boundary_train)

            else:
                On_st1Batch = On_st1[batch : end]
                On_st2Batch = On_st2[batch : end]

                y1, y2 = SpecBatchOuts(y1, y2, On_st1Batch, On_st2Batch, jProbs_st1_batch, jProbs_st2_batch,
                                       Boundary_train, AddOnSites)

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
        rates, disps, SpecsToTrain, jProbs_st1, jProbs_st2, sp_ch, VacSpec,
        start_ep, end_ep, interval, N_train, gNet, batch_size=512, Boundary_train=False,
        DPr=False, jumpSort=True, AddOnSites=True):
    
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
    
    if Boundary_train:
        assert gNet.net[-3].Psi.shape[0] == jProbs_st1.shape[1] == jProbs_st2.shape[1] 
        print("Boundary training indicated. Using jump probabilities.")
        jProbs_st1, jProbs_st2 = sort_jp(jProbs_st1, jProbs_st2, jumpSort)
    
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
                # strict=False to ignore the renamed buffer "JumpVecs" (renamed from JumpUnitVecs)
                # As long as the same crysdats are used, this will not change
                gNet.load_state_dict(pt.load(dirPath + "/ep_{0}.pt".format(epoch), map_location=device), strict=False)
 
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
                    else:
                        jProbs_st1_batch = None
                        jProbs_st2_batch = None

                    y1 = gNet(state1Batch)
                    y2 = gNet(state2Batch)
            
                    if SpecsToTrain==[VacSpec]: 
                        y1, y2 = vacBatchOuts(y1, y2, jProbs_st1_batch, jProbs_st2_batch, Boundary_train)

                    else:
                        On_st1Batch = On_st1[batch : end]
                        On_st2Batch = On_st2[batch : end]
                        y1, y2 = SpecBatchOuts(y1, y2, On_st1Batch, On_st2Batch, jProbs_st1_batch, jProbs_st2_batch,
                                Boundary_train, AddOnSites)

                    dy = y2 - y1
                    loss = pt.sum(rateBatch * pt.norm((dispBatch + dy), dim=1)**2)/6.
                    diff += loss.item()

                diff_epochs.append(diff)

        return np.array(diff_epochs)
    
    train_diff = compute(0, N_train)#/(1.0*N_train)
    test_diff = compute(N_train, Nsamples)#/(1.0*(Nsamples - N_train))

    return train_diff, test_diff


def Gather_Y(T, dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, jProbs_st1, jProbs_st2,
        sp_ch, SpecsToTrain, VacSpec, gNet, Ndim, epoch=None, Boundary_train=False, batch_size=256,
        jumpSort=True, AddOnSites=True):
    
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
    
    if Boundary_train:
        assert gNet.net[-3].Psi.shape[0] == jProbs_st1.shape[1] == jProbs_st2.shape[1] 
        print("Boundary training indicated. Using jump probabilities.")
        jProbs_st1, jProbs_st2 = sort_jp(jProbs_st1, jProbs_st2, jumpSort)

        y1Vecs = np.zeros((Nsamples, 3))
        y2Vecs = np.zeros((Nsamples, 3))

    else:
        y1Vecs = np.zeros((Nsamples, 3))
        y2Vecs = np.zeros((Nsamples, 3))


    gNet.to(device)
    if epoch is not None:
        print("Network: {}".format(dirPath))
    with pt.no_grad():
        ## load checkpoint
        if epoch is not None:
            print("Loading epoch: {}".format(epoch))
            # strict=False to ignore the renamed buffer "JumpVecs" (renamed from JumpUnitVecs)
            # As long as the same crysdats are used, this will not change
            gNet.load_state_dict(pt.load(dirPath + "/ep_{0}.pt".format(epoch), map_location=device), strict=False)
             
        for batch in tqdm(range(0, Nsamples, N_batch), position=0, leave=True):
            end = min(batch + N_batch, Nsamples)

            state1Batch = state1Data[batch : end].double().to(device)
            state2Batch = state2Data[batch : end].double().to(device)
            
            if Boundary_train:
                jProbs_st1_batch = jProbs_st1[batch : end]
                jProbs_st2_batch = jProbs_st2[batch : end]
            else:
                jProbs_st1_batch = None
                jProbs_st2_batch = None

            y1 = gNet(state1Batch)
            y2 = gNet(state2Batch)

            if SpecsToTrain==[VacSpec]:
                y1, y2 = vacBatchOuts(y1, y2, jProbs_st1_batch, jProbs_st2_batch, Boundary_train)

            else:
                On_st1Batch = On_st1[batch : end]
                On_st2Batch = On_st2[batch : end]

                y1, y2 = SpecBatchOuts(y1, y2, On_st1Batch, On_st2Batch, jProbs_st1_batch, jProbs_st2_batch,
                                       Boundary_train, AddOnSites)

            y1Vecs[batch : end] = y1.cpu().numpy()
            y2Vecs[batch : end] = y2.cpu().numpy()

    return y1Vecs, y2Vecs

def GetRep(dirPath, State_Occs, epoch, gNet, LayerInd, batch_size=1000):
    
    N_batch = batch_size
    # Convert compute data to pytorch tensors
    state1Data = pt.tensor(State_Occs)
    Nsamples = state1Data.shape[0]
    Nsites = state1Data.shape[2]
    
    print("computing Representations after layer: {}".format(LayerInd))

    if LayerInd == len(gNet.net):
        ch = gNet.net[LayerInd - 3].Psi.shape[0]
        stReps = np.zeros((Nsamples, ch, 3, Nsites))
    else:
        ch = gNet.net[LayerInd - 2].Psi.shape[0]
        stReps = np.zeros((Nsamples, ch, Nsites))

    gNet.to(device)
    with pt.no_grad():
        ## load checkpoint
        # strict=False to ignore the renamed buffer "JumpVecs" (renamed from JumpUnitVecs)
        # As long as the same crysdats are used, this will not change
        gNet.load_state_dict(pt.load(dirPath + "/ep_{0}.pt".format(epoch), map_location=device), strict=False)
        for batch in tqdm(range(0, Nsamples, N_batch), position=0, leave=True):
            end = min(batch + N_batch, Nsamples)

            state1Batch = state1Data[batch : end].double().to(device)

            y1 = gNet.getRep(state1Batch, LayerInd)
            stReps[batch : end] = y1.cpu().numpy()

    return stReps

def makeDir(args, specsToTrain):
    direcString=""
    if specsToTrain == [args.VacSpec]:
        direcString = "vac"
    else:
        for spec in specsToTrain:
            direcString += "{}".format(spec)

    # 1 This is where networks will be saved to and loaded from
    dirNameNets = "ep_T_{0}_{1}_n{2}c{4}_all_{3}".format(args.TNet, direcString, args.Nlayers, int(args.AllJumps), args.Nchannels)
    if args.Mode == "eval" or args.Mode == "getY" or args.Mode=="getRep":
        prepo = "saved at"
        dirNameNets = "ep_T_{0}_{1}_n{2}c{4}_all_{3}".format(args.TNet, direcString, args.Nlayers, int(args.AllJumpsNetType), args.Nchannels)
    
    elif args.Mode == "train":
        prepo = "saving in"

    # 2 check if a run directory exists
    dirPath = RunPath + dirNameNets
    if not os.path.isdir(dirPath):
        if args.Start_epoch == 0:
            os.mkdir(dirPath)
        elif args.Start_epoch > 0:
            raise ValueError("Training directory does not exist but start epoch greater than zero: {}\ndirectory given: {}".format(args.Start_epoch, dirPath))

    print("Running in Mode {} with networks {} {}".format(args.Mode, prepo, dirPath))
    return dirPath, direcString

def main(args):
    print("Running at : "+ RunPath)
    if args.Mode=="train" and args.Tdata != args.TNet:
        raise ValueError("Different temperatures in training mode not allowed")

    if not (args.Mode == "train" or args.Mode == "eval"):
        print("Mode : {}, setting end epoch to start epoch".format(args.Mode))
        args.End_epoch = args.Start_epoch

    if not (args.Mode == "train" or args.Mode == "eval" or args.Mode == "getY" or args.Mode == "getRep"):
        raise ValueError("Mode needs to be train, eval, getY or getRep but given : {}".format(args.Mode))

    if args.Mode == "train":
        if args.Tdata != args.TNet:
            raise ValueError("Network and data temperature (arguments \"TNet\"/\"tn\" and \"Tdata\"/\"td\") must be the same in train mode")
    
    if args.AllJumps and args.BoundTrain:
        raise NotImplementedError("Cannot do all-jump training with boundary states.")
    
    # 1. Load crystal parameters
    GpermNNIdx, NNsiteList, JumpNewSites, dxJumps = Load_crysDats(args.CrysDatPath)
    N_ngb = NNsiteList.shape[0]
    z = N_ngb - 1

    if args.NoSymmetry:
        print("Switching off convolution symmetries (considering only identity operator).")
        GnnPerms = pt.tensor(GpermNNIdx[:1]).long()
        try:
            assert pt.equal(GnnPerms, pt.arange(N_ngb).unsqueeze(0))
        except:
            raise ValueError("The 0th group operation in GpermNNIdx needs to be the identity.")
        print("Switching off jump due to under non-symmetric condition")
        args.JumpSort = False

    else:
        print("Considering full symmetry group.")
        GnnPerms = pt.tensor(GpermNNIdx).long()

    # Dump args to files if necessary.
    if args.DumpArgs:
        print("Dumping arguments to: {}".format(args.DumpFile))
        opts = vars(args)
        with open(RunPath + args.DumpFile, "w") as fl:
            for key, val in opts.items():
                fl.write("{}: {}\n".format(key, val))

    NNsites = pt.tensor(NNsiteList).long()
    JumpVecs = pt.tensor(dxJumps.T * args.LatParam, dtype=pt.double)
    print("Jump Vectors: \n", JumpVecs.T, "\n")
    Ndim = dxJumps.shape[1]

    # 2. Load data
    state1List, state2List, dispList, rateList, AllJumpRates_st1, AllJumpRates_st2, avgDisps_st1, avgDisps_st2 = Load_Data(args.DataPath, args.NoPerm)
    
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
    
    # 2.3 Make numpy arrays to feed into training/evaluation functions
    specsToTrain = [int(args.SpecTrain[i]) for i in range(len(args.SpecTrain))]
    specsToTrain = sorted(specsToTrain)

    # 3. Make directories if needed
    specs = np.unique(state1List[0])
    NSpec = specs.shape[0] - 1
    dirPath, direcString = makeDir(args, specsToTrain)
    
    if args.BoundTrain:
        assert args.NchLast == z

    gNet = GCNet(GnnPerms.long(), NNsites, JumpVecs, N_ngb=N_ngb, NSpec=NSpec,
            mean=args.Mean_wt, std=args.Std_wt, nl=args.Nlayers, nch=args.Nchannels, nchLast=args.NchLast).double()

    print("No. of channels in last layer: {}".format(gNet.net[-3].Psi.shape[0]))

    # 4. Call Training or evaluating or y-evaluating function here
    N_train_jumps = z*args.N_train if args.AllJumps else args.N_train
    if args.Mode == "train":
        State1_occs, State2_occs, rateData, dispData, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(state1List, state2List, dispList, specsToTrain, args.VacSpec, rateList,
                            AllJumpRates_st1, JumpNewSites, dxJumps, NNsiteList, args.N_train, AllJumps=args.AllJumps,
                            mode=args.Mode)
        print("Done Creating numpy occupancy tensors. Species channels: {}".format(sp_ch))

        Train(args.Tdata, dirPath, State1_occs, State2_occs, OnSites_state1, OnSites_state2,
              rateData, dispData, jProbs_st1, jProbs_st2, specsToTrain, sp_ch, args.VacSpec,
              args.Start_epoch, args.End_epoch, args.Interval, N_train_jumps, gNet,
              lRate=args.Learning_rate, scratch_if_no_init=args.Scratch, batch_size=args.Batch_size,
              DPr=args.DatPar, Boundary_train=args.BoundTrain, jumpSort=args.JumpSort, AddOnSites=args.AddOnSitesJPINN,
              scaleL0=args.ScaleL0, decay=args.Decay)

    elif args.Mode == "eval":
        State1_occs, State2_occs, rateData, dispData, OnSites_state1, OnSites_state2, sp_ch = \
            makeComputeData(state1List, state2List, dispList, specsToTrain, args.VacSpec, rateList,
                            AllJumpRates_st1, JumpNewSites, dxJumps, NNsiteList, args.N_train, AllJumps=args.AllJumps,
                            mode=args.Mode)
        print("Done Creating numpy occupancy tensors. Species channels: {}".format(sp_ch))

        train_diff, valid_diff = Evaluate(args.TNet, dirPath, State1_occs, State2_occs,
                OnSites_state1, OnSites_state2, rateData, dispData,
                specsToTrain, jProbs_st1, jProbs_st2, sp_ch, args.VacSpec, args.Start_epoch, args.End_epoch,
                args.Interval, N_train_jumps, gNet, batch_size=args.Batch_size, Boundary_train=args.BoundTrain,
                DPr=args.DatPar, jumpSort=args.JumpSort, AddOnSites=args.AddOnSitesJPINN)
        np.save("tr_{0}_{1}_{2}_n{3}c{4}_all_{5}.npy".format(direcString, args.Tdata, args.TNet, args.Nlayers, args.Nchannels,
                                                             int(args.AllJumps)), train_diff/(1.0*args.N_train))
        np.save("val_{0}_{1}_{2}_n{3}c{4}_all_{5}.npy".format(direcString, args.Tdata, args.TNet, args.Nlayers, args.Nchannels,
                                                              int(args.AllJumps)), valid_diff/(1.0*args.N_train))

    elif args.Mode == "getY":
        if args.AllJumps:
            State1_occs, State2_occs, OnSites_state1, OnSites_state2, sp_ch = \
                makeStateTensors(state1List, specsToTrain, args.VacSpec, JumpNewSites, AllJumps=True)

            print("Calculating y for state 1  and state 1 exits for {}.".format(args.Tdata))
            y_st1_Vecs, y_st1_Exits = Gather_Y(args.TNet, dirPath, State1_occs, State2_occs,
                                      OnSites_state1, OnSites_state2, jProbs_st1, jProbs_st2, sp_ch,
                                      specsToTrain, args.VacSpec, gNet, Ndim, batch_size=args.Batch_size,
                                      epoch=args.Start_epoch,
                                      Boundary_train=args.BoundTrain, AddOnSites=args.AddOnSitesJPINN)

            State1_occs, State2_occs, OnSites_state1, OnSites_state2, sp_ch = \
                makeStateTensors(state2List, specsToTrain, args.VacSpec, JumpNewSites, AllJumps=True)

            print("Calculating y for state 2  and state 2 exits for {}.".format(args.Tdata))
            y_st2_Vecs, y_st2_Exits = Gather_Y(args.TNet, dirPath, State1_occs, State2_occs,
                                               OnSites_state1, OnSites_state2, jProbs_st1, jProbs_st2, sp_ch,
                                               specsToTrain, args.VacSpec, gNet, Ndim, batch_size=args.Batch_size,
                                               epoch=args.Start_epoch,
                                               Boundary_train=args.BoundTrain, AddOnSites=args.AddOnSitesJPINN)

            np.save("y_st1_{0}_{1}_{2}_n{3}c{4}_all_{5}_{6}.npy".format(direcString, args.Tdata,
                                                                                args.TNet, args.Nlayers,args.Nchannels,
                                                                                int(args.AllJumps), args.Start_epoch), y_st1_Vecs[::z])

            np.save("y_st1_exits_{0}_{1}_{2}_n{3}c{4}_all_{5}_{6}.npy".format(direcString, args.Tdata,
                                                                                args.TNet, args.Nlayers,args.Nchannels,
                                                                                int(args.AllJumps), args.Start_epoch), y_st1_Exits)

            np.save("y_st2_{0}_{1}_{2}_n{3}c{4}_all_{5}_{6}.npy".format(direcString, args.Tdata,
                                                                        args.TNet, args.Nlayers,args.Nchannels,
                                                                        int(args.AllJumps), args.Start_epoch), y_st2_Vecs[::z])

            np.save("y_st2_exits_{0}_{1}_{2}_n{3}c{4}_all_{5}_{6}.npy".format(direcString, args.Tdata,
                                                                                args.TNet, args.Nlayers,args.Nchannels,
                                                                                int(args.AllJumps), args.Start_epoch), y_st2_Exits)

        else:
            State1_occs, OnSites_state1, sp_ch = \
                makeStateTensors(state1List, specsToTrain, args.VacSpec, JumpNewSites, AllJumps=False)

            State2_occs, OnSites_state2, sp_ch = \
                makeStateTensors(state2List, specsToTrain, args.VacSpec, JumpNewSites, AllJumps=False)

            y1Vecs, y2Vecs = Gather_Y(args.TNet, dirPath, State1_occs, State2_occs,
                    OnSites_state1, OnSites_state2, jProbs_st1, jProbs_st2, sp_ch,
                    specsToTrain, args.VacSpec, gNet, Ndim, batch_size=args.Batch_size, epoch=args.Start_epoch,
                    Boundary_train=args.BoundTrain, AddOnSites=args.AddOnSitesJPINN)

            np.save("y_st1_{0}_{1}_{2}_n{3}c{4}_all_{5}_{6}.npy".format(direcString, args.Tdata, args.TNet, args.Nlayers,
                                                                        args.Nchannels, int(args.AllJumps), args.Start_epoch),
                    y1Vecs)
            np.save("y_st2_{0}_{1}_{2}_n{3}c{4}_all_{5}_{6}.npy".format(direcString, args.Tdata, args.TNet, args.Nlayers,
                                                                        args.Nchannels, int(args.AllJumps), args.Start_epoch),
                    y2Vecs)
    
    elif args.Mode == "getRep":
        if args.RepLayer == len(gNet.net):
            print("site wise y vectors will be computed for indicated samples")

        if args.AllJumps:
            State1_occs, State1_exit_occs, _, _, _ = \
                makeStateTensors(state1List[args.RepStart : args.RepStart + args.N_train], specsToTrain, args.VacSpec,
                                 JumpNewSites, AllJumps=True)

            stReps_st1 = GetRep(dirPath, State1_occs, args.Start_epoch, gNet, args.RepLayer,
                                 batch_size=args.Batch_size)
            stReps_st1_exits = GetRep(dirPath, State1_exit_occs, args.Start_epoch, gNet, args.RepLayer,
                                   batch_size=args.Batch_size)

            State2_occs, State2_exit_occs, _, _, _ = \
                makeStateTensors(state2List[args.RepStart : args.RepStart + args.N_train], specsToTrain, args.VacSpec,
                                 JumpNewSites, AllJumps=True)
            stReps_st2 = GetRep(dirPath, State2_occs, args.Start_epoch, gNet, args.RepLayer,
                                   batch_size=args.Batch_size)

            stReps_st2_exits = GetRep(dirPath, State2_exit_occs, args.Start_epoch, gNet, args.RepLayer,
                                batch_size=args.Batch_size)

            np.save("Rep_L_{0}_st1_{1}_{2}_{3}_n{4}c{5}_all_{6}_{7}.npy".format(args.RepLayer, direcString, args.Tdata,
                                                                                args.TNet, args.Nlayers,args.Nchannels,
                                                                                int(args.AllJumps), args.Start_epoch),
                    stReps_st1[::z])

            np.save("Rep_L_{0}_st1Exits_{1}_{2}_{3}_n{4}c{5}_all_{6}_{7}.npy".format(args.RepLayer, direcString, args.Tdata,
                                                                                args.TNet, args.Nlayers,args.Nchannels,
                                                                                int(args.AllJumps), args.Start_epoch),
                    stReps_st1_exits)

            np.save("Rep_L_{0}_st2_{1}_{2}_{3}_n{4}c{5}_all_{6}_{7}.npy".format(args.RepLayer, direcString, args.Tdata,
                                                                                args.TNet, args.Nlayers,args.Nchannels,
                                                                                int(args.AllJumps), args.Start_epoch),
                    stReps_st2[::z])

            np.save("Rep_L_{0}_st2_exits_{1}_{2}_{3}_n{4}c{5}_all_{6}_{7}.npy".format(args.RepLayer, direcString, args.Tdata,
                                                                                args.TNet, args.Nlayers,args.Nchannels,
                                                                                int(args.AllJumps), args.Start_epoch),
                    stReps_st2_exits)

        else:
            State1_occs, _, _ = \
                makeStateTensors(state1List[args.RepStart: args.RepStart + args.N_train],
                                 specsToTrain, args.VacSpec, JumpNewSites, AllJumps=False)
            stReps_st1 = GetRep(dirPath, State1_occs, args.Start_epoch, gNet, args.RepLayer,
                                batch_size=args.Batch_size)

            np.save("Rep_L_{0}_st1_{1}_{2}_{3}_n{4}c{5}_all_{6}_{7}.npy".format(args.RepLayer, direcString, args.Tdata,
                                                                                args.TNet, args.Nlayers,args.Nchannels,
                                                                                int(args.AllJumps), args.Start_epoch),
                    stReps_st1)

            State2_occs, _, _ = \
                makeStateTensors(state2List[args.RepStart: args.RepStart + args.N_train],
                                 specsToTrain, args.VacSpec, JumpNewSites, AllJumps=False)
            stReps_st2 = GetRep(dirPath, State2_occs, args.Start_epoch, gNet, args.RepLayer,
                                batch_size=args.Batch_size)

            np.save("Rep_L_{0}_st2_{1}_{2}_{3}_n{4}c{5}_all_{6}_{7}.npy".format(args.RepLayer, direcString, args.Tdata,
                                                                                args.TNet, args.Nlayers,args.Nchannels,
                                                                                int(args.AllJumps), args.Start_epoch),
                    stReps_st2)

    print("All done\n\n")


if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description="Input parameters for using GCnets", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-DP", "--DataPath", metavar="/path/to/data", type=str, help="Path to Data file.")
    parser.add_argument("-np", "--NoPerm", action="store_true", help="Whether to mix up the data set if a permutation array is found.")
    parser.add_argument("-cr", "--CrysDatPath", metavar="/path/to/crys/dat", type=str, help="Path to crystal Data.")
    parser.add_argument("-a0", "--LatParam", type=float, default=1.0, metavar="3.59", help="Lattice parameter.")

    parser.add_argument("-m", "--Mode", metavar="M", type=str, help="Running mode (one of train, eval, getY, getRep). If getRep, then layer must specified with -RepLayer.")
    parser.add_argument("-rl", "--RepLayer", metavar="L (int)", type=int, help="Which Layer to extract representation from (count starts from 0). Must be (0 or 2 or 5 or 8 ...)")
    parser.add_argument("-rStart", "--RepStart", metavar="L (int)", type=int, default=0, help="Which sample to start computing representations onward (N_train no. of samples will be computed).")
    parser.add_argument("-bt","--BoundTrain", action="store_true", help="Whether to train using boundary state averages.")
    parser.add_argument("-nojsr", "--JumpSort", action="store_false", help="Whether to switch on/off sort jumps by rates. Not doing it will cause symmetry to break.")
    parser.add_argument("-aos","--AddOnSitesJPINN", action="store_true", help="Whether to consider on sites along with vacancy sites in JPINN.")
    parser.add_argument("-nosym", "--NoSymmetry", action="store_true", help="Whether to switch off all symmetry operations except identity.")
    parser.add_argument("-l0", "--ScaleL0", action="store_true", help="Whether to scale transport coefficients during training with uncorrelated value.")

    parser.add_argument("-nl", "--Nlayers",  metavar="L", type=int, default=1, help="No. of intermediate layers of the neural network.")
    parser.add_argument("-nch", "--Nchannels", metavar="Ch", type=int, default=4, help="No. of representation channels in non-input layers.")
    parser.add_argument("-ncL", "--NchLast", metavar="1", type=int, default=1, help="No. channels of the last layers - how many vectors to produce per site.")

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
    parser.add_argument("-dcy", "--Decay", type=float, default=0.0005, help="Weight decay (L2 penalty for the weights).")
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
