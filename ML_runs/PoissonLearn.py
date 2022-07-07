import os
import sys
import argparse
RunPath = os.getcwd() + "/"

sys.path.append(ModulePath)

import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import h5py
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from SymmLayers import *
import GCNetRun
from GCNetRun import makeComputeData, makeProdTensor, makeDataTensors, Gather_Y 

device=None
if pt.cuda.is_available():
    print(pt.cuda.get_device_name())
    device = pt.device("cuda:0")
    DeviceIDList = list(range(pt.cuda.device_count()))
else:
    device = pt.device("cpu")

def Load_crysDats(nn=1, typ="FCC"):
    ## load the crystal data files
    print("Loading crystal data for : {}".format(typ))
    if typ == "FCC":
        CrysDatPath = CrysPath + "CrysDat_FCC/"
    elif typ == "BCC":
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

def Load_Data(FilePath):
    with h5py.File(FileName, "r") as fl:
        
        state1List = np.array(fl["InitStates"])
        state2List = np.array(fl["FinStates"])
        dispList = np.array(fl["SpecDisps"])
        rateList = np.array(fl["rates"])
        
        try:
            AllJumpRates = np.array(fl["AllJumpRates"])
            # Transform rates to probabilities
            rateSums = np.sum(AllJumpRates, axis=1).reshape[-1, 1]
            AllJumpRates /= rateSums
        except:
            AllJumpRates = None
            raise ValueError("All Jump Rates not provided in data set.")
    
    return state1List, state2List, dispList, rateList, AllJumpRates


"""## Write the training loop"""
def Train(T_data, dirPath, state1List, state2List, AllJumpRates, dxJumps,
        NNSites, rateList, dispList, JumpNewSites, 
        specsToTrain, VacSpec, start_ep, end_ep, interval, N_train, 
        gNet, gNetInit, lRate=0.001, scratch_if_no_init=True, batch_size=10)
    
    #1. Call MakeComputeData here to make transition pairs
    State1_Occs, State2_Occs, rates, disps, OnSites_state1, OnSites_state2, sp_ch = makeComputeData(state1List, state2List, dispList,
            specsToTrain, VacSpec, rateList, AllJumpRates, JumpNewSites, dxJumps, NNsiteList, N_train, AllJumps=True, mode="train")

    Ndim = disps.shape[2]
    N_batch = batch_size
    Njmp = AllJumpRates.shape[1]
    
    #2. Convert compute data to pytorch tensors
    state1Data, state2Data, dispData, rateData, On_st1, On_st2 = makeDataTensors(State1_Occs, State2_Occs, rates, disps,
            OnSites_state1, OnSites_state2, specsToTrain, VacSpec, sp_ch, N_train)
    
    state1Data = state1Data[::Njmp] # extract the unique initial states
    if On_st1 is not None:
        On_st1 = On_st1[::Njmp]

    #3. Next we have to compute the relaxation vectors with gNetInit
    y1vecs, y2vecs = Gather_Y(None, None, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, 
            sp_ch, SpecsToTrain, VacSpec, epoch, gNetInit, Ndim, batch_size=256)
    
    #4. Next, compute relaxation changes
    delY = np.zeros((N_train, 3))
    for samp in range(N_train):
        y1 = y1vecs[samp*Njmp]
        y2Av = np.zeros(3)
        AvDisp = np.zeros(3)
        for jmp in range(Njmp):
            y2Av += AllJumpRates[samp, jmp] * y2vecs[samp*Njmp + jmp]
            if state1List[samp, NNsites[jmp+1, 0]] in specsToTrain:
                AvDisp -= dxJumps[jmp] * AllJumpRates[samp, jmp]

        delY[samp] = AvDisp + y2Av - y1

    #5. Try loading saved networks
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
    BackgroundSpecs = [[spec] for spec in range(state1Data.shape[1]) if spec not in specTrainCh] 

    gNet.to(device)
    
    opt = pt.optim.Adam(gNet.parameters(), lr=lRate) #, weight_decay=0.0005)

    print("Starting Training loop") 

    for epoch in tqdm(range(start_ep, end_ep + 1), position=0, leave=True):
        
        ## checkpoint
        if epoch%interval==0:
            pt.save(gNet.state_dict(), dirPath + "/ep_{0}.pt".format(epoch))
            
        for batch in range(0, N_train, N_batch):
            
            opt.zero_grad()
            
            end = min(batch + N_batch, N_train)

            state1Batch = state1Data[batch : end].double().to(device)
            On_st1Batch = On_st1[batch : end].to(device)
            y1 = gNet(state1Batch)
            y1 = pt.sum(y1*On_st1Batch, dim=2)
 
            state2Batch = state2Data[batch * Njmp : (end+1) * Njmp].double().to(device)
            On_st2Batch = On_st1[batch * Njmp : (end + 1) * Njmp].to(device)
            y2Batch = gNet(state2Batch) 
            
            rateBatch = rateData[batch : end]
            
            diff.backward()
            
            opt.step()


def Evaluate(T, dirPath, State1_Occs, State2_Occs, OnSites_st1, OnSites_st2, 
        rates, disps, SpecsToTrain, sp_ch, VacSpec, start_ep, end_ep, interval, N_train,
        gNet, batch_size=512):
    
    for key, item in sp_ch.items():
        if key > VacSpec:
            assert item == key - 1
        else:
            assert key < VacSpec
            assert item == key

    Ndim = disps.shape[2]
    N_batch = batch_size
    # Convert compute data to pytorch tensors
    state1Data = pt.tensor(State1_Occs).double()
    Nsamples = state1Data.shape[0]

    print("Evaluating species: {}, Vacancy label: {}".format(SpecsToTrain, VacSpec))
    print("Sample Jumps: {}, Training: {}, Validation: {}, Batch size: {}".format(Nsamples, N_train, Nsamples-N_train, N_batch))
    print("Evaluating with networks at: {}".format(dirPath))
    
    state2Data = pt.tensor(State2_Occs)
    rateData = pt.tensor(rates)
    On_st1 = None
    On_st2 = None
    
    if SpecsToTrain == [VacSpec]:
        assert OnSites_st1 == OnSites_st2 == None
        dispData = pt.tensor(disps[:, 0, :]).double()
    else:
        dispData = pt.tensor(disps[:, 1, :]).double() 
        On_st1 = makeProdTensor(OnSites_st1, Ndim).long()
        On_st2 = makeProdTensor(OnSites_st2, Ndim).long()

    specTrainCh = [sp_ch[spec] for spec in SpecsToTrain]
    BackgroundSpecs = [[spec] for spec in range(state1Data.shape[1]) if spec not in specTrainCh] 
    
    if isinstance(gNet, GCSubNet) or isinstance(gNet, GCSubNetRes): 
        print("Species Channels to train: {}".format(specTrainCh))
        print("Background species channels: {}".format(BackgroundSpecs))

    def compute(startSample, endSample):
        diff_epochs = []
        with pt.no_grad():
            for epoch in tqdm(range(start_ep, end_ep + 1, interval), position=0, leave=True):
                ## load checkpoint
                gNet.load_state_dict(pt.load(dirPath + "/ep_{0}.pt".format(epoch), map_location=device)) 
                if isinstance(gNet, GCSubNet) or isinstance(gNet, GCSubNetRes): 
                    gNet.Distribute_subNets()
                
                diff = 0 
                for batch in range(startSample, endSample, N_batch):
                    end = min(batch + N_batch, endSample)

                    state1Batch = state1Data[batch : end].double().to(device)
                    state2Batch = state2Data[batch : end].double().to(device)
                    
                    rateBatch = rateData[batch : end].to(device)
                    dispBatch = dispData[batch : end].to(device)
                    
                    
                    if isinstance(gNet, GCNet) or isinstance(gNet, GCNetRes):
                        y1 = gNet(state1Batch)
                        y2 = gNet(state2Batch)

                    else:
                        y1 = gNet.forward(state1Batch, specTrainCh, BackgroundSpecs)
                        y2 = gNet.forward(state2Batch, specTrainCh, BackgroundSpecs)
                    
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

def GetRep(T_net, T_data, dirPath, State1_Occs, State2_Occs, epoch, gNet, LayerIndList, N_train, batch_size=1000,
           avg=True, AllJumps=False):
    
    N_batch = batch_size
    # Convert compute data to pytorch tensors
    state1Data = pt.tensor(State1_Occs)
    Nsamples = state1Data.shape[0]
    Nsites = state1Data.shape[2]
    state2Data = pt.tensor(State2_Occs)
    
    storeDir = RunPath + "StateReps_{}".format(T_net) 
    exists = os.path.isdir(storeDir)
    if not exists:
        os.mkdir(RunPath + "StateReps_{}".format(T_net))
    
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
            
                np.save(storeDir + "/Rep1_Poiss_l{6}_{0}_{1}_n{2}c{5}_all_{3}_{4}.npy".format(T_data, T_net, nLayers, int(AllJumps), epoch, glob_Nch, LayerInd), y1Reps)
                np.save(storeDir + "/Rep2_Poiss_l{6}_{0}_{1}_n{2}c{5}_all_{3}_{4}.npy".format(T_data, T_net, nLayers, int(AllJumps), epoch, glob_Nch, LayerInd), y2Reps)


def main(args):
    print("Running at : "+ RunPath)

    # Get run parameters
    #parser.add_argument("-DP", "--DataFilePath", metavar="/path/to/datafile", type=str, help="Data file Path.")
    #parser.add_argument("-RP", "--InitRunPath", metavar="/path/to/save-or-read/nets", type=str, help="Path to load and save networks.")
    #parser.add_argument("-Crp", "--CrysDatPath", metavar="/path/to/crysdats/", type=str, help="Path to read crystal data out of.")
    
    DataFilePath = args.DataFilePath # Full Path of data file to train on
    InitRunPath = args.InitRunPath # Full Path of the network with which to compute relaxations
    CrysDatPath = args.CrysDatPath
    
    Mode = args.Mode # "train" mode or "eval" mode or "getY" mode
    
    nLayers = args.Nlayers
    
    ch = args.Nchannels
    
    filter_nn = args.ConvNgbRange
    
    nLayersInit = args.NlayersInit
    
    chInit = args.NchannelsInit
    
    filter_nn_init = args.ConvNgbRangeInit
    
    scratch_if_no_init = args.Scratch
    
    T_data = args.Tdata

    start_ep = args.Start_epoch
    end_ep = args.End_epoch

    if not (Mode == "train" or Mode == "eval"):
        print("Mode : {}, setting end epoch to start epoch".format(Mode))
        end_ep = start_ep
    
    specTrain = args.SpecTrain
    
    VacSpec = args.VacSpec
    
    N_train = args.N_train

    batch_size = args.Batch_size
    interval = args.Interval # for train mode, interval to save and for eval mode, interval to load
    learning_Rate = args.Learning_rate

    wt_means = args.Mean_wt
    wt_std = args.Std_wt

    if not (Mode == "train" or Mode == "eval"):
        raise ValueError("Mode needs to be train or eval but given : {}".format(Mode))

    #1. Load data
    state1List, state2List, dispList, rateList, AllJumpRates = Load_Data(DataFilePath)
    
    #2. This is where the Poisson networks will be saved to and loaded from
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

    dirNameNets = "ep_Poiss_T_{0}_{1}_n{2}c{3}".format(T_net, direcString, nLayers, ch)
    if Mode == "eval" or Mode == "getY" or Mode=="getRep":
        prepo = "saved at"
    
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
    print("Computing relaxations with: {}".format(InitRunPath))

    print(pt.__version__)
    
    #3.  Load crystal parameters
    GpermNNIdx, NNsiteList, siteShellIndices, GIndtoGDict, JumpNewSites, dxJumps = Load_crysDats(nn=filter_nn, typ=CrysDatPath)
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
    
    #4. Make the required networks
    #4.1 Make a network to either train from scratch or load saved state into
    gNet = GCNet(GnnPerms, gdiags, NNsites, SitesToShells, Ndim, N_ngb, NSpec,
            mean=wt_means, std=wt_std, b=1.0, nl=nLayers, nch=ch).double().to(device)
    
    #4.2 - load the network to compute relaxations with
    gNetInit = GCNet(GnnPerms, gdiags, NNsites, SitesToShells, Ndim, N_ngb, NSpec,
            mean=wt_means, std=wt_std, b=1.0, nl=nLayersInit, nch=chInit).double()
    
    gNetInit.load_state_dict(pt.load(InitRunPath, map_location="cpu")).to(device)
    
    #5. Call Training or evaluating or y-evaluating function here
    #5.1 - Training mode
    #state1List, state2List, dispList, rateList, AllJumpRates = Load_Data(DataFilePath)
    if Mode == "train":
        Train(T_data, dirPath, state1List, state2List, rateList, dispList, specsToTrain, VacSpec, start_ep, end_ep, interval, N_train,
                gNet, gNetInit, lRate=learning_Rate, scratch_if_no_init=scratch_if_no_init, batch_size=batch_size)
    
    #5.2 - Evauation mode
    elif Mode == "eval":
        train_diff, valid_diff = Evaluate(T_data, dirPath, state1List, state2List, rateList, dispList,
                specsToTrain, sp_ch, VacSpec, start_ep, end_ep,
                interval, N_train, gNet, batch_size=batch_size)
        np.save("tr_Poiss_{3}_{0}_{1}_n{2}c{4}.npy".format(T_data, T_net, nLayers, direcString, ch), train_diff/(1.0*N_train))
        np.save("val_Poiss_{3}_{0}_{1}_n{2}c{4}.npy".format(T_data, T_net, nLayers, direcString, ch), valid_diff/(1.0*N_train))
    
    #5.3 - Get the representations
    elif Mode == "getRep":
        GetRep(T_net, T_data, dirPath, State1_Occs, State2_Occs, start_ep, gNet, args.RepLayer, N_train_jumps, batch_size=batch_size,
           avg=args.RepLayerAvg, AllJumps=AllJumps)
    print("All done\n\n")

# Add argument parser
parser = argparse.ArgumentParser(description="Input parameters for using GCnets", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-DP", "--DataFilePath", metavar="/path/to/datafile", type=str, help="Data file Path.")
parser.add_argument("-RP", "--InitRunPath", metavar="/path/to/save-or-read/nets", type=str, help="Path to load and save networks.")
parser.add_argument("-CrP", "--CrysDatPath", metavar="/path/to/crysdats/", type=str, help="Path to read crystal data out of.")

# We also need a path to the initial trained network to compute the state relaxations
parser.add_argument("-gep", "--GFNetEpoch", metavar="/path/to/nets/", type=str, help="epoch to compute the relaxations.")

parser.add_argument("-m", "--Mode", metavar="M", type=str, help="Running mode (one of train, eval, getY, getRep). If getRep, then layer must specified with -RepLayer.")
parser.add_argument("-rl","--RepLayer", metavar="[L1, L2,..]", type=int, nargs="+", help="Layers to extract representation from (count starts from 0)")
parser.add_argument("-rlavg","--RepLayerAvg", action="store_true", help="Whether to average Representations across samples (training and validation will be made separate)")

parser.add_argument("-nl", "--Nlayers",  metavar="L", type=int, help="No. of layers of the neural network.")
parser.add_argument("-nch", "--Nchannels", metavar="Ch", type=int, help="No. of representation channels in non-input layers.")
parser.add_argument("-cngb", "--ConvNgbRange", type=int, default=1, metavar="NN", help="Nearest neighbor range of convolutional filters.")

parser.add_argument("-nlI", "--NlayersInit",  metavar="L", type=int, help="No. of layers of the initial neural network.")
parser.add_argument("-nchI", "--NchannelsInit", metavar="Ch", type=int, help="No. of representation channels in the initial network.")
parser.add_argument("-cngbI", "--ConvNgbRangeInit", type=int, default=1, metavar="NN", help="Nearest neighbor range of convolutional filter in initial network.")

parser.add_argument("-scr", "--Scratch", action="store_true", help="Whether to create new network and start from scratch")

parser.add_argument("-td", "--Tdata", metavar="T", type=int, help="Temperature to read data from")
parser.add_argument("-sep", "--Start_epoch", metavar="Ep", type=int, help="Starting epoch (for training, this network will be read in.)")
parser.add_argument("-eep", "--End_epoch", metavar="Ep", type=int, help="Ending epoch (for training, this will be the last epoch.)")

parser.add_argument("-sp", "--SpecTrain", metavar="s1s2s3", type=str, help="species to consider, order independent (Eg, 123 or 213 etc for species 1, 2 and 3")
parser.add_argument("-vSp", "--VacSpec", metavar="SpV", type=int, default=0, help="species index of vacancy, must match dataset, default 0")

parser.add_argument("-nt", "--N_train", type=int, default=10000, help="No. of training samples.")
parser.add_argument("-i", "--Interval", type=int, default=1, help="Epoch intervals in which to save or load networks.")
parser.add_argument("-lr", "--Learning_rate", type=float, default=0.001, help="Learning rate for Adam algorithm.")
parser.add_argument("-bs", "--Batch_size", type=int, default=128, help="size of a single batch of samples.")
parser.add_argument("-wm", "--Mean_wt", type=float, default=0.02, help="Initialization mean value of weights.")
parser.add_argument("-ws", "--Std_wt", type=float, default=0.2, help="Initialization standard dev of weights.")

parser.add_argument("-d", "--DumpArgs", action="store_true", help="Whether to dump arguments in a file")
parser.add_argument("-dpf", "--DumpFile", metavar="F", type=str, help="Name of file to dump arguments to (can be the jobID in a cluster for example).")


if __name__ == "__main__":
    # main(list(sys.argv))
    args_file = list(sys.argv)[1]

    count=1
    args = parser.parse_args()
    
    if args.DumpArgs:
        print("Dumping arguments to: {}".format(args.DumpFile))
        opts = vars(args)
        with open(RunPath + args.DumpFile, "w") as fl:
            for key, val in opts.items():
                fl.write("{}: {}\n".format(key, val))

    main(args)
