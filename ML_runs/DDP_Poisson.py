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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import h5py
import pickle
from tqdm import tqdm
from SymmLayers import GCNet
from GCNetRun import Load_crysDats
import copy

class GCNet_NgbAvg(GCNet):
    def forward(self, InstateNgbs):
        pass

# Function to set up parallel process groups
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Function to load the data from the h5py files
def Load_Data(DataPath, f1, f2):
    with h5py.File(DataPath + f1, "r") as fl:
        try:
            perm_1 = fl["Permutation"]
            print("found permuation for step 1")
        except:        
            perm_1 = np.arange(len(fl["InitStates"]))

        state1List_1 = np.array(fl["InitStates"])[perm_1]
        state2List_1 = np.array(fl["FinStates"])[perm_1]
        dispList_1 = np.array(fl["SpecDisps"])[perm_1]
        rateList_1 = np.array(fl["rates"])[perm_1]
        AllJumpRates_1 = np.array(fl["AllJumpRates"])[perm_1]
    
    with h5py.File(DataPath + f2, "r") as fl:
        try:
            perm_2 = fl["Permutation"]
            print("found permuation for step 2")
        except:        
            perm_2 = np.arange(len(fl["InitStates"]))

        state1List_2 = np.array(fl["InitStates"])[perm_2]
        AllJumpRates_2 = np.array(fl["AllJumpRates"])[perm_2]
    
    assert np.array_equa(state2List_1, state1List_2)
    assert np.array_equal(perm_1, perm_2)

    #state1List, state2List, allRates_st1, allRates_st2, dispList, escRateList = Load_Data(DataPath, f1, f2)
    return state1Listi_1, state2List_1, AllJumpRates_1, AllJumpRates_2, dispList_1, rateList_1

# The data partitioning function - extract necessary portion based on rank
def makeData(rank, world_size, state1List, state2List, allRates_st1, allRates_st2,
        JumpNewSites, NNsiteList, dispList, dxJumps, a0, escRateList, vacSpec, specsToTrain):
    
    chunkSize = state1List.shape[0] // world_size
    startIndex = rank * chunkSize
    endIndex = min((rank + 1) * chunkSize, state1List.shape[0])
    if rank == world_size - 1 and endIndex < state1List.shape[0]:
        print("Dropping last {} samples".format(state1List.shape[0] - endIndex))

    NStateSamples = endIndex - startIndex
    N_ngb = dxJumps.shape[0]
    dispJumps = pt.tensor(dxJumps * a0).double()
    
    jumpSites = NNsiteList[1:, 0]

    print("Process : {0}, splitting samples from {1} to {2}".format(rank, startSample, endSample))

    specs = np.unique(state1List[0])
    NSpec = specs.shape[0] - 1
    Nsites = state1List.shape[1]
    
    sp_ch = {}
    for sp in specs:
        if sp == VacSpec:
            continue
        
        if sp < VacSpec:
            sp_ch[sp] = sp
        else:
            sp_ch[sp] = sp-1

    # Tensors to make:
    # state1NgbTens, state2NgbTens, avDispSpecTrain_st1, avDispSpecTrain_st2, rateProbTens_st1, rateProbTens_st2, escRateTens, dispTens

    state1NgbTens = pt.zeros(NStateSamples, N_ngb, NSpec, Nsites)
    state2NgbTens = pt.zeros(NStateSamples, N_ngb, NSpec, Nsites)
    avDispSpecTrain_st1 = pt.zeros(NStateSamples, dxJumps.shape[1]).double()
    avDispSpecTrain_st2 = pt.zeros(NStateSamples, dxJumps.shape[1]).double()
    rateProbTens_st1 = pt.zeros(NStateSamples, N_ngb).double()
    rateProbTens_st2 = pt.zeros(NStateSamples, N_ngb).double()

    escRateTens = pt.tensor(escRateList)
    dispTens = pt.zeros(NStateSamples, dxJumps.shape[1]).double()

    # Now let's construct the tensors
    for sampInd in tqdm(range(startIndex, endIndex), position=0, leave=True):
        state1 = state1List[sampInd]
        state2 = state2List[sampInd]

        rateSum_st1 = np.sum(allRates_st1[sampInd])
        rateSum_st2 = np.sum(allRates_st2[sampInd])
        
        jProbs_st1 = allRates_st1[sampInd]/rateSum_st1
        jProbs_st2 = allRates_st2[sampInd]/rateSum_st2
        
        rateProbTens_st1[sampInd, :] = pt.tensor(jProbs_st1)
        rateProbTens_st2[sampInd, :] = pt.tensor(jProbs_st2)
        
        dispTens[sampInd] = pt.tensor(sum(dispList[sampInd, spec, :] for spec in specsToTrain)).double()

        for jmp in range(N_ngb):
            state1Jmp = state1[JumpNewSites[jmp]]
            state2Jmp = state2[JumpNewSites[jmp]]

            if specsToTrain == [VacSpec]:
                avDispSpecTrain_st1[sampInd] += jProbs_st1[jmp] * dispJumps[jmp]
                avDispSpecTrain_st2[sampInd] += jProbs_st2[jmp] * dispJumps[jmp]

            else:
                if state1[jumpSites[jmp]] in specsToTrain:
                    avDispSpecTrain_st1[sampInd] -= jProbs_st1[jmp] * dispJumps[jmp]
                if state2[jumpSites[jmp]] in specsToTrain:
                    avDispSpecTrain_st2[sampInd] -= jProbs_st2[jmp] * dispJumps[jmp]

            # Now we construct the neighbor tensors
            for site in range(1, Nsites):
                sp1_site_jmpState = state1Jmp[site]
                sp2_site_jmpState = state2Jmp[site]
                state1NgbTens[sampInd, jmp, sp_ch[sp1_site_jmpState], site] = 1.0
                state2NgbTens[sampInd, jmp, sp_ch[sp2_site_jmpState], site] = 1.0

    return state1NgbTens, state2NgbTens, avDispSpecTrain, avDispSpecTrain_st2, rateProbTens_st1, rateProbTens_st2, escRateTens, dispTens, sp_ch

# Compute the multipliers in a flattened manner without a neighbor axis
def FlatMults(state1NgbTens, state2NgbTens, rateProbTens_st1, rateProbTens_st2, specsToTrain, sp_ch, dim):
    N_samps = state1NgbTens.shape[0]
    N_ngb = state1NgbTens.shape[1]
    NspecCh = state1NgbTens.shape[2]
    Nsites = state1NgbTens.shape[3]

    # The goal is to have the site y vectors output as (N_batch * N_ngb, 3, Nsites)
    # Then multiply with the OnSite multipliers and sum along sites -> (N_batch * N_ngb, 3)
    # Then multiply with the jump probabilities -> (N_batch * N_ngb, 3)
    # Then change the view to (N_batch, N_ngb, 3)
    # Then sum axis 1 -> (N_batch, 3)

    OnSites_st_1_ngbs = pt.zeros(N_samps*N_ngb, 3, Nsites)
    OnSites_st_2_ngbs = pt.zeros(N_samps*N_ngb, 3, Nsites)

    rateMult_st1 = pt.zeros(N_samps*N_ngb, 1)
    rateMult_st2 = pt.zeros(N_samps, 1)

    # Now we fill them up
    
    for samp in range(Nsamps):
        for ngb in range(N_ngb):
            rateMult_st1[samp*N_ngb + ngb, 0] = rateProbTens_st1[samp, ngb]
            rateMult_st2[samp*N_ngb + ngb, 0] = rateProbTens_st2[samp, ngb]
            
            for d in range(dim):
                OnSites_st_1_ngbs[samp*N_ngb + ngb, d, :] = sum([state1NgbTens[samp, ngb, sp_ch[sp], :] for sp in specsToTrain])
                OnSites_st_2_ngbs[samp*N_ngb + ngb, d, :] = sum([state2NgbTens[samp, ngb, sp_ch[sp], :] for sp in specsToTrain])

    return OnSites_st1_ngbs, OnSites_st2_ngbs, rateMult_st1, rateMult_st2



# The training function
def train(rank, world_size, state1List, state2List, allRates_st1, allRates_st2,
        JumpNewSites, NNsiteList, dispList, dxJumps, a0, escRateList,
        vacSpec, SpecsToTrain, N_train, batch_size, start_ep, end_ep, interval, gNet,
        net_dir):
    
    # Convert to necessary tensors with  portions of the data extracted based on rank
    state1NgbTens, state2NgbTens, avDispSpecTrain_st1, avDispSpecTrain_st2, rateProbTens_st1, rateProbTens_st2, escRateTens, dispTens , sp_ch =\
            makeData(rank, world_size, state1List[:N_train], state2List[:N_train], allRates_st1[:N_train], allRates_st2[:N_train],
                    JumpNewSites, NNsiteList, dispList[:N_train], dxJumps, a0, escRateList[:N_train], vacSpec, SpecsToTrain)
        
    OnSites_st1_ngbs, OnSites_st2_ngbs, rateMult_st1, rateMult_st2 =\
            FlatMults(state1NgbTens, state2NgbTens, rateProbTens_st1, rateProbTens_st2, SpecsToTrain, sp_ch, dxJumps.shape[1])

    N_ngb = dxJumps.shape[0]
    Ndim = dxJumps.shape[1]
    # Now write the training loop
    opt = pt.optim.Adam(gNet.parameters(), lr=lRate, weight_decay=0.0005) 
    for epoch in tqdm(range(start_ep, end_ep + 1, batch_size), position=0, leave=True):

        if ep % interval == 0:
            if rank == 0:
                pt.save(gNet.module.state_dict(), RunPath + net_dir + "ep_{}.pt".format(epoch))

        dist.barrier() # Halt all processes under saving is complete

        for start_samp in range(0, N_train, batch_size):
            
            opt.zero_grad()

            end_samp = min(start_samp + batch_size, N_train)
            BS = end_samp - start_samp 
            # Flatten the samples
            state1Batch = state1NgbTens[start_samp : end_samp].view(BS*N_ngb, len(sp_ch), state1List.shape[1]).double().to(rank)
            state2Batch = state2NgbTens[start_samp : end_samp].view(BS*N_ngb, len(sp_ch), state1List.shape[1]).double().to(rank)
            
            On_st1Ngb_Batch = OnSites_st1_ngbs[start_samp * N_ngb : end_samp * N_ngb].double().to(rank) 
            On_st2Ngb_Batch = OnSites_st2_ngbs[start_samp * N_ngb : end_samp * N_ngb].double().to(rank) 
            
            rateMults_st1_Batch = ratemult_st1[start_samp * N_ngb : end_samp * N_ngb].double().to(rank) 
            rateMults_st2_Batch = ratemult_st2[start_samp * N_ngb : end_samp * N_ngb].double().to(rank)

            avDisp_st1_Batch = avDispSpecTrain_st1[start_samp : end_samp]
            avDisp_st2_Batch = avDispSpecTrain_st2[start_samp : end_samp]

            # Take in the constant parts
            dispBatch = dispTens[start_samp : end_samp] + avDisp_st2_Batch - avDisp_st1_Batch
            esc_ratesBatch = escRateTens[start_samp : end_samp]

            y1Batch = gNet(state1Batch)
            y2Batch = gNet(state2Batch)

            y1Batch = pt.sum(y1Batch * On_st1Ngb_Batch, dim = 2)
            y2Batch = pt.sum(y2Batch * On_st2Ngb_Batch, dim = 2)
            
            y1Batch = (y1Batch * rateMults_st1_Batch).view(BS, N_ngb, Ndim)
            y2Batch = (y2Batch * rateMults_st2_Batch).view(Bs, N_ngb, Ndim)

            y1Batch = pt.sum(y1Batch, dim=1)
            y2Batch = pt.sum(y2Batch, dim=1)
            
            dy = y2Batch - y1Batch

            diff = pt.sum(esc_ratesBatch * pt.norm(dispBatch + dy, dim=1)**2)/6.

            diff.backward()
            opt.step()


def Eval(rank, world_size, state1List, state2List, allRates_st1, allRates_st2,
        JumpNewSites, NNsiteList, dispList, dxJumps, a0, escRateList,
        vacSpec, SpecsToTrain, N_train, batch_size, start_ep, end_ep, interval, gNet,
        savePath):
    
    # Convert to necessary tensors with  portions of the data extracted based on rank
    state1NgbTens, state2NgbTens, avDispSpecTrain_st1, avDispSpecTrain_st2, rateProbTens_st1, rateProbTens_st2, escRateTens, dispTens , sp_ch =\
            makeData(rank, world_size, state1List, state2List, allRates_st1, allRates_st2,
                    JumpNewSites, NNsiteList, dispList, dxJumps, a0, escRateList, vacSpec, SpecsToTrain)
        
    OnSites_st1_ngbs, OnSites_st2_ngbs, rateMult_st1, rateMult_st2 =\
            FlatMults(state1NgbTens, state2NgbTens, rateProbTens_st1, rateProbTens_st2, SpecsToTrain, sp_ch, dxJumps.shape[1])

    N_ngb = dxJumps.shape[0]
    Ndim = dxJumps.shape[1]

    def calcDiff(startSample, endSample):
        with pt.no_grad:
            for epoch in tqdm(range(start_ep, end_ep + 1, interval), position=0, leave=True):

                if ep % interval == 0:
                    if rank == 0:
                        pt.save(gNet.module.state_dict(), savePath + "/ep_{}.pt".format(epoch))

                dist.barrier() # Halt all processes under saving is complete

                for start_samp in range(startSample, endSample, batch_size):
                    

                    end_samp = min(start_samp + batch_size, N_train)
                    BS = end_samp - start_samp 
                    # Flatten the samples
                    state1Batch = state1NgbTens[start_samp : end_samp].view(BS*N_ngb, len(sp_ch), state1List.shape[1]).double().to(rank)
                    state2Batch = state2NgbTens[start_samp : end_samp].view(BS*N_ngb, len(sp_ch), state1List.shape[1]).double().to(rank)
                    
                    On_st1Ngb_Batch = OnSites_st1_ngbs[start_samp * N_ngb : end_samp * N_ngb].double().to(rank) 
                    On_st2Ngb_Batch = OnSites_st2_ngbs[start_samp * N_ngb : end_samp * N_ngb].double().to(rank) 
                    
                    rateMults_st1_Batch = ratemult_st1[start_samp * N_ngb : end_samp * N_ngb].double().to(rank) 
                    rateMults_st2_Batch = ratemult_st2[start_samp * N_ngb : end_samp * N_ngb].double().to(rank)

                    avDisp_st1_Batch = avDispSpecTrain_st1[start_samp : end_samp]
                    avDisp_st2_Batch = avDispSpecTrain_st2[start_samp : end_samp]

                    # Take in the constant parts
                    dispBatch = dispTens[start_samp : end_samp] + avDisp_st2_Batch - avDisp_st1_Batch
                    esc_ratesBatch = escRateTens[start_samp : end_samp]

                    y1Batch = gNet(state1Batch)
                    y2Batch = gNet(state2Batch)

                    y1Batch = pt.sum(y1Batch * On_st1Ngb_Batch, dim = 2)
                    y2Batch = pt.sum(y2Batch * On_st2Ngb_Batch, dim = 2)
                    
                    y1Batch = (y1Batch * rateMults_st1_Batch).view(BS, N_ngb, Ndim)
                    y2Batch = (y2Batch * rateMults_st2_Batch).view(Bs, N_ngb, Ndim)

                    y1Batch = pt.sum(y1Batch, dim=1)
                    y2Batch = pt.sum(y2Batch, dim=1)
                    
                    dy = y2Batch - y1Batch

                    diff = pt.sum(esc_ratesBatch * pt.norm(dispBatch + dy, dim=1)**2)/6.

                    diffAll += diff.item()

    tr_loss = calcDiff(0, N_train)/(N_train)


# The function to get the Y vectors
def getY():
    pass

# Next, the main function - this main function is the one that will be
# run on parallel instances of the code
def main(rank, world_size, args):

    # Extract parsed arguments

    # Initiate process group
    setup(rank, world_size)
    
    # Load the crystal data
    GpermNNIdx, NNsiteList, siteShellIndices, GIndtoGDict, JumpNewSites, dxJumps = Load_crysDats(filter_nn, CrysDatPath) 
    
    # Load the KMC Trajectory data - we'll need rates from both state 1 and state 2
    state1List, state2List, allRates_st1, allRates_st2, dispList, escRateList = Load_Data(DataPath, f1, f2)

   
    net_dir = "epochs_T_{0}_n{1}c{2}_NgbAvg/".format(T_net, nch, nl)
    # if from scratch, create new network
    gNet = GCNet_NgbAvg().double().to(rank) # pass in arguments to make the GCNet
    
    # Wrap with DDP
    gNet = DDP(gNet, device_ids=[rank], output_device=rank) #, find_unused_parameters=True)
    
    if not from_scratch and mode=="train":
        # load unwrapped state dict
        print("loading net from epoch : {}".format(sep))
        state_dict = torch.load(RunPath + net_dir + "ep_{}.pt".format(sep))
        gNet.load_state_dict(state_dict)


    # Pass the partitioned data to the training function
    if mode == "train":
        # Call training function
    elif mode == "eval":
        # Call evaluation function
    elif mode == "getY":
        # Call getY function


    # Lastly, clean things up by destroying the process group
    dist.destroy_process_group()


# Add argument parser
    # arguments needed:
    #   DataPath, f1, f2 - directory of data files, file for step 1 data, file for step 2 data
    #   a0 (float), from_scratch_bool, T_data, T_net, filter_nn
    #   CrysDatPath, sep (Start epoch - int), nch (int), nl(int)
parser = argparse.ArgumentParser(description="Input parameters for using GCnets", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-DP", "--DataPath", metavar="/path/to/data", type=str, help="Path to Data files.")
parser.add_argument("-f1", "--FileStep1", metavar="data_stp1.h5", type=str, help="Data file for step 1.")
parser.add_argument("-f2", "--FileStep2", metavar="data_stp2.h5", type=str, help="Data file for step 2.")
parser.add_argument("-cr", "--CrysDatPath", metavar="/path/to/crys/dat", type=str, help="Path to crystal Data.")

parser.add_argument("-m", "--Mode", metavar="M", type=str, help="Running mode (one of train, eval, getY, getRep). If getRep, then layer must specified with -RepLayer.")

parser.add_argument("-nl", "--Nlayers",  metavar="L", type=int, help="No. of layers of the neural network.")
parser.add_argument("-nch", "--Nchannels", metavar="Ch", type=int, help="No. of representation channels in non-input layers.")
parser.add_argument("-cngb", "--ConvNgbRange", type=int, default=1, metavar="NN", help="Nearest neighbor range of convolutional filters.")


parser.add_argument("-scr", "--Scratch", action="store_true", help="Whether to create new network and start from scratch")

parser.add_argument("-td", "--Tdata", metavar="T", type=int, help="Temperature to read data from")
parser.add_argument("-tn", "--TNet", metavar="T", type=int, help="Temperature to use networks from\n For example one can evaluate a network trained on 1073 K data, on the 1173 K data, to see what it does.")
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

# Then, we need to spawm multiple processes to run the main function

if __name__ == "__main__":
    
    args = parser.parse_args()    
    if args.DumpArgs:
        print("Dumping arguments to: {}".format(args.DumpFile))
        opts = vars(args)
        with open(RunPath + args.DumpFile, "w") as fl:
            for key, val in opts.items():
                fl.write("{}: {}\n".format(key, val))

    if pt.cuda.is_available():
        DeviceIDList = list(range(pt.cuda.device_count()))
    if len(DeviceIDList == 0):
        raise ValueError("No Gpu found for distributed training.")
        device = pt.device("cpu")

    # Then spawn processes - we'll do one GPU per process
    world_size = len(DeviceIDList)
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
