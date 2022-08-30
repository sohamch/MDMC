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
import matplotlib.pyplot as plt
from SymmLayers import GCNet
from GCNetRun import Load_crysDats
import copy

# Function to set up parallel process groups
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# The data partitioning function
def splitData():
    pass

# The training function
def train():
    pass

# The evaluation function
def Eval():
    pass

# The function to get the Y vectors
def getY():
    pass

# Next, the main function - this main function is the one that will be
# run on parallel instances of the code
def main(rank, world_size, args):
    # Initiate process group
    setup(rank, world_size)
    
    # Get the arguments from argparse - things like batch size, interval to load etc
    parser = argparse.ArgumentParser()


    # Load the crystal data

    # Load the KMC Trajectory data - we'll need rates from both state 1 and state 2
    state1List, state2List, allRates_state1, allRates_state2, dispList, escRateList = Load_Data()

    # Convert to necessary tensors
    state

    gNet = GCNet() # pass in arguments to make the GCNet
    # Wrap the model with DDP
    gNet = DDP(gNet, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # Training and evaluation will involve splitting the data based on rank,
    # Partition the necessary data based on rank by calling SplitData

    # Pass the partitioned data to the training function
    if mode == "train":
        # Call training function
    elif mode == "eval":
        # Call evaluation function
    elif mode == "getY":
        # Call getY function


    # Lastly, clean things up by destroying the process group
    dist.destroy_process_group()

# Then, we need to spawm multiple processes to run the main function

if __name__ == "__main__":
    
    if pt.cuda.is_available():
        DeviceIDList = list(range(pt.cuda.device_count()))
    if len(DeviceIDList == 0):
        raise ValueError("No Gpu found for distributed training.")
        device = pt.device("cpu")

    # Then spawn processes - we'll do one GPU per process
    world_size = len(DeviceIDList)
    mp.spawn(main, args=(world_size), nprocs=world_size)
