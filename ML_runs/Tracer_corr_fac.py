import os
import sys
import argparse
RunPath = os.getcwd() + "/"
ModulePath = "/home/sohamc2/HEA_FCC/MDMC/Symm_Network/"

sys.path.append(ModulePath)
import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

from onsager import crystal, cluster, supercell
from SymmLayers import GCNet


device=None
if pt.cuda.is_available():
    print(pt.cuda.get_device_name())
    device = pt.device("cuda:0")
    DeviceIDList = list(range(pt.cuda.device_count()))
else:
    device = pt.device("cpu")
print(device, " - ", pt.cuda.get_device_name())

def main(args):
    # Read crystal data
    CrysDatPath = args.CrysDatPath

    # Load symmetry operations with which they were constructed
    with open(CrysDatPath + "GroupOpsIndices.pkl", "rb") as fl:
        GIndtoGDict = pickle.load(fl)

    GpermNNIdx = np.load(CrysDatPath + "GroupNNpermutations.npy")

    NNsiteList = np.load(CrysDatPath + "NNsites_sitewise.npy")

    with open(CrysDatPath + "jnet"+args.CrysType+".pkl", "rb") as fl:
        jnet = pickle.load(fl)

    dxList = np.array([dx for (i,j), dx in jnet[0]])
    norm = np.linalg.norm(dxList[0])
    dxUnitVecs = dxList / norm

    N_ngb = NNsiteList.shape[0]
    z = N_ngb - 1
    assert z == dxList.shape[0]
    Nsites = NNsiteList.shape[1]

    with open(CrysDatPath + "supercell"+args.CrysType+".pkl", "rb") as fl:
        superCell = pickle.load(fl)

    # Convert to tensors
    GnnPerms = pt.tensor(GpermNNIdx).long()
    NNsites = pt.tensor(NNsiteList).long()
    JumpUnitVecs = pt.tensor(dxUnitVecs.T).to(device)
    Ng = GnnPerms.shape[0]
    Ndim = dxList.shape[1]

    # Build the indexing of the sites after the jumps
    dxLatVec = np.dot(np.linalg.inv(superCell.crys.lattice), dxList.T).round(decimals=4).T.astype(int)
    assert np.allclose(np.dot(superCell.crys.lattice, dxLatVec.T).T, dxList)

    jumpNewSites = np.zeros((z, Nsites), dtype=int)

    for jumpInd in range(z):
        Rjump = dxLatVec[jumpInd]
        RjumpNeg = -dxLatVec[jumpInd]
        siteExchange, _ = superCell.index(Rjump, (0,0))
        siteExchangeNew, _ = superCell.index(RjumpNeg, (0,0))
        jumpNewSites[jumpInd, siteExchangeNew] = siteExchange
        for siteInd in range(1, Nsites): # vacancy site is always origin
            if siteInd == siteExchange: # exchange site has already been moved so skip that
                continue
            _, Rsite = superCell.ciR(siteInd)
            RsiteNew = Rsite - dxLatVec[jumpInd]
            siteIndNew, _ = superCell.index(RsiteNew, (0,0))
            jumpNewSites[jumpInd, siteIndNew] = siteInd

    # Now convert the jump indexing into a form suitable for gathering batches of vectors
    jumpGather = np.zeros((args.BatchSize * z, Ndim, Nsites), dtype=int)
    for batch in range(args.BatchSize):
        for jumpInd in range(z):
            for site in range(Nsites):
                for dim in range(Ndim):
                    jumpGather[batch * z + jumpInd, dim, site] = jumpNewSites[jumpInd, site]

    GatherTensor = pt.tensor(jumpGather, dtype=pt.long).to(device)

    # Next, we'll record the displacement of the atoms at different sites after each jump
    # Only the neighboring sites of the vacancy move - the rest don't
    dispSites = np.zeros((args.BatchSize * z, Ndim, Nsites))
    for batch in range(args.BatchSize):
        for jumpInd in range(z):
            dispSites[batch * z + jumpInd, :, NNsiteList[1+jumpInd, 0]] = -dxList[jumpInd]

    dispTensor = pt.tensor(dispSites, dtype=pt.double).to(device)

    # Build the network and the host state with a single vacancy
    # we'll make "z" copies of the host for convenience
    hostState = pt.ones(args.BatchSize*z, 1, Nsites, dtype=pt.double)

    # set the vacancy site to unoccupied
    hostState[:, :, 0] = 0

    hostState = hostState.to(device)

    gNet = GCNet(GnnPerms.long(), NNsites, JumpUnitVecs, dim=3, N_ngb=N_ngb,
                 NSpec=1, mean=0.0, std=0.2, nl=args.NLayers, nch=args.NChannels, nchLast=1).double()

    if args.StartEpoch > 0:
        gNet.load_state_dict(pt.load(RunPath + "epochs_tracer_{0}_16_Sup/ep_{1}.pt".format(args.CrysType, args.StartEpoch),map_location="cpu"))
    
    else:
        if not os.path.isdir(RunPath + "epochs_tracer_{0}_16_Sup".format(args.CrysType)):
            os.mkdir(RunPath + "epochs_tracer_{0}_16_Sup".format(args.CrysType))
    
    gNet.to(device)

    try:
        l_epoch = np.load(RunPath + "tcf_epoch.npy")
        l_epoch = list(l_epoch)

    except:
        l_epoch = []

    if args.StartEpoch == 0:
        l_epoch = []
    
    l0 = np.linalg.norm(dxList[0])**2 / (Nsites)

    opt = pt.optim.Adam(gNet.parameters(), lr=0.001)
    for epoch in tqdm(range(args.EndEpoch + 1), ncols=65, position=0, leave=True):
        pt.save(gNet.state_dict(), RunPath + "epochs_tracer_{0}_16_Sup/ep_{1}.pt".format(args.CrysType, args.StartEpoch + epoch))
        for batch_pass in range(args.BatchesPerEpoch): 
            opt.zero_grad()    
            y_jump_init = gNet(hostState)[:, 0, :, :]
            y_jump_fin = pt.gather(y_jump_init, 2, GatherTensor)
            
            dy = y_jump_fin[:, :, 1:] - y_jump_init[:, :, 1:]
            
            dispMod = dispTensor[:, :, 1:] + dy
            
            norm_sq_Sites = pt.norm(dispMod, dim=1)**2
            norm_sq_SiteAv = pt.sum(norm_sq_Sites, dim=1) / (1.0 * Nsites)
            norm_sq_batchAv = pt.sum(norm_sq_SiteAv) / (1.0 * z * args.BatchSize)
            
            norm_sq_batchAv.backward()
        
            opt.step()

        l_epoch.append(norm_sq_batchAv.item()/l0)

    np.save(RunPath + "tcf_epoch.npy", np.array(l_epoch))

# Now set up the arg parser
parser = argparse.ArgumentParser(description="Input parameters for tracer training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-CP", "--CrysDatPath", metavar="/path/to/crys/dat", type=str, help="Path to crystal Data.")
parser.add_argument("-ct", "--CrysType", metavar="FCC/BCC", type=str, help="Crystal type.")
parser.add_argument("-sep", "--StartEpoch", metavar="eg: 0", default=0, type=int, help="starting epoch")
parser.add_argument("-eep", "--EndEpoch", metavar="eg: 0", default=100, type=int, help="Ending epoch")
parser.add_argument("-nl", "--NLayers", metavar="eg: 6", default=6, type=int, help="number of intermediate layers")
parser.add_argument("-nch", "--NChannels", metavar="eg: 8", default=8, type=int, help="number of channels in intermediate layers.")
parser.add_argument("-bep", "--BatchesPerEpoch", metavar="eg: 500", default=500, type=int, help="No. of times to do batch GD in each epoch.")
parser.add_argument("-bs", "--BatchSize", metavar="eg: 10", default=2, type=int, help="No. of times each of the z jumps are considered in a batch.")

parser.add_argument("-d", "--DumpArgs", action="store_true", help="Whether to dump arguments in a file")
parser.add_argument("-dpf", "--DumpFile", metavar="F", type=str, help="Name of file to dump arguments to (can be the jobID in a cluster for example).")

if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.DumpArgs:
        opts = vars(args)
        with open(RunPath + args.DumpFile, "w") as fl:
            for key, val in opts.items():
                fl.write("{}: {}\n".format(key, val))

    main(args)
