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
import h5py
from tqdm import tqdm
from onsager import crystal, supercell
from SymmLayers import GCNet


device=None
if pt.cuda.is_available():
    print(pt.cuda.get_device_name())
    device = pt.device("cuda:0")
    DeviceIDList = list(range(pt.cuda.device_count()))
else:
    device = pt.device("cpu")
print(device, " - ", pt.cuda.get_device_name())

def Load_crysDats(CrysDatPath):
    print("Loading Crystal data at {}".format(CrysDatPath))

    with h5py.File(CrysDatPath, "r") as fl:
        lattice = np.array(fl["Lattice_basis_vectors"])
        superlatt = np.array(fl["SuperLatt"])
        dxList = np.array(fl["dxList_1nn"])
        GpermNNIdx = np.array(fl["GroupNNPermutation"])
        NNsiteList = np.array(fl["NNsiteList_sitewise"])

    crys = crystal.Crystal(lattice=lattice, basis=[[np.array([0., 0., 0.])]], chemistry=["A"])
    superCell = supercell.ClusterSupercell(crys, superlatt)

    return GpermNNIdx, NNsiteList, dxList, superCell

def make_SiteJumpDests(dxList, superCell, NNsiteList):
    # Build the indexing of the sites after the jumps
    z = dxList.shape[0]
    Nsites = NNsiteList.shape[1]
    dxLatVec = np.zeros((z, 3), dtype=int)
    for jInd in range(dxList.shape[0]):
        dx = dxList[jInd]
        dxR, _ = superCell.crys.cart2pos(dx)
        dxLatVec[jInd, :] = dxR[:]

    assert np.allclose(np.dot(superCell.crys.lattice, dxLatVec.T).T, dxList)

    SitesJumpDests = np.zeros((z, Nsites), dtype=int)
    for jumpInd in range(z):
        Rjump = dxLatVec[jumpInd]
        RjumpNeg = -dxLatVec[jumpInd]
        siteExchange, _ = superCell.index(Rjump, (0, 0))
        assert siteExchange == NNsiteList[1 + jumpInd, 0]
        siteExchangeNew, _ = superCell.index(RjumpNeg, (0, 0))
        SitesJumpDests[jumpInd, siteExchange] = siteExchangeNew
        for siteInd in range(1, Nsites):  # vacancy site is always origin
            if siteInd == siteExchange:  # exchange site has already been moved so skip that
                continue
            _, Rsite = superCell.ciR(siteInd)
            RsiteNew = Rsite - dxLatVec[jumpInd]
            siteIndNew, _ = superCell.index(RsiteNew, (0, 0))
            SitesJumpDests[jumpInd, siteInd] = siteIndNew

    return SitesJumpDests

def make_JumpGatherTensor(SitesJumpDests, BatchSize, z, Nsites, Ndim):
    jumpGather = np.zeros((args.BatchSize * z, Ndim, Nsites), dtype=int)
    for batch in range(BatchSize):
        for jumpInd in range(z):
            for site in range(Nsites):
                for dim in range(Ndim):
                    jumpGather[batch * z + jumpInd, dim, site] = SitesJumpDests[jumpInd, site]

    return jumpGather

def get_InputTensors(BatchSize, z, Ndim, Nsites, NNsiteList, dxList):
    # Record the displacement of the atoms at different sites after each jump
    # Only the neighboring sites of the vacancy move - the rest don't
    dispSites = np.zeros((BatchSize * z, Ndim, Nsites))
    for batch in range(args.BatchSize):
        for jumpInd in range(z):
            dispSites[batch * z + jumpInd, :, NNsiteList[1 + jumpInd, 0]] = -dxList[jumpInd]

    dispTensor = pt.tensor(dispSites, dtype=pt.double).to(device)

    # Build the host state with a single vacancy
    # we'll make "z" copies of the host for convenience in batch processing
    hostState = pt.ones(args.BatchSize * z, 1, Nsites, dtype=pt.double)
    # set the vacancy site to unoccupied
    hostState[:, :, 0] = 0.
    hostState = hostState.to(device)

    return dispTensor, hostState

def Train(hostState, dispTensor, GatherTensor, dxList, gNet, Nsites, StartEpoch, EndEpoch, PassesPerEpoch):
    if StartEpoch == 0:
        tcf_epoch = []
    else:
        try:
            tcf_epoch = list(np.load(RunPath + "tcf_epochs_{}_to_{}.npy".format(StartEpoch, EndEpoch)))    
        except:
            tcf_epoch = []

    l0 = np.linalg.norm(dxList[0]) ** 2 / (Nsites)

    opt = pt.optim.Adam(gNet.parameters(), lr=0.001)
    for epoch in tqdm(range(EndEpoch + 1), ncols=65, position=0, leave=True):
        pt.save(gNet.state_dict(),
                RunPath + "epochs_tracer_16_Sup/ep_{0}.pt".format(StartEpoch + epoch))
        for batch_pass in range(PassesPerEpoch):
            opt.zero_grad()
            y_jump_init = gNet(hostState)[:, 0, :, :]
            y_jump_fin = pt.gather(y_jump_init, 2, GatherTensor)

            dy = y_jump_fin[:, :, 1:] - y_jump_init[:, :, 1:]

            dispMod = dispTensor[:, :, 1:] + dy

            norm_sq_Sites = pt.norm(dispMod, dim=1) ** 2
            norm_sq_SiteAv = pt.sum(norm_sq_Sites, dim=1) / (1.0 * Nsites)
            norm_sq_batchAv = pt.sum(norm_sq_SiteAv) / (1.0 * dxList.shape[0] * args.BatchSize)

            norm_sq_batchAv.backward()

            opt.step()

        tcf_epoch.append(norm_sq_batchAv.item() / l0)

    return tcf_epoch

def main(args):
    # Read crystal data
    GpermNNIdx, NNsiteList, dxList, superCell = Load_crysDats(args.CrysDatPath)

    N_ngb = NNsiteList.shape[0]
    z = N_ngb - 1
    assert z == dxList.shape[0]
    Nsites = NNsiteList.shape[1]

    # Convert to tensors
    GnnPerms = pt.tensor(GpermNNIdx).long()
    NNsites = pt.tensor(NNsiteList).long()
    JumpVecs = pt.tensor(dxList.T).to(device)
    Ng = GnnPerms.shape[0]
    Ndim = dxList.shape[1]

    # Get the destination of each site after the jumps
    SitesJumpDests = make_SiteJumpDests(dxList, superCell, NNsiteList)
    # Now convert the jump indexing into a form suitable for gathering batches of vectors
    jumpGather = make_JumpGatherTensor(SitesJumpDests, args.BatchSize, z, Nsites, Ndim)
    GatherTensor = pt.tensor(jumpGather, dtype=pt.long).to(device)

    dispTensor, hostState = get_InputTensors(args.BatchSize, z, Ndim, Nsites, NNsiteList, dxList)

    gNet = GCNet(GnnPerms.long(), NNsites, JumpVecs, N_ngb=N_ngb, NSpec=1,
                 mean=args.Mean_wt, std=args.Std_wt, nl=args.Nlayers, nch=args.Nchannels, nchLast=args.NchLast).double()

    # Load saved networks if starting from checkpoint
    if args.StartEpoch > 0:
        gNet.load_state_dict(pt.load(RunPath + "epochs_tracer_16_Sup/ep_{0}.pt".format(args.StartEpoch),map_location="cpu"))
    
    else:
        if not os.path.isdir(RunPath + "epochs_tracer_16_Sup"):
            os.mkdir(RunPath + "epochs_tracer_16_Sup")
    
    gNet.to(device)
    
    tcf_epoch = Train(hostState, dispTensor, GatherTensor, dxList, gNet, Nsites, args.StartEpoch, args.EndEpoch, args.PassesPerEpoch)
    np.save(RunPath + "tcf_epochs_{}_to_{}.npy".format(args.StartEpoch, args.EndEpoch), np.array(tcf_epoch))

# Now set up the arg parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input parameters for tracer training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-CP", "--CrysDatPath", metavar="/path/to/crys/dat", type=str, help="Path to crystal Data.")
    parser.add_argument("-sep", "--StartEpoch", metavar="eg: 0", default=0, type=int, help="starting epoch")
    parser.add_argument("-eep", "--EndEpoch", metavar="eg: 0", default=250, type=int, help="Ending epoch")
    parser.add_argument("-nl", "--NLayers", metavar="eg: 6", default=6, type=int, help="number of intermediate layers")
    parser.add_argument("-nch", "--NChannels", metavar="eg: 8", default=8, type=int, help="number of channels in intermediate layers.")
    parser.add_argument("-bs", "--BatchSize", metavar="eg: 10", default=2, type=int, help="No. of times each of the z jumps are considered in a batch.")
    parser.add_argument("-pep", "--PassesPerEpoch", metavar="eg: 500", default=500, type=int,
                        help="No. of times to do batch Gradient Descent in each epoch. In each pass, each jump is considered \"BatchSize\" number of times.")

    parser.add_argument("-wm", "--Mean_wt", metavar="eg: 0.02", default=0.02, type=float, help="Mean to initialize weights")
    parser.add_argument("-ws", "--Std_wt", metavar="eg: 0.02", default=0.02, type=float, help="Standard deviation to initialize weights")

    parser.add_argument("-d", "--DumpArgs", action="store_true", help="Whether to dump arguments in a file")
    parser.add_argument("-dpf", "--DumpFile", metavar="F", type=str, help="Name of file to dump arguments to (can be the jobID in a cluster for example).")

    args = parser.parse_args()
    
    if args.DumpArgs:
        opts = vars(args)
        with open(RunPath + args.DumpFile, "w") as fl:
            for key, val in opts.items():
                fl.write("{}: {}\n".format(key, val))

    main(args)
