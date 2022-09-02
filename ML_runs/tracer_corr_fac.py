#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[2]:


from onsager import crystal, cluster, supercell
from SymmLayers import GCNet


# Read crystal data
crysType = "BCC"
CrysDatPath = "../../../CrysDat_"+crysType+"_16Sup/"

# Load symmetry operations with which they were constructed
with open(CrysDatPath + "GroupOpsIndices.pkl", "rb") as fl:
    GIndtoGDict = pickle.load(fl)

GpermNNIdx = np.load(CrysDatPath + "GroupNNpermutations.npy")

NNsiteList = np.load(CrysDatPath + "NNsites_sitewise.npy")

with open(CrysDatPath + "jnet"+crysType+".pkl", "rb") as fl:
    jnet = pickle.load(fl)

dxList = np.array([dx for (i,j), dx in jnet[0]])
norm = np.linalg.norm(dxList[0])
dxUnitVecs = dxList / norm

N_ngb = NNsiteList.shape[0]
z = N_ngb - 1
assert z == dxList.shape[0]
Nsites = NNsiteList.shape[1]

with open(CrysDatPath + "supercell"+crysType+".pkl", "rb") as fl:
    superCell = pickle.load(fl)

# Convert to tensors
GnnPerms = pt.tensor(GpermNNIdx).long()
NNsites = pt.tensor(NNsiteList).long()
JumpUnitVecs = pt.tensor(dxUnitVecs.T)
Ng = GnnPerms.shape[0]
Ndim = dxList.shape[1]


# In[4]:


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


# In[5]:


# Now convert the jump indexing into a form suitable for gathering vectors
jumpGather = np.zeros((z, Ndim, Nsites), dtype=int)
for jumpInd in range(z):
    for site in range(Nsites):
        for dim in range(Ndim):
            jumpGather[jumpInd, dim, site] = jumpNewSites[jumpInd, site]

GatherTensor = pt.tensor(jumpGather, dtype=pt.long)


# In[6]:


# Let's do a test for gatherting y vectors after vacancy jumps
# Let's make a pseudo-batch of y vectors
y0 = pt.rand(3, Nsites)
y = y0.repeat((z,1)).view(z, Ndim, Nsites) # copy the tensor 12 times

# check that repeating is correct
for jumpInd in range(z):
    assert pt.allclose(y0, y[jumpInd])
    
# Now gather the vectors according to gather tensor
# NOTE - this is the operation we'll use in the training code
y_jump = pt.gather(y, 2, GatherTensor)

for jumpInd in range(z):
    Rjump = dxLatVec[jumpInd]
    RjumpNeg = -dxLatVec[jumpInd]
    siteExchange, _ = superCell.index(Rjump, (0,0))
    siteExchangeNew, _ = superCell.index(RjumpNeg, (0,0))
    
    assert pt.equal(y_jump[jumpInd, :, siteExchangeNew], y0[:, siteExchange])
    assert pt.equal(y_jump[jumpInd, :, 0], y0[:, 0])
    
    for siteInd in range(1, Nsites):
        if siteInd == siteExchange: # exchange site has already been moved so skip that
            continue
        _, Rsite = superCell.ciR(siteInd)
        RsiteNew = Rsite - dxLatVec[jumpInd]
        siteIndNew, _ = superCell.index(RsiteNew, (0,0))
        assert pt.equal(y_jump[jumpInd, :, siteIndNew], y0[:, siteInd])

print("y vector gathering test passed")

# Next, we'll record the displacement of the atoms at different sites after each jump
# Only the neighboring sites of the vacancy move - the rest don't
dispSites = np.zeros((z, Ndim, Nsites))
for jumpInd in range(z):
    dispSites[jumpInd, :, NNsiteList[1+jumpInd, 0]] = -dxList[jumpInd]

dispTensor = pt.tensor(dispSites, dtype=pt.double)

# Build the network and the host state with a single vacancy
# we'll make "z" copies of the host for convenience
hostState = pt.ones(batch_size, 1, Nsites, dtype=pt.double).to(device)

# set the vacancy site to unoccupied
hostState[:, :, 0] = 0

# Now write the training loop
#epoch_last = 0
#N_epoch = 300

gNet = GCNet(GnnPerms.long(), NNsites, JumpUnitVecs, dim=3, N_ngb=N_ngb,
             NSpec=1, mean=0.0, std=0.2, nl=6, nch=8, nchLast=1).double()

if epoch_last > 0:
    gNet.load_state_dict(pt.load("epochs_tracer_{0}_16_Sup/ep_{1}.pt".format(crysType, epoch_last)))

try:
    l_epoch = np.load("L_epoch.npy")
    l_epoch = list(l_epoch)

except:
    l_epoch = []

if epoch_last == 0:
    l_epoch = []

opt = pt.optim.Adam(gNet.parameters(), lr=0.001)
for epoch in tqdm(range(N_epoch + 1), ncols=65, position=0, leave=True):
    pt.save(gNet.state_dict(), "epochs_tracer_BCC_16_Sup/ep_{}.pt".format(epoch_last + epoch))
    opt.zero_grad()    
    for batch_pass in range(Updates_per_epoch):
                
        y_jump_init = gNet(hostState)[:, 0, :, :]
        y_jump_fin = pt.gather(y_jump_init, 2, GatherTensor)
        
        dy = y_jump_fin[:, :, 1:] - y_jump_init[:, :, 1:]
        
        dispMod = dispTensor[:, :, 1:] + dy
        
        norm_sq_Sites = pt.norm(dispMod, dim=1)**2/6.
        norm_sq_SiteAv = pt.sum(norm_sq_Sites, dim=1) / (1.0 * Nsites)
        norm_sq_batchAv = pt.sum(norm_sq_SiteAv) / (1.0 * z)
        
        norm_sq_batchAv.backward()
    
    opt.step()
    l_epoch.append(norm_sq_batchAv.item())

np.save("l_epoch.npy", np.array(l_epoch))


# In[ ]:




