#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch as pt
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


class GConv(nn.Module):
    def __init__(self, InChannels, OutChannels, N_ngb, mean=1.0, std=0.1):
        super().__init__()
        self.register_buffer("N_ngb", pt.tensor(N_ngb))
        self.register_buffer("NchIn", pt.tensor(InChannels))
        self.register_buffer("NchOut", pt.tensor(OutChannels))
        
        Layerweights = nn.Parameter(pt.normal(mean, std, size=(OutChannels, InChannels, N_ngb),
                                     requires_grad=True))
        
        LayerBias = nn.Parameter(pt.normal(mean, std, size=(OutChannels, 1)))
            
        self.register_parameter("Psi", Layerweights)
        self.register_parameter("bias", LayerBias)
    
    def RotateParams(self, GnnPerms):
        Ng = GnnPerms.shape[0]
        # First, get the input and output channels
        NchIn = self.NchIn
        NchOut = self.NchOut
        N_ngb = self.N_ngb

        weight = getattr(self, "Psi")
        bias = getattr(self, "bias")
        # First repeat the weight
        weightRepeat = weight.repeat_interleave(Ng, dim=0)

        # Then repeat the permutation indices
        GnnPerm_repeat = GnnPerms.repeat(NchOut, NchIn).view(-1, NchIn, N_ngb)

        # Then gather according to the indices
        self.GWeights = pt.gather(weightRepeat, 2, GnnPerm_repeat).view(-1, NchIn*N_ngb)
        
        # store the repeated biases
        self.Gbias = bias.repeat_interleave(Ng, dim=0)
    
    def RearrangeInput(self, In, NNsites, Ng):
        N_ngb = NNsites.shape[0]
        NNtoRepeat = NNsites.unsqueeze(0)
        
        Nch = In.shape[1]
        
        In = In.repeat_interleave(N_ngb, dim=1)
        NNRepeat = NNToRepeat.repeat(In.shape[0], Nch, 1)
        return pt.gather(In, 2, NNRepeat)
    
    def forward(self, In, NSites, GnnPerms, NNsites act="softplus"):
        
        Nbatch = In.shape[0]
        NchOut = self.NchOut
        Ng = GnnPerms.shape[0]
        
        self.RotateParams(GnnPerms)
        self.RearrangeInput(In, NNsites, Ng)
        Psi = self.GWeights
        bias = self.Gbias

        # do the convolution + group averaging
        out = pt.matmul(Psi, In) + bias
        if act == "softplus":
            out = F.softplus(out)
        else:
            out = F.leaky_relu(out)

        return out.view(Nbatch, NchOut, Ng, NSites)


# In[ ]:


class R3Conv(nn.Module):
    def __init__(self, N_ngb, dim, gdiags, SitesToShells, mean=1.0, std=0.1):
        super().__init__()
        self.register_buffer("N_ngb", pt.tensor(N_ngb))      
        wtVC = nn.Parameter(pt.normal(mean, std, size=(dim, N_ngb), requires_grad=True))
        self.register_parameter("wtVC", wtVC)
        
        # Make the shell parameters
        Nshells = pt.max(SitesToShells)+1
        
        ShellWeights = nn.Parameter(pt.normal(mean, std, size=(Nshells,)
                                                   ,requires_grad = True))
        
        self.register_buffer("SitesToShells", SitesToShells)
        self.register_parameter("ShellWeights", ShellWeights)
        
    
    def RotateParams(self, GnnPerms):
        # First, we repeat the weights
        wtVC_repeat = self.wtVC.repeat(self.Ng, 1)
        
        # The we repeat group permutation indices
        GnnPerm_repeat = self.GnnPerms.repeat_interleave(self.dim, dim=0)            
        self.wtVC_repeat_transf = pt.matmul(self.gdiags, pt.gather(wtVC_repeat, 1, GnnPerm_repeat))
        
        # Repeat the shell indices
        self.SiteShellWeights = self.ShellWeights[self.SitesToShells]
    
    def RearrangeInput(self, In, NNsites, Ng):
        N_ngb = NNsites.shape[0]
        NNtoRepeat = NNsites.unsqueeze(0)
        
        Nch = In.shape[1]
        
        In = In.repeat_interleave(N_ngb, dim=1)
        NNRepeat = NNToRepeat.repeat(In.shape[0], Nch, 1)
        return pt.gather(In, 2, NNRepeat)
    
    def forward(self, In, NSites, GnnPerms, NNsites act="softplus"):
        
        Nbatch = In.shape[0]
        NchOut = self.NchOut
        Ng = GnnPerms.shape[0]
        
        self.RotateParams(GnnPerms)
        self.RearrangeInput(In, NNsites, Ng)
        Psi = self.GWeights
        bias = self.Gbias

        # do the convolution + group averaging
        out = pt.matmul(Psi, In) + bias
        if act == "softplus":
            out = F.softplus(out)
        else:
            out = F.leaky_relu(out)

        return out.view(Nbatch, NchOut, Ng, NSites)


# In[3]:


class GAvg(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(In, NNSites):
        N_ngb = NNsites.shape[0]
        NNtoRepeat = NNsites.unsqueeze(0)
        
        Ng = In.shape[2]
        Nch = In.shape[1]
        
        # sum out the group channels
        In = pt.sum(In, dim==2)/Ng        
        
        return In


# In[ ]:




