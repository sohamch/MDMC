#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch as pt
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


class GConv(nn.Module):
    def __init__(self, InChannels, OutChannels, GnnPerms, NNsites, 
                 N_ngb, mean=1.0, std=0.1):
        
        super().__init__()
        Nsites = NNsites.shape[1]
        self.register_buffer("NSites", pt.tensor(Nsites))
        self.register_buffer("GnnPerms", GnnPerms)
        self.register_buffer("NNsites", NNsites)
        
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
    
    def RearrangeInput(self, In, NNsites):
        N_ngb = NNsites.shape[0]
        
        Nch = In.shape[1]
        
        In = In.repeat_interleave(N_ngb, dim=1)
        NNRepeat = NNsites.unsqueeze(0).repeat(In.shape[0], Nch, 1)
        
        return pt.gather(In, 2, NNRepeat)
    
    def forward(self, In):
        
        Nbatch = In.shape[0]
        NchOut = self.NchOut
        Ng = self.GnnPerms.shape[0]
        NSites = self.NSites
        
        self.RotateParams(self.GnnPerms)
        out = self.RearrangeInput(In, self.NNsites)
        Psi = self.GWeights
        bias = self.Gbias

        # do the convolution + group averaging
        out = pt.matmul(Psi, out) + bias
        
        return out.view(Nbatch, NchOut, Ng, NSites)


# In[ ]:


class R3Conv(nn.Module):
    def __init__(self, SitesToShells, GnnPerms, gdiags, NNsites,
                 N_ngb, dim, mean=1.0, std=0.1):
        super().__init__()
        Nsites = NNsites.shape[1]
        self.register_buffer("NSites", pt.tensor(Nsites))
        self.register_buffer("GnnPerms", GnnPerms)
        self.register_buffer("NNsites", NNsites)
        self.register_buffer("gdiags", gdiags)
        self.register_buffer("N_ngb", pt.tensor(N_ngb))
        self.register_buffer("dim", pt.tensor(dim))      
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
        Ng = GnnPerms.shape[0]
        wtVC_repeat = self.wtVC.repeat(Ng, 1)
        
        # The we repeat group permutation indices
        GnnPerm_repeat = self.GnnPerms.repeat_interleave(self.dim, dim=0)            
        self.wtVC_repeat_transf = pt.matmul(self.gdiags, pt.gather(wtVC_repeat, 1, GnnPerm_repeat))
        
        # Repeat the shell indices
        self.SiteShellWeights = self.ShellWeights[self.SitesToShells]
    
    def RearrangeInput(self, In, NNsites, Ng):
        N_ngb = NNsites.shape[0]        
        Nch = In.shape[1]
        In = In.repeat_interleave(N_ngb, dim=1)
        NNRepeat = NNsites.unsqueeze(0).repeat(In.shape[0], Nch, 1)
        return pt.gather(In, 2, NNRepeat)
    
    def forward(self, In, sites=False):
        
        NSites, GnnPerms, NNsites = self.NSites, self.GnnPerms, self.NNsites
        
        Nbatch = In.shape[0]
        Ng = GnnPerms.shape[0]
        
        self.RotateParams(GnnPerms)
        out = self.RearrangeInput(In, NNsites, Ng)
        
        # Finally, do the R3 convolution
        out = pt.matmul(self.wtVC_repeat_transf, out).view(Nbatch, Ng, self.dim, NSites)
        
        # Then group average
        out = pt.sum(out, dim=1)/Ng
        
        if sites:
            # return site wise results if requested
            return out
        
        else:
            # Site average with shell weights
            out = pt.sum(out*self.SiteShellWeights, dim=2)/NSites
            return out

class GAvg(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, In):
        Ng = In.shape[2]        
        # sum out the group channels
        return pt.sum(In, dim=2)/Ng



