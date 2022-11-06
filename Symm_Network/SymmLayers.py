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


class R3ConvSites(nn.Module):
    def __init__(self, GnnPerms, gdiags, NNsites,
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
        
    def RotateParams(self, GnnPerms):
        # First, we repeat the weights
        Ng = GnnPerms.shape[0]
        wtVC_repeat = self.wtVC.repeat(Ng, 1)
        
        # The we repeat group permutation indices
        GnnPerm_repeat = self.GnnPerms.repeat_interleave(self.dim, dim=0)            
        self.wtVC_repeat_transf = pt.matmul(self.gdiags, pt.gather(wtVC_repeat, 1, GnnPerm_repeat))
        
    
    def RearrangeInput(self, In, NNsites):
        N_ngb = NNsites.shape[0]        
        Nch = In.shape[1]
        In = In.repeat_interleave(N_ngb, dim=1)
        NNRepeat = NNsites.unsqueeze(0).repeat(In.shape[0], Nch, 1)
        return pt.gather(In, 2, NNRepeat)
    
    
    def forward(self, In):
        NSites, GnnPerms, NNsites = self.NSites, self.GnnPerms, self.NNsites

        Nbatch = In.shape[0]
        Ng = GnnPerms.shape[0]

        self.RotateParams(GnnPerms)
        out = self.RearrangeInput(In, NNsites)

        # Finally, do the R3 convolution
        out = pt.matmul(self.wtVC_repeat_transf, out).view(Nbatch, Ng, self.dim, NSites)

        # Then group average
        return pt.sum(out, dim=1) / Ng


class GAvg(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, In):
        Ng = In.shape[2]        
        # sum out the group channels
        return pt.sum(In, dim=2)/Ng



class GCNet(nn.Module):
    def __init__(self, GnnPerms, NNsites, JumpVecs, N_ngb,
            NSpec, mean=1.0, std=0.1, nl=3, nch=8, nchLast=1, relu=False):
        
        super().__init__()
        modules = []
        
        if relu:
            nonLin = nn.ReLU
        else:
            nonLin = nn.Softplus
        
        if nl == -1:
            modules = [
                GConv(NSpec, nchLast, GnnPerms, NNsites, N_ngb, mean=mean, std=std),
                nonLin(),
                GAvg()
            ]

        else:
            modules += [
                GConv(NSpec, nch, GnnPerms, NNsites, N_ngb, mean=mean, std=std),
                nonLin(),
                GAvg()
            ]

            for l in range(nl):
                modules += [
                    GConv(nch, nch, GnnPerms, NNsites, N_ngb, mean=mean, std=std),
                    nonLin(),
                    GAvg()
                ]
            modules += [
                GConv(nch, nchLast, GnnPerms, NNsites, N_ngb, mean=mean, std=std),
                nonLin(),
                GAvg()
            ]
        
        self.net = nn.Sequential(*modules)
        # Store NNsites without the self terms for vector prediction
        # Store them in a buffer so they are saved in the state_dict by pytorch
        self.register_buffer("NNsites", NNsites[1:, :])
        self.register_buffer("JumpVecs", JumpVecs)
    
    def RearrangeInput(self, In):
        N_ngb = self.NNsites.shape[0]
        Nch = In.shape[1]
        Nsites = In.shape[2]
        In = In.repeat_interleave(N_ngb, dim=1)
        NNRepeat = self.NNsites.unsqueeze(0).repeat(In.shape[0], Nch, 1)
        return pt.gather(In, 2, NNRepeat).view(In.shape[0], Nch, N_ngb, Nsites)

    def NgbSum(self, In):
        Nbatch = In.shape[0]
        Nchannels = In.shape[1]
        Nsites = In.shape[2]
        
        # The input has shape (N_batch, Nch, Nsites)
        out = self.RearrangeInput(In)

        out = pt.matmul(self.JumpVecs, out)

        return out

    def forward(self, InState):
        y = self.net(InState)
        return self.NgbSum(y)
    
    def getRep(self, InState, LayerInd):
        # LayerInd is counted starting from zero
        if LayerInd == len(self.net):
            # Just do a forward pass
            return self.forward(InState)

        y = self.net[0](InState)
        for L in range(1, LayerInd + 1):
            y = self.net[L](y)
        return y


# This network was used in previous runs. R3ConvSites is a redundant operation
# for cubic systems, and not appropriate for non-cubic systems for relaxation vectors.
# It is still kept here because it is more general
class GCNet_R3ConvSites(nn.Module):
    def __init__(self, GnnPerms, gdiags, NNsites, dim, N_ngb,
            NSpec, mean=1.0, std=0.1, b=1.0, nl=3, nch=8):
        
        super().__init__()
        modules = []
        modules += [
            GConv(NSpec, nch, GnnPerms, NNsites, N_ngb, mean=mean, std=std),
            nn.Softplus(beta=b),
            GAvg()
        ]
        
        for l in range(nl):
            modules += [
                GConv(nch, nch, GnnPerms, NNsites, N_ngb, mean=mean, std=std),
                nn.Softplus(beta=b),
                GAvg()
            ]
        modules += [
            GConv(nch, 1, GnnPerms, NNsites, N_ngb, mean=mean, std=std),
            nn.Softplus(beta=b),
            GAvg()
        ]
        modules.append(R3ConvSites(GnnPerms, gdiags, NNsites, N_ngb,
                        dim, mean=mean, std=std))
        
        self.net = nn.Sequential(*modules)
    
    def forward(self, InState):
        y = self.net(InState)
        return y
    
    def getRep(self, InState, LayerInd):
        # get the last single channel representation of the state
        # LayerInd is counted starting from zero
        y = self.net[0](InState)
        for L in range(1, LayerInd + 1):
            y = self.net[L](y)
        return y
