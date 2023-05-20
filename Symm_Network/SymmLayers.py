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
        """
        Implements a group-equivariant convolutional layer. Permutations of convolutional filters under
        space group operations are used to build symmetry-equivariant outputs that rotate/transform automatically
        if the input is rotated.
        :param: InChannel - no. of input channels.
        :param: OutChannel - no. of output channels.
        :param: GnnPerms - Permutation of nearest neighbors under inverse group operations - shape (Ng, coordination number + 1)
        :param: NNsites - nearest neighbors of each site - shape(coordination number + 1, Nsites).
        Note - the 0th row of NNsites is just [0, 1, 2...], i.e, the sites themselves are their own 0th neighbors
        :param: N_ngb (coordination number + 1)
        """
        super().__init__()
        Nsites = NNsites.shape[1]
        self.register_buffer("NSites", pt.tensor(Nsites))
        self.register_buffer("GnnPerms", GnnPerms)
        self.register_buffer("NNsites", NNsites)
        
        self.register_buffer("N_ngb", pt.tensor(N_ngb))
        self.register_buffer("NchIn", pt.tensor(InChannels))
        self.register_buffer("NchOut", pt.tensor(OutChannels))
        
        Layerweights = nn.Parameter(pt.normal(mean, std, size=(OutChannels, InChannels, N_ngb)))
        
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


# Let's build a message passing / crystal graph network here
class msgPassLayer(nn.Module):
    """
    Gathers information at each site with a message passing method from nearest neighbors.
    GConv layers need space group information, but the message passing layer does not.
    Message passing is more seamlessly applicable to multi-site lattices.
    """
    def __init__(self, NChannels, NSpec, NNsites, output=False, mean=1.0, std=0.1):
        """
        :param NChannels: No. of mesaage gathering linear transformations in each layer.
        :param NSpec: No. of atomic species (excluding vacancy)
        :param NNsites: nearest neighbors of each site - shape(coordination number + 1, Nsites).
        :param mean: mean to initialize the weight arrays from a normal distribution.
        :param std: standard deviation to initialize the weight arrays from a normal distribution.
        """
        super().__init__()
        Nsites = NNsites.shape[1]
        self.register_buffer("NSites", pt.tensor(Nsites))
        self.register_buffer("NNsites", NNsites)
        self.register_buffer("NSpec", pt.tensor(NSpec))
        self.register_buffer("Z", pt.tensor(NNsites.shape[0] - 1))
        self.register_buffer("NchIn", pt.tensor(NChannels))

        if output:
            Layerweights = nn.Parameter(pt.normal(mean, std, size=(NChannels, 1, 2 * NSpec),
                                                  requires_grad=True))
            LayerBias = nn.Parameter(pt.normal(mean, std, size=(NChannels, NSpec)))
        else:
            Layerweights = nn.Parameter(pt.normal(mean, std, size=(NChannels, NSpec, 2 * NSpec),
                                                  requires_grad=True))
            LayerBias = nn.Parameter(pt.normal(mean, std, size=(NChannels, 1)))

        self.register_parameter("Weights", Layerweights)
        self.register_parameter("bias", LayerBias)

    def forward(self, In):
        """
        :param In: input tensor with shape (N_batch, NSpec, Nsites)
        :return: out: output tensor of shape (N_batch, NSpec, Nsites)
        """

        dt = In.dtype
        dev = In.device
        total = pt.zeros(In.shape[0], 2*In.shape[1], In.shape[2], dtype=dt).to(dev)

        total[:, :In.shape[1], :] = In[:, :, :]

        out = pt.zeros(In.shape[0], In.shape[1], In.shape[2], dtype=dt).to(dev)

        for z in range(self.Z):
            # reindex the site according to the z^th nearest neighbor and append
            total[:, In.shape[1]:2*In.shape[1], :] = In[:, :, self.NNsites[1 + z, :]]

            # Apply mesagge passing
            o = pt.tensordot(total, self.Weights, dims=([1], [2])) + self.bias.view(1, 1, self.NChannels, self.NSpec)
            o = pt.sum(o.transpose(1, -1), dim=2)

            out += F.softplus(o)

        return out

class msgPassNet(nn.Module):
    """
    Constructs a sequence of message passing layers, and return relaxation vector as a linear combination of
    nearest neighbor vectors, with the coefficients of the combination being the output of the last layer.
    """
    def __int__(self, NLayers, NChannels, NSpec, NNsites, JumpVecs, mean=1.0, std=0.1):
        """
        :param NChannels: No. of mesaage gathering linear transformations in each layer.
        :param NSpec: No. of atomic species (excluding vacancy)
        :param NNsites: nearest neighbors of each site - shape(coordination number + 1, Nsites).
        :param JumpVecs: nearest neighbor Jump vectors - shape(coordination number, 3).
        :param mean: mean to initialize the weight arrays from a normal distribution.
        :param std: standard deviation to initialize the weight arrays from a normal distribution.
        """
        super().__init__()
        Nsites = NNsites.shape[1]
        self.register_buffer("JumpVecs", JumpVecs)

        seq = []
        for l in range(NLayers):
            seq += [
                msgPassLayer(NChannels, NSpec, NNsites, mean=mean, std=std)
            ]

        # Apply output layer

        self.net = nn.Sequential(*seq)




