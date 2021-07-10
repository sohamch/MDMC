#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch as pt
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


class SymNetDP(nn.Module):
    def __init__(self, NchOutLayers, N_batch, GnnPerms, GDict, gdiags, NNSites,
                 SitesToShells, dim, mean, std, act="softplus"):
        """               
        :param NchOutLayers : no. of output channels of each layer
        
        :param GnnPerms : the permutation of nearest neighbor vectors due to group operations.
                This will be used to rotate the filters.
        
        :param Gdict : contains the cartesian rotation matrix of each group operation, with
                Indices assigned to each group op. Format: {Index : g}
                Note : every "i^th" row of GnnPerms must correspond to the Group operation 
                with index "i" in GDict.
        
        :param gdiags : block diagonal matrix, containing the cartesian rotation matrices
                        of group operations in the diagonal block from the top left as given
                        by the index of the group operation in GDict
        
        :param NNSites : contains nearest neighbor indices of each site. Sites go along the 
                         columns of the first row, and nn sites of each site appear in the 
                         rows below it in the same column.
        
        :param SitesToShells : Nsite-length torch array that gives which neighbor shell a site
                               belongs to
        
        :param dim : dimensionality of cartesian vectors
        
        Note : the last layer is actually the second last layer, because
        before R3 conv, we'll add another layer that outputs a single channel.
        """
        super().__init__()
        
        self.Nlayers = len(NchOutLayers)
        self.GnnPerms = GnnPerms
        self.gdiags = gdiags
        self.Ng = GnnPerms.shape[0]
        self.N_ngb = GnnPerms.shape[1]
        self.NNSites = NNSites
        self.Nsites = NNSites.shape[1]
        
        if SitesToShells.shape[0] != self.Nsites:
            raise ValueError("Site to shell indexing is not correct")
        
        self.SitesToShells = SitesToShells
        
        self.NchOutLayers = NchOutLayers
        self.GDict = GDict
        self.dim = dim        
        
        if self.NchOutLayers[-1] != 1:
            raise ValueError("The last conv layer must output a single channel")
        
        for layer in range(self.Nlayers):
            if layer == 0:
                # Layer 0 processes the input state
                NchIn = 1
            else:
                NchIn = NchOutLayers[layer-1]
                
            Layerweights = nn.Parameter(pt.normal(mean, std, size=(NchOutLayers[layer], NchIn, self.N_ngb),
                                     requires_grad=True))
            LayerBias = nn.Parameter(pt.normal(mean, std, size=(NchOutLayers[layer], 1)))
            
            self.register_parameter("Psi_{}".format(layer), Layerweights)
            self.register_parameter("bias_{}".format(layer), LayerBias)
            
            
        
#         weightList = []
#         biasList = []
#         # define weights first
#         for layer in range(self.Nlayers):
#             if layer == 0:
#                 # Layer 0 processes the input state
#                 NchIn = 1
#             else:
#                 NchIn = NchOutLayers[layer-1]
            
#             Layerweights = pt.normal(mean, std, size=(NchOutLayers[layer], NchIn, self.N_ngb),
#                                      requires_grad=True).double()
#             LayerBias = pt.normal(mean, std, size=(NchOutLayers[layer], 1)).double()
            
            
#             weightList.append(nn.Parameter(Layerweights))
#             biasList.append(nn.Parameter(LayerBias))
        
#         self.weightList = nn.ParameterList(weightList)
#         self.biasList = nn.ParameterList(biasList)
        
        # Now make the last vector conv layer
        wtVC = nn.Parameter(pt.normal(mean, std, size=(self.dim, self.N_ngb),
                                           requires_grad=True).double())
        
        self.register_parameter("wtVC", wtVC)
        
        # Make the shell parameters
        Nshells = pt.max(SitesToShells)+1
        
        ShellWeights = nn.Parameter(pt.normal(mean, std, size=(Nshells,)
                                                   ,requires_grad = True).double())
        
        self.register_parameter("ShellWeights", ShellWeights)
        
        self.activation = F.relu if act=="relu" else F.softplus
        
        NNToRepeat = self.NNSites.unsqueeze(0)
        
        self.register_buffer("NNToRepeat", NNToRepeat)
        
    def RotateParams(self):
        """
        Constructs symmetrically rotated weight matrices
        """
        print(len(self.weightList))
        self.GWeights = []
        self.Gbias = []
        
        for Idx, weight in enumerate(self.weightList):
            
            # First, get the input and output channels
            NchIn = 1 if Idx == 0 else self.NchOutLayers[Idx-1]
            NchOut = self.NchOutLayers[Idx]
            
            # First repeat the weight
            weightRepeat = weight.repeat_interleave(self.Ng, dim=0)
            
            # Then repeat the permutation indices
            GnnPerm_repeat = self.GnnPerms.repeat(NchOut, NchIn).view(-1, 
                                                                      NchIn,
                                                                      self.N_ngb)
            
            # Then gather according to the indices
            weight_repeat_perm = pt.gather(weightRepeat, 2, GnnPerm_repeat).view(-1, NchIn*self.N_ngb)
            
            # Store the repeated weights
            self.GWeights.append(weight_repeat_perm)
            
            # store the repeated biases
            bias = self.biasList[Idx]
            bias_repeat = bias.repeat_interleave(self.Ng, dim=0)
            
            self.Gbias.append(bias_repeat)
        
        # Now we need to process the vector convolution layer
        
        # First, we repeat the weights
        wtVC_repeat = self.wtVC.repeat(self.Ng, 1)
        
        # The we repeat group permutation indices
        GnnPerm_repeat = self.GnnPerms.repeat_interleave(self.dim, dim=0)            
        self.wtVC_repeat_transf = pt.matmul(self.gdiags, pt.gather(wtVC_repeat, 1, GnnPerm_repeat))
        
        # Repeat the shell indices
        self.SiteShellWeights = self.ShellWeights[self.SitesToShells]
        print("Param Rotations done")
    
    def RearrangeToInput(self, In, layer):
        """
        Takes the output of a layer and rearranges it into a suitable output
        for the next layer.
        
        :param In : the tensor to rearrange.
               "In" should have the shape (N_batch x Nch x Nsites), where
               N_batch is the no. of sample states in the batch
               Nch is the no. of channels in the output
               Nsites is the no. of sites in the supercell.
               
               An additional N_ngb subchannels are added to each channel.
               So, if the incoming image had four channels, the function will
               return an image with N_ngb*4 channels. Each subchannel of a channel
               contains nearest neighbor information of each site indexed in the
               same manner as NNSites.
        """
        Nch = In.shape[1]
        In = In.repeat_interleave(self.N_ngb, dim=1)
        NNRepeat = self.NNToRepeat.repeat(In.shape[0], Nch, 1)
        In = pt.gather(In, 2, NNRepeat)
        return In
    
    def G_conv(self, layer, In, Nbatch, NSites, InLayer, outlayers, Test=False):
        
        
        Psi = self.GWeights[layer]
        bias = self.Gbias[layer]

        NchOut = self.NchOutLayers[layer]

        if Test:
            InLayer.append(In.clone().detach().data)


        # do the convolution
        out = pt.sum(self.activation((pt.matmul(Psi, In) + bias)).view(Nbatch, NchOut, self.Ng, NSites), dim=2)/self.Ng

#         if Test:
#             outlayersG.append(out.clone().detach().data)

#         # do the group averaging
#         out = pt.sum(out, dim=2)/self.Ng

        if Test:
            outlayers.append(out.clone().detach().data)

        # Rearrange input for the next layer
#         out = self.RearrangeToInput(out)
        
        return self.RearrangeToInput(out, layer+1)
    
    def forward(self, InStates, Test=False):
        """
        :param InStates : input states with shape (N_batch, Nch, Nsites)
        """
        self.RotateParams()
        
        Nbatch = InStates.shape[0]
        NSites = InStates.shape[2]
        # Expand to include nearest neighbors
        out = self.RearrangeToInput(InStates, 0)
        
        outlayers = []
        
        InLayer = []
        
        # Now do the scalar kernel convolutions
        for layer in range(self.Nlayers):
            out = self.G_conv(layer, out, Nbatch, NSites, InLayer, outlayers, Test=Test)
        
        # Finally, do the R3 convolution
        # out should now have the shape (N_batch, N_ngb, Nsites)
        out = pt.matmul(self.wtVC_repeat_transf, out).view(Nbatch, self.Ng, self.dim, NSites)
        
        # Then group average
        out = pt.sum(out, dim=1)/self.Ng
        
        out = out*self.SiteShellWeights
        
        outVecSites = out.clone().data
        
        # Then site average
        out = pt.sum(out, dim=2)/NSites
        
        if Test:
            return InLayer, outlayers, outVecSites, out 
        
        return out


# In[ ]:




