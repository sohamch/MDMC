import numpy as np
from numba import jit, jitclass
from numba import int64, float64

# Paste all the function definitions here as comments


MonteCarloSamplerSpec = [
    ("numInteractsSiteSpec", int64[:, :]),
    ("SiteSpecInterArray", int64[:, :, :]),
    ("numSitesInteracts", int64[:]),
    ("SupSitesInteracts", int64[:, :]),
    ("SpecOnInteractSites", int64[:, :]),
    ("Interaction2En", float64[:]),
    ("numVecsInteracts", int64[:]),
    ("VecsInteracts", float64[:]),
    ("jumpFinSites", int64[:]),
    ("jumpFinSpec", int64[:]),
    ("numJumpPointGroups", int64[:]),
    ("numTSInteractsInPtGroups", int64[:, :]),
    ("JumpInteracts", int64[:, :]),
    ("Jump2KRAEng", float64[:, :]),
    ("mobOcc", int64[:, :]),
    ("mobOccNew", int64[:, :]),
    ("vacSiteInd", int64),
    ("Nsites", int64),
    ("Nspecs", int64),
    ("OffSiteCount", int64[:])

]


@jitclass(MonteCarloSamplerSpec)
class MCSamplerClass(object):

    def __init__(self, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts,
                 VecsInteracts, numInteractsSiteSpec, SiteSpecInterArray, jumpFinSites, jumpFinSpec,
                 numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng, vacSiteInd, mobOcc):

        self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites, self.Interaction2En, self.numVecsInteracts, \
        self.VecsInteracts, self.numInteractsSiteSpec, self.SiteSpecInterArray, self.jumpFinSites, self.jumpFinSpec, \
        self.numJumpPointGroups, self.numTSInteractsInPtGroups, self.JumpInteracts, self.Jump2KRAEng, self.vacSiteInd = \
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts, \
            VecsInteracts, numInteractsSiteSpec, SiteSpecInterArray, jumpFinSites, jumpFinSpec, \
            numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng, vacSiteInd

        # check if proper sites and species data are entered
        self.Nsites, self.Nspecs = numInteractsSiteSpec.shape[0], numInteractsSiteSpec.shape[1]
        self.mobOcc = mobOcc
        self.OffSiteCount = np.zeros(len(numSitesInteracts), dtype=int)
        for interactIdx in range(len(numSitesInteracts)):
            numSites = numSitesInteracts[interactIdx]
            for intSiteind in range(numSites):
                interSite = SupSitesInteracts[interactIdx, intSiteind]
                interSpec = SpecOnInteractSites[interactIdx, intSiteind]
                if mobOcc[interSite] != interSpec:
                    self.OffSiteCount[interactIdx] += 1

        # Reformat the array so that the swaps are always between atoms of different species

    def makeMCsweep(self, NswapTrials, beta):
        """
        This is the function that will do the MC sweeps
        :param NswapTrials: the number of site swaps needed to be done in a single MC sweep
        :param beta : 1/(KB*T)
        update the mobile occupance array and the OffSiteCounts for the MC sweeps
        """

        mobOcc = self.mobOcc.copy()
        OffSiteCountOld = self.OffSiteCount.copy()
        OffSiteCountNew = self.OffSiteCount.copy()
        count = 0
        randarr = np.random.rand(NswapTrials)
        while count < NswapTrials:
            # first select two random sites to swap - for now, let's just select naively.
            siteA = np.random.randint(0, self.Nsites)
            siteB = np.random.randint(0, self.Nsites)

            # make sure we are swapping different atoms because otherwise we are in the same state
            if mobOcc[siteA] == mobOcc[siteB] or siteA == self.vacSiteInd or siteB == self.vacSiteInd:
                continue
            delE = 0.
            # Next, switch required sites off
            for interIdx in range(self.numInteractsSiteSpec[siteA, mobOcc[siteA]]):
                # check if an interaction is on
                interMainInd = self.SiteSpecInterArray[siteA, mobOcc[siteA], interIdx]
                offscount= OffSiteCountOld[interMainInd]
                if offscount == 0:
                    delE -= self.Interaction2En[interMainInd]
                OffSiteCountNew[self.SiteSpecInterArray[siteA, mobOcc[siteA], interIdx]] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, mobOcc[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteB, mobOcc[siteB], interIdx]
                offscount = OffSiteCountOld[interMainInd]
                if offscount == 0:
                    delE -= self.Interaction2En[interMainInd]
                OffSiteCountNew[self.SiteSpecInterArray[siteB, mobOcc[siteB], interIdx]] += 1

            # Next, switch required sites on
            for interIdx in range(self.numInteractsSiteSpec[siteA, mobOcc[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteA, mobOcc[siteB], interIdx]
                OffSiteCountNew[interMainInd] -= 1
                if OffSiteCountNew[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]

            for interIdx in range(self.numInteractsSiteSpec[siteB, mobOcc[siteA]]):
                interMainInd = self.SiteSpecInterArray[siteB, mobOcc[siteA], interIdx]
                OffSiteCountNew[interMainInd] -= 1
                if OffSiteCountNew[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]

            # do the selection test
            if np.exp(-beta * delE) > randarr[count]:
                # update the off site counts
                # swap the sites to get to the next state
                temp = mobOcc[siteA]
                mobOcc[siteA] = mobOcc[siteB]
                mobOcc[siteB] = temp
                OffSiteCountOld = OffSiteCountNew.copy()
            else:
                # revert back the off site counts, because the state has not changed
                OffSiteCountNew = OffSiteCountOld.copy()
            count += 1

            # this is for unit testing where only one MC step is tested - will be removed in JIT version
        return mobOcc, OffSiteCountNew