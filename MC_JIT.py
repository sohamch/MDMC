import numpy as np
from numba.experimental import jitclass
from numba import int64, float64

# Paste all the function definitions here as comments


MonteCarloSamplerSpec = [
    ("numInteractsSiteSpec", int64[:, :]),
    ("SiteSpecInterArray", int64[:, :, :]),
    ("numSitesInteracts", int64[:]),
    ("numSitesTSInteracts", int64[:]),
    ("SupSitesInteracts", int64[:, :]),
    ("TSInteractSites", int64[:, :]),
    ("TSInteractSpecs", int64[:, :]),
    ("SpecOnInteractSites", int64[:, :]),
    ("Interaction2En", float64[:]),
    ("numVecsInteracts", int64[:]),
    ("VecsInteracts", float64[:, :, :]),
    ("jumpFinSites", int64[:]),
    ("jumpFinSpec", int64[:]),
    ("numJumpPointGroups", int64[:]),
    ("numTSInteractsInPtGroups", int64[:, :]),
    ("JumpInteracts", int64[:, :, :]),
    ("Jump2KRAEng", float64[:, :, :]),
    ("mobOcc", int64[:]),
    ("vacSiteInd", int64),
    ("Nsites", int64),
    ("Nspecs", int64),
    ("OffSiteCount", int64[:]),
    ("FinSiteFinSpecJumpInd", int64[:, :]),
    ("VecGroupInteracts", int64[:, :])

]


@jitclass(MonteCarloSamplerSpec)
class MCSamplerClass(object):

    def __init__(self, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts,
                 VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray,
                 numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
                 FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng,
                 vacSiteInd, mobOcc, OffSiteCount):

        self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites, self.Interaction2En, self.numVecsInteracts, \
        self.VecsInteracts, self.VecGroupInteracts, self.numInteractsSiteSpec, self.SiteSpecInterArray, self.vacSiteInd = \
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts, \
            VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray, vacSiteInd

        self.numSitesTSInteracts, self.TSInteractSites, self.TSInteractSpecs = \
            numSitesTSInteracts, TSInteractSites, TSInteractSpecs

        self.jumpFinSites, self.jumpFinSpec, self.FinSiteFinSpecJumpInd, self.numJumpPointGroups, self.numTSInteractsInPtGroups, \
        self.JumpInteracts, self.Jump2KRAEng = \
            jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, \
            JumpInteracts, Jump2KRAEng

        # check if proper sites and species data are entered
        self.Nsites, self.Nspecs = numInteractsSiteSpec.shape[0], numInteractsSiteSpec.shape[1]
        self.mobOcc = mobOcc
        self.OffSiteCount = OffSiteCount.copy()
        for interactIdx in range(numSitesInteracts.shape[0]):
            numSites = numSitesInteracts[interactIdx]
            for intSiteind in range(numSites):
                interSite = SupSitesInteracts[interactIdx, intSiteind]
                interSpec = SpecOnInteractSites[interactIdx, intSiteind]
                if mobOcc[interSite] != interSpec:
                    self.OffSiteCount[interactIdx] += 1

        # Reformat the array so that the swaps are always between atoms of different species

    def makeMCsweep(self, mobOcc, OffSiteCount, TransOffSiteCount,
                    SwapTrials, beta, randarr, Nswaptrials):

        # TODO : Need to implement biased sampling methods to select sites from TSinteractions with more prob.
        for swapcount in range(Nswaptrials):
            # first select two random sites to swap - for now, let's just select naively.
            siteA = SwapTrials[swapcount, 0]
            siteB = SwapTrials[swapcount, 1]

            specA = mobOcc[siteA]
            specB = mobOcc[siteB]

            delE = 0.
            # Next, switch required sites off
            for interIdx in range(self.numInteractsSiteSpec[siteA, specA]):
                # check if an interaction is on
                interMainInd = self.SiteSpecInterArray[siteA, specA, interIdx]
                # offscount = OffSiteCount[interMainInd]
                if OffSiteCount[interMainInd] == 0:
                    delE -= self.Interaction2En[interMainInd]
                OffSiteCount[interMainInd] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, specB]):
                interMainInd = self.SiteSpecInterArray[siteB, specB, interIdx]
                # offscount = OffSiteCount[interMainInd]
                if OffSiteCount[interMainInd] == 0:
                    delE -= self.Interaction2En[interMainInd]
                OffSiteCount[interMainInd] += 1

            # Next, switch required sites on
            for interIdx in range(self.numInteractsSiteSpec[siteA, specB]):
                interMainInd = self.SiteSpecInterArray[siteA, specB, interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]

            for interIdx in range(self.numInteractsSiteSpec[siteB, specA]):
                interMainInd = self.SiteSpecInterArray[siteB, specA, interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]
                    
            # do the selection test
            if np.exp(-beta*delE) > randarr[swapcount]:
                # swap the sites to get to the next state
                mobOcc[siteA] = specB
                mobOcc[siteB] = specA
                # OffSiteCount is already updated to that of the new state.

            else:
                # revert back the off site counts, because the state has not changed
                for interIdx in range(self.numInteractsSiteSpec[siteA, specA]):
                    # interMainInd = self.SiteSpecInterArray[siteA, specA, interIdx]
                    OffSiteCount[self.SiteSpecInterArray[siteA, specA, interIdx]] -= 1

                for interIdx in range(self.numInteractsSiteSpec[siteB, specB]):
                    # interMainInd = self.SiteSpecInterArray[siteB, specB, interIdx]
                    OffSiteCount[self.SiteSpecInterArray[siteB, specB, interIdx]] -= 1

                for interIdx in range(self.numInteractsSiteSpec[siteA, specB]):
                    # interMainInd = self.SiteSpecInterArray[siteA, specB, interIdx]
                    OffSiteCount[self.SiteSpecInterArray[siteA, specB, interIdx]] += 1

                for interIdx in range(self.numInteractsSiteSpec[siteB, specA]):
                    # interMainInd = self.SiteSpecInterArray[siteB, specA, interIdx]
                    OffSiteCount[self.SiteSpecInterArray[siteB, specA, interIdx]] += 1

        # make the offsite for the transition states
        for TsInteractIdx in range(len(self.TSInteractSites)):
            TransOffSiteCount[TsInteractIdx] = 0
            for Siteind in range(self.numSitesTSInteracts[TsInteractIdx]):
                if mobOcc[self.TSInteractSites[TsInteractIdx, Siteind]] != self.TSInteractSpecs[TsInteractIdx, Siteind]:
                    TransOffSiteCount[TsInteractIdx] += 1

    def Expand(self, state, ijList, dxList, OffSiteCount, TSOffSiteCount, lenVecClus, beta):

        del_lamb_mat = np.zeros((lenVecClus, lenVecClus, ijList.shape[0]))
        delxDotdelLamb = np.zeros((lenVecClus, ijList.shape[0]))

        ratelist = np.zeros(ijList.shape[0])

        siteA, specA = self.vacSiteInd, self.Nspecs - 1

        Wbar = np.zeros((lenVecClus, lenVecClus))
        Bbar = np.zeros(lenVecClus)
        # go through all the transitions
        for jumpInd in range(ijList.shape[0]):
            del_lamb = np.zeros((lenVecClus, 3))

            # Get the transition index
            siteB, specB = ijList[jumpInd], state[ijList[jumpInd]]
            transInd = self.FinSiteFinSpecJumpInd[siteB, specB]

            # First, work on getting the KRA energy for the jump
            delEKRA = 0.0
            # We need to go through every point group for this jump
            for tsPtGpInd in range(self.numJumpPointGroups[transInd]):
                for interactInd in range(self.numTSInteractsInPtGroups[transInd, tsPtGpInd]):
                    # Check if this interaction is on
                    interactMainInd = self.JumpInteracts[transInd, tsPtGpInd, interactInd]
                    if TSOffSiteCount[interactMainInd] == 0:
                        delEKRA += self.Jump2KRAEng[transInd, tsPtGpInd, interactInd]

            # next, calculate the energy change due to site swapping

            delE = 0.0
            # Switch required sites off
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteA]]):
                # check if an interaction is on
                interMainInd = self.SiteSpecInterArray[siteA, state[siteA], interIdx]
                if OffSiteCount[interMainInd] == 0:
                    delE -= self.Interaction2En[interMainInd]
                    # take away the vectors for this interaction
                    for i in range(self.numVecsInteracts[interMainInd]):
                        del_lamb[self.VecGroupInteracts[interMainInd, i]] -= self.VecsInteracts[interMainInd, i, :]
                OffSiteCount[interMainInd] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteB, state[siteB], interIdx]
                if OffSiteCount[interMainInd] == 0:
                    delE -= self.Interaction2En[interMainInd]
                    for i in range(self.numVecsInteracts[interMainInd]):
                        del_lamb[self.VecGroupInteracts[interMainInd, i]] -= self.VecsInteracts[interMainInd, i, :]
                # OffSiteCount[interMainInd] += 1

            # Next, switch required sites on
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteA, state[siteB], interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]
                    # add the vectors for this interaction
                    for i in range(self.numVecsInteracts[interMainInd]):
                        del_lamb[self.VecGroupInteracts[interMainInd, i]] += self.VecsInteracts[interMainInd, i, :]

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
                interMainInd = self.SiteSpecInterArray[siteB, state[siteA], interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]
                    # add the vectors for this interaction
                    for i in range(self.numVecsInteracts[interMainInd]):
                        del_lamb[self.VecGroupInteracts[interMainInd, i]] += self.VecsInteracts[interMainInd, i, :]

            # Energy change computed, now expand
            ratelist[jumpInd] = np.exp(-(0.5 * delE + delEKRA) * beta)
            del_lamb_mat[:, :, jumpInd] = np.dot(del_lamb, del_lamb.T)

            # delxDotdelLamb[:, jumpInd] = np.tensordot(del_lamb, dxList[jumpInd], axes=(1, 0))
            # let's do the tensordot by hand (work on finding numba support for this)
            for i in range(lenVecClus):
                # replace innder loop with outer product
                delxDotdelLamb[i, jumpInd] = np.dot(del_lamb[i, :], dxList[jumpInd, :])

            # Next, restore OffSiteCounts to original values for next jump, as well as
            # for use in the next MC sweep.
            # During switch-off operations, offsite counts were increased by one.
            # So decrease them back by one
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteA]]):
                OffSiteCount[self.SiteSpecInterArray[siteA, state[siteA], interIdx]] -= 1

            # During switch-on operations, offsite counts were decreased by one.
            # So increase them back by one
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
                OffSiteCount[self.SiteSpecInterArray[siteA, state[siteB], interIdx]] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
                OffSiteCount[self.SiteSpecInterArray[siteB, state[siteA], interIdx]] += 1

        # Wbar = np.tensordot(ratelist, del_lamb_mat, axes=(0, 2))
        for i in range(lenVecClus):
            for j in range(lenVecClus):
                Wbar[i, j] += np.dot(del_lamb_mat[i, j, :], ratelist)

        # Bbar = np.tensordot(ratelist, delxDotdelLamb, axes=(0, 1))
        for i in range(lenVecClus):
            Bbar[i] = np.dot(ratelist, delxDotdelLamb[i, :])

        return Wbar, Bbar