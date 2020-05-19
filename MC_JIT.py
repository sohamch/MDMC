import numpy as np
from numba.experimental import jitclass
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
    ("VecsInteracts", float64[:, :, :]),
    ("jumpFinSites", int64[:]),
    ("jumpFinSpec", int64[:]),
    ("numJumpPointGroups", int64[:]),
    ("numTSInteractsInPtGroups", int64[:, :]),
    ("JumpInteracts", int64[:, :, :]),
    ("Jump2KRAEng", float64[:, :, :]),
    ("mobOcc", int64[:]),
    ("mobOccNew", int64[:]),
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
                 VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray, jumpFinSites, jumpFinSpec,
                 FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng, vacSiteInd,
                 mobOcc, OffSiteCount):

        self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites, self.Interaction2En, self.numVecsInteracts, \
        self.VecsInteracts, self.VecGroupInteracts, self.numInteractsSiteSpec, self.SiteSpecInterArray, self.jumpFinSites,\
        self.jumpFinSpec, self.FinSiteFinSpecJumpInd, self.numJumpPointGroups, self.numTSInteractsInPtGroups, self.JumpInteracts,\
        self.Jump2KRAEng, self.vacSiteInd = \
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts, \
            VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray, jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd,\
            numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng, vacSiteInd

        # check if proper sites and species data are entered
        self.Nsites, self.Nspecs = numInteractsSiteSpec.shape[0], numInteractsSiteSpec.shape[1]
        self.mobOcc = mobOcc
        self.OffSiteCount = OffSiteCount
        for interactIdx in range(numSitesInteracts.shape[0]):
            numSites = numSitesInteracts[interactIdx]
            for intSiteind in range(numSites):
                interSite = SupSitesInteracts[interactIdx, intSiteind]
                interSpec = SpecOnInteractSites[interactIdx, intSiteind]
                if mobOcc[interSite] != interSpec:
                    self.OffSiteCount[interactIdx] += 1

        # Reformat the array so that the swaps are always between atoms of different species

    def makeMCsweep(self, mobOcc, OffSiteCount, SwapTrials, beta, randarr, Nswaptrials):
        """
        This is the function that will do the MC sweeps
        :param NswapTrials: the number of site swaps needed to be done in a single MC sweep
        :param beta : 1/(KB*T)
        update the mobile occupance array and the OffSiteCounts for the MC sweeps
        """
        # TODO : Need to implement biased sampling methods to select sites from TSinteractions with more prob.
        OffSiteCountOld = OffSiteCount.copy()
        OffSiteCountNew = OffSiteCount.copy()
        count = 0
        for swapcount in range(Nswaptrials):
            # first select two random sites to swap - for now, let's just select naively.
            siteA = SwapTrials[swapcount, 0]
            siteB = SwapTrials[swapcount, 1]
            delE = 0.
            # Next, switch required sites off
            for interIdx in range(self.numInteractsSiteSpec[siteA, mobOcc[siteA]]):
                # check if an interaction is on
                interMainInd = self.SiteSpecInterArray[siteA, mobOcc[siteA], interIdx]
                offscount = OffSiteCountOld[interMainInd]
                if offscount == 0:
                    delE -= self.Interaction2En[interMainInd]
                OffSiteCountNew[interMainInd] += 1
                count += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, mobOcc[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteB, mobOcc[siteB], interIdx]
                offscount = OffSiteCountOld[interMainInd]
                if offscount == 0:
                    delE -= self.Interaction2En[interMainInd]
                OffSiteCountNew[interMainInd] += 1
                count += 1

            # Next, switch required sites on
            for interIdx in range(self.numInteractsSiteSpec[siteA, mobOcc[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteA, mobOcc[siteB], interIdx]
                OffSiteCountNew[interMainInd] -= 1
                if OffSiteCountNew[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]
                count += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, mobOcc[siteA]]):
                interMainInd = self.SiteSpecInterArray[siteB, mobOcc[siteA], interIdx]
                OffSiteCountNew[interMainInd] -= 1
                if OffSiteCountNew[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]
                count += 1

            # do the selection test
            if np.exp(-beta*delE) > randarr[swapcount]:
                # update the off site counts
                # swap the sites to get to the next state
                temp = mobOcc[siteA]
                mobOcc[siteA] = mobOcc[siteB]
                mobOcc[siteB] = temp
                OffSiteCountOld = OffSiteCountNew.copy()
            else:
                # revert back the off site counts, because the state has not changed
                OffSiteCountNew = OffSiteCountOld.copy()

            # this is for unit testing where only one MC step is tested - will be removed in JIT version
            # if test_single:
            #     return siteA, siteB, delE, mobOcc, randarr[0]
        # print("trials: {}".format(trialCount))
        # print("total iterations: {}".format(count))
        return mobOcc, OffSiteCountNew

    def Expand(self, state, ijList, dxList, OSCount, lenVecClus, beta):

        OffSiteCount = OSCount.copy()

        del_lamb_mat = np.zeros((lenVecClus, lenVecClus, ijList.shape[0]))
        delxDotdelLamb = np.zeros((lenVecClus, ijList.shape[0]))

        ratelist = np.zeros(ijList.shape[0])

        Wbar = np.zeros((lenVecClus, lenVecClus))
        Bbar = np.zeros(lenVecClus)

        siteA, specA = self.vacSiteInd, self.Nspecs - 1
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
                    if OffSiteCount[interactMainInd] == 0:
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
                OffSiteCount[interMainInd] += 1

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

        # Wbar = np.tensordot(ratelist, del_lamb_mat, axes=(0, 2))
        for i in range(lenVecClus):
            for j in range(lenVecClus):
                Wbar[i, j] += np.dot(del_lamb_mat[i, j, :], ratelist)

        # Bbar = np.tensordot(ratelist, delxDotdelLamb, axes=(0, 1))
        for i in range(lenVecClus):
            Bbar[i] = np.dot(ratelist, delxDotdelLamb[i, :])

        return Wbar, Bbar