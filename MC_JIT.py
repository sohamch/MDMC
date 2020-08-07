import numpy as np
from numba.experimental import jitclass
from numba import int64, float64

# Paste all the function definitions here as comments

np.seterr(all='raise')
MonteCarloSamplerSpec = [
    ("numInteractsSiteSpec", int64[:, :]),
    ("SiteSpecInterArray", int64[:, :, :]),
    ("numSitesInteracts", int64[:]),
    ("InteractToRepClus", int64[:]),
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
    ("totRepClus", int64),
    ("OffSiteCount", int64[:]),
    ("FinSiteFinSpecJumpInd", int64[:, :]),
    ("VecGroupInteracts", int64[:, :])

]


@jitclass(MonteCarloSamplerSpec)
class MCSamplerClass(object):

    def __init__(self, numSitesInteracts, InteractToRepClus, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts,
                 VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray,
                 numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
                 FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng,
                 vacSiteInd):

        self.numSitesInteracts, self.InteractToRepClus,self.SupSitesInteracts, self.SpecOnInteractSites, self.Interaction2En,\
        self.numVecsInteracts, self.VecsInteracts, self.VecGroupInteracts, self.numInteractsSiteSpec, self.SiteSpecInterArray,\
        self.vacSiteInd =\
            numSitesInteracts, InteractToRepClus, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts, \
            VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray, vacSiteInd

        self.numSitesTSInteracts, self.TSInteractSites, self.TSInteractSpecs = \
            numSitesTSInteracts, TSInteractSites, TSInteractSpecs

        self.jumpFinSites, self.jumpFinSpec, self.FinSiteFinSpecJumpInd, self.numJumpPointGroups, self.numTSInteractsInPtGroups, \
        self.JumpInteracts, self.Jump2KRAEng = \
            jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, \
            JumpInteracts, Jump2KRAEng

        l = np.max(self.InteractToRepClus)
        self.totRepClus = l+1

        # check if proper sites and species data are entered
        self.Nsites, self.Nspecs = numInteractsSiteSpec.shape[0], numInteractsSiteSpec.shape[1]

        # Reformat the array so that the swaps are always between atoms of different species

    def makeRepClusOnCounter(self, offsc):
        repclusOncounter = np.zeros(self.totRepClus, dtype=int64)
        for interactInd in range(offsc.shape[0]):
            repClusInd = self.InteractToRepClus[interactInd]
            if offsc[interactInd] == 0:
                repclusOncounter[repClusInd] += 1
        return repclusOncounter

    def makeMCsweep(self, mobOcc, OffSiteCount, repClustOnCount, TransOffSiteCount,
                    SwapTrials, beta, randarr, Nswaptrials):

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
                if OffSiteCount[interMainInd] == 0:
                    delE -= self.Interaction2En[interMainInd]
                    # This interaction was on but will get switched off, so
                    # it's representative cluster will have one less "on" interaction.
                    repClustOnCount[self.InteractToRepClus[interMainInd]] -= 1
                OffSiteCount[interMainInd] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, specB]):
                interMainInd = self.SiteSpecInterArray[siteB, specB, interIdx]
                # offscount = OffSiteCount[interMainInd]
                if OffSiteCount[interMainInd] == 0:
                    delE -= self.Interaction2En[interMainInd]
                    repClustOnCount[self.InteractToRepClus[interMainInd]] -= 1
                OffSiteCount[interMainInd] += 1

            # Next, switch required sites on
            for interIdx in range(self.numInteractsSiteSpec[siteA, specB]):
                interMainInd = self.SiteSpecInterArray[siteA, specB, interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]
                    repClustOnCount[self.InteractToRepClus[interMainInd]] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, specA]):
                interMainInd = self.SiteSpecInterArray[siteB, specA, interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]
                    repClustOnCount[self.InteractToRepClus[interMainInd]] += 1
                    
            # do the selection test
            if -beta*delE > randarr[swapcount]:
                # swap the sites to get to the next state
                mobOcc[siteA] = specB
                mobOcc[siteB] = specA
                # OffSiteCount is already updated to that of the new state.

            else:
                # revert back the off site counts, because the state has not changed
                for interIdx in range(self.numInteractsSiteSpec[siteA, specA]):
                    interMainInd = self.SiteSpecInterArray[siteA, specA, interIdx]
                    OffSiteCount[interMainInd] -= 1
                    if OffSiteCount[interMainInd] == 0:
                        repClustOnCount[self.InteractToRepClus[interMainInd]] += 1

                for interIdx in range(self.numInteractsSiteSpec[siteB, specB]):
                    interMainInd = self.SiteSpecInterArray[siteB, specB, interIdx]
                    OffSiteCount[interMainInd] -= 1
                    if OffSiteCount[interMainInd] == 0:
                        repClustOnCount[self.InteractToRepClus[interMainInd]] += 1

                for interIdx in range(self.numInteractsSiteSpec[siteA, specB]):
                    interMainInd = self.SiteSpecInterArray[siteA, specB, interIdx]
                    if OffSiteCount[interMainInd] == 0:
                        repClustOnCount[self.InteractToRepClus[interMainInd]] -= 1
                    OffSiteCount[interMainInd] += 1

                for interIdx in range(self.numInteractsSiteSpec[siteB, specA]):
                    interMainInd = self.SiteSpecInterArray[siteB, specA, interIdx]
                    if OffSiteCount[interMainInd] == 0:
                        repClustOnCount[self.InteractToRepClus[interMainInd]] -= 1
                    OffSiteCount[interMainInd] += 1

        # make the offsite for the transition states
        for TsInteractIdx in range(len(self.TSInteractSites)):
            TransOffSiteCount[TsInteractIdx] = 0
            for Siteind in range(self.numSitesTSInteracts[TsInteractIdx]):
                if mobOcc[self.TSInteractSites[TsInteractIdx, Siteind]] != self.TSInteractSpecs[TsInteractIdx, Siteind]:
                    TransOffSiteCount[TsInteractIdx] += 1

    # For testing, use this signature.
    # def Expand(self, state, ijList, dxList, OffSiteCount, TSOffSiteCount, lenVecClus, beta, delEKRAarray, delEarray,
    #            SiteTransArray, SpecTransArray, WBar, BBar):

    def Expand(self, state, ijList, dxList, OffSiteCount, TSOffSiteCount, lenVecClus, beta):

        del_lamb_mat = np.zeros((lenVecClus, lenVecClus, ijList.shape[0]))
        delxDotdelLamb = np.zeros((lenVecClus, ijList.shape[0]))

        ratelist = np.zeros(ijList.shape[0])

        siteA, specA = self.vacSiteInd, self.Nspecs - 1
        # go through all the transition

        for jumpInd in range(ijList.shape[0]):
            del_lamb = np.zeros((lenVecClus, 3))

            # Get the transition index
            siteB, specB = ijList[jumpInd], state[ijList[jumpInd]]
            transInd = self.FinSiteFinSpecJumpInd[siteB, specB]

            # SiteTransArray[jumpInd] = siteB
            # SpecTransArray[jumpInd] = specB

            # First, work on getting the KRA energy for the jump
            delEKRA = 0.0
            # We need to go through every point group for this jump
            for tsPtGpInd in range(self.numJumpPointGroups[transInd]):
                for interactInd in range(self.numTSInteractsInPtGroups[transInd, tsPtGpInd]):
                    # Check if this interaction is on
                    interactMainInd = self.JumpInteracts[transInd, tsPtGpInd, interactInd]
                    if TSOffSiteCount[interactMainInd] == 0:
                        delEKRA += self.Jump2KRAEng[transInd, tsPtGpInd, interactInd]

            # delEKRAarray[jumpInd] = delEKRA

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
                    # for interactions with zero vector basis, numVecsInteracts[interMainInd] = -1 and the
                    # loop doesn't run
                    for i in range(self.numVecsInteracts[interMainInd]):
                        del_lamb[self.VecGroupInteracts[interMainInd, i]] += self.VecsInteracts[interMainInd, i, :]

            # delEarray[jumpInd] = delE
            # Energy change computed, now expand
            # if 0.5 * delE + delEKRA < 0:
            #     print(delE, delEKRA)
            #     raise ValueError("negative activation barrier")
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

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
                OffSiteCount[self.SiteSpecInterArray[siteB, state[siteB], interIdx]] -= 1

            # During switch-on operations, offsite counts were decreased by one.
            # So increase them back by one
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
                OffSiteCount[self.SiteSpecInterArray[siteA, state[siteB], interIdx]] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
                OffSiteCount[self.SiteSpecInterArray[siteB, state[siteA], interIdx]] += 1

        # Wbar = np.tensordot(ratelist, del_lamb_mat, axes=(0, 2))
        WBar = np.zeros((lenVecClus, lenVecClus))
        for i in range(lenVecClus):
            for j in range(lenVecClus):
                WBar[i, j] += np.dot(del_lamb_mat[i, j, :], ratelist)

        # assert np.allclose(Wbar, WBar)

        # Bbar = np.tensordot(ratelist, delxDotdelLamb, axes=(0, 1))
        BBar = np.zeros(lenVecClus)
        for i in range(lenVecClus):
            BBar[i] = np.dot(ratelist, delxDotdelLamb[i, :])

        # assert np.allclose(Bbar, BBar)

        return WBar, BBar

    def GetNewRandState(self, mobOcc, OffSiteCount, Energy, SwapTrials, Nswaptrials):

        En = Energy
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
            # swap the sites to get to the next state
            mobOcc[siteA] = specB
            mobOcc[siteB] = specA
            # add the energy to get the energy of the next state
            En += delE

        return En

    def getExitData(self, state, ijList, OffSiteCount, TSOffSiteCount, beta, Nsites):

        statesTrans = np.zeros((ijList.shape[0], Nsites), dtype=int64)
        ratelist = np.zeros(ijList.shape[0])

        siteA, specA = self.vacSiteInd, self.Nspecs - 1
        # go through all the transition

        for jumpInd in range(ijList.shape[0]):
            # Get the transition index
            siteB, specB = ijList[jumpInd], state[ijList[jumpInd]]
            transInd = self.FinSiteFinSpecJumpInd[siteB, specB]

            # copy the state
            statesTrans[jumpInd, :] = state
            # for siteInd in range(Nsites):
            #     statesTrans[jumpInd, siteInd] = state[siteInd]

            # swap occupancies after jump
            statesTrans[jumpInd, siteB] = state[siteA]
            statesTrans[jumpInd, siteA] = state[siteB]

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
                OffSiteCount[interMainInd] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteB, state[siteB], interIdx]
                if OffSiteCount[interMainInd] == 0:
                    delE -= self.Interaction2En[interMainInd]
                OffSiteCount[interMainInd] += 1

            # Next, switch required sites on
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteA, state[siteB], interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
                interMainInd = self.SiteSpecInterArray[siteB, state[siteA], interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]

            ratelist[jumpInd] = np.exp(-(0.5 * delE + delEKRA) * beta)

            # Next, restore OffSiteCounts to original values for next jump, as well as
            # for use in the next MC sweep.
            # During switch-off operations, offsite counts were increased by one.
            # So decrease them back by one
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteA]]):
                OffSiteCount[self.SiteSpecInterArray[siteA, state[siteA], interIdx]] -= 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
                OffSiteCount[self.SiteSpecInterArray[siteB, state[siteB], interIdx]] -= 1

            # During switch-on operations, offsite counts were decreased by one.
            # So increase them back by one
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
                OffSiteCount[self.SiteSpecInterArray[siteA, state[siteB], interIdx]] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
                OffSiteCount[self.SiteSpecInterArray[siteB, state[siteA], interIdx]] += 1

        return statesTrans, ratelist
