import numpy as np
from numba.experimental import jitclass
from numba import jit, int64, float64

# Paste all the function definitions here as comments
MonteCarloSamplerSpec = [
    ("numInteractsSiteSpec", int64[:, :]),
    ("SiteSpecInterArray", int64[:, :, :]),
    ("numSitesInteracts", int64[:]),
    # ("numSitesTSInteracts", int64[:]),
    ("SupSitesInteracts", int64[:, :]),
    # ("TSInteractSites", int64[:, :]),
    # ("TSInteractSpecs", int64[:, :]),
    ("SpecOnInteractSites", int64[:, :]),
    ("Interaction2En", float64[:]),
    # ("jumpFinSites", int64[:]),
    # ("jumpFinSpec", int64[:]),
    # ("numJumpPointGroups", int64[:]),
    # ("numTSInteractsInPtGroups", int64[:, :]),
    # ("JumpInteracts", int64[:, :, :]),
    # ("Jump2KRAEng", float64[:, :, :]),
    # ("KRASpecConstants", float64[:]),
    ("mobOcc", int64[:]),
    ("Nsites", int64),
    ("Nspecs", int64),
    ("OffSiteCount", int64[:]),
    ("delEArray", float64[:]),
    ("delETotal", float64),
    ("vacSpec", int64),
    # ("FinSiteFinSpecJumpInd", int64[:, :])
]


@jitclass(MonteCarloSamplerSpec)
class MCSamplerClass(object):

    def __init__(self, vacSpec, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numInteractsSiteSpec, SiteSpecInterArray,
                 numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
                 FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng, KRASpecConstants):

        self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites, self.Interaction2En, self.numInteractsSiteSpec,\
        self.SiteSpecInterArray = \
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numInteractsSiteSpec, SiteSpecInterArray

        self.numSitesTSInteracts, self.TSInteractSites, self.TSInteractSpecs = \
            numSitesTSInteracts, TSInteractSites, TSInteractSpecs

        self.jumpFinSites, self.jumpFinSpec, self.FinSiteFinSpecJumpInd, self.numJumpPointGroups, self.numTSInteractsInPtGroups, \
        self.JumpInteracts, self.Jump2KRAEng = \
            jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, \
            JumpInteracts, Jump2KRAEng

        self.vacSpec = vacSpec
        self.KRASpecConstants = KRASpecConstants  # a constant to be added to KRA energies depending on which species jumps
        # This can be kept to just zero if not required
        assert KRASpecConstants.shape[0] == numInteractsSiteSpec.shape[1]
        assert KRASpecConstants[self.vacSpec] == 0.

        # check if proper sites and species data are entered
        self.Nsites, self.Nspecs = numInteractsSiteSpec.shape[0], numInteractsSiteSpec.shape[1]

    def DoSwapUpdate(self, state, siteA, siteB, lenVecClus, OffSiteCount,
                     numVecsInteracts, VecGroupInteracts, VecsInteracts):

        del_lamb = np.zeros((lenVecClus, 3))
        delE = 0.0
        # Switch required sites off
        for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteA]]):
            # check if an interaction is on
            interMainInd = self.SiteSpecInterArray[siteA, state[siteA], interIdx]
            if OffSiteCount[interMainInd] == 0:
                delE -= self.Interaction2En[interMainInd]
                # take away the vectors for this interaction
                if numVecsInteracts is not None:
                    for i in range(numVecsInteracts[interMainInd]):
                        del_lamb[VecGroupInteracts[interMainInd, i]] -= VecsInteracts[interMainInd, i, :]
            OffSiteCount[interMainInd] += 1

        for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
            interMainInd = self.SiteSpecInterArray[siteB, state[siteB], interIdx]
            if OffSiteCount[interMainInd] == 0:
                delE -= self.Interaction2En[interMainInd]
                if numVecsInteracts is not None:
                    for i in range(numVecsInteracts[interMainInd]):
                        del_lamb[VecGroupInteracts[interMainInd, i]] -= VecsInteracts[interMainInd, i, :]
            OffSiteCount[interMainInd] += 1

        # Next, switch required sites on
        for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
            interMainInd = self.SiteSpecInterArray[siteA, state[siteB], interIdx]
            OffSiteCount[interMainInd] -= 1
            if OffSiteCount[interMainInd] == 0:
                delE += self.Interaction2En[interMainInd]
                # add the vectors for this interaction
                if numVecsInteracts is not None:
                    for i in range(numVecsInteracts[interMainInd]):
                        del_lamb[VecGroupInteracts[interMainInd, i]] += VecsInteracts[interMainInd, i, :]

        for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
            interMainInd = self.SiteSpecInterArray[siteB, state[siteA], interIdx]
            OffSiteCount[interMainInd] -= 1
            if OffSiteCount[interMainInd] == 0:
                delE += self.Interaction2En[interMainInd]
                # add the vectors for this interaction
                # for interactions with zero vector basis, numVecsInteracts[interMainInd] = -1 and the
                # loop doesn't run
                if numVecsInteracts is not None:
                    for i in range(numVecsInteracts[interMainInd]):
                        del_lamb[VecGroupInteracts[interMainInd, i]] += VecsInteracts[interMainInd, i, :]

        return delE, del_lamb

    def revert(self, offsc, state, siteA, siteB):
        for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteA]]):
            offsc[self.SiteSpecInterArray[siteA, state[siteA], interIdx]] -= 1

        for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
            offsc[self.SiteSpecInterArray[siteB, state[siteB], interIdx]] -= 1

        for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
            offsc[self.SiteSpecInterArray[siteA, state[siteB], interIdx]] += 1

        for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
            offsc[self.SiteSpecInterArray[siteB, state[siteA], interIdx]] += 1

    def makeMCsweep(self, state, N_Specs, OffSiteCount, TransOffSiteCount,
                    beta, randLogarr, Nswaptrials, vacSiteInd):
        """

        :param state: the starting state
        :param N_Specs: how many of each species are there in the state
        Example input array[10, 501]- 10 species 0 atoms and 501 specis 1 atoms.
        :param OffSiteCount: interaction off site counts for the current state
        :param TransOffSiteCount: transition state interaction off site counts
        :param beta: 1/kT
        :param randLogarr: log of random numbers for acceptance criterion
        :param Nswaptrials: How many site swaps we want to attempt
        :param vacSiteInd: where the vacancy is
        """
        acceptCount = 0
        self.delETotal = 0.0
        self.delEArray = np.zeros(Nswaptrials)

        Nsites = state.shape[0]
        MaxCount = np.max(N_Specs)
        # print(N_Specs)
        # Next, fill up where the atoms are located
        specMemberCounts = np.zeros_like(N_Specs, dtype=int64)
        SpecLocations = np.full((N_Specs.shape[0], MaxCount), -1, dtype=int64)
        for siteInd in range(Nsites):
            if siteInd == vacSiteInd:  # If vacancy, do nothing
                assert state[siteInd] == self.vacSpec
            else:
                assert state[siteInd] != self.vacSpec
            spec = state[siteInd]
            specMemIdx = specMemberCounts[spec]
            SpecLocations[spec, specMemIdx] = siteInd
            specMemIdx += 1
            specMemberCounts[spec] = specMemIdx

        count = 0  # to keep a count of accepted moves

        NonVacLabels = np.zeros(N_Specs.shape[0] - 1, dtype=int64)
        for spInd in range(N_Specs.shape[0]):
            if spInd == self.vacSpec:
                continue
            elif spInd < self.vacSpec:
                NonVacLabels[spInd] = spInd
            else:
                NonVacLabels[spInd-1] = spInd

        for swapcount in range(Nswaptrials):

            # first select two random species to swap
            NonVacLabels = np.random.permutation(NonVacLabels)
            spASelect = NonVacLabels[0]
            spBSelect = NonVacLabels[1]

            # randomly select two different locations to swap
            siteALocIdx = np.random.randint(0, N_Specs[spASelect])
            siteBLocIdx = np.random.randint(0, N_Specs[spBSelect])

            siteA = SpecLocations[spASelect, siteALocIdx]
            siteB = SpecLocations[spBSelect, siteBLocIdx]

            specA = state[siteA]
            specB = state[siteB]

            assert -1 < siteA < Nsites and siteA != vacSiteInd
            assert -1 < siteB < Nsites and siteB != vacSiteInd
            assert specA == spASelect
            assert specB == spBSelect
            assert specA != specB

            # Do the swap
            delE, _ = self.DoSwapUpdate(state, siteA, siteB, 1, OffSiteCount, None, None, None)

            self.delEArray[swapcount] = delE

            # do the selection test
            if -beta*delE > randLogarr[swapcount]:
                # swap the sites to get to the next state
                self.delETotal += delE
                state[siteA] = specB
                state[siteB] = specA

                # swap the site indices stored for these two atoms
                SpecLocations[spASelect, siteALocIdx] = siteB
                SpecLocations[spBSelect, siteBLocIdx] = siteA

                # OffSiteCount is already updated to that of the new state.
                acceptCount += 1
                count += 1
                # make the offsite for the transition states
                for TsInteractIdx in range(len(self.TSInteractSites)):
                    TransOffSiteCount[TsInteractIdx] = 0
                    for Siteind in range(self.numSitesTSInteracts[TsInteractIdx]):
                        if state[self.TSInteractSites[TsInteractIdx, Siteind]] != self.TSInteractSpecs[TsInteractIdx, Siteind]:
                            TransOffSiteCount[TsInteractIdx] += 1

            else:
                # revert back the off site counts, because the state has not changed
                self.revert(OffSiteCount, state, siteA, siteB)

        return SpecLocations, acceptCount

    def getLambda(self, offsc, NVclus, numVecsInteracts, VecGroupInteracts, VecsInteracts):
        lamb = np.zeros((NVclus, 3))
        for interactInd in range(offsc.shape[0]):
            if offsc[interactInd] == 0:
                for vGInd in range(numVecsInteracts[interactInd]):
                    vGroup = VecGroupInteracts[interactInd, vGInd]
                    vec = VecsInteracts[interactInd, vGInd]
                    lamb[vGroup, :] += vec
        return lamb

    def getDelLamb(self, state, offsc, siteA, siteB, lenVecClus,
                   numVecsInteracts, VecGroupInteracts, VecsInteracts):


        _, del_lamb = self.DoSwapUpdate(state, siteA, siteB, lenVecClus, offsc,
                                        numVecsInteracts, VecGroupInteracts, VecsInteracts)

        # Then revert back off site count to original values
        self.revert(offsc, state, siteA, siteB)

        return del_lamb

    def Expand(self, state, ijList, dxList, spec, OffSiteCount, TSOffSiteCount,
               numVecsInteracts, VecGroupInteracts, VecsInteracts,
               lenVecClus, beta, vacSiteInd, RateList):

        del_lamb_mat = np.zeros((lenVecClus, lenVecClus, ijList.shape[0]))
        delxDotdelLamb = np.zeros((lenVecClus, ijList.shape[0]))

        delElist = np.zeros(ijList.shape[0])
        delEKRAlist = np.zeros(ijList.shape[0])
        if RateList is None:
            ratelist = np.zeros(ijList.shape[0])
        else:
            ratelist = RateList.copy()

        siteA, specA = vacSiteInd, state[vacSiteInd]
        # go through all the transition

        for jumpInd in range(ijList.shape[0]):
            # Get the transition site and species
            siteB, specB = ijList[jumpInd], state[ijList[jumpInd]]

            # next, calculate the energy and basis function change due to site swapping
            delE, del_lamb = self.DoSwapUpdate(state, siteA, siteB, lenVecClus, OffSiteCount,
                                               numVecsInteracts, VecGroupInteracts, VecsInteracts)

            # Next, restore OffSiteCounts to original values for next jump
            self.revert(OffSiteCount, state, siteA, siteB)
            
            # record new rates only if none were provided
            if RateList is None:
                # Do KRA expansion first
                delEKRA = self.KRASpecConstants[specB]  # Start with the constant term for the jumping species.
                transInd = self.FinSiteFinSpecJumpInd[siteB, specB]
                # First, work on getting the KRA energy for the jump
                # We need to go through every point group for this jump
                for tsPtGpInd in range(self.numJumpPointGroups[transInd]):
                    for interactInd in range(self.numTSInteractsInPtGroups[transInd, tsPtGpInd]):
                        # Check if this interaction is on
                        interactMainInd = self.JumpInteracts[transInd, tsPtGpInd, interactInd]
                        if TSOffSiteCount[interactMainInd] == 0:
                            delEKRA += self.Jump2KRAEng[transInd, tsPtGpInd, interactInd]
                
                ratelist[jumpInd] = np.exp(-(0.5 * delE + delEKRA) * beta)
                delElist[jumpInd] = delE
                delEKRAlist[jumpInd] = delEKRA

            del_lamb_mat[:, :, jumpInd] = np.dot(del_lamb, del_lamb.T)

            # let's do the tensordot by hand (numba doesn't support np.tensordot)
            for i in range(lenVecClus):
                if spec == specA:
                    delxDotdelLamb[i, jumpInd] = np.dot(del_lamb[i, :], dxList[jumpInd, :])
                elif spec == specB:
                    delxDotdelLamb[i, jumpInd] = np.dot(del_lamb[i, :], -dxList[jumpInd, :])

        WBar = np.zeros((lenVecClus, lenVecClus))
        for i in range(lenVecClus):
            WBar[i, i] += np.dot(del_lamb_mat[i, i, :], ratelist)
            for j in range(i):
                WBar[i, j] += np.dot(del_lamb_mat[i, j, :], ratelist)
                WBar[j, i] = WBar[i, j]

        BBar = np.zeros(lenVecClus)
        for i in range(lenVecClus):
            BBar[i] = np.dot(ratelist, delxDotdelLamb[i, :])

        return WBar, BBar, ratelist, delElist, delEKRAlist