import numpy as np
from numba.experimental import jitclass
from numba import jit, int64, float64

# Paste all the function definitions here as comments

np.seterr(all='raise')


@jit(nopython=True)
def DoRandSwap(stateIn, Ntrials, vacSiteInd):
    state = stateIn.copy()
    Nsite = state.shape[0]
    initSiteList = np.zeros(Ntrials, dtype=int64)
    finSiteList = np.zeros(Ntrials, dtype=int64)

    count = 0
    while count < Ntrials:
        siteA = np.random.randint(0, Nsite)
        siteB = np.random.randint(0, Nsite)

        if state[siteA] == state[siteB] or siteA == vacSiteInd or siteB == vacSiteInd:
            continue

        # Otherwise, store the index of the site that was swapped
        initSiteList[count] = siteA
        finSiteList[count] = siteB

        # Now do the swap
        temp = state[siteA]
        state[siteA] = state[siteB]
        state[siteB] = temp
        count += 1

    return initSiteList, finSiteList

@jit(nopython=True)
def GetOffSite(state, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites):
    """
    :param state: State for which to count off sites of interactions
    :return: OffSiteCount array (N_interaction x 1)
    """
    OffSiteCount = np.zeros(numSitesInteracts.shape[0], dtype=int64)
    for interactIdx in range(numSitesInteracts.shape[0]):
        for intSiteind in range(numSitesInteracts[interactIdx]):
            if state[SupSitesInteracts[interactIdx, intSiteind]] != \
                    SpecOnInteractSites[interactIdx, intSiteind]:
                OffSiteCount[interactIdx] += 1
    return OffSiteCount

@ jit(nopython=True)
def GetTSOffSite(state, numSitesTSInteracts, TSInteractSites, TSInteractSpecs):
    """
    :param state: State for which to count off sites of TS interactions
    :return: OffSiteCount array (N_interaction x 1)
    """
    TransOffSiteCount = np.zeros(numSitesTSInteracts.shape[0], dtype=int64)
    for TsInteractIdx in range(numSitesTSInteracts.shape[0]):
        for Siteind in range(numSitesTSInteracts[TsInteractIdx]):
            if state[TSInteractSites[TsInteractIdx, Siteind]] != TSInteractSpecs[TsInteractIdx, Siteind]:
                TransOffSiteCount[TsInteractIdx] += 1
    return TransOffSiteCount

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
    ("jumpFinSites", int64[:]),
    ("jumpFinSpec", int64[:]),
    ("numJumpPointGroups", int64[:]),
    ("numTSInteractsInPtGroups", int64[:, :]),
    ("JumpInteracts", int64[:, :, :]),
    ("Jump2KRAEng", float64[:, :, :]),
    ("KRASpecConstants", float64[:]),
    ("mobOcc", int64[:]),
    ("Nsites", int64),
    ("Nspecs", int64),
    ("OffSiteCount", int64[:]),
    ("delEArray", float64[:]),
    ("delETotal", float64),
    ("FinSiteFinSpecJumpInd", int64[:, :])
]


@jitclass(MonteCarloSamplerSpec)
class MCSamplerClass(object):

    def __init__(self, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numInteractsSiteSpec, SiteSpecInterArray,
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

        self.KRASpecConstants = KRASpecConstants  # a constant to be added to KRA energies depending on which species jumps
        # This can be kept to just zero if not required
        assert KRASpecConstants.shape[0] == numInteractsSiteSpec.shape[1] - 1

        # check if proper sites and species data are entered
        self.Nsites, self.Nspecs = numInteractsSiteSpec.shape[0], numInteractsSiteSpec.shape[1]

    def makeMCsweep(self, state, N_nonVacSpecs, OffSiteCount, TransOffSiteCount,
                    beta, randLogarr, Nswaptrials, vacSiteInd):
        """

        :param state: the starting state
        :param N_nonVacSpecs: how many of non-vacancy species are there in the state
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
        MaxCount = np.max(N_nonVacSpecs)
        # print(N_nonVacSpecs)
        # Next, fill up where the atoms are located
        specMemberCounts = np.zeros_like(N_nonVacSpecs, dtype=int64)
        SpecLocations = np.full((N_nonVacSpecs.shape[0], MaxCount), -1, dtype=int64)
        for siteInd in range(Nsites):
            if siteInd == vacSiteInd:  # If vacancy, do nothing
                assert state[vacSiteInd] == N_nonVacSpecs.shape[0]
                continue
            spec = state[siteInd]
            assert spec < N_nonVacSpecs.shape[0]
            specMemIdx = specMemberCounts[spec]
            SpecLocations[spec, specMemIdx] = siteInd
            specMemIdx += 1
            specMemberCounts[spec] = specMemIdx

        count = 0  # to keep a steady count of accepted moves

        NonVacLabels = np.arange(N_nonVacSpecs.shape[0])
        for swapcount in range(Nswaptrials):

            # first select two random species to swap
            NonVacLabels = np.random.permutation(NonVacLabels)
            spASelect = NonVacLabels[0]
            spBSelect = NonVacLabels[1]

            # randomly select two different locations to swap
            siteALocIdx = np.random.randint(0, N_nonVacSpecs[spASelect])
            siteBLocIdx = np.random.randint(0, N_nonVacSpecs[spBSelect])

            siteA = SpecLocations[spASelect, siteALocIdx]
            siteB = SpecLocations[spBSelect, siteBLocIdx]

            specA = state[siteA]
            specB = state[siteB]

            assert -1 < siteA < Nsites
            assert -1 < siteB < Nsites
            assert specA == spASelect
            assert specB == spBSelect
            assert specA != specB

            delE = 0.
            # Next, switch required interactions off
            for interIdx in range(self.numInteractsSiteSpec[siteA, specA]):
                interMainInd = self.SiteSpecInterArray[siteA, specA, interIdx]
                if OffSiteCount[interMainInd] == 0:
                    delE -= self.Interaction2En[interMainInd]
                OffSiteCount[interMainInd] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, specB]):
                interMainInd = self.SiteSpecInterArray[siteB, specB, interIdx]
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
                for interIdx in range(self.numInteractsSiteSpec[siteA, specA]):
                    OffSiteCount[self.SiteSpecInterArray[siteA, specA, interIdx]] -= 1

                for interIdx in range(self.numInteractsSiteSpec[siteB, specB]):
                    OffSiteCount[self.SiteSpecInterArray[siteB, specB, interIdx]] -= 1

                for interIdx in range(self.numInteractsSiteSpec[siteA, specB]):
                    OffSiteCount[self.SiteSpecInterArray[siteA, specB, interIdx]] += 1

                for interIdx in range(self.numInteractsSiteSpec[siteB, specA]):
                    OffSiteCount[self.SiteSpecInterArray[siteB, specA, interIdx]] += 1

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

    def Expand(self, state, ijList, dxList, spec, OffSiteCount, TSOffSiteCount,
               numVecsInteracts, VecGroupInteracts, VecsInteracts,
               lenVecClus, beta, vacSiteInd=0):

        del_lamb_mat = np.zeros((lenVecClus, lenVecClus, ijList.shape[0]))
        delxDotdelLamb = np.zeros((lenVecClus, ijList.shape[0]))

        ratelist = np.zeros(ijList.shape[0])

        siteA, specA = vacSiteInd, self.Nspecs - 1
        # go through all the transition

        for jumpInd in range(ijList.shape[0]):
            del_lamb = np.zeros((lenVecClus, 3))

            # Get the transition index
            siteB, specB = ijList[jumpInd], state[ijList[jumpInd]]
            transInd = self.FinSiteFinSpecJumpInd[siteB, specB]
            # First, work on getting the KRA energy for the jump
            delEKRA = self.KRASpecConstants[specB]  # Start with the constant term for the jumping species.
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
                    for i in range(numVecsInteracts[interMainInd]):
                        del_lamb[VecGroupInteracts[interMainInd, i]] -= VecsInteracts[interMainInd, i, :]
                OffSiteCount[interMainInd] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteB, state[siteB], interIdx]
                if OffSiteCount[interMainInd] == 0:
                    delE -= self.Interaction2En[interMainInd]
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
                    for i in range(numVecsInteracts[interMainInd]):
                        del_lamb[VecGroupInteracts[interMainInd, i]] += VecsInteracts[interMainInd, i, :]

            ratelist[jumpInd] = np.exp(-(0.5 * delE + delEKRA) * beta)
            del_lamb_mat[:, :, jumpInd] = np.dot(del_lamb, del_lamb.T)

            # let's do the tensordot by hand (work on finding numba support for this)
            for i in range(lenVecClus):
                # replace innder loop with outer product
                if spec == specA:
                    delxDotdelLamb[i, jumpInd] = np.dot(del_lamb[i, :], dxList[jumpInd, :])
                elif spec == specB:
                    delxDotdelLamb[i, jumpInd] = np.dot(del_lamb[i, :], -dxList[jumpInd, :])

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

        WBar = np.zeros((lenVecClus, lenVecClus))
        for i in range(lenVecClus):
            for j in range(lenVecClus):
                WBar[i, j] += np.dot(del_lamb_mat[i, j, :], ratelist)

        BBar = np.zeros(lenVecClus)
        for i in range(lenVecClus):
            BBar[i] = np.dot(ratelist, delxDotdelLamb[i, :])

        return WBar, BBar

    def ExpandLatGas(self, state, ijList, dxList, spec, OffSiteCount, specRates, lenVecClus,
                     numVecsInteracts, VecGroupInteracts, VecsInteracts, vacSiteInd=0):

        del_lamb_mat = np.zeros((lenVecClus, lenVecClus, ijList.shape[0]))
        delxDotdelLamb = np.zeros((lenVecClus, ijList.shape[0]))

        ratelist = np.zeros(ijList.shape[0])

        siteA, specA = vacSiteInd, self.Nspecs - 1
        # go through all the transition

        for jumpInd in range(ijList.shape[0]):
            del_lamb = np.zeros((lenVecClus, 3))

            # Get the transition index
            siteB, specB = ijList[jumpInd], state[ijList[jumpInd]]

            ratelist[jumpInd] = specRates[specB]

            # Switch required sites off
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteA]]):
                # check if an interaction is on
                interMainInd = self.SiteSpecInterArray[siteA, state[siteA], interIdx]
                if OffSiteCount[interMainInd] == 0:
                    # take away the vectors for this interaction
                    for i in range(numVecsInteracts[interMainInd]):
                        del_lamb[VecGroupInteracts[interMainInd, i]] -= VecsInteracts[interMainInd, i, :]
                OffSiteCount[interMainInd] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteB, state[siteB], interIdx]
                if OffSiteCount[interMainInd] == 0:
                    for i in range(numVecsInteracts[interMainInd]):
                        del_lamb[VecGroupInteracts[interMainInd, i]] -= VecsInteracts[interMainInd, i, :]
                OffSiteCount[interMainInd] += 1

            # Next, switch required sites on
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteA, state[siteB], interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    # add the vectors for this interaction
                    for i in range(numVecsInteracts[interMainInd]):
                        del_lamb[VecGroupInteracts[interMainInd, i]] += VecsInteracts[interMainInd, i, :]

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
                interMainInd = self.SiteSpecInterArray[siteB, state[siteA], interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    # add the vectors for this interaction
                    # for interactions with zero vector basis, numVecsInteracts[interMainInd] = -1 and the
                    # loop doesn't run
                    for i in range(numVecsInteracts[interMainInd]):
                        del_lamb[VecGroupInteracts[interMainInd, i]] += VecsInteracts[interMainInd, i, :]

            del_lamb_mat[:, :, jumpInd] = np.dot(del_lamb, del_lamb.T)

            # delxDotdelLamb[:, jumpInd] = np.tensordot(del_lamb, dxList[jumpInd], axes=(1, 0))
            # let's do the tensordot by hand (work on finding numba support for this)
            for i in range(lenVecClus):

                if spec == specA: # vacancy
                    delxDotdelLamb[i, jumpInd] = np.dot(del_lamb[i, :], dxList[jumpInd, :])

                elif spec == specB: # if it is the desired species
                    delxDotdelLamb[i, jumpInd] = np.dot(del_lamb[i, :], -dxList[jumpInd, :])

            # Next, restore OffSiteCounts to original values
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

        return WBar, BBar

    @staticmethod
    def ExpandDirect(lamb1, lamb2, rate, dx):
        """
        This is to get the expansion for a jump explicitly by specifying the basis functions,
        the rates and displacements.
        This should help when rates have been evaluated with something else other than cluster expansion
        :param lamb1: basis functions for state 1 (N_basis x 3) shape
        :param lamb2: basis functions for state 2 (N_basis x 3) shape
        :param rate: rate of the the jump
        :param dx: displacement of the jumps
        :return: Wbar, Bbar - the expanded transition rate and bias vectors in the new basis
        """
        Nbasis = lamb1.shape[0]
        WBar = np.zeros((Nbasis, Nbasis))
        bBar = np.zeros(Nbasis)

        del_lamb = lamb2 - lamb1

        for i in range(Nbasis):
            bBar[i] = np.dot(dx, del_lamb[i])*rate
            for j in range(Nbasis):
                WBar[i, j] = rate*np.dot(del_lamb[i], del_lamb[j])

        return WBar, bBar

    def getExitData(self, state, ijList, dxList, OffSiteCount, TSOffSiteCount, beta, Nsites, vacSiteInd=0):

        statesTrans = np.zeros((ijList.shape[0], Nsites), dtype=int64)
        ratelist = np.zeros(ijList.shape[0])
        Specdisps = np.zeros((ijList.shape[0], self.Nspecs, 3))  # To store the displacement of each species during every jump

        siteA, specA = vacSiteInd, self.Nspecs - 1
        # go through all the transition

        for jumpInd in range(ijList.shape[0]):
            # Get the transition index
            siteB, specB = ijList[jumpInd], state[ijList[jumpInd]]
            transInd = self.FinSiteFinSpecJumpInd[siteB, specB]

            # copy the state
            statesTrans[jumpInd, :] = state

            # swap occupancies after jump
            statesTrans[jumpInd, siteB] = state[siteA]
            statesTrans[jumpInd, siteA] = state[siteB]

            # First, work on getting the KRA energy for the jump
            delEKRA = self.KRASpecConstants[specB]
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
            Specdisps[jumpInd, specB, :] = -dxList[jumpInd, :]
            Specdisps[jumpInd, -1, :] = dxList[jumpInd, :]

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

        return statesTrans, ratelist, Specdisps



KMC_additional_spec = [
    ("siteIndtoR", int64[:, :]),
    ("RtoSiteInd", int64[:, :, :]),
    ("N_unit", int64),
]


@jitclass(MonteCarloSamplerSpec+KMC_additional_spec)
class KMC_JIT(object):
    # TODO: Implement periodic boundary conditions for non-diag primitive supercells

    def __init__(self, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En,numInteractsSiteSpec, SiteSpecInterArray,
                 numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
                 FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng, KRASpecConstants,
                 siteIndtoR, RtoSiteInd, N_unit):

        self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites, self.Interaction2En, self.numInteractsSiteSpec,\
        self.SiteSpecInterArray = \
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numInteractsSiteSpec, SiteSpecInterArray

        self.numSitesTSInteracts, self.TSInteractSites, self.TSInteractSpecs = \
            numSitesTSInteracts, TSInteractSites, TSInteractSpecs

        self.jumpFinSites, self.jumpFinSpec, self.FinSiteFinSpecJumpInd, self.numJumpPointGroups, self.numTSInteractsInPtGroups, \
        self.JumpInteracts, self.Jump2KRAEng = \
            jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, \
            JumpInteracts, Jump2KRAEng

        self.KRASpecConstants = KRASpecConstants

        self.RtoSiteInd = RtoSiteInd
        self.siteIndtoR = siteIndtoR

        self.Nsites, self.Nspecs = numInteractsSiteSpec.shape[0], numInteractsSiteSpec.shape[1]

        assert KRASpecConstants.shape[0] == numInteractsSiteSpec.shape[1] - 1

        self.N_unit = N_unit

    def TranslateState(self, state, siteFin, siteInit):

        dR = self.siteIndtoR[siteFin, :] - self.siteIndtoR[siteInit, :]
        stateTrans = np.zeros_like(state, dtype=int64)
        for siteInd in range(state.shape[0]):
            Rnew = (self.siteIndtoR[siteInd, :] + dR) % self.N_unit  # to apply PBC
            siteIndNew = self.RtoSiteInd[Rnew[0], Rnew[1], Rnew[2]]
            stateTrans[siteIndNew] = state[siteInd]
        return stateTrans

    def getJumpVibfreq(self, state, SpecVibFreq, jmpFinSiteListTrans):
        freqs = np.zeros(jmpFinSiteListTrans.shape)
        for jmpInd in range(jmpFinSiteListTrans.shape[0]):
            siteB = jmpFinSiteListTrans[jmpInd]
            specB = state[siteB]
            freqs[jmpInd] = SpecVibFreq[specB]
        return freqs

    def getKRAEnergies(self, state, TSOffSiteCount, ijList):

        delEKRA = np.zeros(ijList.shape[0], dtype=float64)
        for jumpInd in range(ijList.shape[0]):
            # Get the transition index
            siteB, specB = ijList[jumpInd], state[ijList[jumpInd]]
            delE = self.KRASpecConstants[specB]
            transInd = self.FinSiteFinSpecJumpInd[siteB, specB]
            # We need to go through every point group for this jump
            for tsPtGpInd in range(self.numJumpPointGroups[transInd]):
                for interactInd in range(self.numTSInteractsInPtGroups[transInd, tsPtGpInd]):
                    # Check if this interaction is on
                    interactMainInd = self.JumpInteracts[transInd, tsPtGpInd, interactInd]
                    if TSOffSiteCount[interactMainInd] == 0:
                        delE += self.Jump2KRAEng[transInd, tsPtGpInd, interactInd]
            delEKRA[jumpInd] = delE

        return delEKRA

    def getEnergyChangeJumps(self, state, OffSiteCount, siteA, jmpFinSiteListTrans):

        delEArray = np.zeros(jmpFinSiteListTrans.shape[0], dtype=float64)

        for jmpInd in range(jmpFinSiteListTrans.shape[0]):
            siteB = jmpFinSiteListTrans[jmpInd]
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

            delEArray[jmpInd] = delE

            # Now revert offsitecounts for next jump
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

        return delEArray

    def updateState(self, state, OffSiteCount, siteA, siteB):

        # update offsitecounts and symmetry classes
        for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteA]]):
            interMainInd = self.SiteSpecInterArray[siteA, state[siteA], interIdx]
            OffSiteCount[interMainInd] += 1

        for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
            interMainInd = self.SiteSpecInterArray[siteB, state[siteB], interIdx]
            OffSiteCount[interMainInd] += 1

        # Next, switch required sites on
        for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
            interMainInd = self.SiteSpecInterArray[siteA, state[siteB], interIdx]
            OffSiteCount[interMainInd] -= 1

        for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
            interMainInd = self.SiteSpecInterArray[siteB, state[siteA], interIdx]
            OffSiteCount[interMainInd] -= 1

        # swap sites
        temp = state[siteA]
        state[siteA] = state[siteB]
        state[siteB] = temp

    def getTraj(self, state, offsc, vacSiteFix, jumpFinSiteList, dxList, NSpec, SpecVibFreq, Nsteps, beta):

        X = np.zeros((NSpec, 3), dtype=float64)
        t = 0.

        X_steps = np.zeros((Nsteps, NSpec, 3), dtype=float64)
        t_steps = np.zeros(Nsteps, dtype=float64)
        jmpSelectArray = np.zeros(Nsteps, dtype=int64)

        jumpFinSiteListTrans = np.zeros_like(jumpFinSiteList, dtype=int64)
        vacIndNow = vacSiteFix

        # track displacements of individual atoms
        Nsites = state.shape[0]
        Specs, SpecCounts = np.unique(state, return_counts=True)
        AtomId2AtomPos = np.full((NSpec, np.max(SpecCounts)), -1, dtype=int64)
        AtomPos2Atomtype = np.zeros((Nsites), dtype=int64)
        AtomPos2AtomId = np.zeros((Nsites), dtype=int64)
        AtomIdtoAtomDisp = np.full((NSpec, np.max(SpecCounts), 3), -1, dtype=int64)

        # Now assign IDs to each atom
        spIDcounts = np.zeros(NSpec, dtype=int64)  # to track the ID of each atom of each species
        for siteInd in range(Nsites):
            sp = state[siteInd]
            AtomID = spIDcounts[sp]

            # Store the site this atom is present in
            AtomId2AtomPos[sp, AtomID] = siteInd
            AtomPos2Atomtype[siteInd] = sp
            AtomPos2AtomId[siteInd] = AtomID

            # Increment the index
            spIDcounts[sp] += 1

        for step in range(Nsteps):

            # Translate the states so that vacancy is taken from vacIndnow to vacSiteFix
            stateTrans = self.TranslateState(state, vacSiteFix, vacIndNow)
            TSoffsc = GetTSOffSite(stateTrans, self.numSitesTSInteracts, self.TSInteractSites, self.TSInteractSpecs)

            delEKRA = self.getKRAEnergies(stateTrans, TSoffsc, jumpFinSiteList)

            dR = self.siteIndtoR[vacIndNow] - self.siteIndtoR[vacSiteFix]

            for jmp in range(jumpFinSiteList.shape[0]):
                RfinSiteNew = (dR + self.siteIndtoR[jumpFinSiteList[jmp]]) % self.N_unit
                jumpFinSiteListTrans[jmp] = self.RtoSiteInd[RfinSiteNew[0], RfinSiteNew[1], RfinSiteNew[2]]

            delE = self.getEnergyChangeJumps(state, offsc, vacIndNow, jumpFinSiteListTrans)
            frq = self.getJumpVibfreq(state, SpecVibFreq, jumpFinSiteListTrans)

            rates = frq*np.exp(-(0.5 * delE + delEKRA) * beta)
            rateTot = np.sum(rates)
            t += 1.0/rateTot

            rates /= rateTot
            rates_cm = np.cumsum(rates)
            rn = np.random.rand()
            jmpSelect = np.searchsorted(rates_cm, rn)
            jmpSelectArray[step] = jmpSelect
            vacIndNext = jumpFinSiteListTrans[jmpSelect]

            X[NSpec - 1, :] += dxList[jmpSelect]
            specB = state[vacIndNext]
            X[specB, :] -= dxList[jmpSelect]

            # Now get the ID of this atom
            # specBID =

            X_steps[step, :, :] = X.copy()
            t_steps[step] = t

            self.updateState(state, offsc, vacIndNow, vacIndNext)

            vacIndNow = vacIndNext

        return X_steps, t_steps, jmpSelectArray