"""
Functions to perform KMC simulations on mono-atomic lattice gases.
"""
import numpy as np
from onsager import cluster
from numba import jit, int64, float64


def makeSiteIndtoR(supercell):
    """
    Function to map supercell sites to corresponding lattice coordinates and vice versa.
    Works only for mono-atomic lattices for now on supercells constructed by just scaling unit cell vectors.
    :param supercell:
    :return: siteIndtoR - takes a supercell site index and returns the corresponding lattice coordinates.
             RtoSiteInd - takes integer lattice coordinates and returns the supercell sites.
    """

    Nsites = len(supercell.mobilepos)

    N_units = np.array([supercell.superlatt[0, 0], supercell.superlatt[1, 1], supercell.superlatt[2, 2]])
    # superlatt must be diagonal right now for this to work.

    siteIndtoR = np.zeros((Nsites, 3), dtype=int)
    RtoSiteInd = np.zeros((N_units[0], N_units[1], N_units[2]), dtype=int)

    for siteInd in range(Nsites):
        R = supercell.ciR(siteInd)[1]
        siteIndtoR[siteInd, :] = R
        RtoSiteInd[R[0], R[1], R[2]] = siteInd

    return RtoSiteInd, siteIndtoR

def makeSupJumps(supercell, jumpnetwork, chem):

    """
    Function to re-index jumps with supercell indices for final sites after vacancy jumo for mono-atomic lattices.
    The initial vacancy site is assumed to be [0, 0, 0]
    :param supercell: supercell object to get site indices from
    :param jumpnetwork: vacancy jumps indexed as (i, j), dx from crystal class
    :param chem: vacancy jump sublattice
    :param vacsite: clusterSite object for vacancy site in supercell
    :param vacsiteInd: supercell index for vacancy site
    :return: jList - list of final super cell site indices of the jumps.
             dxList - list of displacements of the jumps
             dxtoR - unit cell displacements of jumps
    """

    # Set up the jump network
    ijList = []
    dxList = []
    dxtoR = []
    crys = supercell.crys
    for jump in [jmp for jmpList in jumpnetwork for jmp in jmpList]:
        siteA = cluster.ClusterSite(ci=(chem, jump[0][0]), R=np.zeros(3, dtype=int))

        # Rj + uj = ui + dx (since Ri is zero in the jumpnetwork)
        Rj, (c, cj) = crys.cart2pos(jump[1] + np.dot(crys.lattice, crys.basis[chem][jump[0][0]]))
        # check we have the correct site
        if not cj == jump[0][1]:
            raise ValueError("improper coordinate transformation, did not get same final jump site\n")

        dx = jump[1]
        # Since we are in a monoatomic lattice, we can directly check the jump
        # dx2 = np.dot(crys.lattice, Rj)
        # assert np.allclose(dx2, dx), "dx: {}\n dx2: {}".format(dx, dx2)

        siteB = cluster.ClusterSite(ci=(chem, jump[0][1]), R=Rj)

        indB = supercell.index(siteB.R, siteB.ci)[0]

        ijList.append(indB)  # store the final site of the jump
        dxList.append(dx)
        dxtoR.append(Rj)

    return ijList, dxList, dxtoR

@jit(nopython=True)
def TrajAv(X_steps, t_steps, diff):
    """
    To produce average displacements divided by time across trajectories
    :param X_steps: Nsteps x Nspec x 3 array to store accumulated displacements at each step in a trajectory
    :param t_steps: Nsteps array to store the accumulated time at each step in a trajectory
    :param diff: array to accumulate the diffusivity in
    """
    Nsteps = X_steps.shape[0]
    for step in range(Nsteps):
        t = t_steps[step]
        X = X_steps[step].copy()
        for spec in range(X_steps.shape[1]):
            diff[spec, step] += np.dot(X[spec], X[spec])/(6*t)

@jit(nopython=True)
def translateState(state, vacSiteNow, vacsiteDes, RtoSiteInd, siteIndtoR, N_units):
    """
    Function to translate a state so that the vacancy is at a desired location
    in a monoatomic lattice
    :param state: The state to translate
    :param vacSiteNow: Where the vacancy is at the current state
    :param vacsiteDes: Where the vacancy should be at the end state
    :param RtoSiteInd: Takes the lattice coordinates of a site and returns the site index (see makeSiteIndtoR)
    :param siteIndtoR: Takes the site index and returns the lattice coordinates of a site
    :param N_units: Integer - supercell scaling units (assumes same scaling of all unit cell vectors for now)
    :return: state2New : The translated state
    """
    state2New = np.zeros_like(state, dtype=int64)
    dR = siteIndtoR[vacsiteDes] - siteIndtoR[vacSiteNow]  # this is the vector by which everything has to move
    for siteInd in range(siteIndtoR.shape[0]):
        Rsite = siteIndtoR[siteInd]
        spec = state[Rsite[0], Rsite[1], Rsite[2]]
        RsiteNew = (Rsite + dR) % N_units
        state2New[RsiteNew[0], RsiteNew[1], RsiteNew[2]] = spec
    return state2New

@jit(nopython=True)
def LatGasKMCTraj(state, SpecRates, Nsteps, ijList, dxList,
                  vacSiteInit, N_unit, siteIndtoR, RtoSiteInd):
    """
    Function to generate a KMC trajectory on a lattice gas where there aren't any energetic interactions, and
    vacancy transition rates with all species are pre-defined
    :param state - the starting state for the trajectory
    :param SpecRates - the exchange rates of the vacancy with the different species
    :param Nsteps - the no. of KMC steps to take
    :param ijList - array containing the final site indices of each vacancy jump out of the initial state.
    :param dxList - array containing the displacement of each jump
    :param vacSiteInit - Integer, representing the site at which the vacancy is initially present.
    :param N_unit - Supercell size.
    :param siteIndtoR - contains the lattice vectors pointing to specific supercell sites
    :param RtoSiteInd - contains the index of the site to which a lattice vector points.

    :returns
    X_steps - (NstepsxNSpecx3) the displacement at each step for each of the NSpec species
    t_steps - (Nsteps-size array)the residence time at each step
    jmpSelectSteps - (Nsteps-size array) which jump was selected at each step - for reproduction of a trajectory during testing.
    jmpFinSites - The exit site index list for the vacancy out of the final state - for testing.
    """
    NSpec = SpecRates.shape[0] + 1
    X = np.zeros((NSpec, 3))
    t = 0.

    X_steps = np.zeros((Nsteps, NSpec, 3))
    t_steps = np.zeros(Nsteps)

    rateArr = np.zeros(ijList.shape[0])

    jmpFinSiteList = ijList.copy()
    vacSiteNow = vacSiteInit

    jmpSelectSteps = np.zeros(Nsteps, dtype=int64)  # To store which jump was selected in each step

    for step in range(Nsteps):

        # first get the exit rates out of this state
        for jmpInd in range(jmpFinSiteList.shape[0]):
            specB = state[jmpFinSiteList[jmpInd]]  # Get the species occupying the exit site.
            rateArr[jmpInd] = SpecRates[specB]  # Get the exit rate corresponding to this species.

        # Next get the escape time
        rateTot = np.sum(rateArr)
        if rateTot < 1e-8:  # If escape rate is zero, then nothing will move and time will be infinite
            X[:, :] += 0.
            t = np.inf
        else:
            # convert the rates to cumulative probability
            rateArr /= rateTot

            t += 1. / rateTot
            rates_cm = np.cumsum(rateArr)

            # Then select the jump
            rn = np.random.rand()
            jmpSelect = np.searchsorted(rates_cm, rn)

            jmpSelectSteps[step] = jmpSelect  # Store which jump was selected

            # Store the displacement and the time for this step
            X[NSpec - 1, :] += dxList[jmpSelect]

            siteB = jmpFinSiteList[jmpSelect]
            specB = state[siteB]
            X[specB, :] -= dxList[jmpSelect]

            dR = siteIndtoR[siteB] - siteIndtoR[vacSiteInit]

            # Update the final jump sites
            for jmp in range(jmpFinSiteList.shape[0]):
                RfinSiteNew = (dR + siteIndtoR[ijList[jmp]]) % N_unit
                jmpFinSiteList[jmp] = RtoSiteInd[RfinSiteNew[0], RfinSiteNew[1], RfinSiteNew[2]]

            # Next, do the site swap to update the state
            temp = state[vacSiteNow]
            state[vacSiteNow] = specB
            state[siteB] = temp

            vacSiteNow = siteB

        X_steps[step, :, :] = X.copy()
        t_steps[step] = t

    return X_steps, t_steps, jmpSelectSteps, jmpFinSiteList