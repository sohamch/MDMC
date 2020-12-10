from onsager import crystal, supercell, cluster
import numpy as np
import LatGas
import unittest

class Test_latGasKMC(unittest.TestCase):

    def setUp(self):
        self.NSpec = 5
        self.chem = 0
        self.crys = crystal.Crystal.BCC(0.2836, chemistry="A")
        self.jnetBCC = self.crys.jumpnetwork(0, 0.26)
        self.N_units = 8
        self.superlatt = self.N_units * np.eye(3, dtype=int)
        self.superBCC = supercell.ClusterSupercell(self.crys, self.superlatt)

        # The initial vacancy needs to be at [0, 0, 0].
        self.vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
        self.vacsiteInd = self.superBCC.index(np.zeros(3, dtype=int), (0, 0))[0]

        self.RtoSiteInd, self.siteIndtoR = LatGas.makeSiteIndtoR(self.superBCC)

        # make the ijList and dxList from the jump network
        self.ijList, self.dxList, self.dxtoR = LatGas.makeSupJumps(self.superBCC, self.jnetBCC, self.chem)

        initState = np.zeros(len(self.superBCC.mobilepos), dtype=int)
        # Now assign random species (excluding the vacancy)
        for i in range(len(self.superBCC.mobilepos)):
            initState[i] = np.random.randint(0, self.NSpec - 1)

        # Now put in the vacancy at the vacancy site
        initState[self.vacsiteInd] = self.NSpec - 1
        self.initState = initState
        print("Done setting up")

    def test_RtoSiteInd(self):
        for siteInd in range(self.siteIndtoR.shape[0]):
            Rsite = self.siteIndtoR[siteInd]
            self.assertEqual(self.RtoSiteInd[Rsite[0], Rsite[1], Rsite[2]], siteInd)
            ci, R = self.superBCC.ciR(siteInd)
            np.array_equal(Rsite, R)

    def test_makeSupJumps(self):

        superBCC = self.superBCC
        N_units = np.array([superBCC.superlatt[0, 0], superBCC.superlatt[1, 1], superBCC.superlatt[2, 2]])

        for jmp in range(self.ijList.shape[0]):
            dx = self.dxList[jmp]
            dxR = self.dxtoR[jmp]
            self.assertTrue(np.allclose(np.dot(self.crys.lattice, dxR), dx))

            # Apply PBC to the lattice vector
            dxRInCell = dxR % N_units

            # Then see the location of the site in ijList
            siteR = superBCC.ciR(self.ijList[jmp])[1]

            self.assertTrue(np.array_equal(dxRInCell, siteR))


    def testStep(self):

        state = self.initState.copy()

        RtoSiteInd, siteIndtoR = self.RtoSiteInd.copy(), self.siteIndtoR.copy()

        N_unit = self.N_units

        SpecRates = np.array(range(1, self.NSpec))

        ijList, dxList = self.ijList.copy(), self.dxList.copy()

        vacSiteInit = self.vacsiteInd
        self.assertEqual(vacSiteInit, 0)

        Nsteps = 5

        X_steps, t_steps, jmpSelectSteps, jmpFinSiteList = LatGas.LatGasKMCTraj(state, SpecRates, Nsteps, ijList, dxList,
                                                                                vacSiteInit, N_unit, siteIndtoR, RtoSiteInd)

        dxtoR = [np.around(np.dot(np.linalg.inv(self.crys.lattice), dx), decimals=4).astype(int) for dx in dxList]

        state0 = self.initState.copy()

        dxRun = np.zeros(3)
        dxRunR = np.zeros(3, dtype=int)
        dxRunSpecs = np.zeros((self.NSpec-1, 3))
        tRun = 0
        for step in range(Nsteps):
            # get the jump selected
            jmpStep = jmpSelectSteps[step]
            dxRun += dxList[jmpStep]
            dxRunR += dxtoR[jmpStep]

            # Check the vacancy displacements
            self.assertTrue(np.allclose(X_steps[step, self.NSpec-1], dxRun))

            # Next, we need to check if the correct species has been exchanged.

            # # First, get where the vacancy is in the current state
            vacNow = np.where(state0 == self.NSpec-1)[0][0]
            # Get the vacancy position
            Rvac = siteIndtoR[vacNow]

            # add the displacement
            RExchange = (Rvac + dxtoR[jmpStep]) % N_unit

            # Get the exchange site
            vacNext = RtoSiteInd[RExchange[0], RExchange[1], RExchange[2]]

            # Check that the correct displacement has been recorded for the species that is jumping
            specB = state0[vacNext]

            dxRunSpecs[specB] -= dxList[jmpStep]

            self.assertTrue(np.allclose(X_steps[step, specB], dxRunSpecs[specB]))

            # Check that the correct residence time is calculated
            rateTot = 0.
            for jmp in range(ijList.shape[0]):
                dxR = dxtoR[jmp]
                # get the exchange site
                REx = (Rvac + dxR) % N_unit
                siteEx = RtoSiteInd[REx[0], REx[1], REx[2]]
                specEx = state0[siteEx]
                rateTot += SpecRates[specEx]

            tRun += 1/rateTot

            self.assertAlmostEqual(tRun, t_steps[step])

            # update the state
            temp = state0[vacNext]
            state0[vacNext] = state0[vacNow]
            state0[vacNow] = temp


        # get the vacancy postiton vector in lattice coordinates
        poscart2R = np.around(np.dot(np.linalg.inv(self.crys.lattice), dxRun), decimals=3)  # sometimes round off will be observed
                                                                                            # during type casting
        poscart2R = poscart2R.astype(int)

        poscart2R %= N_unit
        dxRunR %= N_unit

        # Check if vacancy is tracked correctly always.
        self.assertTrue(np.array_equal(dxRunR, poscart2R))
        self.assertEqual(state[RtoSiteInd[dxRunR[0], dxRunR[1], dxRunR[2]]], self.NSpec-1)

        print("finished testing steps")
