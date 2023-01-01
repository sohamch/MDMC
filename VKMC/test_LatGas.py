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
        self.superCell = supercell.ClusterSupercell(self.crys, self.superlatt)

        # The initial vacancy needs to be at [0, 0, 0].
        self.vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
        self.vacsiteInd = self.superCell.index(np.zeros(3, dtype=int), (0, 0))[0]

        self.RtoSiteInd, self.siteIndtoR = LatGas.makeSiteIndtoR(self.superCell)

        # make the ijList and dxList from the jump network
        self.ijList, self.dxList, self.dxtoR = LatGas.makeSupJumps(self.superCell, self.jnetBCC, self.chem)

        initState = np.zeros(len(self.superCell.mobilepos), dtype=int)
        # Now assign random species (excluding the vacancy)
        for i in range(len(self.superCell.mobilepos)):
            initState[i] = np.random.randint(0, self.NSpec - 1)

        # Now put in the vacancy at the vacancy site
        initState[self.vacsiteInd] = self.NSpec - 1
        self.initState = initState
        print("Done setting up")

    def test_RtoSiteInd(self):
        for siteInd in range(self.siteIndtoR.shape[0]):
            Rsite = self.siteIndtoR[siteInd]
            self.assertEqual(self.RtoSiteInd[Rsite[0], Rsite[1], Rsite[2]], siteInd)
            ci, R = self.superCell.ciR(siteInd)
            np.array_equal(Rsite, R)

    def test_makeSupJumps(self):

        superCell = self.superCell
        N_units = np.array([superCell.superlatt[0, 0], superCell.superlatt[1, 1], superCell.superlatt[2, 2]])

        for jmp in range(self.ijList.shape[0]):
            dx = self.dxList[jmp]
            dxR = self.dxtoR[jmp]
            self.assertTrue(np.allclose(np.dot(self.crys.lattice, dxR), dx))

            # Apply PBC to the lattice vector
            dxRInCell = dxR % N_units

            # Then see the location of the site in ijList
            siteR = superCell.ciR(self.ijList[jmp])[1]

            self.assertTrue(np.array_equal(dxRInCell, siteR))

    def test_gridState(self):
        superCell = self.superCell
        state = self.initState.copy()

        N_units = np.array([superCell.superlatt[0, 0], superCell.superlatt[1, 1], superCell.superlatt[2, 2]])
        stateGrid = LatGas.gridState(state, self.siteIndtoR, N_units)

        for siteInd in range(self.siteIndtoR.shape[0]):
            spec = state[siteInd]
            Rsite = self.siteIndtoR[siteInd]
            self.assertEqual(stateGrid[Rsite[0], Rsite[1], Rsite[2]], spec)

    def test_Translate(self):

        superCell = self.superCell
        N_units = np.array([superCell.superlatt[0, 0], superCell.superlatt[1, 1], superCell.superlatt[2, 2]])

        state = self.initState.copy()
        state = np.random.permutation(state)

        stateGrid = LatGas.gridState(state, self.siteIndtoR, N_units)

        vacIndNow = np.where(state == self.NSpec-1)[0][0]
        print(vacIndNow)

        vacDes = self.vacsiteInd

        stateTransGrid = LatGas.translateState(stateGrid, vacIndNow, vacDes, self.RtoSiteInd, self.siteIndtoR, N_units)

        dR = -self.siteIndtoR[vacDes] + self.siteIndtoR[vacIndNow]
        for R0 in range(N_units[0]):
            for R1 in range(N_units[1]):
                for R2 in range(N_units[2]):
                    R0T = (R0 + dR[0]) % N_units[0]
                    R1T = (R1 + dR[1]) % N_units[1]
                    R2T = (R2 + dR[2]) % N_units[2]
                    assert stateGrid[R0T, R1T, R2T] == stateTransGrid[R0, R1, R2]

    def test_LatGasKMCTraj(self):

        state = self.initState.copy()

        RtoSiteInd, siteIndtoR = self.RtoSiteInd.copy(), self.siteIndtoR.copy()

        N_unit = self.N_units

        SpecRates = np.array(range(1, self.NSpec))

        ijList, dxList = self.ijList.copy(), self.dxList.copy()

        vacSiteInit = self.vacsiteInd
        self.assertEqual(vacSiteInit, 0)

        Nsteps = 50

        X_steps, t_steps, jmpSelectSteps, randSteps, jmpFinSiteList =\
            LatGas.LatGasKMCTraj(state, SpecRates, Nsteps, ijList, dxList,
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
            rates = np.zeros(ijList.shape[0])
            for jmp in range(ijList.shape[0]):
                dxR = dxtoR[jmp]
                # get the exchange site
                REx = (Rvac + dxR) % N_unit
                siteEx = RtoSiteInd[REx[0], REx[1], REx[2]]
                specEx = state0[siteEx]
                rates[jmp] = SpecRates[specEx]
                rateTot += SpecRates[specEx]

            tRun += 1/rateTot

            self.assertAlmostEqual(tRun, t_steps[step])

            rateProbs = rates / rateTot
            rateSum = np.cumsum(rateProbs)
            rn = randSteps[step]
            self.assertEqual(np.searchsorted(rateSum, rn), jmpStep)

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

    def test_Traj_Av(self):

        state1 = self.initState.copy()
        state2 = self.initState.copy()
        state2[1:] = np.random.permutation(state2[1:])

        RtoSiteInd, siteIndtoR = self.RtoSiteInd.copy(), self.siteIndtoR.copy()

        N_unit = self.N_units

        SpecRates = np.array(range(1, self.NSpec))

        ijList, dxList = self.ijList.copy(), self.dxList.copy()

        vacSiteInit = self.vacsiteInd
        self.assertEqual(vacSiteInit, 0)

        Nsteps = 50

        X_steps_1 = np.zeros((Nsteps, self.NSpec, 3))
        t_steps_1 = np.zeros(Nsteps)
        X_steps_2 = np.zeros((Nsteps, self.NSpec, 3))
        t_steps_2 = np.zeros(Nsteps)
        diff = np.zeros((self.NSpec, Nsteps))

        X_steps, t_steps, jmpSelectSteps, randSteps, jmpFinSiteList = \
            LatGas.LatGasKMCTraj(state1, SpecRates, Nsteps, ijList, dxList,
                                 vacSiteInit, N_unit, siteIndtoR, RtoSiteInd)

        X_steps_1[:] = X_steps[:]
        t_steps_1[:] = t_steps[:]
        LatGas.TrajAv(X_steps, t_steps, diff)

        X_steps, t_steps, jmpSelectSteps, randSteps, jmpFinSiteList = \
            LatGas.LatGasKMCTraj(state2, SpecRates, Nsteps, ijList, dxList,
                                 vacSiteInit, N_unit, siteIndtoR, RtoSiteInd)

        X_steps_2[:] = X_steps[:]
        t_steps_2[:] = t_steps[:]
        LatGas.TrajAv(X_steps, t_steps, diff)

        for spec in range(self.NSpec):
            for step in range(Nsteps):
                Traj1R = X_steps_1[step, spec]
                Traj1T = t_steps_1[step]

                Traj1R2by6t = np.dot(Traj1R, Traj1R) / (6*Traj1T)

                Traj2R = X_steps_2[step, spec]
                Traj2T = t_steps_2[step]

                Traj2R2by6t = np.dot(Traj2R, Traj2R) / (6 * Traj2T)

                self.assertAlmostEqual(diff[spec, step], Traj2R2by6t + Traj1R2by6t, msg="{} {} {}".format(diff[spec, step], Traj2R2by6t, Traj1R2by6t))

    def test_getJumpRates(self):
        state = self.initState
        Nsites = state.shape[0]
        siteGPerms = np.zeros((len(self.crys.G), self.initState.shape[0]), dtype=int)
        GList = list(self.crys.G)
        for gInd, g in enumerate(GList):
            for site in range(Nsites):
                ci, Rsite = self.superCell.ciR(site)
                Rnew, ciNew = self.superCell.crys.g_pos(g, Rsite, ci)
                siteNew = self.superCell.index(Rnew, ciNew)[0]
                siteGPerms[gInd, siteNew] = site  # where does the current site go after group op

        # First, we'll check for detailed balance
        # Let's make up some mu and std
        muArray = np.random.rand(self.NSpec - 1)
        stdArray = np.random.rand(self.NSpec - 1)
        # Then let's consider the sites that will be considered
        stringSites = np.random.randint(0, Nsites, size=100)

        state[1:] = np.random.permutation(state[1:])
        assert state[0] == self.NSpec - 1
        assert self.vacsiteInd == 0

        for jmp in range(self.dxList.shape[0]):
            print(jmp)
            jSite = self.ijList[jmp]

            # Check the jump site
            dxR, _ = self.superCell.crys.cart2pos(self.dxList[jmp])
            siteIndSup = self.superCell.index(dxR, (0, 0))[0]
            self.assertEqual(siteIndSup, jSite)

            # do the exchange
            state2 = state.copy()
            spec = state2[jSite]
            state2[self.vacsiteInd] = spec
            state2[jSite] = self.NSpec - 1

            # Now translate sites back
            state2Trans  = np.zeros_like(state2, dtype=int)
            for siteInd in range(Nsites):
                ciSite, Rsite = self.superCell.ciR(siteInd)
                assert ciSite == (0, 0)
                RsiteNew = Rsite - dxR
                siteIndNew, _ = self.superCell.index(RsiteNew, ciSite)
                state2Trans[siteIndNew] = state2[siteInd]

            assert state2Trans[self.vacsiteInd] == self.NSpec - 1

            # Test the rates for detailed balance (since all states have same energy)
            rate1 = LatGas.getJumpRate(state, state2, siteGPerms, stringSites, muArray[spec], stdArray[spec])
            rate2 = LatGas.getJumpRate(state2, state, siteGPerms, stringSites, muArray[spec], stdArray[spec])
            self.assertTrue(np.math.isclose(rate1, rate2))




