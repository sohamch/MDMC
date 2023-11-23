import numpy as np
import h5py
import unittest
from onsager import crystal, supercell
from Scaled_bias_funcs import ScaledBiasBasis, ScaledResBias, Expand_scaled_bias
from tqdm import tqdm

class TestScaledBias(unittest.TestCase):

    def setUp(self):
        DataPath = "../Symm_Network/Test_Data/testData_HEA_MEAM_orthogonal.h5"

        with h5py.File(DataPath, "r") as fl:
            self.init_step1 = np.array(fl["InitStates"])
            self.fin_step1 = np.array(fl["FinStates"])
            self.AllRates_step1 = np.array(fl["AllJumpRates_Init"])
            self.AllRates_step2 = np.array(fl["AllJumpRates_Fin"])
            self.escapeRates_step1 = np.array(fl["rates"])
            self.escapeRates_step2 = np.array(fl["rates_step2"])
            self.JumpSelects_step1 = np.array(fl["JumpSelects"])
            self.dispList_step1 = np.array(fl["SpecDisps"])

        with h5py.File("../CrysDat_FCC/CrystData_ortho_5_cube.h5", "r") as fl:
            lattice = np.array(fl["Lattice_basis_vectors"])
            superlatt = np.array(fl["SuperLatt"])
            basis_cubic = np.array(fl["basis_sites"])
            dxList = np.array(fl["dxList_1nn"])
            NNList = np.array(fl["NNsiteList_sitewise"])
            jumpNewSties = np.array(fl["JumpSiteIndexPermutation"])
            GroupOpLatticeCartRotMatrices = np.array(fl["GroupOpLatticeCartRotMatrices"])
            GpermNNIdx = np.array(fl["GroupNNPermutation"])

        crys = crystal.Crystal(lattice=lattice, basis=[[b for b in basis_cubic]], chemistry=["A"], noreduce=True)
        self.superCell = supercell.ClusterSupercell(crys, superlatt)

        self.dxList = dxList
        self.NNList = NNList

        self.JumpProbs_step1 = self.AllRates_step1 / (self.escapeRates_step1.reshape(self.init_step1.shape[0], 1))
        self.JumpProbs_step2 = self.AllRates_step2 / (self.escapeRates_step2.reshape(self.init_step1.shape[0], 1))

        self.a0 = np.linalg.norm(self.dispList_step1[0, 0, :]) / np.linalg.norm(dxList[0])

        assert np.allclose(np.sum(self.JumpProbs_step1, axis=1), 1.0)
        assert np.allclose(np.sum(self.JumpProbs_step2, axis=1), 1.0)

        self.vacSpec = 0
        assert np.all(self.init_step1[:, 0] == self.vacSpec)
        assert np.all(self.fin_step1[:, 0] == self.vacSpec)

        self.Nspec = np.unique(self.init_step1[0]).shape[0]

        print("Done Setting up test system. Lattice parameter: {:.3f}. No. of species: {}".format(self.a0, self.Nspec))

    def test_ScaledBiasBasis(self):

        NNvacList = self.NNList[1:, 0]
        Basis_spec_init_states = ScaledBiasBasis(self.init_step1, self.dxList * self.a0, NNvacList,
                                                self.JumpProbs_step1, self.vacSpec)

        Basis_spec_fin_states = ScaledBiasBasis(self.fin_step1, self.dxList * self.a0, NNvacList,
                                                self.JumpProbs_step2, self.vacSpec)

        print("checking average displacements", flush=True)
        for stateInd in tqdm(range(self.init_step1.shape[0])):
            state = self.init_step1[stateInd]

            for sp in range(self.Nspec):
                phi_spec = np.zeros(3)
                if sp != self.vacSpec:
                    for jmp in range(self.dxList.shape[0]):
                        if state[NNvacList[jmp]] == sp:
                            phi_spec -= self.JumpProbs_step1[stateInd, jmp] * self.dxList[jmp] * self.a0
                else:
                    for jmp in range(self.dxList.shape[0]):
                        phi_spec += self.JumpProbs_step1[stateInd, jmp] * self.dxList[jmp] * self.a0

                self.assertTrue(np.allclose(Basis_spec_init_states[sp, stateInd], phi_spec))

            #  Now the final state
            state = self.fin_step1[stateInd]
            for sp in range(self.Nspec):
                phi_spec = np.zeros(3)
                if sp != self.vacSpec:
                    for jmp in range(self.dxList.shape[0]):
                        if state[NNvacList[jmp]] == sp:
                            phi_spec -= self.JumpProbs_step2[stateInd, jmp] * self.dxList[jmp] * self.a0
                else:
                    for jmp in range(self.dxList.shape[0]):
                        phi_spec += self.JumpProbs_step2[stateInd, jmp] * self.dxList[jmp] * self.a0

                self.assertTrue(np.allclose(Basis_spec_fin_states[sp, stateInd], phi_spec, atol=1e-10, rtol=0))


    def test_ScaledResBias_non_vac(self):
        NNvacList = self.NNList[1:, 0]

        # Pick a random non-vacancy species
        specExpand = np.random.randint(0, self.Nspec)
        while specExpand == self.vacSpec:
            specExpand = np.random.randint(0, self.Nspec)

        # set  random relaxations for it
        y_st = np.random.rand(self.init_step1.shape[0], 3)
        y_st_exits = np.random.rand(self.init_step1.shape[0] * self.dxList.shape[0], 3)

        # Compute the residual bias basis functions
        Basis_spec_init_states = ScaledResBias(self.init_step1, y_st, y_st_exits, self.dxList * self.a0,
                                               NNvacList, self.JumpProbs_step1, self.vacSpec, specExpand)

        Basis_spec_fin_states = ScaledResBias(self.fin_step1, y_st, y_st_exits, self.dxList * self.a0,
                                               NNvacList, self.JumpProbs_step2, self.vacSpec, specExpand)

        # check the residual displacements
        zeroVec  = np.zeros(3)
        print("checking average residual displacements", flush=True)

        for lst in [(self.init_step1, self.JumpProbs_step1, Basis_spec_init_states),
                    (self.fin_step1, self.JumpProbs_step2, Basis_spec_fin_states)]:

            stateList = lst[0]
            jProbs = lst[1]
            basisToCheck = lst[2]

            for stateInd in tqdm(range(stateList.shape[0])):
                state = stateList[stateInd]

                for sp in range(self.Nspec):
                    phi_spec = np.zeros(3)
                    if sp == self.vacSpec:
                        for jmp in range(self.dxList.shape[0]):
                            phi_spec += jProbs[stateInd, jmp] * self.dxList[jmp] * self.a0

                    else:
                        if sp == specExpand:
                            for jmp in range(self.dxList.shape[0]):
                                if state[NNvacList[jmp]] == sp:
                                    dx = -self.dxList[jmp] * self.a0
                                else:
                                    dx = zeroVec

                                dy = y_st_exits[stateInd * self.dxList.shape[0] + jmp] - y_st[stateInd]
                                phi_spec += jProbs[stateInd, jmp] * (dx + dy)

                        else:
                            for jmp in range(self.dxList.shape[0]):
                                if state[NNvacList[jmp]] == sp:
                                    phi_spec -= jProbs[stateInd, jmp] * self.dxList[jmp] * self.a0

                    self.assertTrue(np.allclose(basisToCheck[sp, stateInd], phi_spec, atol=1e-10, rtol=0),
                                    msg="sp: {}, vacSpec:{}, spExp: {}".format(sp, self.vacSpec, specExpand))

    def test_ScaledResBias_vac(self):
        NNvacList = self.NNList[1:, 0]

        specExpand = self.vacSpec
        # set  random relaxations
        y_st = np.random.rand(self.init_step1.shape[0], 3)
        y_st_exits = np.random.rand(self.init_step1.shape[0] * self.dxList.shape[0], 3)

        # Compute the scaled residual bias basis functions
        Basis_spec_init_states = ScaledResBias(self.init_step1, y_st, y_st_exits, self.dxList * self.a0,
                                               NNvacList, self.JumpProbs_step1, self.vacSpec, specExpand)

        Basis_spec_fin_states = ScaledResBias(self.fin_step1, y_st, y_st_exits, self.dxList * self.a0,
                                               NNvacList, self.JumpProbs_step2, self.vacSpec, specExpand)

        # check the residual displacements
        print("checking average residual displacements", flush=True)
        for lst in [(self.init_step1, self.JumpProbs_step1, Basis_spec_init_states),
                    (self.fin_step1, self.JumpProbs_step2, Basis_spec_fin_states)]:

            stateList = lst[0]
            jProbs = lst[1]
            basisToCheck = lst[2]

            for stateInd in tqdm(range(stateList.shape[0])):
                state = stateList[stateInd]

                for sp in range(self.Nspec):
                    phi_spec = np.zeros(3)
                    if sp == self.vacSpec:
                        for jmp in range(self.dxList.shape[0]):
                            dy = y_st_exits[stateInd * self.dxList.shape[0] + jmp] - y_st[stateInd]
                            dx = self.dxList[jmp] * self.a0
                            phi_spec += jProbs[stateInd, jmp] * (dx + dy)

                    else:
                        for jmp in range(self.dxList.shape[0]):
                            if state[NNvacList[jmp]] == sp:
                                phi_spec -= jProbs[stateInd, jmp] * self.dxList[jmp] * self.a0

                    self.assertTrue(np.allclose(basisToCheck[sp, stateInd], phi_spec), msg="sp: {}, vacSpec:{}, spExp: {}".format(sp, self.vacSpec, specExpand))


    def test_Expand_scaled_bias(self):
        NNvacList = self.NNList[1:, 0]
        Basis_spec_init_states = ScaledBiasBasis(self.init_step1, self.dxList * self.a0, NNvacList,
                                                 self.JumpProbs_step1, self.vacSpec)

        Basis_spec_fin_states = ScaledBiasBasis(self.fin_step1, self.dxList * self.a0, NNvacList,
                                                self.JumpProbs_step2, self.vacSpec)

        specExpand = np.random.randint(0, self.Nspec)
        disp_spec_expand = self.dispList_step1[:, specExpand, :]
        Ntrain = 500

        Wbar, bBar = Expand_scaled_bias(Basis_spec_init_states, Basis_spec_fin_states, disp_spec_expand, self.escapeRates_step1, Ntrain)

        Wbar_comp = np.zeros((self.Nspec, self.Nspec))
        bBar_comp = np.zeros(self.Nspec)

        for stateInd in range(Ntrain):
            basis_state_1 = Basis_spec_init_states[:, stateInd, :]
            basis_state_2 = Basis_spec_fin_states[:, stateInd, :]

            Wstate = np.zeros_like(Wbar_comp)
            bstate = np.zeros_like(bBar_comp)
            for sp1 in range(self.Nspec):
                del_basi_sp1 = basis_state_2[sp1] - basis_state_1[sp1]

                bstate[sp1] = self.escapeRates_step1[stateInd] * np.dot(disp_spec_expand[stateInd], del_basi_sp1)

                for sp2 in range(self.Nspec):
                    del_basi_sp2 = basis_state_2[sp2] - basis_state_1[sp2]
                    Wstate[sp1, sp2] = self.escapeRates_step1[stateInd] * np.dot(del_basi_sp2, del_basi_sp1)

            Wbar_comp += Wstate
            bBar_comp += bstate

        self.assertTrue(np.allclose(Wbar, Wbar_comp, rtol=0, atol=1e-10))
        self.assertTrue(np.allclose(bBar, bBar_comp, rtol=0, atol=1e-10))
