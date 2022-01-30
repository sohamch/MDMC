import numpy as np
import subprocess
import pickle
import time
from ase.spacegroup import crystal
from ase.build import make_supercell
from ase.io.lammpsdata import write_lammps_data, read_lammps_data
from Init_state_MC import write_lammps_input, MC_Run
from scipy.constants import physical_constants
import unittest
import sys

class Test_MC(unittest.TestCase):
    def setUp(self):
        kB = physical_constants["Boltzmann constant in eV/K"][0]
        T = 1073
        N_therm = 2  # thermalization steps (until this many moves accepted)
        N_units = 8  # dimensions of unit cell
        N_proc = 1  # No. of procs to parallelize over
        jobID = 0

        __test__ = False
        chk_cmd = subprocess.Popen("mkdir chkpt", shell=True)
        rt = chk_cmd.wait()
        assert rt == 0

        # Create an FCC primitive unit cell
        a = 3.59
        fcc = crystal('Ni', [(0, 0, 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90], primitive_cell=True)

        # Form a supercell with a vacancy at the centre
        superlatt = np.identity(3) * N_units
        superFCC = make_supercell(fcc, superlatt)
        Nsites = len(superFCC.get_positions())
        # randomize occupancies of the sites
        Nperm = 10
        Indices = np.arange(Nsites)
        for i in range(Nperm):
            Indices = np.random.permutation(Indices)

        NSpec = 5
        partition = Nsites // NSpec

        elems = ["Co", "Ni", "Cr", "Fe", "Mn"]
        elemsToNum = {}
        for elemInd, el in enumerate(elems):
            elemsToNum[el] = elemInd + 1

        for i in range(NSpec):
            for at_Ind in range(i * partition, (i + 1) * partition):
                permInd = Indices[at_Ind]
                superFCC[permInd].symbol = elems[i]
        del (superFCC[0])
        Natoms = len(superFCC)

        # save the supercell: will be useful for getting site positions
        with open("superInitial_{}.pkl".format(jobID), "wb") as fl:
            pickle.dump(superFCC, fl)

        # First thermalize the starting state
        write_lammps_input(jobID)
        start = time.time()

        self.supercell_init = superFCC.copy()

        self.N_total, self.N_accept, self.Eng_steps_accept, self.Eng_steps_all, self.rand_steps, self.swap_steps =\
            MC_Run(N_therm, superFCC, N_proc, jobID, elems, __test__=__test__)