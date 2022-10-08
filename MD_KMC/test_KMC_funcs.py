import unittest
import numpy as np
from ase.spacegroup import crystal
from ase.build import make_supercell
from ase.io.lammpsdata import write_lammps_data, read_lammps_data
import sys
from tqdm import tqdm
from KMC_funcs import write_init_states, write_final_states, updateStates, getJumpSelects

class test_KMC_funcs(unittest.TestCase):
    def setUp(self):
        self.N_units = 8
        # Create an FCC primitive unit cell
        self.a = 3.59
        a = self.a
        self.fcc = crystal('Ni', [(0, 0, 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90], primitive_cell=True)

        # Form a supercell with a vacancy at the centre
        self.superlatt = np.identity(3) * self.N_units
        self.superFCCASE = make_supercell(self.fcc, self.superlatt)
        Nsites = len(self.superFCCASE.get_positions())
        self.Nsites = Nsites

        # Write the lammps box
        write_lammps_data("lammpsBox.txt", self.superFCCASE)
        self.Sup_lammps_unrelax_coords = read_lammps_data("lammpsBox.txt", style="atomic")
        self.assertEqual(len(self.Sup_lammps_unrelax_coords), Nsites)

        # Store the lammps-basis coordinate of each site
        self.SiteIndToCartPos = np.zeros((Nsites, 3))
        for i in range(Nsites):
            self.SiteIndToCartPos[i, :] = self.Sup_lammps_unrelax_coords[i].position[:]

        # give some random occupancies to 2 random trajectories
        self.NtrajTest = 5
        self.SiteIndToSpec = np.random.randint(1, 6, (self.NtrajTest, self.SiteIndToCartPos.shape[0]))
        self.vacSiteInd = np.random.randint(0, Nsites, self.NtrajTest)

        for traj in range(self.NtrajTest):
            self.SiteIndToSpec[traj, self.vacSiteInd[traj]] = 0

        np.save("TestOccs.npy", self.SiteIndToSpec)

    def test_write_init_states(self):

        # First, write the output
        with open("lammpsBox.txt", "r") as fl:
            Initlines = fl.readlines()

        siteCount = int(Initlines[2].split()[0])
        InitialSpecCount = int(Initlines[3].split()[0])

        self.assertEqual(siteCount, self.Nsites)
        self.assertEqual(InitialSpecCount, 1)  # should have only the Nickel case

        specs = np.unique(self.SiteIndToSpec[0])
        Nspecs = len(specs) - 1

        Initlines[2] = "{} \t atoms\n".format(self.Nsites - 1)
        Initlines[3] = "{} atom types\n".format(Nspecs - 1)

        # Write out the initial state
        write_init_states(self.SiteIndToSpec, self.SiteIndToCartPos, self.vacSiteInd, Initlines)

        # Now read in the initial states and spot the coordinates
        for traj in range(self.NtrajTest):
            vacInd = self.vacSiteInd[traj]
            # read the input file
            with open("initial_{}.data".format(traj), "r") as fl:
                fileLines = fl.readlines()

            self.assertEqual(fileLines[:12], Initlines[:12])
            atomLines = fileLines[12:]
            self.assertEqual(len(atomLines), self.Nsites - 1)
            # except the vacancy site, things should have been skipped
            for lineInd, writtenSite in enumerate(atomLines):
                splitInfo = writtenSite.split()
                lammpsSiteInd = int(splitInfo[0])
                WrittenSpec = int(splitInfo[1])
                x = float(splitInfo[2])
                y = float(splitInfo[3])
                z = float(splitInfo[4])

                if lineInd >= vacInd:
                    mainSiteInd = lammpsSiteInd
                else:
                    mainSiteInd = lammpsSiteInd - 1

                self.assertEqual(WrittenSpec, self.SiteIndToSpec[traj, mainSiteInd],
                                 msg="{} {}\n{}".format(traj, lineInd + 1, writtenSite))
                self.assertEqual(x, self.SiteIndToCartPos[mainSiteInd, 0])
                self.assertEqual(y, self.SiteIndToCartPos[mainSiteInd, 1])
                self.assertEqual(z, self.SiteIndToCartPos[mainSiteInd, 2])







