# Example script to set up KMC calculation by making the necessary files containing lammps coordinates.

import numpy as np
from ase.spacegroup import crystal
from ase.build import make_supercell
import pickle
from ase.io.lammpsdata import write_lammps_data, read_lammps_data
import sys
from tqdm import tqdm

args = list(sys.argv)
N_units = 8

# Create an FCC primitive unit cell
a = 3.59
fcc = crystal('Ni', [(0, 0, 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90], primitive_cell=True)

# Form a supercell with a vacancy at the centre
superlatt = np.identity(3) * N_units
superFCC = make_supercell(fcc, superlatt)
Nsites = len(superFCC.get_positions())

write_lammps_data("lammpsBox.txt", superFCC)
Sup_lammps_unrelax_coords = read_lammps_data("lammpsBox.txt", style="atomic")

# Save the lammps-basis coordinate of each site
SiteIndToCartPos = np.zeros((Nsites, 3))
for i in range(Nsites):
    SiteIndToCartPos[i, :] = Sup_lammps_unrelax_coords[i].position[:]
np.save("SiteIndToLmpCartPos.npy", SiteIndToCartPos)

print(SiteIndToCartPos[:10])
