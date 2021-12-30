import numpy as np
from ase.spacegroup import crystal
from ase.build import make_supercell
import pickle
from ase.io.lammpsdata import write_lammps_data
import sys

args = list(sys.argv)
N_units = int(args[1])

# Create an FCC primitive unit cell
a = 3.59
fcc = crystal('Ni', [(0, 0, 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90], primitive_cell=True)

# Form a supercell with a vacancy at the centre
superlatt = np.identity(3) * N_units
superFCC = make_supercell(fcc, superlatt)
Nsites = len(superFCC.get_positions())

# Get the neighborhood of each site
# Next, we need to build the jump neighborhood of every site
with open("CrysDat/jnetFCC.pkl", "rb") as fl:
    jnetFCC = pickle.load(fl)
jumpsFCC = [dx*a for (i, j), dx in jnetFCC[0]]