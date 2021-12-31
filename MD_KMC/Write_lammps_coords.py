import numpy as np
from ase.spacegroup import crystal
from ase.build import make_supercell
import pickle
from ase.io.lammpsdata import write_lammps_data, read_lammps_data
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

write_lammps_data("lammpsBox.txt", superFCC)

# get the "a"-normalized jump vectors
with open("CrysDat/jnetFCC.pkl", "rb") as fl:
    jnetFCC = pickle.load(fl)

# multiply by "a" to get the jump vectors.
jumpsFCC = [dx*a for (i, j), dx in jnetFCC[0]]
Sup_lammps_unrelax_coords = read_lammps_data("lammpsBox.txt", style="atomic")

# Save the lammps-basis coordinate of each site
SiteIndToCartPos = np.zeros((Nsites, 3))
for i in range(Nsites):
    SiteIndToCartPos[i, :] = Sup_lammps_unrelax_coords[i-1].position[:]
np.save("SiteIndToLmpCartPos.npy", SiteIndToCartPos)

# Next, save the neighborhood of each site

# Then form the neighborhood of all sites
siteIndtoNgbSiteInd = np.zeros((Nsites, len(jumpsFCC)), dtype=int)
for atMain in superFCC:
    CurrentSiteId = atMain.index
    for ngbInd, dx in enumerate(jumpsFCC):
        foundCount = 0
        dxNgb = atMain.position[:] + dx

        # Apply periodic boundary condition
        dxR = np.dot(np.linalg.inv(fcc.cell[:]), dxNgb).round(decimals=4).astype(int) % N_units
        posInCell = np.dot(fcc.cell[:], dxR)
        atIndex = None
        for at in superFCC:
            if np.allclose(at.position, posInCell):
                foundCount += 1
                atIndex = at.index
        assert foundCount == 1, "{} {}".format(dxR, dxNgb)
        assert atIndex is not None
        siteIndtoNgbSiteInd[CurrentSiteId, ngbInd] = atIndex

# Next, test the neighborhood indices with onsager results - they should match
# Either way, one confirms the other
NNsites = np.load("CrysDat/NNsites_sitewise.npy")
for i in range(Nsites):
    assert np.array_equal(np.sort(NNsites[1:, i]), np.sort(siteIndtoNgbSiteInd[i, :])), "{}\n{}".format(NNsites[1:, i], siteIndtoNgbSiteInd[i, :])
