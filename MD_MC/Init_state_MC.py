#!/usr/bin/env python
# coding: utf-8
import numpy as np
import subprocess
import pickle
import time
from ase.spacegroup import crystal
from ase.build import make_supercell
from ase.io.lammpsdata import write_lammps_data, read_lammps_data
import sys
from scipy.constants import physical_constants

kB = physical_constants["Boltzmann constant in eV/K"][0]

args = list(sys.argv)
T = float(args[1])
N_therm = int(args[2]) # thermalization steps (until this many moves accepted)
N_swaps = int(args[3]) # intervals to draw samples from after thermalization (until this many moves accepted)
N_units = int(args[4]) # dimensions of unit cell
N_proc = int(args[5]) # No. of procs to parallelize over
N_samples = int(args[6]) # How many samples we want to draw from this run
jobID = int(args[7])

__test__ = False

# Create an FCC primitive unit cell
a = 3.59
fcc = crystal('Ni', [(0, 0, 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90], primitive_cell=True)

# Form a supercell with a vacancy at the centre
superlatt = np.identity(3)*N_units
superFCC = make_supercell(fcc, superlatt)
Nsites = len(superFCC.get_positions())
# randomize occupancies of the sites
Nperm = 10
Indices = np.arange(Nsites)
for i in range(Nperm):
    Indices = np.random.permutation(Indices)

NSpec = 5
partition = Nsites//NSpec

elems = ["Co", "Ni", "Cr", "Fe", "Mn"]
elemsToNum = {}
for elemInd, el in enumerate(elems):
    elemsToNum[el] = elemInd + 1
    
for i in range(NSpec):
    for at_Ind in range(i*partition, (i+1)*partition):
        permInd = Indices[at_Ind]
        superFCC[permInd].symbol = elems[i]
del(superFCC[0])
Natoms = len(superFCC)

# save the supercell: will be useful for getting site positions
with open("superInitial_{}.pkl".format(jobID),"wb") as fl:
    pickle.dump(superFCC, fl)
# Write the supercell as a lammps file
write_lammps_data("lammpsCoords.txt", superFCC, specorder=elems)

# First, we write a lammps input script for this run

# Next, we write the MC loop
def MC_Run(SwapRun, ASE_Super, Nprocs, serial=True):
    if serial:
        cmdString = "$LMPPATH/lmp -in in_{0}.minim > out_{0}.txt".format(jobID)
    else:
        cmdString = "mpirun -np {0} $LMPPATH/lmp -in in_{1}.minim > out_{1}.txt".format(Nprocs, jobID)

    N_accept = 0
    N_total = 0
    cond=True

    # write the supercell as a lammps file
    write_lammps_data("inp_MC_{0}.data".format(jobID), ASE_Super, specorder=elems)

    # evaluate the energy
    cmd = subprocess.Popen(cmdString, shell=True)
    rt = cmd.wait()
    assert rt == 0

    # read the energy
    with open("Eng_{0}.txt".format(jobID), "r") as fl_en:
        e1 = fl_en.readline().split()[0]
        e1 = float(e1)

    while cond:
        if __test__:
            write_lammps_data("inp_MC_init_{0}.data".format(jobID), ASE_Super, specorder=elems)
            print("e1: {}".format(e1))
        # Now randomize the atomic occupancies
        site1 = np.random.randint(0, Natoms)
        site2 = np.random.randint(0, Natoms)
        while ASE_Super[site1].symbol == ASE_Super[site2].symbol:
            site1 = np.random.randint(0, Natoms)
            site2 = np.random.randint(0, Natoms)

        if __test__:
            print(site1, site2)

        # change the occupancies
        tmp = ASE_Super[site1].symbol
        ASE_Super[site1].symbol = ASE_Super[site2].symbol
        ASE_Super[site2].symbol = tmp

        # write the supercell again as a lammps file
        write_lammps_data("inp_MC_{0}.data".format(jobID), ASE_Super, specorder=elems)
        if __test__:
            write_lammps_data("inp_MC_final_{0}.data".format(jobID), ASE_Super, specorder=elems)

        # evaluate the energy
        cmd = subprocess.Popen(cmdString, shell=True)
        rt = cmd.wait()
        assert rt == 0
        # read the energy
        with open("Eng_{0}.txt".format(jobID), "r") as fl_en:
            e2 = fl_en.readline().split()[0]
            e2 = float(e2)

        # make decision
        de = e2 - e1
        if __test__:
            print("e2: {}".format(e2))
            print("de: {}".format(de))
        rn = np.random.rand()
        if __test__:
            print("random number: {}".format(rn))
            print("relative prob (e2:e1): {}".format(np.exp(-de/(kB*T))))

        if rn < np.exp(-de/(kB*T)):
            # Then accept the move
            N_accept += 1
            e1 = e2
        else:
            # reject the move by reverting the occupancies
            tmp = ASE_Super[site1].symbol
            ASE_Super[site1].symbol = ASE_Super[site2].symbol
            ASE_Super[site2].symbol = tmp

        if __test__:
            write_lammps_data("Result_{0}.data".format(jobID), ASE_Super, specorder=elems)

        N_total += 1
        if __test__:
            cond = N_total < 1
        else:
            cond = N_accept <= SwapRun

    return N_total, N_accept

# First thermalize the starting state
start = time.time()
N_total, N_accept = MC_Run(N_therm, superFCC, N_proc)
end = time.time()
print("Thermalization Run acceptance ratio : {}".format(N_accept/N_total))
print("Thermalization Time Per iteration : {}".format((end-start)/N_total))

if not __test__:
    occs = np.zeros((N_samples, Nsites), dtype=np.int16)
    occs[:, 0] = -1 # The vacancy is always at 0,0,0
    accept_ratios = np.zeros(N_samples)
    # Now draw samples
    start = time.time()
    for smp in range(N_samples):
        # Update the state
        N_total, N_accept = MC_Run(N_swaps, superFCC, N_proc)
        accept_ratios[smp] = (1.0*N_accept)/N_total
        # store the occupancies
        for at in superFCC:
            idx = at.index
            occs[smp, idx+1] = elemsToNum[at.symbol]
    end = time.time()
    np.save("Occs_{0}.npy".format(jobID), occs)
    print("{} samples drawn with {} swaps. Time: {:.4f} minutes".format(N_samples, N_swaps, (end-start)/60.))
