import numpy as np
import pickle
import time

from ase.spacegroup import crystal
from ase.build import make_supercell
from ase.io.lammpsdata import write_lammps_data

import os
import subprocess
import glob
import argparse
# Next, write a lammps input script for this run
def write_lammps_input(potPath, etol=1e-7, ftol=0.001):
    lines = ["units \t metal\n",
             "atom_style \t atomic\n",
             "atom_modify \t map array\n",
             "boundary \t p p p\n",
             "atom_modify \t sort 0 0.0\n",
             "read_data \t inp_MC_check_disp.data\n",
             "pair_style \t meam\n",
             "pair_coeff \t * * {0}/library.meam Co Ni Cr Fe Mn {0}/params.meam Co Ni Cr Fe Mn".format(potPath),
             "\n",
             "min_style cg\n",
             "min_modify norm max\n"
             "minimize	\t {0} {1} 1000 1000000\n".format(etol, ftol),
             "\n",
             "variable x equal pe\n",
             "print \"$x\" file Eng_check_disp.txt\n",
             "dump \t 1 all xyz 1 relaxed_conf_check_disp.xyz\n",
             "run 0\n"
             ]

    with open("in.minim", "w") as fl:
        fl.writelines(lines)


def getminDist(x, N_units=5, a0=3.595):
    corners = np.array([[n1*a0, n2*a0, n3*a0] for n1 in (0, N_units) for n2 in (0, N_units) for n3 in (0, N_units)])

    dxPos = np.zeros((corners.shape[0], 3))
    dxNorms = np.zeros(corners.shape[0])

    for cInd in range(corners.shape[0]):
        dx = x - corners[cInd]
        dxPos[cInd, :] = dx
        dxNorms[cInd] = np.linalg.norm(dx)

    mn = np.argmin(dxNorms)
    return dxPos[mn]

def check_atomic_displacements(sup, N_units=5, a0=3.595, threshold=1.0):
    with open("relaxed_conf_check_disp.xyz", "r") as fl:
        lines = fl.readlines()

    Natoms_saved = int(lines[0].split()[0])
    Natoms = len(sup)
    assert Natoms_saved == Natoms

    mapping = True
    elems = ["Co", "Ni", "Cr", "Fe", "Mn"]
    lines = lines[2:]  # the atoms start from the third line onwards
    for at in range(Natoms):
        # check that we have the correct atom
        spec = int(lines[at].split()[0])
        el = elems[spec - 1]
        assert sup[at].symbol == el

        # Then get the starting position
        pos_at_init = sup[at].position
        pos_at_fin = np.array([float(x) for x in lines[at].split()[1:]])

        pos_at_init_min = getminDist(pos_at_init, N_units=N_units, a0=a0)
        pos_at_fin_min = getminDist(pos_at_fin, N_units=N_units, a0=a0)

        displacement = np.linalg.norm(pos_at_fin_min - pos_at_init_min)

        if displacement > threshold:
            mapping = False

    return mapping

def main(args):

    elems = ["Co", "Ni", "Cr", "Fe", "Mn"]


    badCheckpoints = []
    badCheckpoints_energies = []
    write_lammps_input(args.PotPath, etol=args.EnTol, ftol=args.ForceTol)

    if not args.NoSrun:
        cmdString = "srun --ntasks=1 --cpus-per-task=1 $LMPPATH/lmp -in in.minim > out_check_disp.txt"
    else:
        cmdString = "$LMPPATH/lmp -in in.minim > out_check_disp.txt"

    start = args.Start
    Nsamps = args.Nckp
    interval = args.Interval
    En = np.load("Eng_all_steps.npy")

    for ckp in range(start, start + (Nsamps - 1) * interval + 1, interval):

        with open("chkpt/supercell_{}.pkl".format(ckp), "rb") as fl:
            sup = pickle.load(fl)

        with open("Check_disp_run.txt", "a") as fl:
            fl.write("Loaded checkpoint: {}\n".format(ckp))

        write_lammps_data("inp_MC_check_disp.data", sup, specorder=elems)

        cmd = subprocess.run(cmdString, shell=True, check=True)

        with open("Eng_check_disp.txt", "r") as fl_en:
            e = fl_en.readline().split()[0]
            e = float(e)

        e_check = En[ckp]
        assert np.math.isclose(e_check, e, rel_tol=0, abs_tol=1e-6)

        check_good = check_atomic_displacements(sup, N_units=args.Nunits, a0=args.LatPar, threshold=args.Threshold)

        if not check_good:
            badCheckpoints.append(ckp)
            badCheckpoints_energies.append(e)

    if len(badCheckpoints) > 0:
        with open("NonLatticeCheckPoints.txt", "w") as fl:
            fl.write("Chkpt \t Energy\n")
            for ckp, en in zip(badCheckpoints, badCheckpoints_energies):
                fl.write("{} \t {}\n".format(ckp, en))

if __name__ == "__main__":
    # N_units, a0, NoVac, T, N_swap, N_eqb, N_save

    parser = argparse.ArgumentParser(description="Input parameters for Metropolis Monte Carlo simulations with ASE.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-pp", "--PotPath", metavar="/path/to/potential/file", type=str,
                        help="Path to the LAMMPS MEAM potential.")

    parser.add_argument("-u", "--Nunits", metavar="int", type=int, nargs="+", default=[5, 5, 5],
                        help="Number of unit cells in the supercell.")

    parser.add_argument("-a0", "--LatPar", metavar="float", type=float, default=3.595,
                        help="Lattice parameter.")

    parser.add_argument("-th", "--Threshold", metavar="float", type=float, default=1.0,
                        help="Lattice parameter.")

    parser.add_argument("-ftol", "--ForceTol", metavar="float", type=float, default=0.001,
                        help="Force tolerance to stop CG minimization of energies.")

    parser.add_argument("-etol", "--EnTol", metavar="float", type=float, default=0.0,
                        help="Relative energy change tolerance to stop CG minimization of energies.")

    parser.add_argument("-nosr", "--NoSrun", action="store_true",
                        help="Whether to use srun on not to launch lammps jobs.")

    parser.add_argument("-st", "--Start", metavar="int", type=int, default=10000,
                        help="Which checkpoints to start from.")

    parser.add_argument("-i", "--Interval", metavar="int", type=int, default=1000,
                        help="At what interval to load the checkpoints starting from the first.")

    parser.add_argument("-ns", "--Nckp", metavar="int", type=int, default=100,
                        help="How many total checkpoints to check (The starting sample will be included in the count).")

    args = parser.parse_args()

    if args.DumpArguments:
        print("Dumping arguments to: {}".format(args.DumpFile))
        opts = vars(args)
        with open(os.getcwd() + '/' + args.DumpFile, "w") as fl:
            for key, val in opts.items():
                fl.write("{}:\t{}\n".format(key, val))

    main(args)







