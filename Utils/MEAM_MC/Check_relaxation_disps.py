import numpy as np
import pickle

from ase.spacegroup import crystal
from ase.build import make_supercell
from ase.io.lammpsdata import write_lammps_data

import os
import subprocess
import argparse

def write_lammps_input(potPath, etol=1e-7, ftol=0.001, threshold=1.0):
    lines = ["units \t metal\n",
             "atom_style \t atomic\n",
             "atom_modify \t map array\n",
             "boundary \t p p p\n",
             "atom_modify \t sort 0 0.0\n",
             "read_data \t inp_MC_check_disp.data\n",
             "pair_style \t meam\n",
             "pair_coeff \t * * {0}/library.meam Co Ni Cr Fe Mn {0}/params.meam Co Ni Cr Fe Mn".format(potPath),
             "\n",
             "min_style \t cg\n",
             "min_modify \t norm max\n\n",

             "variable \t Drel equal {}\n".format(threshold),
             "compute \t dsp all displace/atom\n\n",

             "minimize	\t {0} {1} 1000 1000000\n\n".format(etol, ftol),

             "variable x equal pe\n",
             "print \"$x\" file Eng_check_disp.txt\n\n",

             "dump \t 1 all custom 1 disps.dump id type c_dsp[4]\n",
             "dump_modify \t 1 append no thresh c_dsp[4] > ${Drel}\n",
             "run 0\n"
             ]

    with open("in_check_dips.minim", "w") as fl:
        fl.writelines(lines)

def main(args):

    elems = ["Co", "Ni", "Cr", "Fe", "Mn"]

    badCheckpoints = []

    write_lammps_input(args.PotPath, etol=args.EnTol, ftol=args.ForceTol, threshold=args.Threshold)

    if not args.NoSrun:
        cmdString = "srun --ntasks=1 --cpus-per-task=1 $LMPPATH/lmp -in in_check_dips.minim > out_check_disp.txt"
    else:
        cmdString = "$LMPPATH/lmp -in in_check_dips.minim > out_check_disp.txt"

    start = args.Start
    Nsamps = args.Nckp
    interval = args.Interval
    En = np.load("Eng_all_steps.npy")

    with open("Check_disp_run.txt", "w") as fl:
        fl.write("")

    for ckp in range(start, start + (Nsamps - 1) * interval + 1, interval):

        with open("chkpt/supercell_{}.pkl".format(ckp), "rb") as fl:
            sup = pickle.load(fl)

        with open("Check_disp_run.txt", "a") as fl:
            fl.write("Loaded checkpoint: {}\n".format(ckp))

        write_lammps_data("inp_MC_check_disp.data", sup, specorder=elems)

        subprocess.run(cmdString, shell=True, check=True)

        with open("Eng_check_disp.txt", "r") as fl_en:
            e = fl_en.readline().split()[0]
            e = float(e)

        e_check = En[ckp]
        assert np.math.isclose(e_check, e, rel_tol=0, abs_tol=1e-6)

        with open("disps.dump", "r") as fl:
            lines = fl.readlines()

        if len(lines) == 9:
            assert lines[-1].split()[0] == "ITEM:"

        else:
            badCheckpoints.append(ckp)

    if len(badCheckpoints) > 0:
        with open("NonLatticeCheckPoints.txt", "w") as fl:
            s = ""
            for ckp in badCheckpoints:
                s += "{} ".format(ckp)

            fl.write(s)

if __name__ == "__main__":

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

    parser.add_argument("-dmp", "--DumpArguments", action="store_true",
                        help="Whether to dump all the parsed arguments into a text file.")

    parser.add_argument("-dpf", "--DumpFile", metavar="string", type=str, default="ArgFiles",
                        help="The file in the run directory where all the args will be dumped.")

    args = parser.parse_args()

    if args.DumpArguments:
        print("Dumping arguments to: {}".format(args.DumpFile))
        opts = vars(args)
        with open(os.getcwd() + '/' + args.DumpFile, "w") as fl:
            for key, val in opts.items():
                fl.write("{}:\t{}\n".format(key, val))

    main(args)







