#!/usr/bin/env python
# coding: utf-8

# In[5]:
from ase.spacegroup import crystal
from ase.build import make_supercell
from ase.io.lammpsdata import write_lammps_data, read_lammps_data

from KMC_funcs import *
import os
from scipy.constants import physical_constants
kB = physical_constants["Boltzmann constant in eV/K"][0]

import argparse

# Need to get rid of these argument
# NImage = 3
RunPath = os.getcwd()+'/'
print("Running from : " + RunPath)

def CreateLammpsData(N_units, a, prim=False):
    # Create an FCC primitive unit cell
    fcc = crystal('Ni', [(0, 0, 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90], primitive_cell=prim)

    # Form a supercell
    superlatt = np.identity(3) * N_units
    superFCC = make_supercell(fcc, superlatt)
    Nsites = len(superFCC.get_positions())

    write_lammps_data("lammpsBox.txt", superFCC)
    Sup_lammps_unrelax_coords = read_lammps_data("lammpsBox.txt", style="atomic")

    # Save the lammps-basis coordinate of each site
    SiteIndToCartPos = np.zeros((Nsites, 3))
    for i in range(Nsites):
        SiteIndToCartPos[i, :] = Sup_lammps_unrelax_coords[i].position[:]
        if not prim:
            assert np.allclose(SiteIndToCartPos[i, :], superFCC[i].position)
    np.save("SiteIndToLmpCartPos.npy", SiteIndToCartPos)  # In case we want to check later
    return SiteIndToCartPos

def Load_crysDat(CrysDatPath, a0):
    with h5py.File(CrysDatPath, "r") as fl:
        dxList = np.array(fl["dxList_1nn"])
        SiteIndToNgb = np.array(fl["NNsiteList_sitewise"])[1:, :].T

    assert SiteIndToNgb.shape[1] == dxList.shape[0]
    dxList *= a0
    return dxList, SiteIndToNgb

# load the data
def load_Data(T, startStep, StateStart, batchSize, InitStateFile):
    if startStep > 0:

        with h5py.File(RunPath + "data_{0}_{1}_{2}.h5".format(T, startStep, StateStart), "r") as fl:
            batchStates = np.array(fl["FinalStates"])

        try:
            assert batchStates.shape[0] == batchSize
        except AssertionError:
            raise AssertionError("The checkpointed number of states does not match batchSize argument.")

        SiteIndToSpecAll = batchStates

        vacSiteIndAll = np.zeros(batchSize, dtype=int)
        for stateInd in range(SiteIndToSpecAll.shape[0]):
            state = SiteIndToSpecAll[stateInd]
            vacSite = np.where(state == 0)[0][0]
            vacSiteIndAll[stateInd] = vacSite

        print("Starting from checkpointed step {}".format(startStep))
        np.save("states_{0}_{1}_{2}.npy".format(T, startStep, StateStart), SiteIndToSpecAll)
        np.save("vacSites_{0}_{1}_{2}.npy".format(T, startStep, StateStart), vacSiteIndAll)

    else:
        assert startStep == 0
        try:
            allStates = np.load(InitStateFile)
            print("Starting from step zero.")
        except FileNotFoundError:
            raise FileNotFoundError("Initial states not found.")

        SiteIndToSpecAll = allStates[StateStart: StateStart + batchSize]

        assert np.all(SiteIndToSpecAll[:, 0] == 0), "All vacancies must be at the 0th site initially."
        vacSiteIndAll = np.zeros(SiteIndToSpecAll.shape[0], dtype = int)
        np.save("states_step0_{}.npy".format(T), SiteIndToSpecAll)

    return SiteIndToSpecAll, vacSiteIndAll

def main(args):

    # SourcePath = os.path.split(os.path.realpath(__file__))[0] # the directory where the main script is
    # SourcePath += "/"

    # check number of images is odd
    if args.NImages % 2 == 0:
        raise ValueError("Can only use odd values for the number of images (-ni, --NImages).")

    # Create the Lammps cartesian positions - first check if they have already been made.
    try:
        SiteIndToPos = np.load("SiteIndToLmpCartPos.npy")

    except FileNotFoundError:
        SiteIndToPos = CreateLammpsData(args.Nunits, args.LatPar, prim=args.Prim)

    # Load the crystal data
    dxList, SiteIndToNgb = Load_crysDat(args.CrysDatPath, args.LatPar)

    # Load the initial states
    SiteIndToSpecAll, vacSiteIndAll = load_Data(args.Temp, args.startStep, args.StateStart,
                                                args.batchSize, args.InitStateFile)

    # Run a check on the vacancy positions
    try:
        assert args.batchSize == SiteIndToSpecAll.shape[0]
    except AssertionError:
        raise TypeError("Different batch size (entered as argument) than loaded data detected.")

    for traj in range(SiteIndToSpecAll.shape[0]):
        state = SiteIndToSpecAll[traj]
        vacInd = np.where(state == 0)[0][0]
        assert vacInd == vacSiteIndAll[traj]
    print("Checked vacancy occupancies.")

    # Then do the KMC steps
    if args.SimpleNEB:
        print("Starting KMC calculations with simple NEB from on-lattice states.")
        DoKMC(args.Temp, args.startStep, args.Nsteps, args.StateStart, dxList,
              SiteIndToSpecAll, vacSiteIndAll, args.batchSize, SiteIndToNgb, args.chunkSize, args.PotPath,
              SiteIndToPos, WriteAllJumps=args.WriteAllJumps, etol=args.EnTol, ftol=args.ForceTol, NImages=args.NImages)

    else:
        print("Starting KMC calculations with pre-relaxed initial and final states.")
        DoKMC_Relax_Fix(args.Temp, args.startStep, args.Nsteps, args.StateStart, dxList,
              SiteIndToSpecAll, vacSiteIndAll, args.batchSize, SiteIndToNgb, args.chunkSize, args.PotPath,
              SiteIndToPos, WriteAllJumps=args.WriteAllJumps, etol=args.EnTol, ftol=args.ForceTol, NImages=args.NImages,
              etol_relax=args.EnTolRel)


if __name__ == "__main__":

    # Add argument parser
    parser = argparse.ArgumentParser(description="Input parameters for Kinetic Monte Carlo simulations with LAMMPS.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-cr", "--CrysDatPath", metavar="/path/to/crys/dat", type=str, help="Path to crystal Data.")
    parser.add_argument("-pp", "--PotPath", metavar="/path/to/potential/file", type=str, help="Path to the LAMMPS MEAM potential.")
    parser.add_argument("-if", "--InitStateFile", metavar="/path/to/initial/file.npy", type=str, default=None,
                        help="Path to the .npy file storing the 0-step states from Metropolis Monte Carlo.")

    parser.add_argument("-a0", "--LatPar", metavar="float", type=float, default=3.595,
                        help="Lattice parameter - multiplied to displacements and used"
                                                                           "to construct LAMMPS coordinates.")

    parser.add_argument("-pr", "--Prim", action="store_true",
                        help="Whether to use primitive cell")

    parser.add_argument("-T", "--Temp", metavar="int", type=int, help="Temperature to read data from")

    parser.add_argument("-st", "--startStep", metavar="int", type=int, default=0,
                        help="From which step to start the simulation. Note - checkpointed data file must be present in running directory if value > 0.")

    parser.add_argument("-smp", "--SimpleNEB", action="store_true",
                        help="Whether to just do simple NEB from on-lattice initial and final states with all images being relaxed.")

    parser.add_argument("-ni", "--NImages", metavar="int", type=int, default=5,
                        help="How many NEB Images to use. Must be odd number.")

    parser.add_argument("-ns", "--Nsteps", metavar="int", type=int, default=100,
                        help="How many steps to continue AFTER \"starStep\" argument.")

    parser.add_argument("-ftol", "--ForceTol", metavar="float", type=float, default=0.0,
                        help="Force tolerance for ending NEB calculations.")

    parser.add_argument("-etol", "--EnTol", metavar="float", type=float, default=1e-6,
                        help="Relative Energy change tolerance for ending NEB calculations.")

    parser.add_argument("-etol", "--EnTolRel", metavar="float", type=float, default=1e-6,
                        help="Energy tolerance for CG pre-relaxation of initial and final states.")

    parser.add_argument("-u", "--Nunits", metavar="int", type=int, default=8,
                        help="Number of unit cells in the supercell.")

    parser.add_argument("-idx", "--StateStart", metavar="int", type=int, default=0,
                        help="The starting index of the state for this run from the whole data set of starting states. "
                             "The whole data set is loaded, and then samples starting from this index to the next "
                             "\"batchSize\" number of states are loaded.")

    parser.add_argument("-bs", "--batchSize", metavar="int", type=int, default=200,
                        help="How many initial states starting from StateStart should initially be loaded.")

    parser.add_argument("-cs", "--chunkSize", metavar="int", type=int, default=20,
                        help="How many samples to do NEB calculations for at a time.")

    parser.add_argument("-wa", "--WriteAllJumps", action="store_true",
                        help="Whether to store final state files for all jumps separately.")

    parser.add_argument("-dmp", "--DumpArguments", action="store_true",
                        help="Whether to dump all the parsed arguments into a text file.")

    parser.add_argument("-dpf", "--DumpFile", metavar="string", type=str, default="ArgFiles",
                        help="The file in the run directory where all the args will be dumped.")

    args = parser.parse_args()

    if args.DumpArguments:
        print("Dumping arguments to: {}".format(args.DumpFile))
        opts = vars(args)
        with open(RunPath + args.DumpFile, "w") as fl:
            for key, val in opts.items():
                fl.write("{}:\t{}\n".format(key, val))

    main(args)

