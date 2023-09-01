import numpy as np
import pickle
import subprocess

from tqdm import tqdm
from ase.spacegroup import crystal
from ase.build import make_supercell
from ase.io.lammpsdata import write_lammps_data, read_lammps_data

import os
RunPath = os.getcwd() + '/'
import argparse

from KMC_funcs import *
from NEB_steps_multiTraj import Load_crysDat, CreateLammpsData

Input_Dynamical_Matrix = \
    """
    units \t metal
    atom_style \t atomic
    atom_modify \t map array
    boundary \t p p p
    atom_modify \t sort 0 0.0
    read_data \t Image_{0}_{1}.data
    pair_style \t meam
    pair_coeff \t * * {2}/library.meam Co Ni Cr Fe Mn {2}/params.meam Co Ni Cr Fe Mn

    group \t move id {3}
    dynamical_matrix move eskm 0.000001 file dyn_{0}.dat
    """
def write_dynamical_matrix_commands(traj, Image, JumpAtomIndex, potpath):

   with open("in.dyn_{0}".format(traj), "w") as fl:
       fl.write(Input_Dynamical_Matrix.format(traj, Image, potpath, JumpAtomIndex))

def compute_dynamical_matrix(Ntraj):
    commands = [
        "$LMPPATH/lmp -in in.dyn_{0} -log out_dyn_{0}.txt -screen screen_dyn_{0}.txt".format(
            traj)
        for traj in range(Ntraj)
    ]

    cmdList = [subprocess.Popen(cmd, shell=True) for cmd in commands]

    # Now read the dynamical matrix

    dynMat = np.zeros((Ntraj, 3, 3))
    # Now read the forces and the dynamical matrix
    for traj in range(Ntraj):
        with open("dyn_{0}.dat".format(traj), "r") as fl:
            lines = fl.readlines()
            assert len(lines) == 3
            for i in range(3):
                ln = lines[i].split()
                assert len(ln) == 3
                for j in range(3):
                    dynMat[traj, i, j] = float(ln[j])

    return dynMat

# Load the initial states - all states must have the vacancy at the 0th site
def load_states(InitStateFilePath, startIndex, batchSize):
    InitStates = np.load(InitStateFilePath)
    assert np.all(InitStates[:, 0] == 0)  # assert that all initial states have vacancy at the origin
    return InitStates[startIndex : startIndex + batchSize]

def main(args):

    # make a test directory if required
    if args.Test:
        if not os.path.isdir(RunPath + "Test"):
            os.mkdir(RunPath + "Test")

    # write the NEB and relaxation input files
    write_input_files(args.chunkSize, potPath=args.PotPath, etol=args.etol, ftol=args.ftol, ts=args.TimeStep,
                      k=args.SpringConstant, perp=args.PerpSpringConstant, threshold=args.DispThreshold,
                      writeImageData=True, NImages=args.NImages)

    # Load the initial states
    InitStates = load_states(args.InitStateFile, args.StateStart, args.batchSize)

    # Load the nearest neighbors
    _, SiteIndToNgb = Load_crysDat(args.CrysDatPath, a0=args.LatPar)

    # Make the LAMMPS coordinates
    SiteIndToPos = CreateLammpsData(N_units=args.Nunits, a=args.LatPar, prim=args.Prim)
    try:
        with open("lammpsBox.txt", "r") as fl:
            Initlines = fl.readlines()
            lineStartCoords = None
            for lineInd, line in enumerate(Initlines):
                if "Atoms" in line:
                    lineStartCoords = lineInd + 2
                    break

            # check that we have the correct line
            firstLine = Initlines[lineStartCoords].split()
            assert len(firstLine) == 5
            assert firstLine[0] == "1"  # the atom index

    except FileNotFoundError:
        raise FileNotFoundError("Template lammps data file not found.")

    assert InitStates.shape[1] == len(Initlines[lineStartCoords:])
    Nsites = InitStates.shape[1]
    specs, counts = np.unique(InitStates[0], return_counts=True)
    Nspec = len(specs)  # including the vacancy

    Initlines[2] = "{} \t atoms\n".format(Nsites - 1)  # minus one for the vacant site
    Initlines[3] = "{} atom types\n".format(Nspec - 1)  # minus one for the vacancy

    # Now do the NEB calculations and find the dynamical matrices
    elems = ["Co", "Ni", "Cr", "Fe", "Mn"]
    AttemptFreqs = {"Co": [], "Ni": [], "Cr": [], "Fe": [], "Mn": []}

    Reject_Unstable_init_state = 0
    Reject_moreThanOneUnstableMode_TS = 0
    Reject_allPositiveModes_TS = 0
    total = 0
    for chunk in tqdm(range(0, InitStates.shape[0], args.chunkSize), position=0, leave=True, ncols=65):
        start = chunk
        end = min(InitStates.shape[0], start + args.chunkSize)
        samples = InitStates[start:end, :]
        assert np.all(samples[:, 0] == 0)
        vacSiteInd = np.zeros(samples.shape[0], dtype=int)

        write_init_states(samples, SiteIndToPos, vacSiteInd, Initlines[:lineStartCoords])

        for jumpInd in range(SiteIndToNgb.shape[1]):
            JumpAtomIndex = SiteIndToNgb[0, jumpInd]

            # Write the final states in NEB format for lammps
            write_final_states(SiteIndToPos, vacSiteInd, SiteIndToNgb, jumpInd, writeAll=False)

            # Then run lammps
            commands = [
                "srun --exclusive -n {0} --cpus-per-task=1 --mem-per-cpu={2}M $LMPPATH/lmp -log out_{1}.txt -screen screen_{1}.txt -p {0}x1 -in in.neb_{1}".format(
                    args.NImages, traj, args.MemPerCpu)
                for traj in range(samples.shape[0])
            ]

            cmdList = [subprocess.Popen(cmd, shell=True) for cmd in commands]

            # wait for the lammps commands to complete
            for c in cmdList:
                rt_code = c.wait()
                assert rt_code == 0  # check for system errors

            if args.Test:
                # Copy state data to the test directory
                for traj in range(samples.shape[0]):
                    for im in range(1, args.NImages + 1):
                        subprocess.run(
                            "cp Image_{0}_{1}.data Test/Image_{3}_{1}_{2}.data".format(traj, im, jumpInd, start + traj),
                            shell=True, check=True)
                        subprocess.run(
                            "cp final_{0}.data Test/final_{2}_{1}.data".format(traj, jumpInd, start + traj),
                            shell=True, check=True)

                        subprocess.run(
                            "cp initial_{0}.data Test/initial_{2}_{1}.data".format(traj, jumpInd, start + traj),
                            shell=True, check=True)

            # Write the final states for end-state relaxation
            write_final_states_relaxation(samples, SiteIndToPos, vacSiteInd, SiteIndToNgb,
                                          jumpInd, Initlines[:lineStartCoords])

            # Relax the final states
            commands = [
                "$LMPPATH/lmp -in in.relax_final_{0} -log out_relax_final_{0}.txt -screen screen_relax_final_{0}.txt".format(
                    traj)
                for traj in range(samples.shape[0])
            ]

            cmdList = [subprocess.Popen(cmd, shell=True) for cmd in commands]

            # wait for the lammps commands to complete
            for c in cmdList:
                rt_code = c.wait()
                assert rt_code == 0  # check for system errors

            # Write dynamical matrix commands for the initial state
            for traj in range(samples.shape[0]):
                # Write dynamical matrix inputs

                # 1. First, for the initial state
                im_init = 0  # the first image is the initial state
                write_dynamical_matrix_commands(traj, im_init, JumpAtomIndex, args.PotPath)

            # calculate and read dynamical matrices for the initial state
            dynMat_Init = compute_dynamical_matrix(Ntraj=samples.shape[0])

            if args.Test:
                np.save("Test/dynMat_Initial_{}_to_{}_jump_{}.npy".format(start, start+samples.shape[0], jumpInd), dynMat_Init)

            # Write dynamical matrix commands for the transition states
            TSImages = np.zeros(samples.shape[0], dtype=int)
            for traj in range(samples.shape[0]):
                with open("out_{0}.txt".format(traj), "r") as fl:
                    lines = fl.readlines()

                assert "Climbing" in lines[-3]  # check correct stopping after regular stage
                ebfLine = lines[-4].split()
                ImageEns = np.array([float(x) for x in ebfLine[10::2]])
                TS = np.argmax(ImageEns)
                TSImages[traj] = TS
                write_dynamical_matrix_commands(traj, TS + 1, JumpAtomIndex, args.PotPath)
                if args.Test:
                    np.save("Test/Image_Ens_{}_{}.npy".format(start + traj, jumpInd), ImageEns)

            # calculate and read dynamical matrices for the transition states
            dynMat_Trans = compute_dynamical_matrix(Ntraj=samples.shape[0])
            if args.Test:
                np.save("Test/dynMat_TS_{}_to_{}_jump_{}.npy".format(start, start+samples.shape[0], jumpInd), dynMat_Trans)

            # Now compute attempt frequencies if the dynamical matrices satisfy all the necessary conditions
            for traj in range(samples.shape[0]):

                if TSImages[traj] == args.NImages - 1:  # 1. jump to unstable state
                    continue

                with open("disps_final_{0}.dump".format(traj), "r") as fl:
                    Displines_fin = fl.readlines()

                if len(Displines_fin) > 9:  # 2. At least one atom moved a larger distance than the threshold
                    continue

                total += 1
                # compute initial state eigenvalues
                initD = dynMat_Init[traj, :, :]
                vals, vecs = np.linalg.eig(initD)

                # take the product of the square root of the eigenvalues
                prod_init = np.prod(np.sqrt(vals))

                if np.any(vals < 0):  # 3. if the initial state has less than 3 positive modes
                    Reject_Unstable_init_state += 1
                    continue

                TSD = dynMat_Trans[traj, :, :]
                vals, vecs = np.linalg.eig(TSD)
                negative = 0
                prod_TS = 1
                for v in vals:
                    if v < 0:
                        negative += 1
                    else:
                        # take the product of the square root of the positive eigenvalues
                        prod_TS *= np.sqrt(v)

                if negative == 0:  # 4. metastable TS state or more than one unstable mode
                    Reject_allPositiveModes_TS += 1
                    continue

                if negative > 1:  # 5. more than one unstable mode in the transition state
                    Reject_moreThanOneUnstableMode_TS += 1
                    continue

                # If all the conditions are satisfied, then take the attempt frequency of the jumping atom
                spJump = samples[traj, JumpAtomIndex]
                elemJump = elems[spJump - 1]
                freq = prod_init / prod_TS
                AttemptFreqs[elemJump].append(freq)

    # Next, save all the data and rejection counts
    with open("Statistics.txt", "w") as fl:
        fl.write("Total Jumps Simulated: {}\n".format(total))
        fl.write("Initial states with negative mode: {}\n".format(Reject_Unstable_init_state))
        fl.write("Transition states with all positive modes: {}\n".format(Reject_allPositiveModes_TS))
        fl.write("Transition states with more than one negative mode: {}\n".format(Reject_moreThanOneUnstableMode_TS))

        frac = total / (Reject_Unstable_init_state + Reject_moreThanOneUnstableMode_TS + Reject_allPositiveModes_TS)

        fl.write("Fraction considered: {:.6f}".format(frac))

    with open("AttemptFrequencies.pkl", "wb") as fl:
        pickle.dump(AttemptFreqs, fl)

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

    parser.add_argument("-test", "--Test", action="store_true",
                        help="Whether to store data for testing.")

    parser.add_argument("-ni", "--NImages", metavar="int", type=int, default=11,
                        help="How many NEB Images to use. Must be odd number.")

    parser.add_argument("-ftol", "--ForceTol", metavar="float", type=float, default=0.01,
                        help="Force tolerance for ending NEB calculations.")

    parser.add_argument("-etol", "--EnTol", metavar="float", type=float, default=0.0,
                        help="Relative Energy change tolerance for ending NEB calculations.")

    parser.add_argument("-th", "--DispThreshold", metavar="float", type=float, default=1.0,
                        help="Maximum allowed displacement after relaxation.")

    parser.add_argument("-ts", "--TimeStep", metavar="float", type=float, default=0.001,
                        help="Relative Energy change tolerance for ending NEB calculations.")

    parser.add_argument("-k", "--SpringConstant", metavar="float", type=float, default=1.0,
                        help="Parallel spring constant for NEB calculations.")

    parser.add_argument("-p", "--PerpSpringConstant", metavar="float", type=float, default=1.0,
                        help="Perpendicular spring constant for NEB calculations.")

    parser.add_argument("-u", "--Nunits", metavar="int", type=int, default=8,
                        help="Number of unit cells in the supercell.")

    parser.add_argument("-idx", "--StateStart", metavar="int", type=int, default=0,
                        help="The starting index of the state for this run from the whole data set of starting states."
                             "The whole data set is loaded, and then samples starting from this index to the next "
                             "\"batchSize\" number of states are extracted.")

    parser.add_argument("-bs", "--batchSize", metavar="int", type=int, default=200,
                        help="How many initial states starting from StateStart should initially be loaded.")

    parser.add_argument("-cs", "--chunkSize", metavar="int", type=int, default=20,
                        help="How many samples to do NEB calculations for at a time.")

    parser.add_argument("-mpc", "--MemPerCpu", metavar="int", type=int, default=1000,
                        help="Memory per cpu  (integer, in megabytes)for NEB calculations.")

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