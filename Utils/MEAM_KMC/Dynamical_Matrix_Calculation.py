import numpy as np
import h5py
import subprocess

from ase.spacegroup import crystal
from ase.build import make_supercell
from ase.io.lammpsdata import write_lammps_data, read_lammps_data

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
    InitStates = np.load(InitStateFilePath)[startIndex : startIndex + batchSize]
    assert np.all(InitStates[:, 0] == 0) # assert that all initial states have vacancy at the origin
    return InitStates

def main(args):

    # write the NEB and relaxation input files
    write_input_files(args.chunkSize, potPath=args.potPath, etol=args.etol, ftol=args.ftol, ts=args.timeStep,
                      k=args.springConstant, perp=args.perpSpringConstant, threshold=args.dispThreshold)

    # Load the initial states
    InitStates = load_states(args.InitStateFile, args.startSample, args.batchSize)

    # Load the nearest neighbors
    _, SiteIndToNgb = Load_crysDat(args.CrysDatPath, a0=args.LatPar)

    # Make the LAMMPS coordinates
    SiteIndToPos = CreateLammpsData(N_units=args.N_units, a=args.LatPar, prim=args.Prim)
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
    for chunk in range(0, InitStates.shape[0], args.chunkSize):
        start = chunk
        end = min(InitStates.shape[0], start + args.chunkSize)
        samples = InitStates[start : end]
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
                write_dynamical_matrix_commands(traj, im_init, JumpAtomIndex, args.potPath)

            # calculate and read dynamical matrices for the initial state
            dynMat_Init = compute_dynamical_matrix(Ntraj=samples.shape[0])

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
                write_dynamical_matrix_commands(traj, TS, JumpAtomIndex, args.potPath)

            # calculate and read dynamical matrices for the transition states
            dynMat_Trans = compute_dynamical_matrix(Ntraj=samples.shape[0])

            # Now compute attempt frequencies if the dynamical matrices satisfy all the necessary conditions



