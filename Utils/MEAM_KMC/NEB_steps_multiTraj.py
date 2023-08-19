#!/usr/bin/env python
# coding: utf-8

# In[5]:
import numpy as np
import subprocess
import time
import h5py

from ase.spacegroup import crystal
from ase.build import make_supercell
from ase.io.lammpsdata import write_lammps_data, read_lammps_data

from KMC_funcs import *
import os
import glob
from scipy.constants import physical_constants
kB = physical_constants["Boltzmann constant in eV/K"][0]

import argparse

# Need to get rid of these argument
# NImage = 3
RunPath = os.getcwd()+'/'
print("Running from : " + RunPath)

def CreateLammpsData(N_units, a, prim=False):
    # Create an FCC unit cell
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
def load_Data(StateStart, batchSize, InitStateFile):

    Existing = glob.glob("data*.h5")
    startStep = len(Existing)

    if startStep > 0:

        with h5py.File(RunPath + "data_{0}_{1}.h5".format(startStep, StateStart), "r") as fl:
            SiteIndToSpecAll = np.array(fl["FinalStates"])

        vacSiteIndAll = np.zeros(SiteIndToSpecAll.shape[0], dtype=int)
        for stateInd in range(SiteIndToSpecAll.shape[0]):
            state = SiteIndToSpecAll[stateInd]
            vacSite = np.where(state == 0)[0][0]
            vacSiteIndAll[stateInd] = vacSite
        
        print("Starting from checkpointed step {}".format(startStep))
        np.save("states_{0}_{1}.npy".format(startStep, StateStart), SiteIndToSpecAll)
        np.save("vacSites_{0}_{1}.npy".format(startStep, StateStart), vacSiteIndAll)

    else:
        assert startStep == 0
        try:
            allStates = np.load(InitStateFile)
            assert allStates.dtype != float
            print("Starting from step zero.")
        except FileNotFoundError:
            raise FileNotFoundError("Initial states not found.")

        SiteIndToSpecAll = allStates[StateStart: StateStart + batchSize]

        # Do a small check on the batchSize
        if SiteIndToSpecAll.shape[0] < batchSize:
            assert StateStart + SiteIndToSpecAll.shape[0] == allStates.shape[0]

        assert np.all(SiteIndToSpecAll[:, 0] == 0), "All vacancies must be at the 0th site initially."
        vacSiteIndAll = np.zeros(SiteIndToSpecAll.shape[0], dtype=int)
        np.save("states_step0.npy", SiteIndToSpecAll)
        
    return SiteIndToSpecAll, vacSiteIndAll, startStep

def DoKMC(T, startStep, Nsteps, StateStart, dxList,
          SiteIndToSpecAll, vacSiteIndAll, SiteIndToNgb, chunkSize, PotPath,
          SiteIndToPos, WriteAllJumps=False, ftol=0.01, etol=0.0, ts=0.001, NImages=11, k=1.0,
          perp=1.0, threshold=1.0):

    try:
        with open("lammpsBox.txt", "r") as fl:
            Initlines = fl.readlines()
            lineStartCoords = None
            for lineInd, line in enumerate(Initlines):
                if "Atoms" in line:
                    lineStartCoords = lineInd + 2
                    break
    except:
        raise FileNotFoundError("Template lammps data file not found.")

    assert SiteIndToSpecAll.shape[1] == len(Initlines[lineStartCoords:])

    specs, counts = np.unique(SiteIndToSpecAll[0], return_counts=True)
    Nspec = len(specs)  # including the vacancy
    Ntraj = SiteIndToSpecAll.shape[0]

    print("No. of samples : {}".format(Ntraj))

    Nsites = SiteIndToSpecAll.shape[1]
    Initlines[2] = "{} \t atoms\n".format(Nsites - 1)
    Initlines[3] = "{} atom types\n".format(Nspec - 1)

    # Begin KMC loop below
    FinalStates = SiteIndToSpecAll
    FinalVacSites = vacSiteIndAll

    # Make arrays for storing the data
    SpecDisps = np.zeros((Ntraj, Nspec, 3))
    tarr = np.zeros(Ntraj)
    JumpSelects = np.zeros(Ntraj, dtype=np.int8)  # which jump is chosen for each trajectory
    TestRandomNums = np.zeros(Ntraj)  # store the random numbers at all steps

    AllJumpRates = np.zeros((Ntraj, SiteIndToNgb.shape[1]))
    AllJumpBarriers = np.zeros((Ntraj, SiteIndToNgb.shape[1]))
    AllJumpImageE = np.zeros((Ntraj, NImages, SiteIndToNgb.shape[1]))
    AllJumpImageRD = np.zeros((Ntraj, NImages, SiteIndToNgb.shape[1]))
    AllJumpMaxForceAtom = np.zeros((Ntraj, SiteIndToNgb.shape[1]))
    AllJumpMaxIteration = np.zeros((Ntraj, SiteIndToNgb.shape[1]), dtype=int)

    # Before starting, write the lammps input files
    if startStep == 0:
        write_input_files(chunkSize, potPath=PotPath, etol=etol, ftol=ftol, ts=ts, k=k, perp=perp,
                          threshold=threshold, NImages=NImages)

    TimeStart = time.time()

    for step in range(Nsteps - startStep):
        BadSamples = []
        for chunk in range(0, Ntraj, chunkSize):
            # Write the initial states from last accepted state
            sampleStart = chunk
            sampleEnd = min(chunk + chunkSize, Ntraj)

            SiteIndToSpec = FinalStates[sampleStart: sampleEnd].copy()
            vacSiteInd = FinalVacSites[sampleStart: sampleEnd].copy()

            write_init_states(SiteIndToSpec, SiteIndToPos, vacSiteInd, Initlines[:lineStartCoords])

            # declare arrays to store quantities
            rates = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]))
            barriers = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]))
            ImageEn_Jumps = np.zeros((SiteIndToSpec.shape[0], NImages, SiteIndToNgb.shape[1]))
            ImageRD_Jumps = np.zeros((SiteIndToSpec.shape[0], NImages, SiteIndToNgb.shape[1]))
            MaxForceAtom = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]))
            MaxIteration = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]), dtype=int)

            # iterate through the 12 jumps
            for jumpInd in range(SiteIndToNgb.shape[1]):
                # Write the final states in NEB format for lammps
                write_final_states(SiteIndToPos, vacSiteInd, SiteIndToNgb, jumpInd, writeAll=WriteAllJumps)

                # Then run lammps
                commands = [
                    "srun --exclusive -n {0} --cpus-per-task=1 --mem-per-cpu=1000M $LMPPATH/lmp -log out_{1}.txt -screen screen_{1}.txt -p {0}x1 -in in.neb_{1}".format(NImages, traj)
                    for traj in range(SiteIndToSpec.shape[0])
                ]

                cmdList = [subprocess.Popen(cmd, shell=True) for cmd in commands]

                # wait for the lammps commands to complete
                for c in cmdList:
                    rt_code = c.wait()
                    assert rt_code == 0  # check for system errors

                # Then read the results for each trajectory
                for traj in range(SiteIndToSpec.shape[0]):

                    with open("out_{0}.txt".format(traj), "r") as fl:
                        lines = fl.readlines()

                    assert "Climbing" in lines[-3]  # check correct stopping after regular stage
                    ebfLine = lines[-4].split()

                    ImageEns = np.array([float(x) for x in ebfLine[10::2]])
                    ImageRDs = np.array([float(x) for x in ebfLine[9::2]])

                    assert ImageEns.shape[0] == ImageRDs.shape[0] == NImages

                    ImageEn_Jumps[traj, :, jumpInd] = ImageEns[:]
                    ImageRD_Jumps[traj, :, jumpInd] = ImageRDs[:]

                    ebf = float(ebfLine[6])
                    maxForce = float(ebfLine[2])

                    # Assert that the initial state does not have more than threshold displacement
                    with open("disps_{0}_{1}.dump".format(traj, 1), "r") as fl:
                        Displines_init = fl.readlines()

                    # store the bad sample to check later one
                    if len(Displines_init) != 9:
                        BadSamples.append(chunk + traj)

                    # check displacements in the final state during neb minimization
                    with open("disps_{0}_{1}.dump".format(traj, NImages), "r") as fl:
                        Displines_fin = fl.readlines()

                    if len(Displines_fin) == 9:  # No atom was displaced by more than the threshold in the final image
                        rates[traj, jumpInd] = np.exp(-ebf / (kB * T))
                        barriers[traj, jumpInd] = ebf
                        MaxForceAtom[traj, jumpInd] = maxForce
                        MaxIteration[traj, jumpInd] = int(ebfLine[0])

                    else:  # at least one atom will have moved by more than the threshold in the final image
                        rates[traj, jumpInd] = 0.0
                        barriers[traj, jumpInd] = np.inf
                        MaxForceAtom[traj, jumpInd] = maxForce
                        MaxIteration[traj, jumpInd] = int(ebfLine[0])


            # store all the rates
            AllJumpRates[sampleStart:sampleEnd] = rates[:, :]
            AllJumpBarriers[sampleStart:sampleEnd] = barriers[:, :]
            AllJumpImageE[sampleStart:sampleEnd, :, :] = ImageEn_Jumps[:, :, :]
            AllJumpImageRD[sampleStart:sampleEnd, :, :] = ImageRD_Jumps[:, :, :]

            AllJumpMaxForceAtom[sampleStart:sampleEnd] = MaxForceAtom[:, :]
            AllJumpMaxIteration[sampleStart:sampleEnd] = MaxIteration[:, :]

            # Then do selection
            jumpID, rateProbs, ratesCsum, rndNums, time_step = getJumpSelects(rates)
            # store the selected jumps
            JumpSelects[sampleStart: sampleEnd] = jumpID[:]

            # store the random numbers for testing
            TestRandomNums[sampleStart: sampleEnd] = rndNums[:]

            # Then do the final exchange
            jumpAtomSelectArray, X_traj = updateStates(SiteIndToNgb, Nspec, SiteIndToSpec, vacSiteInd, jumpID, dxList)
            # def updateStates(SiteIndToNgb, Nspec,  SiteIndToSpec, vacSiteInd, jumpID, dxList):

            # store final states, displacements and times
            FinalStates[sampleStart: sampleEnd, :] = SiteIndToSpec[:, :]
            FinalVacSites[sampleStart: sampleEnd] = vacSiteInd[:]
            SpecDisps[sampleStart:sampleEnd, :, :] = X_traj[:, :, :]
            tarr[sampleStart:sampleEnd] = time_step[:]
            with open("ChunkTiming.txt", "a") as fl:
                fl.write(
                    "Chunk {0} of {1} in step {3} completed in : {2} seconds\n".format(chunk//chunkSize + 1,
                                                                                       int(np.ceil(Ntraj/chunkSize)),
                                                                                       time.time() - TimeStart, startStep + step + 1))

                # fl.write("Maxiterations: {}\n".format(np.unique(MaxIteration, return_counts=True)))

        with open("StepTiming.txt", "a") as fl:
            fl.write("Time per step up to {0} of {1} steps : {2} seconds\n".format(startStep + step + 1, Nsteps, (time.time() - TimeStart)/(step + 1)))

        # Next, save all the arrays in an hdf5 file for the current step.
        # For the first 10 steps, store test random numbers.
        with h5py.File("data_{0}_{1}.h5".format(startStep + step + 1, StateStart), "w") as fl:
            fl.create_dataset("BadSamples", data=BadSamples)
            fl.create_dataset("FinalStates", data=FinalStates)
            fl.create_dataset("SpecDisps", data=SpecDisps)
            fl.create_dataset("times", data=tarr)
            fl.create_dataset("AllJumpRates", data=AllJumpRates)
            fl.create_dataset("AllJumpBarriers", data=AllJumpBarriers)
            fl.create_dataset("AllJumpImageEnergies", data=AllJumpImageE)
            fl.create_dataset("AllJumpImageRDs", data=AllJumpImageRD)
            fl.create_dataset("AllJumpMaxForceComponent", data=AllJumpMaxForceAtom)
            fl.create_dataset("AllJumpMaxIterations", data=AllJumpMaxIteration)
            fl.create_dataset("JumpSelects", data=JumpSelects)
            fl.create_dataset("TestRandNums", data=TestRandomNums)

def main(args):

    # Create the Lammps cartesian positions - first check if they have already been made.
    try:
        SiteIndToPos = np.load("SiteIndToLmpCartPos.npy")

    except FileNotFoundError:
        SiteIndToPos = CreateLammpsData(args.Nunits, args.LatPar, prim=args.Prim)

    # Load the crystal data
    dxList, SiteIndToNgb = Load_crysDat(args.CrysDatPath, args.LatPar)

    # Load the initial states
    SiteIndToSpecAll, vacSiteIndAll, startStep = load_Data(args.StateStart,
                                                args.batchSize, args.InitStateFile)

    # Run a check on the vacancy position

    for traj in range(SiteIndToSpecAll.shape[0]):
        state = SiteIndToSpecAll[traj]
        assert np.where(state == 0)[0].shape[0] == 1
        vacInd = np.where(state == 0)[0][0]
        assert vacInd == vacSiteIndAll[traj]
    print("Checked vacancy occupancies.")

    print("Starting KMC NEB calculations.")
    DoKMC(args.Temp, startStep, args.Nsteps, args.StateStart, dxList,
          SiteIndToSpecAll, vacSiteIndAll, SiteIndToNgb, args.chunkSize, args.PotPath,
          SiteIndToPos, WriteAllJumps=args.WriteAllJumps, etol=args.EnTol, ftol=args.ForceTol, NImages=args.NImages,
          ts=args.TimeStep, k=args.SpringConstant, perp=args.PerpSpringConstant, threshold=args.DispThreshold)

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

    parser.add_argument("-T", "--Temp", metavar="int", type=float, help="Temperature to read data from")

    parser.add_argument("-ni", "--NImages", metavar="int", type=int, default=11,
                        help="How many NEB Images to use. Must be odd number.")

    parser.add_argument("-ns", "--Nsteps", metavar="int", type=int, default=100,
                        help="How many steps to run.")

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
                        help="The starting index of the state for this run from the whole data set of starting states. "
                             "The whole data set is loaded, and then samples starting from this index to the next "
                             "\"batchSize\" number of states are loaded.")

    parser.add_argument("-bs", "--batchSize", metavar="int", type=int, default=200,
                        help="How many initial states starting from StateStart should initially be loaded.")

    parser.add_argument("-cs", "--chunkSize", metavar="int", type=int, default=20,
                        help="How many samples to do NEB calculations for at a time.")

    parser.add_argument("-wa", "--WriteAllJumps", action="store_true",
                        help="Whether to store final style NEB files for all jumps separately.")

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