#!/usr/bin/env python
# coding: utf-8

# In[5]:
import numpy as np
import subprocess
import time
import h5py
import pickle

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
        
        try:
            with open("JumpsToAvoid.pkl", "rb") as fl:
                JumpsToAvoid = pickle.load(fl)
        
        except FileNotFoundError:
            print("No Jumps found to avoid.")
            JumpsToAvoid = set()
        
        print("Starting from checkpointed step {}".format(startStep))
        np.save("states_{0}_{1}_{2}.npy".format(T, startStep, StateStart), SiteIndToSpecAll)
        np.save("vacSites_{0}_{1}_{2}.npy".format(T, startStep, StateStart), vacSiteIndAll)

    else:
        assert startStep == 0
        try:
            allStates = np.load(InitStateFile)
            assert allStates.dtype == np.int8
            print("Starting from step zero.")
        except FileNotFoundError:
            raise FileNotFoundError("Initial states not found.")

        SiteIndToSpecAll = allStates[StateStart: StateStart + batchSize]

        assert np.all(SiteIndToSpecAll[:, 0] == 0), "All vacancies must be at the 0th site initially."
        vacSiteIndAll = np.zeros(SiteIndToSpecAll.shape[0], dtype=int)
        np.save("states_step0_{}.npy".format(T), SiteIndToSpecAll)
        JumpsToAvoid = set()
        
    return SiteIndToSpecAll, vacSiteIndAll, JumpsToAvoid

def DoKMC(T, startStep, Nsteps, StateStart, dxList,
          SiteIndToSpecAll, vacSiteIndAll, JumpsToAvoid, batchSize, SiteIndToNgb, chunkSize, PotPath,
          SiteIndToPos, WriteAllJumps=False, ftol=0.01, etol=0.0, ts=0.001, NImages=11):
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
    assert Ntraj == batchSize
    print("No. of samples : {}".format(Ntraj))

    Nsites = SiteIndToSpecAll.shape[1]
    Initlines[2] = "{} \t atoms\n".format(Nsites - 1)
    Initlines[3] = "{} atom types\n".format(Nspec - 1)

    # Begin KMC loop below
    FinalStates = SiteIndToSpecAll
    FinalVacSites = vacSiteIndAll
    SpecDisps = np.zeros((Ntraj, Nspec, 3))
    tarr = np.zeros(Ntraj)
    JumpSelects = np.zeros(Ntraj, dtype=np.int8)  # which jump is chosen for each trajectory
    TestRandomNums = np.zeros(Ntraj)  # store the random numbers at all steps

    AllJumpRates = np.zeros((Ntraj, SiteIndToNgb.shape[1]))
    AllJumpBarriers = np.zeros((Ntraj, SiteIndToNgb.shape[1]))
    AllJumpISE = np.zeros((Ntraj, SiteIndToNgb.shape[1]))
    AllJumpTSE = np.zeros((Ntraj, NImages-2, SiteIndToNgb.shape[1]))
    AllJumpFSE = np.zeros((Ntraj, SiteIndToNgb.shape[1]))
    AllJumpMaxForceAtom = np.zeros((Ntraj, SiteIndToNgb.shape[1]))
    AllJumpChooseCI = np.zeros((Ntraj, SiteIndToNgb.shape[1]))
    AllJumpChooseRegular = np.zeros((Ntraj, SiteIndToNgb.shape[1]))

    # Before starting, write the lammps input files
    write_input_files(chunkSize, potPath=PotPath, etol=etol, ftol=ftol, ts=ts)

    start = time.time()

    for step in range(Nsteps):
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
            ISE = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]))
            TSE = np.zeros((SiteIndToSpec.shape[0], NImages-2, SiteIndToNgb.shape[1]))
            FSE = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]))
            MaxForceAtom = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]))
            ChooseRegular = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]), dtype=np.int8)
            ChooseCI = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]), dtype=np.int8)

            # iterate through the 12 jumps
            for jumpInd in range(SiteIndToNgb.shape[1]):
                # Write the final states in NEB format for lammps
                write_final_states(SiteIndToPos, vacSiteInd, SiteIndToNgb, jumpInd, writeAll=WriteAllJumps)

                # Then run lammps
                commands = [
                    "srun -n {0} --cpus-per-task=1 $LMPPATH/lmp -log out_{1}.txt -screen screen_{1}.txt -p {0}x1 -in in.neb_{1}".format(NImages, traj)
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

                    for lInd, l in enumerate(lines):
                        if "Climbing" in l:
                            break

                    LastLine_CI = lines[-1].split()
                    LastLine_Regular = lines[lInd - 1].split()

                    iters_total = int(LastLine_CI[0])
                    iters_regular = int(LastLine_Regular[0])
                    iters_CI = iters_total - iters_regular

                    st = SiteIndToSpec[traj]
                    if (tuple(st), jumpInd) in JumpsToAvoid: # Check if this is a jump to avoid.
                        ebfLine = None

                    else:
                        if iters_CI < 500 and iters_regular < 5000: # both CI and regular NEB are converged
                            ebfLine = LastLine_CI
                            ChooseCI[traj, jumpInd] = 1

                        elif iters_CI >= 500 and iters_regular < 5000:  # regular converged but CI is not
                            ebfLine = LastLine_Regular
                            ChooseRegular[traj, jumpInd] = 1

                        else:  # regular NEB not converged within 5000 iterations (we'll set 0 rates for these and their reverse jumps).
                            ebfLine = None

                            # put the state, the jump and the reverse state and jump in jumps to avoid
                            JumpsToAvoid.add((tuple(st), jumpInd))

                            # First get the reverse jump
                            jIndRev = None
                            count = 0
                            for jInd in range(dxList.shape[0]):
                                if np.allclose(dxList[jInd] + dxList[jumpInd], 0):
                                    count += 1
                                    jIndRev = jInd

                            assert count == 1 and jIndRev is not None

                            # Then get the reverse jump's initial state
                            vac = vacSiteInd[traj]
                            vacngb = SiteIndToNgb[vac, jumpInd]
                            assert st[vac] == 0  # check the vacancy site
                            assert SiteIndToNgb[vacngb, jIndRev] == vac
                            stRev = st.copy()
                            stRev[vac] = st[vacngb]
                            stRev[vacngb] = st[vac]

                            JumpsToAvoid.add((tuple(stRev), jIndRev))

                    if ebfLine is None:
                        assert ChooseRegular[traj, jumpInd] == ChooseCI[traj, jumpInd] == 0
                        rates[traj, jumpInd] = 0.0
                        barriers[traj, jumpInd] = np.inf
                        MaxForceAtom[traj, jumpInd] = np.inf

                    else:
                        assert ChooseRegular[traj, jumpInd] + ChooseCI[traj, jumpInd] == 1
                        ebf = float(ebfLine[6])
                        maxForce = float(ebfLine[2])

                        rates[traj, jumpInd] = np.exp(-ebf / (kB * T))
                        barriers[traj, jumpInd] = ebf
                        MaxForceAtom[traj, jumpInd] = maxForce

                        Is = float(ebfLine[10])

                        for im in range(NImages-2):
                            Ts = float(ebfLine[10 + 2 * (im + 1)])
                            TSE[traj, im, jumpInd] = Ts

                        Fs = float(ebfLine[10 + 2 * (NImages-1)])

                        ISE[traj, jumpInd] = Is
                        FSE[traj, jumpInd] = Fs


            # store all the rates
            AllJumpRates[sampleStart:sampleEnd] = rates[:, :]
            AllJumpBarriers[sampleStart:sampleEnd] = barriers[:, :]
            AllJumpISE[sampleStart:sampleEnd] = ISE[:, :]
            AllJumpTSE[sampleStart:sampleEnd, :, :] = TSE[:, :, :]
            AllJumpFSE[sampleStart:sampleEnd] = FSE[:, :]

            AllJumpMaxForceAtom[sampleStart:sampleEnd] = MaxForceAtom[:, :]
            AllJumpChooseCI[sampleStart:sampleEnd] = ChooseCI[:, :]
            AllJumpChooseRegular[sampleStart:sampleEnd] = ChooseRegular[:, :]

            # Then do selection
            jumpID, rateProbs, ratesCsum, rndNums, time_step = getJumpSelects(rates)
            # store the selected jumps
            JumpSelects[sampleStart: sampleEnd] = jumpID[:]

            # store the random numbers for testing
            TestRandomNums[sampleStart: sampleEnd] = rndNums[:]

            # Then do the final exchange
            jumpAtomSelectArray, X_traj = updateStates(SiteIndToNgb, Nspec, SiteIndToSpec, vacSiteInd, jumpID, dxList)
            # def updateStates(SiteIndToNgb, Nspec,  SiteIndToSpec, vacSiteInd, jumpID, dxList):

            # save final states, displacements and times
            FinalStates[sampleStart: sampleEnd, :] = SiteIndToSpec[:, :]
            FinalVacSites[sampleStart: sampleEnd] = vacSiteInd[:]
            SpecDisps[sampleStart:sampleEnd, :, :] = X_traj[:, :, :]
            tarr[sampleStart:sampleEnd] = time_step[:]
            with open("ChunkTiming.txt", "a") as fl:
                fl.write(
                    "Chunk {0} of {1} in step {3} completed in : {2} seconds\n".format(chunk//chunkSize + 1,
                                                                                       int(np.ceil(Ntraj/chunkSize)),
                                                                                       time.time() - start, step + 1))

        with open("StepTiming.txt", "a") as fl:
            fl.write("Time per step up to {0} of {1} steps : {2} seconds\n".format(step + 1, Nsteps, (time.time() - start)/(step + 1)))

        # Next, save all the arrays in an hdf5 file for the current step.
        # For the first 10 steps, store test random numbers.
        with h5py.File("data_{0}_{1}_{2}.h5".format(T, startStep + step + 1, StateStart), "w") as fl:
            fl.create_dataset("FinalStates", data=FinalStates)
            fl.create_dataset("SpecDisps", data=SpecDisps)
            fl.create_dataset("times", data=tarr)
            fl.create_dataset("AllJumpRates", data=AllJumpRates)
            fl.create_dataset("AllJumpBarriers", data=AllJumpBarriers)
            fl.create_dataset("AllJumpISEnergy", data=AllJumpISE)
            fl.create_dataset("AllJumpTSEnergy", data=AllJumpTSE)
            fl.create_dataset("AllJumpFSEnergy", data=AllJumpFSE)
            fl.create_dataset("AllJumpMaxForceComponent", data=AllJumpMaxForceAtom)
            fl.create_dataset("AllJumpChooseRegular", data=AllJumpChooseRegular)
            fl.create_dataset("AllJumpChooseCI", data=AllJumpChooseCI)
            fl.create_dataset("JumpSelects", data=JumpSelects)
            fl.create_dataset("TestRandNums", data=TestRandomNums)

        # save if there are some jumps to avoid
        if len(JumpsToAvoid) > 0:
            with open("JumpsToAvoid.pkl", "wb") as fl:
                pickle.dump(JumpsToAvoid, fl)

def main(args):

    # Create the Lammps cartesian positions - first check if they have already been made.
    try:
        SiteIndToPos = np.load("SiteIndToLmpCartPos.npy")

    except FileNotFoundError:
        SiteIndToPos = CreateLammpsData(args.Nunits, args.LatPar, prim=args.Prim)

    # Load the crystal data
    dxList, SiteIndToNgb = Load_crysDat(args.CrysDatPath, args.LatPar)

    # Load the initial states
    SiteIndToSpecAll, vacSiteIndAll, JumpsToAvoid = load_Data(args.Temp, args.startStep, args.StateStart,
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
    print("Starting KMC NEB calculations.")
    DoKMC(args.Temp, args.startStep, args.Nsteps, args.StateStart, dxList,
          SiteIndToSpecAll, vacSiteIndAll, JumpsToAvoid, args.batchSize, SiteIndToNgb, args.chunkSize, args.PotPath,
          SiteIndToPos, WriteAllJumps=args.WriteAllJumps, etol=args.EnTol, ftol=args.ForceTol, NImages=args.NImages,
          ts=args.TimeStep)

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

    parser.add_argument("-ni", "--NImages", metavar="int", type=int, default=11,
                        help="How many NEB Images to use. Must be odd number.")

    parser.add_argument("-ns", "--Nsteps", metavar="int", type=int, default=100,
                        help="How many steps to continue AFTER \"starStep\" argument.")

    parser.add_argument("-ftol", "--ForceTol", metavar="float", type=float, default=0.01,
                        help="Force tolerance for ending NEB calculations.")

    parser.add_argument("-etol", "--EnTol", metavar="float", type=float, default=0.0,
                        help="Relative Energy change tolerance for ending NEB calculations.")

    parser.add_argument("-ts", "--TimeStep", metavar="float", type=float, default=0.001,
                        help="Relative Energy change tolerance for ending NEB calculations.")

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

