#!/usr/bin/env python
# coding: utf-8

# In[5]:
import numpy as np
import subprocess
import time
import h5py

from KMC_funcs import *
from scipy.constants import physical_constants
kB = physical_constants["Boltzmann constant in eV/K"][0]

def DoKMC_Relax_Fix(T, startStep, Nsteps, StateStart, dxList,
          SiteIndToSpecAll, vacSiteIndAll, batchSize, SiteIndToNgb, chunkSize, PotPath,
          SiteIndToPos, ftol=0.001, etol=1e-7, etol_relax=1e-8, NImages=5):
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

    # Before starting, write the lammps input files
    # write_input_files(chunkSize, potPath=PotPath, etol=etol, ftol=ftol)
    write_input_files_relax_fix(chunkSize, potPath=PotPath, etol=etol, etol_relax=etol_relax,
                                ftol=ftol, NImages=NImages)

    start = time.time()

    for step in range(Nsteps):
        for chunk in range(0, Ntraj, chunkSize):
            # Write the initial states from last accepted state
            sampleStart = chunk
            sampleEnd = min(chunk + chunkSize, Ntraj)

            SiteIndToSpec = FinalStates[sampleStart: sampleEnd].copy()
            vacSiteInd = FinalVacSites[sampleStart: sampleEnd].copy()

            write_init_states(SiteIndToSpec, SiteIndToPos, vacSiteInd, Initlines[:lineStartCoords])

            # Relax the initial states
            commands = [
                "$LMPPATH/lmp -log out_rel_init_{0}.txt -screen screen_rel_init_{0}.txt -in in.minim_init_{0}".format(traj)
                for traj in range(SiteIndToSpec.shape[0])
            ]
            cmdList = [subprocess.Popen(cmd, shell=True) for cmd in commands]

            # wait for the lammps commands to complete
            for c in cmdList:
                rt_code = c.wait()
                assert rt_code == 0  # check for system errors

            rates = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]))
            barriers = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]))
            ISE = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]))
            TSE = np.zeros((SiteIndToSpec.shape[0], NImages-2, SiteIndToNgb.shape[1]))
            FSE = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]))
            for jumpInd in range(SiteIndToNgb.shape[1]):

                # Create the minimization input file for the final states
                write_minim_final_states_relax_fix(SiteIndToSpec, SiteIndToPos, vacSiteInd,
                                                   SiteIndToNgb, jumpInd, Initlines[:lineStartCoords])

                # Then relax the final states
                commands = [
                    "$LMPPATH/lmp -log out_rel_fin_{0}.txt -screen screen_rel_fin_{0}.txt -in in.minim_fin_{0}".format(traj)
                    for traj in range(SiteIndToSpec.shape[0])
                ]
                cmdList = [subprocess.Popen(cmd, shell=True) for cmd in commands]

                # wait for the lammps commands to complete
                for c in cmdList:
                    rt_code = c.wait()
                    assert rt_code == 0  # check for system errors

                # Write the RELAXED final states in NEB format for lammps
                write_final_NEB_relax_fix(SiteIndToSpec.shape[0], Natoms=SiteIndToSpec.shape[1] - 1)
                # write_final_states(SiteIndToPos, vacSiteInd, SiteIndToNgb, jumpInd, writeAll=WriteAllJumps)

                # Then run lammps
                commands = [
                    "mpirun -np {0} --oversubscribe $LMPPATH/lmp -log out_NEB_{1}.txt -screen screen_NEB_{1}.txt -p {0}x1 -in in.neb_{1}".format(NImages, traj)
                    for traj in range(SiteIndToSpec.shape[0])
                ]
                cmdList = [subprocess.Popen(cmd, shell=True) for cmd in commands]

                # wait for the lammps commands to complete
                for c in cmdList:
                    rt_code = c.wait()
                    assert rt_code == 0  # check for system errors

                # Then read the forward barrier -> ebf
                for traj in range(SiteIndToSpec.shape[0]):
                    with open("out_NEB_{0}.txt".format(traj), "r") as fl:
                        for line in fl:
                            continue
                    ebfLine = line.split()
                    ebf = float(ebfLine[6])
                    rates[traj, jumpInd] = np.exp(-ebf / (kB * T))
                    barriers[traj, jumpInd] = ebf

                    Is = float(ebfLine[10])

                    for im in range(NImages-2):
                        Ts = float(ebfLine[10 + 2 * (im + 1)])
                        TSE[traj, im, jumpInd] = Ts

                    Fs = float(ebfLine[10 + 2 * (NImages-1)])

                    ISE[traj, jumpInd] = Is
                    FSE[traj, jumpInd] = Fs

                    # get the jumping species and store the barrier for later use
                    vInd = vacSiteInd[traj]
                    vacNgb = SiteIndToNgb[vInd, jumpInd]
                    jAtom = SiteIndToSpec[traj, vacNgb]

            # store all the rates
            AllJumpRates[sampleStart:sampleEnd] = rates[:, :]
            AllJumpBarriers[sampleStart:sampleEnd] = barriers[:, :]
            AllJumpISE[sampleStart:sampleEnd] = ISE[:, :]
            AllJumpTSE[sampleStart:sampleEnd, :, :] = TSE[:, :, :]
            AllJumpFSE[sampleStart:sampleEnd] = FSE[:, :]

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
            fl.create_dataset("JumpSelects", data=JumpSelects)
            fl.create_dataset("TestRandNums", data=TestRandomNums)