import numpy as np
import subprocess
import time
import h5py

from scipy.constants import physical_constants
kB = physical_constants["Boltzmann constant in eV/K"][0]
def write_input_files(Ntr, potPath=None, etol=1e-8, ftol=0.00):
    for traj in range(Ntr):
        with open("in.neb_{0}".format(traj), "w") as fl:
            fl.write("units \t metal\n")
            fl.write("atom_style \t atomic\n")
            fl.write("atom_modify \t map array\n")
            fl.write("boundary \t p p p\n")
            fl.write("atom_modify \t sort 0 0.0\n")
            fl.write("read_data \t initial_{0}.data\n".format(traj))
            fl.write("pair_style \t meam\n")
            if potPath is None:
                fl.write("pair_coeff \t * * pot/library.meam Co Ni Cr Fe Mn pot/params.meam Co Ni Cr Fe Mn\n")
            else:
                fl.write("pair_coeff \t * * "+ potPath + "/library.meam Co Ni Cr Fe Mn " +
                         potPath + "/params.meam Co Ni Cr Fe Mn\n")
            fl.write("fix \t 1 all neb 1.0\n")
            fl.write("timestep \t 0.01\n")
            fl.write("min_style \t quickmin\n")
            fl.write("neb \t {2} {0} 500 500 10 final final_{1}.data".format(ftol, traj, etol))


def write_input_files_relax_fix(Ntr, potPath=None, etol=1e-8, ftol=0.00, etol_relax=1e-8, NImages=5):
    for traj in range(Ntr):
        with open("in.neb_{0}".format(traj), "w") as fl:
            fl.write("units \t metal\n")
            fl.write("atom_style \t atomic\n")
            fl.write("atom_modify \t map array\n")
            fl.write("boundary \t p p p\n")
            fl.write("atom_modify \t sort 0 0.0\n")
            fl.write("read_data \t initial_relax_{0}.data\n".format(traj)) # read RELAXED initial state
            fl.write("pair_style \t meam\n")
            if potPath is None:
                fl.write("pair_coeff \t * * pot/library.meam Co Ni Cr Fe Mn pot/params.meam Co Ni Cr Fe Mn\n")
            else:
                fl.write("pair_coeff \t * * " + potPath + "/library.meam Co Ni Cr Fe Mn " +
                         potPath + "/params.meam Co Ni Cr Fe Mn\n")
            fl.write("timestep \t 0.01\n")
            fl.write("min_style \t quickmin\n")
            fl.write("partition no 2*{0} fix 2 all setforce 0.0 0.0 0.0\n".format(NImages-1))
            fl.write("fix \t 1 all neb 1.0\n")
            fl.write("neb \t {2} {0} 500 500 10 final final_NEB_coords_{1}.data".format(ftol, traj, etol))

        with open("in.minim_init_{0}".format(traj), "w") as fl:
            fl.write("units \t metal\n")
            fl.write("atom_style \t atomic\n")
            fl.write("atom_modify \t map array\n")
            fl.write("boundary \t p p p\n")
            fl.write("atom_modify \t sort 0 0.0\n")
            fl.write("read_data \t initial_{0}.data\n".format(traj))
            fl.write("pair_style \t meam\n")
            if potPath is None:
                fl.write("pair_coeff \t * * pot/library.meam Co Ni Cr Fe Mn pot/params.meam Co Ni Cr Fe Mn\n")
            else:
                fl.write("pair_coeff \t * * " + potPath + "/library.meam Co Ni Cr Fe Mn " +
                         potPath + "/params.meam Co Ni Cr Fe Mn\n")
            fl.write("minimize {1} {0} 500 100000\n".format(ftol, etol_relax))
            fl.write("write_data initial_relax_{0}.data".format(traj))

        with open("in.minim_fin_{0}".format(traj), "w") as fl:
            fl.write("units \t metal\n")
            fl.write("atom_style \t atomic\n")
            fl.write("atom_modify \t map array\n")
            fl.write("boundary \t p p p\n")
            fl.write("atom_modify \t sort 0 0.0\n")
            fl.write("read_data \t final_{0}.data\n".format(traj))
            fl.write("pair_style \t meam\n")
            if potPath is None:
                fl.write("pair_coeff \t * * pot/library.meam Co Ni Cr Fe Mn pot/params.meam Co Ni Cr Fe Mn\n")
            else:
                fl.write("pair_coeff \t * * " + potPath + "/library.meam Co Ni Cr Fe Mn " +
                         potPath + "/params.meam Co Ni Cr Fe Mn\n")
            fl.write("minimize {1} {0} 500 100000\n".format(ftol, etol_relax))
            fl.write("write_data final_relax_{0}.data".format(traj))


def write_final_NEB_relax_fix(Ntr, Natoms=499):
    for tr in range(Ntr):
        with open("final_relax_{0}.data".format(tr), "r") as fl:
            lines = fl.readlines()

        for lineInd, l in enumerate(lines):
            if "Atoms" in l:
                break

        lineInd += 2
        coords = []
        for lInd, l in enumerate(lines[lineInd: lineInd + Natoms]):
            l_s = l.split()
            coords.append("{} {} {} {}\n".format(lInd + 1, l_s[2], l_s[3], l_s[4]))
        assert int(l_s[0]) == Natoms

        with open("final_NEB_coords_{0}.data".format(tr), "w") as fl:
            fl.write("{}\n".format(Natoms))
            fl.writelines(coords)


def write_minim_final_states_relax_fix(SiteIndToSpec, SiteIndToPos, vacSiteInd,
                                       siteIndToNgb, jmp, TopLines, writeAll=False):

    for traj in range(SiteIndToSpec.shape[0]):
        with open("final_{}.data".format(traj), "w") as fl:
            fl.writelines(TopLines)
            counter = 1
            for idx in range(SiteIndToSpec.shape[1]):

                spec = SiteIndToSpec[traj, idx]
                if spec == 0:  # if the site is vacant
                    assert idx == vacSiteInd[traj], "{} {}".format(idx, SiteIndToSpec[traj, idx])
                    continue

                # the neighbor will move to the vacancy site
                if idx == siteIndToNgb[vacSiteInd[traj], jmp]:
                    pos = SiteIndToPos[vacSiteInd[traj]]
                else:
                    pos = SiteIndToPos[idx]
                fl.write("{} {} {} {} {}\n".format(counter, spec, pos[0], pos[1], pos[2]))
                counter += 1

        with open("JumpSite_{}_{}.data".format(traj, jmp), "w") as fl:
            fl.write("{}  {}  {}".format(jmp, siteIndToNgb[vacSiteInd[traj], jmp], SiteIndToPos[siteIndToNgb[vacSiteInd[traj], jmp]]))

        if writeAll:
            with open("final_{}.data".format(traj), "r") as fl:
                lines = fl.readlines()
            with open("final_{}_{}.data".format(traj, jmp), "w") as fl:
                fl.writelines(lines)


def write_init_states(SiteIndToSpec, SiteIndToPos, vacSiteInd, TopLines):
    Ntr = vacSiteInd.shape[0]
    for traj in range(Ntr):
        with open("initial_{}.data".format(traj), "w") as fl:
            fl.writelines(TopLines)
            counter = 1
            for idx in range(SiteIndToSpec.shape[1]):
                spec = SiteIndToSpec[traj, idx]
                if spec == 0:  # if the site is vacant
                    assert idx == vacSiteInd[traj], "{} {}".format(idx, SiteIndToSpec[traj, idx])
                    continue
                pos = SiteIndToPos[idx]
                fl.write("{} {} {} {} {}\n".format(counter, spec, pos[0], pos[1], pos[2]))
                counter += 1


def write_final_states(SiteIndToPos, vacSiteInd, siteIndToNgb, jInd, writeAll=False):
    Ntr = vacSiteInd.shape[0]
    for traj in range(Ntr):
        with open("final_{}.data".format(traj), "w") as fl:
            fl.write("{}\n".format(1))
            pos = SiteIndToPos[vacSiteInd[traj]]
            ngbInd = siteIndToNgb[vacSiteInd[traj], jInd]
            if ngbInd > vacSiteInd[traj]:
                LammpsAtomInd = ngbInd
            else:
                LammpsAtomInd = ngbInd + 1
            fl.write("{} {} {} {}\n".format(LammpsAtomInd, pos[0], pos[1], pos[2]))

        if writeAll:
            with open("final_{}.data".format(traj), "r") as fl:
                lines = fl.readlines()
            with open("final_{}_{}.data".format(traj, jInd), "w") as fl:
                fl.writelines(lines)


def getJumpSelects(rates):
    Ntr = rates.shape[0]
    timeStep = 1. / np.sum(rates, axis=1)
    ratesProb = rates * timeStep.reshape(Ntr, 1)
    ratesProbSum = np.cumsum(ratesProb, axis=1)
    rn = np.random.rand(Ntr)
    jumpID = np.zeros(Ntr, dtype=int)
    for tr in range(Ntr):
        jSelect = np.searchsorted(ratesProbSum[tr, :], rn[tr])
        jumpID[tr] = jSelect
    # jumpID, rateProbs, ratesCum, rndNums, time_step
    return jumpID, ratesProb, ratesProbSum, rn, timeStep


def updateStates(SiteIndToNgb, Nspec, SiteIndToSpec, vacSiteInd, jumpID, dxList):
    Ntraj = jumpID.shape[0]
    jumpAtomSelectArray = np.zeros(Ntraj, dtype=int)
    X = np.zeros((Ntraj, Nspec, 3), dtype=float)
    for tr in range(Ntraj):
        assert SiteIndToSpec[tr, vacSiteInd[tr]] == 0
        jumpSiteSelect = SiteIndToNgb[vacSiteInd[tr], jumpID[tr]]
        jumpAtomSelect = SiteIndToSpec[tr, jumpSiteSelect]
        jumpAtomSelectArray[tr] = jumpAtomSelect
        SiteIndToSpec[tr, vacSiteInd[tr]] = jumpAtomSelect
        SiteIndToSpec[tr, jumpSiteSelect] = 0  # The next vacancy site
        vacSiteInd[tr] = jumpSiteSelect
        X[tr, 0, :] = dxList[jumpID[tr]]
        X[tr, jumpAtomSelect, :] = -dxList[jumpID[tr]]

    return jumpAtomSelectArray, X


def DoKMC(T, startStep, Nsteps, StateStart, dxList,
          SiteIndToSpecAll, vacSiteIndAll, batchSize, SiteIndToNgb, chunkSize, PotPath,
          SiteIndToPos, WriteAllJumps=False, ftol=0.001, etol=1e-7, NImages=5):
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
    write_input_files(chunkSize, potPath=PotPath, etol=etol, ftol=ftol)

    start = time.time()

    for step in range(Nsteps):
        for chunk in range(0, Ntraj, chunkSize):
            # Write the initial states from last accepted state
            sampleStart = chunk
            sampleEnd = min(chunk + chunkSize, Ntraj)

            SiteIndToSpec = FinalStates[sampleStart: sampleEnd].copy()
            vacSiteInd = FinalVacSites[sampleStart: sampleEnd].copy()

            write_init_states(SiteIndToSpec, SiteIndToPos, vacSiteInd, Initlines[:lineStartCoords])

            rates = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]))
            barriers = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]))
            ISE = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]))
            TSE = np.zeros((SiteIndToSpec.shape[0], NImages-2, SiteIndToNgb.shape[1]))
            FSE = np.zeros((SiteIndToSpec.shape[0], SiteIndToNgb.shape[1]))
            for jumpInd in range(SiteIndToNgb.shape[1]):
                # Write the final states in NEB format for lammps
                write_final_states(SiteIndToPos, vacSiteInd, SiteIndToNgb, jumpInd, writeAll=WriteAllJumps)

                # Then run lammps
                commands = [
                    "mpirun -np {0} --oversubscribe $LMPPATH/lmp -log out_{1}.txt -screen screen_{1}.txt -p {0}x1 -in in.neb_{1}".format(NImages, traj)
                    for traj in range(SiteIndToSpec.shape[0])
                ]
                cmdList = [subprocess.Popen(cmd, shell=True) for cmd in commands]

                # wait for the lammps commands to complete
                for c in cmdList:
                    rt_code = c.wait()
                    assert rt_code == 0  # check for system errors

                # Then read the forward barrier -> ebf
                for traj in range(SiteIndToSpec.shape[0]):
                    with open("out_{0}.txt".format(traj), "r") as fl:
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


# ToDo: This is similar to the previous function, can we modularize things further?
def DoKMC_Relax_Fix(T, startStep, Nsteps, StateStart, dxList,
          SiteIndToSpecAll, vacSiteIndAll, batchSize, SiteIndToNgb, chunkSize, PotPath,
          SiteIndToPos, WriteAllJumps=False, ftol=0.001, etol=1e-7, etol_relax=1e-8, NImages=5):
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
                                                   SiteIndToNgb, jumpInd, Initlines[:lineStartCoords],
                                                   writeAll=WriteAllJumps)

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
