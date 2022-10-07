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
import os
import glob

# First, we write a lammps input script for this run
def write_lammps_input(jobID):

    seed = np.random.randint(0, 10000)
    st = "units \t metal\n"
    st += "atom_style \t atomic\n"
    st += "atom_modify \t map array\n"
    st += "boundary \t p p p\n"
    st += "atom_modify \t sort 0 0.0\n"
    st += "read_data \t inp_MC_{0}.data\n".format(jobID)
    st += "pair_style \t meam\n"
    st += "pair_coeff \t * * ../../../pot/library.meam Co Ni Cr Fe Mn ../../../pot/params.meam Co Ni Cr Fe Mn\n"
    st += "neighbor \t 0.3 bin\n"
    st += "neigh_modify \t delay 0 every 1 check yes\n"
    st += "variable x equal pe\n"
    st += "displace_atoms all random 0.1 0.1 0.1 {0}\n".format(seed)
    st += "minimize	\t 1e-5 0.0 1000 10000\n"
    st += "run 0\n"
    st += "print \"$x\" file Eng_{0}.txt".format(jobID)
    with open("in_{0}.minim".format(jobID), "w") as fl:
        fl.write(st)


# Next, we write the MC loop
def MC_Run(T, SwapRun, ASE_Super, Nprocs, jobID, elems,
           N_therm=2000, N_save=200, serial=True, lastChkPt=0):
    if serial:
        cmdString = "$LMPPATH/lmp -in in_{0}.minim > out_{0}.txt".format(jobID)
    else:
        cmdString = "mpirun -np {0} $LMPPATH/lmp -in in_{1}.minim > out_{1}.txt".format(Nprocs, jobID)

    Natoms = len(ASE_Super)
    N_accept = 0
    N_total = 0
    Eng_steps_all = []
    accepts = []
    rand_steps = []
    swap_steps = []
    
    # Store the starting supercell in the test directory if doing from scratch
    if lastChkPt == 0:
        write_lammps_data("test/inp_MC_test_job_{0}_step_{1}.data".format(jobID, N_total), ASE_Super, specorder=elems)
        with open("test/supercell_{}_test.pkl".format(N_total), "wb") as fl_sup:
            pickle.dump(ASE_Super, fl_sup)

    else:
        Eng_steps_all = list(np.load("Eng_all_steps.npy")[:lastChkPt])
        accepts = list(np.load("accepts_all_steps.npy")[:lastChkPt])
        rand_steps = list(np.load("rands_all_steps.npy")[:lastChkPt])
        swap_steps = list(np.load("swap_atoms_all_steps.npy")[:lastChkPt])


    # write the supercell as a lammps file
    write_lammps_data("inp_MC_{0}.data".format(jobID), ASE_Super, specorder=elems)

    # evaluate the energy
    cmd = subprocess.Popen(cmdString, shell=True)
    rt = cmd.wait()
    assert rt == 0, "job ID : {}".format(jobID)
    start_time = time.time()
    # read the energy
    with open("Eng_{0}.txt".format(jobID), "r") as fl_en:
        e1 = fl_en.readline().split()[0]
        e1 = float(e1)
    
    beta = 1.0/(kB * T)

    while N_total + lastChkPt < SwapRun:
        Eng_steps_all.append(e1)
        
        # Now randomize the atomic occupancies
        site1 = np.random.randint(0, Natoms)
        site2 = np.random.randint(0, Natoms)
        while ASE_Super[site1].symbol == ASE_Super[site2].symbol:
            site1 = np.random.randint(0, Natoms)
            site2 = np.random.randint(0, Natoms)

        # change the occupancies
        tmp = ASE_Super[site1].symbol
        ASE_Super[site1].symbol = ASE_Super[site2].symbol
        ASE_Super[site2].symbol = tmp

        # write the supercell again as a lammps file
        write_lammps_data("inp_MC_{0}.data".format(jobID), ASE_Super, specorder=elems)

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
        rn = np.random.rand()

        if rn < np.exp(-beta * de):
            # Then accept the move
            N_accept += 1
            e1 = e2  # set the next initial state energy to the current final energy
            accepts.append(1)

        else:
            # reject the move by reverting the occupancies to initial state values
            tmp = ASE_Super[site1].symbol
            ASE_Super[site1].symbol = ASE_Super[site2].symbol
            ASE_Super[site2].symbol = tmp
            accepts.append(0)

        rand_steps.append(rn)
        swap_steps.append([site1, site2])
        
        # save the history at all steps
        np.save("Eng_all_steps.npy", np.array(Eng_steps_all))
        np.save("accepts_all_steps.npy", np.array(accepts))
        np.save("rands_all_steps.npy", np.array(rand_steps))
        np.save("swap_atoms_all_steps.npy", np.array(swap_steps))

        N_total += 1

        if N_total%N_save == 0:
            with open("timing.txt", "w") as fl_timer:
                t_now = time.time()
                fl_timer.write("Time Per step ({0} steps): {1}\n".format(N_total, (t_now-start_time)/N_total))

            if N_total + lastChkPt >= N_therm:
                with open("chkpt/supercell_{}.pkl".format(N_total + lastChkPt), "wb") as fl_sup:
                    pickle.dump(ASE_Super, fl_sup)

                with open("chkpt/counter.txt", "w") as fl_counter:
                    fl_counter.write("last step saved\n{}".format(N_total))

        # For the first 20 steps, store all the supercells as well to a test directory
        if N_total <= 20 and lastChkPt == 0:
            
            np.save("test/Eng_steps_test.npy", np.array(Eng_steps_all))
            np.save("test/rand_steps_test.npy", np.array(rand_steps))
            np.save("test/swap_atoms_test.npy", np.array(swap_steps))
            np.save("test/acceptances_test.npy", np.array(accepts))
            # store the supercells and lammps files too
            write_lammps_data("test/inp_MC_test_job_{0}_step_{1}.data".format(jobID, N_total), ASE_Super, specorder=elems)
            with open("test/supercell_{}_test.pkl".format(N_total), "wb") as fl_sup:
                pickle.dump(ASE_Super, fl_sup)

    return N_total, N_accept


if __name__ == "__main__":
    args = list(sys.argv)
    T = float(args[1])
    N_swap = int(args[2])
    N_units = int(args[3])  # dimensions of unit cell
    N_proc = int(args[4])  # No. of procs to parallelize over
    jobID = int(args[5])
    N_save = int(args[6])
    N_eqb = int(args[7])
    MakeVac = bool(int(args[8]))
    UseLastChkPt = bool(int(args[9])) if len(args)==10 else False

    print("Using CheckPoint : {}".format(UseLastChkPt))

    elems = ["Co", "Ni", "Cr", "Fe", "Mn"]

    elemsToNum = {}
    for elemInd, el in enumerate(elems):
        elemsToNum[el] = elemInd + 1

    if UseLastChkPt: 
        ChkPtFiles=os.getcwd() + "/chkpt/*.pkl"
        files=glob.glob(ChkPtFiles)
        
        if len(files) == 0:
            raise FileNotFoundError("No checkpoint found.")

        else:
            max_file = max(files, key=os.path.getctime) # Get the file created last
            with open(max_file, "rb") as fl:
                superFCC = pickle.load(fl)
        
            lastFlName=max_file.split("/")[-1]
            lastSave=int(lastFlName[10:-4])
            print("Loading checkpointed step : {} for run : {}".format(lastSave, jobID))
        

    else:
        lastSave=0
        
        RunPath = os.getcwd() + "/"
        if not os.path.isdir(RunPath + "chkpt"):
            os.mkdir(RunPath + "chkpt")
        if not os.path.isdir(RunPath + "test"):
            os.mkdir(RunPath + "test")

        # Create an FCC primitive unit cell
        a = 3.59
        fcc = crystal('Ni', [(0, 0, 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90], primitive_cell=True)

        # Form a supercell with a vacancy at the centre
        superlatt = np.identity(3) * N_units
        superFCC = make_supercell(fcc, superlatt)
        Nsites = len(superFCC.get_positions())
        
        
        # randomize occupancies of the sites
        Nperm = 10
        Indices = np.arange(Nsites)
        for i in range(Nperm):
            Indices = np.random.permutation(Indices)
        
        NSpec = len(elems)
        partition = Nsites // NSpec

        for i in range(NSpec):
            for at_Ind in range(i * partition, (i + 1) * partition):
                permInd = Indices[at_Ind]
                superFCC[permInd].symbol = elems[i]
 
        if MakeVac:
            print("Putting vacancy at site 0")
            assert np.allclose(superFCC[0].position, 0)
            del (superFCC[0])
        
        Natoms = len(superFCC)

        # save the initial supercell
        with open("superInitial_{}.pkl".format(jobID), "wb") as fl:
            pickle.dump(superFCC, fl)
        
        
    
    # Run MC
    if not UseLastChkPt:
        # A new random seed should not be generated if continuing.
        write_lammps_input(jobID)

    start = time.time()
    N_total, N_accept = MC_Run(T, N_swap, superFCC, N_proc, jobID, elems, N_therm=N_eqb, N_save=N_save, lastChkPt=lastSave)
    end = time.time()
    print("Thermalization Run acceptance ratio : {}".format(N_accept/N_total))
    print("Thermalization Run accepted moves : {}".format(N_accept))
    print("Thermalization Run total moves : {}".format(N_total))
    print("Thermalization Time Per iteration : {}".format((end-start)/N_total))
    with open("superFCC_therm.pkl", "wb") as fl:
        pickle.dump(superFCC, fl)

