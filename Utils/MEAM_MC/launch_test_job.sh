#!/bin/bash
potpath="/mnt/WorkPartition/Work/Research/UIUC/MDMC/Utils/pot"

mkdir MC_test_traj
cd MC_test_traj

mkdir 1
cd 1
python3 ../../Init_state_MC.py -nosr -pp $potpath -T 1073 -etol 1e-8 -ftol 0.0 -na 99 100 100 100 100 -nt 400 -ne 5 -ns 1 -dmp -dpf test_args_first_run.txt

cp in.minim in_run_1.minim

# Now re-run from last checkpoint
python3 ../../Init_state_MC.py -etol 1e-8 -ftol 0.0 -nosr -pp $potpath -T 1073 -ckp -nt 450 -ne 5 -ns 1 -dmp -dpf test_args_second_run.txt

# Do it again, but this time load the backed up history
rm *.npy # delete the running array so the code is forced to load a backup
python3 ../../Init_state_MC.py -etol 1e-8 -ftol 0.0 -nosr -pp $potpath -T 1073 -ckp -nt 500 -ne 5 -ns 1 -dmp -dpf test_args_second_run.txt

