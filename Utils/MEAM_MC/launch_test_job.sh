#!/bin/bash
potpath="/mnt/WorkPartition/Work/Research/UIUC/MDMC/Utils/pot"

mkdir MC_test_traj
cd MC_test_traj

mkdir 1
cd 1
python3 ../../Init_state_MC.py -pp $potpath -T 1073 -na 99 100 100 100 100 -nt 500 -ne 5 -ns 1 -dmp -dpf test_args_first_run.txt

# remove the last 20 files to simulate incomplete run
for i in {480..500}
do
	rm chkpt/supercell_${i}.pkl
done

# copy the lammps command file before continuation runs
# we'll compare the two command files later to ensure
# no new command files were created.
cp in.minim in_run_1.minim

# Now re-run from last checkpoint
python3 ../../Init_state_MC.py -pp $potpath -T 1073 -ckp -nt 500 -ne 5 -ns 1 -dmp -dpf test_args_second_run.txt
