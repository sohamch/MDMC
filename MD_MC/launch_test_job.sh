#!/bin/bash
mkdir MC_test_traj
cd MC_test_traj

mkdir 1
cd 1
python3 ../../Init_state_MC.py 1073 100 8 1 1 1 1 1

# remove the last 10 checkpoint files to simulate incomplete run
for i in {90..100}
do
	rm chkpt/supercell_${i}.pkl
done

# copy the lammps command file before continuation runs
# we'll compare the two command files later to ensure
# no new command files were created and that the
# same random seed was used to displace atoms in both the previous
# and the continuation runs
cp in_1.minim in_1_run_1.minim

# Now re-run from last checkpoint
python3 ../../Init_state_MC.py 1073 100 8 1 1 1 1 1 1
