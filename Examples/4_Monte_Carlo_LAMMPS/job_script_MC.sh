#!/bin/bash
## The following lines describe slurm job parameters to run the calculations in an HPC environment with SLURM
## They might need to be modified to the particular HPC facility's specification.

#SBATCH --job-name="MyJob"
#SBATCH --output="MyJob_%j.out"
#SBATCH --error="MyJob_%j.err"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# The following lines may need to be included to activate a conda enviroment of your choice
# module load avalaiable_conda_module_in_the_HPC_facility
# conda activate My_fav_Env

# Path for the Lammps potential
VKMC=/path/to/the/local/copy/of/the/VKMC/repository # We Need to add the full path to the local copy of the VKMC respository here.
potpath=$VKMC/Utils/pot

# Path to the directory containing the Monte Carlo Python module
$LMP_MC=$VKMC/Utils/MEAM_MC/


# Runs from scratch
# We'll simulate 16 independent trajectories at 1073K, each starting from a random state, and 250 states gathered from each trajectory.
# We will initialize 512 atom supercells with one vacancy (default option) and use 103, 102, 102, 102, 102 atoms of Co, Ni, Cr Fe and Mn
# respectively.
# In our paper, 5 total such simulations were run (16 trjaectories each, for a total of 80 trajectories), each with one extra atom of
# Co, Ni, Cr, Fe and Mn respectively.

T=1073
jobID=1 # or 2 or 3 or 4 or 5 depending on which simulation is being done.
for i in {1..16} # For each jobID, we generate 16 trajectories with specified atom counts.
do
	mkdir ${T}_${jobID}_${i} # This is the directory in which all run results will be stored.
	cd ${T}_${jobID}_${i}
	python -u $LMP_MC/Init_state_MC.py -pp $potpath -T $T -na 103 102 102 102 102 -nt 60000 -ne 2000 -ns 200 -dmp -dpf args_run_${i}.txt > JobOut_${i}.txt 2>&1 & 
	cd ../
done
wait

# continuation runs - to continue, we specify the -ckp flag (see the ReadMe.txt file)
# This will restart the simulations from two supercells before the last saved ones (in case there were problems due to early job termination etc)
#for i in {1..16}
#do
#	cd ${T}_${jobID}_${i}
#	cp History_backup/* . # Copy all saved Monte Carlo decisions and energies. These will be updated from the checkpoint onward.
#	python -u $LMP_MC/Init_state_MC.py -pp $potpath -T $T -ckp -nt 60000 -ne 2000 -ns 200 -dmp -dpf args_run_${i}.txt > JobOut_${i}.txt 2>&1 & 
#	cd ../
#done
#wait
