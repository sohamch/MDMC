#!/bin/bash
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --job-name="MyKMCJob"
#SBATCH --output="MyKMCJob_%j.out"
#SBATCH --error="MyKMCJob_%j.err"

# The following lines may need to be included to activate a conda enviroment of your choice
# module load avalaiable_conda_module_in_the_HPC_facility
# conda activate My_fav_Env

VKMC=/path/to/the/local/copy/of/the/VKMC/repository # We Need to add the full path to the local copy of the VKMC respository here.
LMP_KMC=$VKMC/Utils/MEAM_KMC # The path to the directory containing the NEB_steps_multiTraj.py module to perform KMC steps with LAMMPS NEB calculations for multiple KMC trajectories.
potpath=$VKMC/Utils/pot # Path for the Lammps potential
CD=$VKMC/CrysDat_FCC/CrystData_ortho_5_cube.h5 # Path for the Crystal data. In our simulations we used orthogonal supercells.
T=1073 # Temperature

IF=InitStates/statesAll_${T}.npy
# path to initial states extracted from Monte Carlo runs.

# The option -ns dictates how many total KMC steps we want to do. For each step, an hdf5 (.h5) file is created in the directory that contains information about that KMC
# step such as the chosen jump from each of "s" to "s+e"th state, the displacements of the species, the barriers of the jumps, etc. The format of such files are described in
# the ReadMe.txt file.


# We will perform KMC steps out of our starting states by computing barriers using LAMMPS.
# We will do so in this example in groups of 400 states up to the 2000th state in the file statesAll_${T}.npy

mkdir $T # make a directory for this temperature if the first time.
cd $T
for s in 0 400 800 1200 1600 # The index of the starting states - 0th, 400th...1600th state in the numpy file statesAll_${T}.npy
do
e=$((s + 400)) # the ending state index - 400 from the starting index
mkdir states_${s}_${e} # make a directory to store results for this set of starting states.
cd states_${s}_${e}
# The input arguments below were also used in our simulations.
python $LMP_KMC/NEB_steps_multiTraj.py -ni 11 -etol 0.0 -ftol 0.001 -k 10.0 -p 10.0 -T $T -cr $CD -u 5 -pp $MP -if $IF -ns 2 -idx {0} -bs {1} -cs 1 -dmp -dpf args_${T}_${SLURM_JOBID}.txt > JobOut_${s}_${e} 2>&1 &
cd ../
done
wait

cd ../ # step out to the job launch directory
