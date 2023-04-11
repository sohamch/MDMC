# VKMC

## A repository to compute transport coefficients in FCC CoNiCrFeMn High Entropy Alloys using non-linear Deep Learning and Linear vector-valued Cluster Expansion in combination with the Variational Principle of Mass Transport (Dallas R. Trinkle, Phys. Rev. Lett., 2018).

In this project, FCC-based CoNiCrFeMn alloys were simulated using the classical MEAM potential by Choi et. al. (https://www.nature.com/articles/s41524-017-0060-9). This repository contains code to generate the necessary data sets as well as to train the Deep networks and cluster expansion models. An installation of the onsager module by Dallas. R. Trinkle, with the cluster modules is required (https://github.com/DallasTrinkle/Onsager/tree/cluster). The structure of this repository is described as follows:

## (1) Crystal Data

The directory "CrysDat_FCC" contains codes necessary to generate required information about the FCC crystal that are going to be used to generate the neural network and cluster expansion models. The notebook "Supercell_data_generation_1nn.ipynb" contains code to generate this data, while the notebook "Test_Crystal_Data_FCC.ipynb" has code to test the generated crystal data. An example hdf5 crystal data file "CrysData.h5" is also provided.

## (2) Directories "pot", "MEAM_MC" and "MEAM_KMC".

 - The "pot" directory contains the MEAM potential files obtained from Choi et.al.'s paper (https://www.nature.com/articles/s41524-017-0060-9) in the format required by LAMMPS for a CoNiCrFeMn High Entropy Alloy.
 
 - The "MEAM_MC" directory contains a python script "Init_state_MC.py" to generate equilibrium states at different temperatures using Metropolis Monte Carlo. The bash script "launch_test_job.sh" launches a single test calculation, including restarting from checkpoints, and the generated states are then tested in the jupyter notebook "test_MC_states.ipynb".

- The "MEAM_KMC" directory contain the python script "NEB_steps_multiTraj.py" to do kinetic Monte Carlo simulations of atomic diffusion using NEB calculations in LAMMPS.

(More updates to come - Aprill 11, 2023)
