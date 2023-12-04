-This file desribes the options to use the NEB_steps_multiTraj.py module in the VKM/Utils/MEAM_KMC directory
to perform KMC steps with LAMMPS to do the climbing-image NEB calculations to compute migration barriers.

Section 1 - Prerequisites:
-To start NEB calculations, we need to have initial states saved into numpy arrays as described in the example directory
"4_Monte_Carlo_LAMMPS".
-The location of the LAMMPS executable must be saved in the $PATH environment variable with the name LMPPATH.

Note - In these NEB calculations, we use 11 total images. The climbing image method was not used.
We start from on-lattice positions and relax the initial, intermediate and final states using the ABCFIRE algorithm with
a maximum force tolerance of 1e-3 per atom.

To get started:
	1. The file "job_script_KMC.sb" is given as an example to show how to run the KMC simulations in an HPC environment.
	2. The notebook "Extract_data.ipynb" is provided to show how data is saved during the KMC runs and how we can extract
	that data and save it in a format required by the neural network and cluster expansion codes. The format of the data
	is also discussed next in Section 2.


Section 2 - Description of code output
After every "i"th KMC step, the code outputs an hdf5 file with the name "data_{0}_{1}.h5"
where the fields are:
	{0}: startStep (option --startStep) + i + 1
	{1}: option --idx (which state in the initial state file we started from)

The information contained in this file are present as hdf5 fields. The most relevant ones are as follows:

"FinalStates" - the final atomic configurations after the vacancy jump (for each of the states simulated). (shape Nsamples x Nsites)
"SpecDisps" - the displacement of each species during the KMC step (for each of the states simulated). (shape Nsamples x Nspecies x 3)
"times" - the escape time of the jump (for each of the states simulated), i.e, the inverse of the total rate. (shape Nsamples)
"AllJumpRates" - the rate of each of the possible vacancy jumps. (shape Nsamples x z)
"AllJumpBarriers" - the barrier of each of the possible vacancy jumps computed by NEB calculations. (shape Nsamples x z)
"AllJumpImageEnergies" - Energies of each of the NEB images. (shape Nsamples x NImages x z)
"AllJumpImageRDs" - Reaction coordiantes of each NEB image. (shape Nsamples x NImages x z)
"JumpSelects" - The jump selected (index 0 to z-1) from each sample during the KMC step. (shape Nsamples)
"TestRandNums" - The random number used for each sample to decide the KMC step. (shape Nsamples).

Suppose we want to continue our simulation after 3 KMC steps have been performed, The code will read in the states stored in the last file saved as the initial states
to continue the simulation from.


Section 3 - Input parameters for Kinetic Monte Carlo simulations with LAMMPS.
Input parameters for Kinetic Monte Carlo simulations with LAMMPS.

optional arguments:
  -h, --help            show this help message and exit
  -cr /path/to/crys/dat, --CrysDatPath /path/to/crys/dat
                        Path to crystal Data. (default: None)
  -pp /path/to/potential/file, --PotPath /path/to/potential/file
                        Path to the LAMMPS MEAM potential. (default: None)
  -if /path/to/initial/file.npy, --InitStateFile /path/to/initial/file.npy
                        Path to the .npy file storing the 0-step states from
                        Metropolis Monte Carlo. (default: None)
  -a0 float, --LatPar float
                        Lattice parameter - multiplied to displacements and
                        usedto construct LAMMPS coordinates. (default: 3.595)
  -pr, --Prim           Whether to use primitive cell (default: False)

  -T int, --Temp int    Temperature to read data from (default: None)

  -ni int, --NImages int
                        How many NEB Images to use. Must be odd number.
                        (default: 11)

  -ns int, --Nsteps int
                        How many steps to run. (default: 100)

  -ftol float, --ForceTol float
                        Force tolerance for ending NEB calculations. (default:
                        0.001)

  -etol float, --EnTol float
                        Relative Energy change tolerance for ending NEB
                        calculations. (default: 0.0)

  -th float, --DispThreshold float
                        Maximum allowed displacement after relaxation in Agnstroms.
                        (default: 1.0)

  -ts float, --TimeStep float
                        Relative Energy change tolerance for ending NEB
                        calculations. (default: 0.001)

  -k float, --SpringConstant float
                        Parallel spring constant for NEB calculations.
                        (default: 10.0)

  -p float, --PerpSpringConstant float
                        Perpendicular spring constant for NEB calculations.
                        (default: 10.0)

  -u int, --Nunits int  Number of unit cells in the supercell. (default: 8)

  -idx int, --StateStart int
                        The starting index of the state for this run from the
                        whole data set of starting states. The whole data set
                        is loaded, and then samples starting from this index
                        to the next "batchSize" number of states are loaded.
                        (default: 0)

  -bs int, --batchSize int
                        How many initial states starting from StateStart
                        should initially be loaded. (default: 200)

  -cs int, --chunkSize int
                        How many samples to do NEB calculations for at a time.
                        (default: 20)

  -wa, --WriteAllJumps  Whether to store final style NEB files for all jumps
                        separately. (default: False)

  -mpc int, --MemPerCpu int
                        Memory per cpu (integer, in megabytes)for NEB
                        calculations. (default: 1000)

  -dmp, --DumpArguments
                        Whether to dump all the parsed arguments into a text
                        file. (default: False)

  -dpf string, --DumpFile string
                        The file in the run directory where all the args will
                        be dumped. (default: ArgFiles)