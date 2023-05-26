-This file desribes the options to use the NEB_steps_multiTraj.py module in the VKM/Utils/MEAM_KMC directory
to perform KMC steps with LAMMPS to do the climbing-image NEB calculations to compute migration barriers.

Section 1 - Prerequisites:
-To start NEB calculations, we need to have initial states saved into numpy arrays as described in the example directory
"1_Monte_Carlo_LAMMPS".
-The location of the LAMMPS executable must be saved in the $PATH environment variable with the name LMPPATH.

Note - In these climbing image NEB calculations, we use 3 images for the initial, final and transition states respectively.
We start from on-lattice positions and relax the initial, intermediate and final states using the quickmin algorithm with
relative energy tolerance of 1e-5, and total force tolerance of 1e-5.

To get started:
	1. The file "job_script_KMC.sb" is given as an example to show how to run the KMC simulations in an HPC environment.
	2. The notebook "Extract_data.ipynb" is provided to show how data is saved during the KMC runs and how we can extract
	that data and save it in a format required by the neural network and cluster expansion codes. The format of the data
	is also discussed next in Section 2.


Section 2 - Description of code output
After every "i"th KMC step, the code outputs an hdf5 file with the name "data_{0}_{1}_{2}.h5"
where the fields are:
	{0}: T (option --Temp)
	{1}: startStep (option --startStep) + i + 1
	{2}: option --idx (which state in the initial state file we started from)

The information contained in this file are present as hdf5 fields with the following names ((see the "Extract_data.ipynb" notebook to see how to read them and what information
they contain):
"FinalStates" - the final atomic configurations after the vacancy jump (for each of the states simulated).
"SpecDisps" - the displacement of each species during the KMC step (for each of the states simulated).
"times" - the time of the jump (for each of the states simulated), i.e, the inverse of the total rate of all possible vacancy jumps.
"AllJumpRates" - the rate of each of the possible vacancy jumps.
"AllJumpBarriers" - the barrier of each of the possible vacancy jumps computed by NEB calculations.
"AllJumpISEnergy" - the energies of each of the initial states computed during the NEB calculations.
"AllJumpTSEnergy" - the transition state energies for each jump computed by the NEB calculations.
"AllJumpFSEnergy" - the final state energies for each jump computed by the NEB calculations.
"JumpSelects" - From the jump rates, which particular jump was selected by the KMC algorithm.
"TestRandNums" - The random number used to select the jump, which will be used to test the correctness of the decisions in the "Extract_data.ipynb" notebook.

Suppose we want to continue our simulation after 3 KMC steps have been performed, at which point a data file with {1} = "3" was saved.
We pass 3 as the --startStep argument, and the code will search for this file in the working directory, and read in the states stored there as the initial states
to continue the simulation from.


Section 3 - Input parameters for Kinetic Monte Carlo simulations with LAMMPS.
usage: NEB_steps_multiTraj.py [-h] [-cr /path/to/crys/dat]
                              [-pp /path/to/potential/file]
                              [-if /path/to/initial/file.npy] [-a0 float]
                              [-T int] [-st int] [-ns int] [-u int] [-idx int]
                              [-bs int] [-cs int] [-wa] [-dmp] [-dpf string]

optional arguments:
  -h, --help            show the help message and exit
  
  -cr /path/to/crys/dat, --CrysDatPath /path/to/crys/dat
                        Path to crystal Data. (default: None)
                        
  -pp /path/to/potential/file, --PotPath /path/to/potential/file
                        Path to the LAMMPS MEAM potential. (default: None)
                        
  -if /path/to/initial/file.npy, --InitStateFile /path/to/initial/file.npy
                        Path to the .npy file storing the 0-step states from
                        Metropolis Monte Carlo. (default: None)
                        
  -a0 float, --LatPar float
                        Lattice parameter - multiplied to displacements and
                        usedto construct LAMMPS coordinates. (default: 3.59)
                        
  -T int, --Temp int    Temperature to read data from (default: None)
  
  -st int, --startStep int
                        From which step to start the simulation. Note -
                        checkpointed data file must be present in running
                        directory if value > 0. (default: 0)
                        
                        (see Section 3 below for more details on the save data files)
                        
  -ns int, --Nsteps int
                        How many steps to continue AFTER "starStep" argument.
                        (default: 100)
                        
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
                        
  -dmp, --DumpArguments
                        Whether to dump all the parsed arguments into a text
                        file. (default: False)
                        
  -dpf string, --DumpFile string
                        The file in the run directory where all the args will
                        be dumped. (default: ArgFiles)
