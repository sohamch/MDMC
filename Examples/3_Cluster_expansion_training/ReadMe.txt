The provided job script example shows how to use the cluster expansion code LBAM_dataset.py in the "VCE" directory of the repository home to compute the transport coefficient.

Although the example job script provides sufficient options to get started using the cluster expansion code for predicting transport coefficient,
all the other options are printed below.

usage: LBAM_DataSet.py [-h] [-T int] [-DP /path/to/data] [-prm]
                       [-cr /path/to/crys/dat] [-mo int] [-cc float] [-sp int]
                       [-vsp int] [-nt int] [-aj] [-scr] [-ait] [-svc] [-svj]
                       [-ao] [-eo] [-rc RCOND] [-nex] [-d] [-dpf F]

Input parameters for cluster expansion transport coefficient prediction

optional arguments:
  -h, --help            show this help message and exit

  -T int, --Temp int    Temperature of data set (or composition for the binary
                        alloys. (default: None)

  -DP /path/to/data, --DataPath /path/to/data
                        Path to Data file. (default: None)

  -cr /path/to/crys/dat, --CrysDatPath /path/to/crys/dat
                        Path to crystal Data. (default: None)

  -red, --ReduceToPrimitve
                        Whether to reduce the crystal from the crystal data
                        file to a primitive crystal. Used to map sites in an orthogonal supercell to a primitve supercell (default: False)

  -mo int, --MaxOrder int
                        Maximum sites to consider in a cluster. (default:
                        None)

  -cc float, --ClustCut float
                        Maximum distance between sites to consider in a
                        cluster in lattice parameter units. (default: None)

  -sp int, --SpecExpand int
                        Which species to compute transport coefficients for. (default: 5)

  -vsp int, --VacSpec int
                        Index of vacancy species. (default: 0)

  -nt int, --NTrain int
                        No. of training samples. (default: 10000)

  -aj, --AllJumps       Whether to train on all jumps or train KMC-style.
                        (default: False)

  -scr, --Scratch       Whether to create new network and start from scratch
                        (default: False)

  -ait, --AllInteracts  Whether to consider all interactions, or just the ones
                        that contain the jump sites. (default: False)

  -svc, --SaveCE        Whether to save the cluster expansion. (default:
                        False)

  -svj, --SaveJitArrays
                        Whether to store arrays for JIT calculations.
                        (default: False)

  -ao, --ArrayOnly      Use the run to only generate the Jit arrays - no
                        transport calculation. (default: False)

  -eo, --ExpandOnly     Use the run to only generate and save rate and
                        velocity expansions to compute transport calculations-
                        no transport calculation. These expansions can then be
                        loaded and reused later on if needed. (default: False)

  -rc RCOND, --rcond RCOND
                        Threshold for zero singular values. (default: 1e-08)

  -nex, --NoExpand      Use the run to use saved rate and velocity expansions
                        to compute transport calculations .without having to
                        generate them again. (default: False)

  -d, --DumpArgs        Whether to dump arguments in a file (default: False)

  -dpf F, --DumpFile F  Name of file to dump arguments to (can be the jobID in
                        a cluster for example). (default: None)