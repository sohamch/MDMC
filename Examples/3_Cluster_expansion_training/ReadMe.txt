The provided job script example shows how to use the cluster expansion code LBAM_dataset.py in the "VCE" directory of the repository home to compute the transport coefficient.

In our example, the calculation typically proceeds in two stages:

1. First, a "dry run" is done in which geometrical and species information about all the clusters are gathered and stored in an hdf5 file called "JitArrays.h5".

- The advantage is that this hdf5 file can be reused for all the binary FCC alloy systems
with any composition of the slow species, or 5-component High entropy alloy datasets at any temperature.

2. In the second stage, for all the datasets of the same system under different conditions as described in the previous point's examples,
the cluster information stored in "JitArrays.h5" is directly used to initiate a Just-in-time compiled linear basis approximation solver which
computes the transport coefficients without having to generate cluster expansions all over again.

The example job script gives a hands-on introduction to this for the case of predicting transport coefficients in the "fast" species in the
binary alloy in our first example directory ("1_Binary_Lattice_Gas_data_Generation"). In the example, we use 2-body, first nearest neighbor clusters
since they are quite fast to compute.

Although the example job script provides sufficient options to get started using the cluster expansion code for predicting transport coefficient,
all the other options are printed below for the interested reader.

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

  -prm, --Perm          Whether to mix all the data before splitting into
                        training and testing. A mixing array is required to be
                        present in the h5 data set. (default: False)

  -cr /path/to/crys/dat, --CrysDatPath /path/to/crys/dat
                        Path to crystal Data. (default: None)

  -mo int, --MaxOrder int
                        Maximum sites to consider in a cluster. (default:
                        None)

  -cc float, --ClustCut float
                        Maximum distance between sites to consider in a
                        cluster. (default: None)

  -sp int, --SpecExpand int
                        Which species to expand. (default: 5)

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

