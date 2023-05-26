The example notebooks in this directory give hand-on introduction into the basics of data set generation to train our cluster expansion 
and neural network models for predicting transport coefficients. To do this, we take the example of the simplest system
in our paper, the SR-2 system, or the binary lattice gas.

In this systems, vacancy exchange rates are fixed for the two atomic species. One of them, called the "fast" species has
an exchange rate of 1.0, while the other, the "slow" species, has an exchange rate of 0.001. For more details about this
kind of system, see the original variation principle by Dallas R. Trinkle in phys. rev. lett. (2018) as well as the supplementary
materials of our paper.

Pre-requisites:
-We need to specify a path to a crystal data file that contains and supercell and symmetry-related information.
Since in this repository we simulate only FCC-based alloys, an example crystal data file for FCC solids named "CrystData.h5" can be found
in the "CrysDat_FCC" directory in the repository home directory, where jupyter notebooks are also given that illustrate how to generate the
crystal data and what data they contain. This file has also been included in our data base along with all our data sets since it was used
in all our simulations.

IMPORTANT: Note that to generate the datasets and train/evaluate the networks or cluster expansion models, the same crystal data
file must be used, since it contains the group operations and jump directions in a specified order.

We first show how to generate single-step data sets to train the cluster expansion and neural networks in the notebook
"1_makeDataSet.ipynb", where we use the LatGas.py module (in the Utils directory) to simulate lattice gases.

Then, we test the data set in "2_test_Dataset.ipynb" giving a more hands-on view into what the quantities in the dataset
represent.
