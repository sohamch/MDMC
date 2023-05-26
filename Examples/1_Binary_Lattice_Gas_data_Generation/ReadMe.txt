The example notebooks in this directory give hand-on introduction into the basics of data set generation to train our cluster expansion 
and neural network models for predicting transport coefficients. To do this, we take the example of the simplest system
in our paper, the SR-2 system, or the binary lattice gas.

In this systems, vacancy exchange rates are fixed for the two atomic species. One of them, called the "fast" species has
an exchange rate of 1.0, while the other, the "slow" species, has an exchange rate of 0.001. For more details about this
kind of system, see the original variation principle by Dallas R. Trinkle in phys. rev. lett. (2018) as well as the supplementary
materials of our paper.

We first show how to generate single-step data sets to train the cluster expansion and neural networks in the notebook
"1_makeDataSet.ipynb", where we use the LatGas.py module (in the Utils directory) to simulate lattice gases.

Then, we test the data set in "2_test_Dataset.ipynb" giving a more hands-on view into what the quantities in the dataset
represent.
