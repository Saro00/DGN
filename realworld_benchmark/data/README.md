# Datasets

## Overview

The following datasets can be downloaded with the script download_datasets.sh':

- `ZINC`, a dataset of molecules. The file 'molecules.py' contains the classes used to load and pre-process the pkl file. The pkl file contains an object of the class 'MoleculeDataset'. The first 5 eigenvectors are added to DGLGraphs as nodes' features with key 'eig'. They are computed once before starting the training.

- `CIFAR10`, a dataset of images saved as graphs. The file 'superpixels.py' contains the classes used to load and pre-process the pkl file. The pkl file contains an object of the class 'SuperPixDataset'. The first 5 eigenvectors are added to DGLGraphs as nodes' features with key 'eig'. They are computed once before starting the training.

- `PATTERN`, a synthetic dataset generated with Stochastic Block Models. The file 'SBMs.py' contains the classes used to load and pre-process the pkl file. The pkl file contains an object of the class 'SBMsDataset'. The first 5 eigenvectors are added to DGLGraphs as nodes' features with key 'eig'. They are computed once before starting the training.
