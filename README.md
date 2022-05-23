fair-baselines

## Overview ##
Implementation of various methods for fair classification.
Methods include:

1) Variational Fair Autoencoder (VFAE), Louizos et al. 2016
2) Controllable Invariance through Adversarial Feature Learning, Xie et al. 2017
3) FR-Train: FR-Train: A Mutual Information-Based Approach to Fair and Robust Training, Roh et al. 2020
4) Wasserstein Fair Classification, Jiang et al. 2020

## Code Structure ##
Various neural architectures are defined within the ``models`` directory.

Data utilities are managed under the ``data_parsing`` directory and bundle information about the German and Adult datasets from csv files.

Actual data for the German and Adult tasks are included under ``data``

Training scripts are included in the ``scripts`` directory. They could almost certainly be consolidated, but right now it's just a lot of separate scripts for each method and dataset.

If you find this useful, please cite 1) whatever baseline methods you used and 2) "Prototype Based Classification from Hierarchy to Fairness" by Tucker and Shah (2022), for which this code was developed.

