<h1 align="center">Adaptive Confidence Intervals</h1>

Models for paper [_Confidence Intervals for Policy Evaluation in Adaptive Experiments_](https://arxiv.org/abs/1911.02768).

<p align="center">
  Table of contents </br>
  <a href="#overview">Overview</a> •
  <a href="#development-setup">Development Setup</a> •
  <a href="#quickstart-with-model">Quickstart</a> •
  <a href="#acknowledgements">Acknowledgements</a> 
</p>


# Overview

*Note: For any questions, please file an issue.*

Adaptive experimental designs can dramatically improve efficiency in randomized trials. But adaptivity also makes offline policy inference challenging. In the paper [_Confidence Intervals for Policy Evaluation in Adaptive Experiments_](https://arxiv.org/abs/1911.02768), we propose a class of estimators that lead to asymptotically normal and consistent policy evaluation. This repo contains reproducible code for the results shown in the paper. 

We organize the code into two directories:
- [./adaptive_CI](https://github.com/gsbDBI/adaptive-confidence-intervals/tree/master/adaptive_CI) is a Python module for doing adaptive inference developed in the paper. This directory also contains other methods for developing confidence intervals using adaptive data that are compared in the paper, including:
   - naive sample mean using the usual variance estimate;
   - non-asymptotic confidence intervals for the sample mean, based on the method of time-uniform confidence sequences described in [Howard et al. (2021)](https://arxiv.org/pdf/1810.08240.pdf);
   - w-decorrelation confidence intervals, based on method described in [Deshpande et al. (2017)](https://arxiv.org/pdf/1712.06695.pdf).

- [./experiments](https://github.com/gsbDBI/adaptive-confidence-intervals/tree/master/experiments) contains python scripts to run experiments and make plots shown in the paper, including:
   - collecting multi-armed bandits data with a Thompson sampling agent;
   - doing adaptive inference using collected data;
   - saving results and making plots. 

# Development setup

We recommend creating the following conda environment for computation.
```bash
conda create --name adaptive_CI python=3.7
conda activate adaptive_CI
python setup.py develop
```

# Quickstart with model

- To use the adaptive inference methods, please follow the instructions in [./experiments/README.md](https://github.com/gsbDBI/adaptive-confidence-intervals/tree/master/experiments).
- To reproduce results shown in the paper, use
```bash
source activate adaptive
cd experiments
python compute_wdecorrelation_lambda.py
python simulations.py
```
Then open [`./experiments/plots.ipynb`](https://github.com/gsbDBI/adaptive-confidence-intervals/blob/master/experiments/plots.ipynb) to load results and make plots. 


# Acknowledgements
We are grateful for the generous financial support provided by the Sloan Foundation, Office of Naval Research grant N00014-17-1-2131, National Science Foundation grant DMS-1916163, Schmidt Futures, Golub Capital Social Impact Lab, and the Stanford Institute for Human-Centered Artificial Intelligence. Ruohan Zhan acknowledges generous support from the Total Innovation graduate fellowship. In addition, we thank Steve Howard, Sylvia Klosin, Sanath Kumar Krishnamurthy and Aaditya Ramdas for helpful advice.

To reference, please cite the paper: [_Confidence Intervals for Policy Evaluation in Adaptive Experiments_](https://arxiv.org/abs/1911.02768).
