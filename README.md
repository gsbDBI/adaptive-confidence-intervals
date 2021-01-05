<h1 align="center">Adaptive Confidence Intervals</h1>
Models for paper [_Confidence Intervals for Policy Evaluation in Adaptive Experiments_](https://arxiv.org/abs/1911.02768).
<p align="center">
  Table of contents </br>
  <a href="#overview">Overview</a> •
  <a href="#development-setup">Development Setup</a> •
  <a href="#quickstart-with-model">Quickstart</a> •
  <a href="#acknowledgements">Acknowledgements</a> 
</p>

---
# Overview

---
# Development setup (recommended)

```bash
conda create --name adaptive_CI python=3.7
conda activate adaptive_CI
python setup.py develop
```
---
# Quickstart with model

- Directory `./experiments` contains scripts to run experiments and make plots shown in the paper. 
- Directory `./adaptive_CI` is a Python module for doing adaptive inference developed in the paper.

To reproduce results, the simplest way is to enter into [./experiments](https://github.com/gsbDBI/adaptive-confidence-intervals/tree/master/experiments) and follow the instruction in `./experiments/README.md`.

Interesting readers can modify the `./adaptive_CI` module to adapt to their specific use. 

---
# Acknowlegements

To reference, please cite the paper: [_Confidence Intervals for Policy Evaluation in Adaptive Experiments_](https://arxiv.org/abs/1911.02768).
