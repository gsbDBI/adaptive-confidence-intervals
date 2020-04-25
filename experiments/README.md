This directory contains scripts to run experiments and reproduce plots in our paper. 

### File description
- `script.py`: Python script to run experiments.
- `compute_W_decorrelation_lambda.py`: use Monte Carlo simulation to precompute bias-variance tradeoff parameter _lambda_ in W_decorrelation [paper](https://arxiv.org/abs/1712.06695).
- `plot_utils.py`: helper functions to make plots.
- `plot.ipynb`: make plots in the paper.

### Instructions for running experiments and making plots
1. run `compute_W_decorrelation_lambda.py` to precompute _lambda_ for W-decorrelation. (make sure the Monte Carlo setup is consistent with your experiments).
2. run `script.py` for experiments. Results will be saved in the ./results/ directory.
3. run `plot.ipynb` to generate plots. 
