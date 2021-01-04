This directory contains scripts to run experiments and make plots shown in the paper [_Confidence Intervals for Policy Evaluation in Adaptive Experiments_](https://arxiv.org/abs/1911.02768).

## File description
- `simulations.py`: Python script to run experiments and save results computed by different weighting schemes.
- `compute_wdecorrelation_lambda.py`: use Monte Carlo simulation to precompute bias-variance tradeoff parameter _lambda_ in the paper [Accurate Inference for Adaptive Linear Models](https://arxiv.org/abs/1712.06695).
- `plot-two-point-new.ipynb`: Jupyter notebook to make plots shown in the paper.
- `jobfile.job`: SLURM job script to run simulations.py.
- `jobfile-wdecorr.job`: SLURM job script to run compute_wdecorrelation_lambda.py.

## Instructions for running experiments and making plots
1. `python compute_wdecorrelation_lambda.py {experiment_name} {arm values} {noise shape}` to precompute bias-variance tradeoff parameter _lambda_ of W-decorrelation. (make sure the experiment configuration is consistent with that in `script.py`).
2. `python simulation.py` to run experiments and save results.
3. Open `plot-two-point-new.ipynb` to generate plots based on the saved results. 

