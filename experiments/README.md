This directory contains scripts to run experiments and make plots shown in the paper [_Confidence Intervals for Policy Evaluation in Adaptive Experiments_](https://arxiv.org/abs/1911.02768).

## File description
- `script.py`: Python script to run experiments and save results computed by different weighting schemes.
- `compute_W_decorrelation_lambda.py`: use Monte Carlo simulation to precompute bias-variance tradeoff parameter _lambda_ in the paper [Accurate Inference for Adaptive Linear Models](https://arxiv.org/abs/1712.06695).
- `plot_utils.py`: helper functions to make plots.
- `plot.ipynb`: Jupyter notebook to make plots.

## Instructions for running experiments and making plots
1. `python compute_W_decorrelation_lambda.py {experiment_name} {arm values} {noise shape}` to precompute bias-variance tradeoff parameter _lambda_ of W-decorrelation. (make sure the experiment configuration is consistent with that in `script.py`).
2. `python script.py {experiment_name} {arm values} {noise shape}` to run experiments and save results in the ./results/ directory.
3. Open `plot.ipynb` to generate plots based on the results saved by `script.py`. 

For instance, to run experiments with configuration `experiment_name=nosignal`, `arm values=[1.0, 1.0, 1.0]` and `noise_shape=uniform`, do the following:
```bash
python compute_W_decorrelation_lambda.py nosignal 1.0,1.0,1.0 uniform
python scripy.py nosignal 1.0,1.0,1.0 uniform
```
