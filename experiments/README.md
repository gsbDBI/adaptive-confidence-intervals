This directory contains scripts to run experiments and make plots shown in the paper [_Confidence Intervals for Policy Evaluation in Adaptive Experiments_](https://arxiv.org/abs/1911.02768).

## File description
- `simulations.py`: Python script to run experiments and save results in `./results/` computed by different weighting schemes.
- `compute_wdecorrelation_lambda.py`: use Monte Carlo simulation to precompute bias-variance tradeoff parameter _lambda_ in the paper [Accurate Inference for Adaptive Linear Models](https://arxiv.org/abs/1712.06695).
- `plots.ipynb`: Jupyter notebook to make plots shown in the paper using saved results in `./results/`.


## Quick start for running experiments using two-point allocation rate and making plots
### 1. Collecting bandits data
```python
import os
from adaptive_CI.experiments import run_mab_experiment
from adaptive_CI.compute import stick_breaking
from adaptive_CI.saving import *
from adaptive_CI.inference import *
from adaptive_CI.weights import *

noise_func = 'uniform'
K = 3 # Number of arms
T = 100000 # Sample size
truth = np.array([1., 1., 1.]) # no signal arm truth
noise = np.random.uniform(-1.0, 1.0, size=(T, K))
ys = truth + noise
data = run_mab_experiment(ys, initial=5, floor_start=1/K, floor_decay=0.7, exploration='TS')
```
### 2. Computing two_point_allocation_rate adaptive weights
```python
muhat = np.row_stack([np.zeros(K), sample_mean(rewards, arms, K)[:-1]])
scores = aw_scores(rewards, arms, probs, muhat)
twopoint_ratio = twopoint_stable_var_ratio(e=probs, alpha=floor_decay)
twopoint_h2es = stick_breaking(twopoint_ratio)
stats = evaluate_aipw_stats(scores, wts_twopoint, truth) 
contrasts = evaluate_aipw_contrasts(scores, wts_twopoint, truth)
```

### 3. Saving results
```python
config = dict(T=T, K=K, noise_func='uniform', noise_scale=1.0, floor_start=1/K, floor_decay=0.7,
        initial=5, dgp='nosignal')
ratios = dict(two_point=twopoint_ratio)
df_stats = pd.DataFrame({"statistic": ["estimate", "stderr", "bias", "90% coverage of t-stat", "t-stat", "mse", "CI_width", "truth"] * stats.shape[1],
                                  "policy": np.repeat(np.arange(K), stats.shape[0]),
                                  "value":  stats.flatten(order='F'),
                                  "method": 'two_point',
                                 **config})
df_contrasts = pd.DataFrame({"statistic": ["estimate", "stderr", "bias", "90% coverage of t-stat", "t-stat", "mse", "CI_width", "truth"] * contrasts.shape[1],
                                      "policy": np.repeat([f"(0,{k})" for k in np.arange(1, K)], contrasts.shape[0]),
                                      "value": contrast.flatten(order='F'),
                                      "method": 'two_point',
                                     **config})
df = pd.concat([df_stats, df_contrasts])  
df.to_pickle(os.path.join('./results', compose_filename('stats','pkl')))
```

### 4. Repeating steps 1-3 for 1000 times
to collect results of 1000 simulations.

### 5. Making plots
```python
import matplotlib.pyplot as plt
from glob import glob
from plot_utils import *
# Load data
stats_files = glob(f"results/stats*.pkl")
dfs = []
for k, file in enumerate(stats_files):
    dfs.append(pd.read_pickle(file))
df = pd.concat(dfs)
df['value'] = df['value'].astype(float)

# Making plots
plot_arm_values(df.query(f"policy==0"), hue_order=['two_point'], labels=['Two-point allocation'], name="arm_values_0")
plot_contrast(df.query(f"policy=='(0,2)'"), hue_order=['two_point'], col_names=['RMSE', 'Bias', 'Confidence Interval Radius', '90% coverage'],
              labels=['Two-point allocation'], name="contrast_good_bad")
```


## Reproducibility 
To reproduce results shown in the paper, do
1. `python compute_wdecorrelation_lambda.py` to precompute bias-variance tradeoff parameter _lambda_ of W-decorrelation. (make sure the experiment configuration is consistent with that in `simulation.py`).
2. `python simulation.py` to run experiments and save results in `./results/`.
3. Open `plots.ipynb`, follow the instructions in the notebook to generate plots based on the saved results in `./results/`. 

