This directory contains the Python module of adaptive inference developed in the paper [Confidence Intervals for Policy Evaluation in Adaptive Experiments](https://arxiv.org/abs/1911.02768), which includes 
- running a multi-armed bandit experiment with different agents (Thompson sampling agent, epsilon-greedy agent, etc.), see function `run_mab_experiment` in `experiment.py`;
- computing two-point allocation rate, see function `twopoint_stable_var_ratio` in `weight.py`;
- policy & contrast inference using different methods (following notations in the paper):
    - uniform/ constant allocation rate/ two-point allocation rate: see functions `evaluate_aipw_stats` and `evaluate_aipw_contrasts` in `inference.py`; 
    - w-decorrelation: see function `wdecorr_stats` in `inference.py`; 
    - sample mean (Howard et al CI): see functions `evaluatie_gamma_exponential_stats` and `evaluatie_gamma_exponential_contrasts` in `inference.py`;
    - sample mean (normal CI): see functions `evaluate_sample_mean_naive_stats` and `evaluate_sample_mean_naive_contrasts` in `inference.py`;


To reproduce results presented in the paper, please go to directory [../experiments](https://github.com/gsbDBI/adaptive-confidence-intervals/tree/master/experiments) and follow the instructions in the README there. 

## File description
- `compute.py` contains helper functions to speed up computation. 
- `experiment.py` contains helper functions to generate data, including functions of agents, environemnt and data generating process. 
- `inference.py` contains helper functions to do inference on policy value and constrast, using adaptive inference, w-decorrelation, sample mean (normal CI), and sample mean (Howard et al CI).
- `saving.py` contains helping functions for better result-saving format. 
- `weights.py` contains helper functions to compute evaluation weights. 
- `inequalities.py` contains helper functions to compute Bernstein-typed, Bennett-typed, and Hoeffding-typed confidence intervals.
