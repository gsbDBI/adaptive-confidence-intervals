This directory contains the Python module of adaptive inference developed in the paper [Confidence Intervals for Policy Evaluation in Adaptive Experiments](https://arxiv.org/abs/1911.02768).

To reproduce results presented in the paper, please go to directory [../experiments](https://github.com/gsbDBI/adaptive-confidence-intervals/tree/master/experiments). 

## File description
- `compute.py` contains helper functions to speed up computation. 
- `experiment.py` contains helper functions to generate data, including functions of agents, environemnt and data generating process. 
- `inference.py` contains helper functions to do inference, including computing scores and statistics of arm values and contrasts.
- `saving.py` contains helping functions for better result-saving format. 
- `weights.py` contains helper functions to compute evaluation weights. 
