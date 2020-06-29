"""
This script pre-computes bias-variance tradeoff parameter W_lambda in W-decorrelation.
Paper reference: Deshpande, Y., Mackey, L., Syrgkanis, V., & Taddy, M. (2017). Accurate inference for adaptive linear models. arXiv preprint arXiv:1712.06695.
"""

from time import time
from adaptive_CI.experiments import run_mab_experiment
import numpy as np
import sys
sys.path.insert(0, "/home/rhzhan/adaptive-confidence-intervals/")


def calculate_W_lambda(config, pcts, TT):
    """
    Compute bias-variance tradeoff parameter W_lambda in W-decorrelation.

    INPUT:
        - config: a dictionary specifying configurations of experiments including
            * sim_W: number of simulations to compute lambda
            * truth: mean values of arms
            * initial: initial number of samples for each arm to do pure exploration.
            * floor_start: assignment probability floor starting value
            * floor_decay: assignment probability floor decaying rate
            (assignment probability floor = floor start * t ^ {-floor_decay})
            * exploration: agent
        - pcts: pencitiles to compute W_lambda
        - TT: a list of sample sizes 

    OUTPUT:
        - W_lambdas: a list of W_lambda for sample size in TT
    """
    noise_func, noise_scale = config['noise_func'], config['noise_scale']
    K = config['K']
    T = max(TT)

    arms = []  # (#sim_W, 20000)
    for s in range(config['sim_W']):
        # Draw potential outcomes.
        if noise_func == 'uniform':
            noise = np.random.uniform(-noise_scale, noise_scale, size=(T, K))
        else:
            noise = np.random.exponential(
                noise_scale, size=(T, K)) - noise_scale
        ys = noise + config["truth"]

        # Run the experiment.
        data = run_mab_experiment(
            ys,
            initial=config["initial"],
            floor_start=config["floor_start"],
            floor_decay=config["floor_decay"],
            exploration=config["exploration"])

        arms.append(data['arms'])

    W_lambdas = []
    for t in TT:
        arm_counts = []  # size (sim_W, #arms)
        # Record selected arms for W-decorrelation construction
        for arm in arms:
            arm_counts.append([np.sum(arm[:t] == w)
                               for w in range(K)])  # size (sim_W, #arms)
            assert(
                np.sum(arm_counts[-1]) == t and f'{np.sum(arm_counts[-1])} not equal to  {t}')
        arm_counts = np.array(arm_counts)

        """
        # size (#percentiles, #arms)
        pct_arm_counts = np.percentile(arm_counts, pcts, axis=0)
        # size (#percentiles)
        min_pct_arm_counts = np.amin(pct_arm_counts, axis=1)

        # size (#percentile, t), a list of W_lambdas for different percentiles
        W_lambdas_t = [np.ones(t)*min_pct/np.log(t)
                       for min_pct in min_pct_arm_counts]
        """
        def compute_lambda_per_arm(sythetic_arm_counts):
            pct_arm_counts = np.percentile(synthetic_arm_counts, pcts, axis=0) # size (#percentiles, 2)
            min_pct_arm_counts = np.amin(pct_arm_counts, axis=1) # size (#percentiles)
            W_lambdas_t_per_arm = [np.ones(t) * min_pct / np.log(t)
                    for min_pct in min_pct_arm_counts] # size (#percentile, t)
            return np.array(W_lambdas_t_per_arm)

        W_lambdas_t = np.zeros((len(pcts), K, t)) # size(#percentile, K, t)
        for w in range(K):
            synthetic_arm_counts = np.zeros((config['sim_W'], 2))
            synthetic_arm_counts[:, 0] = arm_counts[:, w]
            synthetic_arm_counts[:, 1] = t - arm_counts[:, w]
            W_lambdas_t[:, w, :] = compute_lambda_per_arm(synthetic_arm_counts)
        
        W_lambdas.append(W_lambdas_t)
    return W_lambdas


"""
script to do Monte Carlo simulations and compute bias-variance tradeoff parameter in W-decorrelation.
"""
start_time = time()

# Read experiment configuration
experiment, truth, noise_func = sys.argv[1], sys.argv[2], sys.argv[3]
noise_scale = 1.0
truth = [float(a) for a in truth.split(',')]
config = dict(
    K=len(truth),
    sim_W=5000,
    truth=truth,
    noise_func=noise_func,
    noise_scale=noise_scale,
    initial=5,
    floor_start=0.1,
    floor_decay=0.5,
    exploration='TS',
)
percentiles = [5, 15, 35, 50]
TT = [1000, 5000, 10000, 20000]
# Do Monte-Carlo simulations to compute W_lambda.
W_lambdas = calculate_W_lambda(config, percentiles, TT)
for t, W_lam in zip(TT, W_lambdas):
    name = f'W_lambdas_{experiment}-{noise_func}-{t}.npz'
    np.savez(name, percentiles=percentiles, W_lambdas=W_lam)
print(f'time passed {time()-start_time}')
