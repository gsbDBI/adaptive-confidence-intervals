"""
This script pre-computes bias-variance tradeoff parameter W_lambda in W-decorrelation.
Paper reference: Deshpande, Y., Mackey, L., Syrgkanis, V., & Taddy, M. (2017). Accurate inference for adaptive linear models. arXiv preprint arXiv:1712.06695.
"""

import sys
import os
import numpy as np

from time import time

from adaptive_CI.experiments import run_mab_experiment


def on_sherlock():
    """ Checks if running locally or on sherlock """
    return 'GROUP_SCRATCH' in os.environ
    
    
def calculate_W_lambda(config, pcts, TT, num_sims, verbose=True):
    """
    Compute bias-variance tradeoff parameter W_lambda in W-decorrelation.

    INPUT:
        - config: a dictionary specifying configurations of experiments including
            * truth: mean values of arms
            * initial: initial number of samples for each arm to do pure exploration.
            * floor_start: assignment probability floor starting value
            * floor_decay: assignment probability floor decaying rate
            (assignment probability floor = floor start * t ^ {-floor_decay})
            * exploration: agent
        - pcts: pencitiles to compute W_lambda
        - TT: a list of sample sizes 
        - num_sims: number of simulations to run. A larger number (>1000) recommended.
        - verbose: if True prints out progress

    OUTPUT:
        - W_lambdas: a list of W_lambda for sample size in TT
    """
    def message(s):
        if verbose:
            print(s)
    
    noise_func, noise_scale = config['noise_func'], config['noise_scale']
    K = config['K']
    T = max(TT)

    arms = []  # (#sim_W, T)
    message("Part 1/2: Running experiments")
        
    for s in range(num_sims):
        message(f"Simulation {s+1}/{num_sims}")
        
        # Draw potential outcomes.
        if noise_func == 'uniform':
            noise = np.random.uniform(-noise_scale, noise_scale, size=(T, K))
        else:
            noise = np.random.exponential(noise_scale, size=(T, K)) - noise_scale
        ys = noise + config["truth"]

        # Run the experiment.
        data = run_mab_experiment(
            ys,
            initial=config["initial"],
            floor_start=config["floor_start"],
            floor_decay=config["floor_decay"],
            exploration=config["exploration"])

        arms.append(data['arms'])

    message("Done simulating. Computing and saving w-decorrelation values.")
    W_lambdas = []
    for t in TT:
        arm_counts = []  # size (sim_W, #arms)
        # Record selected arms for W-decorrelation construction
        for arm in arms:
            arm_counts.append([np.sum(arm[:t] == w) for w in range(K)])  # size (sim_W, #arms)
            assert(
                np.sum(arm_counts[-1]) == t and f'{np.sum(arm_counts[-1])} not equal to  {t}')
        arm_counts = np.array(arm_counts)

        def compute_lambda_per_arm(sythetic_arm_counts):
            pct_arm_counts = np.percentile(synthetic_arm_counts, pcts, axis=0) # size (#percentiles, 2)
            min_pct_arm_counts = np.amin(pct_arm_counts, axis=1) # size (#percentiles)
            W_lambdas_t_per_arm = [np.ones(t) * min_pct / np.log(t)
                    for min_pct in min_pct_arm_counts] # size (#percentile, t)
            return np.array(W_lambdas_t_per_arm)

        W_lambdas_t = np.zeros((len(pcts), K, t)) # size(#percentile, K, t)
        for w in range(K):
            synthetic_arm_counts = np.zeros((num_sims, 2))
            synthetic_arm_counts[:, 0] = arm_counts[:, w]
            synthetic_arm_counts[:, 1] = t - arm_counts[:, w]
            W_lambdas_t[:, w, :] = compute_lambda_per_arm(synthetic_arm_counts)
        
        W_lambdas.append(W_lambdas_t)
    return W_lambdas


"""
script to do Monte Carlo simulations and compute bias-variance tradeoff parameter in W-decorrelation.
"""
start_time = time()

""" Available configurations """
percentiles = [5, 15, 35, 50]
TT = [1_000, 5_000, 10_000, 50_000, 100_000]
truths = {
    'nosignal': np.array([1., 1., 1.]),
    'lowSNR': np.array([.9, 1., 1.1]),
    'highSNR': np.array([.5, 1., 1.5])
}
floor_decays = [.7] # [.25, .5, .6, .7, .8, .9, .99]
experiments = truths.keys()
initial = 5  # initial number of samples of each arm to do pure exploration
exploration = 'TS'
noise_scale = 1.0
noise_func = "uniform"
num_sims = 200 if on_sherlock() else 1
save = on_sherlock()

for experiment in experiments:
    print(f"Running experiment {experiment}")
    for floor_decay in floor_decays:
        print(f"Floor decay: {floor_decay}")
        truth = truths[experiment]
        K = len(truth)  # number of arms    
        config = dict(
            K=len(truth),
            truth=truth,
            noise_func=noise_func,
            noise_scale=noise_scale,
            initial=initial,
            floor_start=1/K,
            floor_decay=floor_decay,
            exploration=exploration,
        )
        W_lambdas = calculate_W_lambda(config, percentiles, TT, num_sims=num_sims)
        for t, W_lam in zip(TT, W_lambdas):
            name = f'W_lambdas_{experiment}-{noise_func}-{t}-{floor_decay}.npz'
            np.savez(name, percentiles=percentiles, W_lambdas=W_lam)
        print(f'time passed {time()-start_time}')







