"""
This script runs simulations reported in our paper Confidence Intervals for Policy Evaluation in Adaptive Experiments (https://arxiv.org/abs/1911.02768)
"""

import sys
# sys.path.insert(0, "/home/rhzhan/adaptive-confidence-intervals/")
from time import time
from sys import argv
from random import choice
import pickle
import os
import numpy as np
from adaptive_CI.experiments import run_mab_experiment
from adaptive_CI.compute import stick_breaking
from adaptive_CI.saving import *
from adaptive_CI.inference import *
from adaptive_CI.weights import twopoint_stable_var_ratio


# ----------------------------------------------------
# Read DGP specification
noise_func = 'uniform'


experiment = choice(['nosignal', 'lowsignal', 'highsignal'])
truth = np.array([0.0, 0.0, 0.0])

results_list = []
start_time = time()
num_sims = 1000
save_every = 10


# ----------------------------------------------------
# Run simulations
for s in range(num_sims):

    """ Experiment configuration """
    T = choice([1000, 5000, 10000, 20000])  # number of samples
    K = len(truth)  # number of arms
    initial = 5  # initial number of samples of each arm to do pure exploration
    floor_start = 0.1
    floor_decay = 0.5
    exploration = 'TS'
    noise_scale = 1.0

    """ Generate data """
    if noise_func == 'uniform':
        noise = np.random.uniform(-noise_scale, noise_scale, size=(T, K))
    else:
        noise = np.random.exponential(noise_scale, size=(T, K)) - noise_scale
    ys = truth + noise

    """ Run experiment """
    data = run_mab_experiment(
        ys,
        initial=initial,
        floor_start=floor_start,
        floor_decay=floor_decay,
        exploration=exploration)

    probs = data['probs']
    rewards = data['rewards']
    arms = data['arms']

    """ Compute AIPW scores """
    muhat = np.row_stack([np.zeros(K), sample_mean(rewards, arms, K)[:-1]])
    scores = aw_scores(rewards, arms, probs, muhat)

    """ Compute weights """
    # Two-point allocation rate
    twopoint_ratio = twopoint_stable_var_ratio(e=probs, alpha=floor_decay)
    twopoint_h2es = stick_breaking(twopoint_ratio)
    wts_twopoint = np.sqrt(np.maximum(0., twopoint_h2es * probs))

    # Other weights: lvdl(constant allocation rate), propscore and uniform
    wts_lvdl = np.sqrt(probs)
    wts_propscore = probs
    wts_uniform = np.ones_like(probs)

    """ Estimate arm values """
    # for each weighting scheme, return [estimate, S.E, bias, 95%-coverage, t-stat, mse, truth]
    stats = dict(
        uniform=aw_stats(scores, wts_uniform, truth),
        propscore=aw_stats(scores, wts_propscore, truth),
        lvdl=aw_stats(scores, wts_lvdl, truth),
        two_point=aw_stats(scores, wts_twopoint, truth),
    )

    # # add estimates of W_decorrelation
    # W_names = f'W_lambdas_{experiment}-{noise_func}-{T}.npz'
    # W_save = np.load(W_names)  # load presaved W-lambdas
    # for percentile, W_lambda in zip(W_save['percentiles'], W_save['W_lambdas']):
    #     stats[f'W-decorrelation_{percentile}'] = wdecorr_stats(
    #         arms, rewards, K, W_lambda, truth)

    """ Estimate contrasts """
    contrasts = dict(
        uniform=aw_contrasts(scores, wts_uniform, truth),
        propscore=aw_contrasts(scores, wts_propscore, truth),
        lvdl=aw_contrasts(scores, wts_lvdl, truth),
        two_point=aw_contrasts(scores, wts_twopoint, truth),
    )

    """ Save results """
    weights = dict(
        uniform=wts_uniform,
        propscore=wts_propscore,
        lvdl=wts_lvdl,
        two_point=wts_twopoint,
    )

    ratios = dict(
        lvdl=np.ones((T, K)) / np.arange(T, 0, -1)[:, np.newaxis],
        two_point=twopoint_ratio,
    )

    config = dict(
        T=T,
        K=K,
        noise_func=noise_func,
        noise_scale=noise_scale,
        floor_start=floor_start,
        floor_decay=floor_decay,
        initial=initial,
        truth=truth,
    )

    # only save at saved_timepoints for assigmnment probabilities, conditional variance, weights, ratios(lambdas)
    saved_timepoints = list(range(0, T, T//100))
    condVars = dict()
    for method, weight in weights.items():
        condVar = weight ** 2 / probs / np.sum(weight, 0) ** 2 * T
        weight = weight / np.sum(weight, 0) * T
        condVars[method] = condVar[saved_timepoints, :]
        weights[method] = weight[saved_timepoints, :]
    for ratio in ratios:
        ratios[ratio] = ratios[ratio][saved_timepoints, :]
    probs = probs[saved_timepoints, :]
    results = dict(
        config=config,
        probs=probs,
        stats=stats,
        contrasts=contrasts,
        weights=weights,
        ratios=ratios,
        condVars=condVars)

    results_list.append(results)

    # save results every _save_every_ simulations
    if (s+1) % save_every == 0 or s == num_sims-1:
        write_dir = os.path.join(os.getcwd(), 'results')
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        filename = compose_filename(
            f'weight_experiment_{experiment}_{noise_func}', 'pkl')
        write_path = os.path.join(write_dir, filename)
        print(f"Saving at {write_path}")
        with open(write_path, "wb") as f:
            pickle.dump(results_list, f)

        results_list = []

print(f"Time passed {time()-start_time}s")
