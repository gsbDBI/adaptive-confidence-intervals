#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
This script runs simulations reported in our paper Confidence Intervals for Policy Evaluation in Adaptive Experiments (https://arxiv.org/abs/1911.02768)
"""

import sys
import subprocess
import pickle
import os
import numpy as np
import pandas as pd

from time import time
from sys import argv
from random import choice
from time import time
from os.path import dirname, realpath, join, exists
from os import makedirs, chmod
from getpass import getuser

from adaptive_CI.experiments import run_mab_experiment
from adaptive_CI.compute import stick_breaking
from adaptive_CI.inference import *
from adaptive_CI.weights import *

# magics removed
# magics removed


# # Main Figures
# 
# This simulation script produces most figures in the paper, with the _exception_ of Figure 1 in the introduction, and its counterpart Figure 13 in Appendix A5.
# 
# 
# **To non-Stanford members**
# 
# Each time this script is called, it selects a random configuration (e.g., a signal strength, an experiment horizon, etc) and completes a single simulation using that configuration.
# 
# In order to produce the figures in our paper, we recommend that this script be run at least $10ˆ5$ times.
# 
# 
# 
# **To Stanford members with access to the Sherlock cluster**
# 
# Each time this script is called on sherlock, it selects a random configuration (e.g., a signal strength, an experiment horizon, etc) and completes 200 simulations using that configuration.
# 
# In order to produce the figures in our paper, we recommend that this script be run a few thousand times.
# 

# In[2]:


start_time = time()


# In[3]:


def on_sherlock():
    """ 
    Note: This can be ignored by non-Stanford members.

    Checks if running on Stanford's Sherlock cluster
    """
    return 'GROUP_SCRATCH' in os.environ


def get_sherlock_dir(project, *tail, create=True):
    """
    Note: This can be ignored by non-Stanford members.
    
    Output consistent folder name in Sherlock.
    If create=True and on Sherlock, also makes folder with group permissions.
    If create=True and not on Sherlock, does not create anything.

    '/scratch/groups/athey/username/project/tail1/tail2/.../tailn'.

    >>> get_sherlock_dir('adaptive-inference')
    '/scratch/groups/athey/adaptive-inference/vitorh'

    >>> get_sherlock_dir('toronto')
    '/scratch/groups/athey/toronto/vitorh/'

    >>> get_sherlock_dir('adaptive-inference', 'experiments', 'exp_out')
    '/scratch/groups/athey/adaptive-inference/vitorh/experiments/exp_out'
    """
    base = join("/", "scratch", "groups", "athey", project, getuser())
    path = join(base, *tail)
    if not exists(path) and create and on_sherlock():
        makedirs(path, exist_ok=True)
        # Correct permissions for the whole directory branch
        chmod_path = base
        chmod(base, 0o775)
        for child in tail:
            chmod_path = join(chmod_path, child)
            chmod(chmod_path, 0o775)
    return path


def compose_filename(prefix, extension):
    """
    Creates a unique filename based on Github commit id and time.
    Useful when running in parallel on server.

    INPUT:
        - prefix: file name prefix
        - extension: file extension

    OUTPUT:
        - fname: unique filename
    """
    # Tries to find a commit hash
    try:
        commit = subprocess            .check_output(['git', 'rev-parse', '--short', 'HEAD'],
                          stderr=subprocess.DEVNULL)\
            .strip()\
            .decode('ascii')
    except subprocess.CalledProcessError:
        commit = ''

    # Other unique identifiers
    rnd = str(int(time() * 1e8 % 1e8))
    sid = tid = jid = ''
    ident = filter(None, [prefix, commit, jid, sid, tid, rnd])
    basename = "_".join(ident)
    fname = f"{basename}.{extension}"
    return fname


# In[4]:


num_sims = 200 if on_sherlock() else 1

# DGP specification
# ----------------------------------------------------
noise_func = 'uniform'
truths = {
    'nosignal': np.array([1., 1., 1.]),
    'lowSNR': np.array([.9, 1., 1.1]),
    'highSNR': np.array([.5, 1., 1.5])
}
if on_sherlock():
    Ts = [1_000, 5_000, 10_000, 50_000, 100_000]
else:
    Ts = [100_000]
floor_decays = [.7]
initial = 5  # initial number of samples of each arm to do pure exploration
exploration = 'TS'
noise_scale = 1.


# In[5]:


df_stats = []
df_lambdas = []


# In[6]:


# Run simulations
for s in range(num_sims):
    if (s+1) % 10 == 0:
        print(f'Running simulation {s+1}/{num_sims}')

    """ Experiment configuration """
    T = choice(Ts)  # number of samples
    experiment = choice(list(truths.keys()))
    truth = truths[experiment]
    K = len(truth)  # number of arms
    floor_start = 1/K
    floor_decay = choice(floor_decays)

    """ Generate data """
    noise = np.random.uniform(-noise_scale, noise_scale, size=(T, K))
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
    twopoint_ratio_old = twopoint_stable_var_ratio_old(probs, floor_start, floor_decay)
    twopoint_h2es = stick_breaking(twopoint_ratio)
    twopoint_h2es_old = stick_breaking(twopoint_ratio_old)
    wts_twopoint = np.sqrt(np.maximum(0., twopoint_h2es * probs))

    # Other weights: lvdl(constant allocation rate), propscore and uniform
    wts_lvdl = np.sqrt(probs)
    wts_propscore = probs
    wts_uniform = np.ones_like(probs)

    """ Estimate arm values """
    # for each weighting scheme, return [estimate, S.E, bias, 90%-coverage, t-stat, mse, truth]
    stats = dict(
        uniform=evaluate_aipw_stats(scores, wts_uniform, truth),
        propscore=evaluate_aipw_stats(scores, wts_propscore, truth),
        lvdl=evaluate_aipw_stats(scores, wts_lvdl, truth),
        two_point=evaluate_aipw_stats(scores, wts_twopoint, truth),
        beta_bernoulli=evaluate_beta_bernoulli_stats(rewards, arms, truth, K, floor_decay, alpha=.1),
        gamma_exponential=evaluate_gamma_exponential_stats(rewards, arms, truth, K, floor_decay, c=2, expected_noise_variance=1/3, alpha=.1),
        sample_mean_naive=evaluate_sample_mean_naive_stats(rewards, arms, truth, K, alpha=.1)
    )
    
    # # add estimates of W_decorrelation
    W_name = f'wdecorr_results/W_lambdas_{experiment}-{noise_func}-{T}-{floor_decay}.npz'
    try:
        W_save = np.load(W_name)  # load presaved W-lambdas
        for percentile, W_lambda in zip(W_save['percentiles'], W_save['W_lambdas']):
            stats[f'W-decorrelation_{percentile}'] = wdecorr_stats(arms, rewards, K, W_lambda, truth)
    except FileNotFoundError:
        print(f'Could not find relevant w-decorrelation file {W_name}. Ignoring.')
        
    
    """ Estimate contrasts """
    contrasts = dict(
        uniform=evaluate_aipw_contrasts(scores, wts_uniform, truth),
        propscore=evaluate_aipw_contrasts(scores, wts_propscore, truth),
        lvdl=evaluate_aipw_contrasts(scores, wts_lvdl, truth),
        two_point=evaluate_aipw_contrasts(scores, wts_twopoint, truth),
        beta_bernoulli=evaluate_beta_bernoulli_contrasts(rewards, arms, truth, K, floor_decay, alpha=.1),
        gamma_exponential=evaluate_gamma_exponential_contrasts(rewards, arms, truth, K, floor_decay, c=2, expected_noise_variance=1/3, alpha=.1),
        sample_mean_naive=evaluate_sample_mean_naive_contrasts(rewards, arms, truth, K, alpha=.1)
    )

    
    """ Save results """
    config = dict(
        T=T,
        K=K,
        noise_func=noise_func,
        noise_scale=noise_scale,
        floor_start=floor_start,
        floor_decay=floor_decay,
        initial=initial,
        dgp=experiment,
    )

    ratios = dict(
        lvdl=np.ones((T, K)) / np.arange(T, 0, -1)[:, np.newaxis],
        two_point=twopoint_ratio,
    )
    
    # save lambda values at selected timepoints
    saved_timepoints = list(range(0, T, T // 250)) + [T-1]
    for ratio in ratios:
        ratios[ratio] = ratios[ratio][saved_timepoints, :]
    
    # tabulate arm values
    tabs_stats = []
    for method, stat in stats.items():
        tab_stats = pd.DataFrame({"statistic": ["estimate", "stderr", "bias", "90% coverage of t-stat", "t-stat", "mse", "CI_width", "truth"] * stat.shape[1],
                                  "policy": np.repeat(np.arange(K), stat.shape[0]),
                                  "value":  stat.flatten(order='F'),
                                  "method": method,
                                 **config})
        tabs_stats.append(tab_stats)


    # tabulate arm contrasts
    tabs_contrasts = []
    for method, contrast in contrasts.items():
        tabs_contrast = pd.DataFrame({"statistic": ["estimate", "stderr", "bias", "90% coverage of t-stat", "t-stat", "mse", "CI_width", "truth"] * contrast.shape[1],
                                      "policy": np.repeat([f"(0,{k})" for k in np.arange(1, K)], contrast.shape[0]),
                                      "value": contrast.flatten(order='F'),
                                      "method": method,
                                     **config})
        tabs_contrasts.append(tabs_contrast)

    
    df_stats.extend(tabs_stats)
    df_stats.extend(tabs_contrasts)
    
    
    """ Save relevant lambda weights, if applicable """
    if T == max(Ts):
        saved_timepoints = list(range(0, T, T // 500))
        lambdas = twopoint_ratio[saved_timepoints] * (T - np.array(saved_timepoints)[:,np.newaxis])
        lambdas = {key: value for key, value in enumerate(lambdas.T)}
        dfl = pd.DataFrame({**lambdas, **config, 'time': saved_timepoints})
        dfl = pd.melt(dfl, id_vars=list(config.keys()) + ['time'], var_name='policy', value_vars=list(range(K)))
        df_lambdas.append(dfl)
        
    print(f"Time passed {time()-start_time}s")


# ----

# ### Saving

# Break down the output into different chunks, so it will be easier to pick what to load and plot.

# Get the directory (the first statement here is specific to the Stanford cluster).

# In[7]:


if on_sherlock():
    write_dir = get_sherlock_dir('adaptive-confidence-intervals', 'simulations', create=True)
    print(f"saving at {write_dir}")
else:
     write_dir = join(os.getcwd(), 'results')
        


# In[8]:


df_stats = pd.concat(df_stats, ignore_index=True, sort=False)


# Saving information about contrasts.

# In[9]:


filename_contrast = compose_filename(f'contrast', 'pkl')
write_path_contrast = os.path.join(write_dir, filename_contrast)

df_contrast = df_stats.query(
            'policy == "(0,2)" and '
            'statistic == ["mse", "bias", "90% coverage of t-stat", "CI_width"] and '
            "method == ['uniform', 'lvdl', 'two_point',  'sample_mean_naive', 'gamma_exponential', 'W-decorrelation_15']")
df_contrast.to_pickle(write_path_contrast)


# Save information about arms.

# In[10]:


filename_arms = compose_filename(f'arm', 'pkl')
write_path_arms = os.path.join(write_dir, filename_arms)

df_arms = df_stats.query(
            'policy == [0, 1, 2] and '
            'statistic == ["mse", "bias", "90% coverage of t-stat", "CI_width"] and '
            "method == ['uniform', 'lvdl', 'two_point',  'sample_mean_naive', 'gamma_exponential', 'W-decorrelation_15']")
df_arms.to_pickle(write_path_arms)


# Save information about "t-stats" (i.e., our studentized 'statistics').

# In[11]:


filename_tstats = compose_filename(f'tstat', 'pkl')
write_path_tstats = os.path.join(write_dir, filename_tstats)

T_max = max(Ts)
df_tstats = df_stats.query(
    "method == ['uniform', 'lvdl', 'two_point',  'sample_mean_naive', 'gamma_exponential', 'W-decorrelation_15'] and "
    "T == @T_max and "
    "statistic == 't-stat'"
)
df_tstats.to_pickle(write_path_tstats)


# Save information about $\lambda$ behavior, if appropriate.

# In[12]:


filename_lambdas = compose_filename(f'lambdas', 'pkl')
write_path_lambdas = os.path.join(write_dir, filename_lambdas)
if len(df_lambdas) > 0:
    df_lambdas = pd.concat(df_lambdas)


# In[13]:


print("All done.")

