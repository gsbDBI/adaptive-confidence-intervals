import pandas as pd
import numpy as np


import pickle
from pickle import UnpicklingError
from glob import glob
import os
from os.path import dirname, realpath, join, exists
from os import makedirs, chmod
from getpass import getuser


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

if on_sherlock():
    base_dir = get_sherlock_dir('adaptive-confidence-intervals')
else:
    base_dir = "results"
    
    
contrast_files = glob(join(f"{base_dir}", "*contrast*.pkl"))
arm_files = glob(join(f"{base_dir}", "*arm*.pkl"))
lambda_files = glob(join(f"{base_dir}", "*lambda*.pkl"))
tstat_files = glob(join(f"{base_dir}", "*tstat*.pkl"))

print(f"Found {len(contrast_files)} contrast files.")
print(f"Found {len(arm_files)} arm files.")
print(f"Found {len(lambda_files)} lambda files.")
print(f"Found {len(tstat_files)} t-stat files.")


# CONTRASTS
print("Aggregating contrast information.")
df = []
for k, file in enumerate(contrast_files):
    if k % 100 == 0:
        print(f"\tReading contrasts file {k}.")
    try:
        df_tmp = pd.read_pickle(file)
        df_tmp = df_tmp.query(
            "method == ['uniform', 'lvdl', 'two_point',  'sample_mean_naive', 'gamma_exponential', 'W-decorrelation_15']"
        )
        df_tmp['value'] = df_tmp['value'].astype(float)
        df.append(df_tmp)
    except Exception as e:
        print(f"\tError when reading file {file}.")

print('\tConcatenating.')
df = pd.concat(df, ignore_index=True, verify_integrity=False, sort=False, copy=False)
df.to_pickle(join(base_dir, f'contrast_results.pkl'))
print("\tDone aggregating contrasts.\n")
    
            
# ARMS
print("Aggregating arms information.")
df = []
for k, file in enumerate(arm_files):
    if k % 100 == 0:
        print(f"\tReading arm file {k}.")
    try:
        df_tmp = pd.read_pickle(file)
        df_tmp = df_tmp.query(
            "method == ['uniform', 'lvdl', 'two_point',  'sample_mean_naive', 'gamma_exponential', 'W-decorrelation_15']"
        )
        df_tmp['value'] = df_tmp['value'].astype(float)
        df.append(df_tmp)
    except Exception as e:
        print(f"\tError when reading file {file}.")

print('\tConcatenating.')
df = pd.concat(df, ignore_index=True, verify_integrity=False, sort=False, copy=False)
df.to_pickle(join(base_dir, f'arm_results.pkl'))
print("\tDone aggregating arms.\n")


# LAMBDA
print("Aggregating lambda information.")
df = []
for k, file in enumerate(lambda_files):
    if k % 100 == 0:
        print(f"\tReading lambda file {k}.")
    try:
        df.append(pd.read_pickle(file))
    except Exception as e:
        print(f"\tError when reading file {file}.")

print('\tConcatenating.')
df = pd.concat(df, ignore_index=True, verify_integrity=False, sort=False, copy=False)
df['value'] = df['value'].astype(float)
df['time'] = df['time'].astype(float)
df.to_pickle(join(base_dir, f'lambda_results.pkl'))
print("\tDone aggregating lambdas.\n")


# T-STATS
print("Aggregating tstat information.")
df = []
for k, file in enumerate(tstat_files):
    print(f"\tReading tstat file {k}.")
    try:
        df.append(pd.read_pickle(file))
    except Exception as e:
        print(f"\tError when reading file {file}.")

print('\tConcatenating.')
df = pd.concat(df, ignore_index=True, verify_integrity=False, sort=False, copy=False)
df.to_pickle(join(base_dir, f'tstat_results.pkl'))
print("\tDone aggregating tstats.\n")


