import pandas as pd
import numpy as np


import pickle
from pickle import UnpicklingError
from glob import glob
from os.path import join


commit = "4fb095"

base_dir = "/scratch/groups/athey/adaptive-confidence-intervals/vitorh/simulations/"
stats_files = glob(f"{base_dir}/contrast*{commit}*.pkl")
print(f"Found {len(stats_files)} files associated with statistics.")


df = []
for k, file in enumerate(stats_files):
    print(f"Reading file {k}.")
    try:
        df_tmp = pd.read_pickle(file)
        df_tmp = df_tmp.query(
            "statistic == ['mse', 'bias', 'CI_width', '90% coverage of t-stat'] and "
            "method == ['uniform', 'lvdl', 'two_point',  'sample_mean_naive', 'gamma_exponential', 'W-decorrelation_15'] and "
            "policy == [0, 1, 2]"
        )
        df_tmp['value'] = df_tmp['value'].astype(float)
        df.append(df_tmp)
    except Exception as e:
        print(f"Error when reading file {file}.")

print('concatenating')
df = pd.concat(df, ignore_index=True, verify_integrity=False, sort=False, copy=False)
df.to_pickle(join(base_dir, f'arm_results.pkl'))
    
            



base_dir = "/scratch/groups/athey/adaptive-confidence-intervals/vitorh/simulations/"
stats_files = glob(f"{base_dir}/total_results_contrast*.pkl")
stats_files = [f for f in stats_files if 'tight' not in f]
print(f"Found {len(stats_files)} files.")

df = []
for k, file in enumerate(stats_files):
    print(k, file)
    try:
        df_tmp = pd.read_pickle(file)
        df_tmp = df_tmp.query(
            "method == ['uniform', 'lvdl', 'two_point',  'sample_mean_naive', 'gamma_exponential']"
        )
        df.append(df_tmp)
    except Exception as e:
        print(f"Error when reading file {file}.")

df = pd.concat(df, ignore_index=True, verify_integrity=False, sort=False, copy=False)
df.to_pickle(join(base_dir, f'total_results_contrast_tight_all.pkl'))
print(df.shape)