import seaborn as sns
import pandas as pd
import numpy as np

import pickle
from pickle import UnpicklingError
from glob import glob
from os.path import join

sns.set_context("notebook", font_scale=1.4)


commit = "72121e"

base_dir = "/scratch/groups/athey/adaptive-confidence-intervals/vitorh/simulations/"
stats_files = glob(f"{base_dir}/stats*{commit}*.pkl")
print(f"Found {len(stats_files)} files associated with statistics.")


dfs = []
for k, file in enumerate(stats_files[:10]):
    if k % 200 == 0:
        print(f"Reading file {k}.")
    try:
        dfs.append(pd.read_pickle(file))
    except Exception as e:
        print(f"Error when reading file {file}.")
        
print(f"Loaded {len(dfs)} files.")
df = pd.concat(dfs)
df['value'] = df['value'].astype(float)
decay_rates = np.sort(df['floor_decay'].unique())
print("df shape:", df.shape)
df.to_pickle(join(base_dir, 'total_results.pkl'))