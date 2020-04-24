import numpy as np
import sys
sys.path.insert(0, "/home/rhzhan/adaptive-confidence-intervals/")
from adaptive_CI.experiments import run_mab_experiment
from time import time

def calculate_W_lambda(config, pcts, TT):
    noise_func, noise_scale = config['noise_func'], config['noise_scale']
    K = config['K']
    T = max(TT)

    arms = [] # (#sim_W, 20000)
    for s in range(config['sim_W']):    
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

    W_lambdas = [] 
    for t in TT:
        arm_counts = [] # size (sim_W, #arms)
        # Record selected arms for W-decorrelation construction
        for arm in arms:
            arm_counts.append([np.sum(arm[:t] == w) for w in range(K)])  # size (sim_W, #arms)
            assert(np.sum(arm_counts[-1])==t and f'{np.sum(arm_counts[-1])} not equal to  {t}')

        # size (#percentiles, #arms)
        pct_arm_counts = np.percentile(arm_counts, pcts, axis=0)
        # size (#percentiles)
        min_pct_arm_counts = np.amin(pct_arm_counts, axis=1)

        # size (#percentile, t), a list of W_lambdas for different percentiles
        W_lambdas_t = [np.ones(t)*min_pct/np.log(t) for min_pct in min_pct_arm_counts]
        W_lambdas.append(W_lambdas_t)
    return W_lambdas

start_time = time()


experiment, truth, noise_func = sys.argv[1], sys.argv[2], sys.argv[3]
noise_scale = 1.0
truth = [float(a) for a in truth.split(',')]
config = dict(
        K = len(truth),
        sim_W = 5000,
        truth = truth,
        noise_func = noise_func,
        noise_scale = noise_scale,
        initial = 5,
        floor_start = 0.1,
        floor_decay = 0.5,
        exploration = 'TS',
        )
percentiles = [5, 15, 35, 50]
TT = [1000, 5000, 10000, 20000]
W_lambdas = calculate_W_lambda(config, percentiles, TT)
for t, W_lam in zip(TT, W_lambdas):
    name = f'W_lambdas_{experiment}-{noise_func}-{t}.npz'
    np.savez(name, percentiles=percentiles, W_lambdas=W_lam)
print(f'time passed {time()-start_time}')
