#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import norm
from time import time
import matplotlib.pyplot as plt

from adaptive_CI.saving import *

# magics removed
# magics removed


# # Introduction example

# The code below simulates the example in the introduction, but also computes the adaptive estimator with constant allocation (LvdL) weights.

# In[2]:


begin_time = time()


# In[3]:


# Experiment length
T = 1_000_000 

# Number of replications
num_sims = 1000


# In[4]:


avg_estimate = np.empty(num_sims, dtype=float)
aw_estimate = np.empty(num_sims, dtype=float)
ipw_estimate = np.empty(num_sims, dtype=float)

avg_stderr = np.empty(num_sims, dtype=float)
aw_stderr = np.empty(num_sims, dtype=float)
ipw_stderr = np.empty(num_sims, dtype=float)

avg_student = np.empty(num_sims, dtype=float)
aw_student = np.empty(num_sims, dtype=float)
ipw_student = np.empty(num_sims, dtype=float)

Tw = np.empty(num_sims, dtype=int)

for s in range(num_sims):
    
    print(f'Simulation {s}')
    
    # potential outcomes for first arm
    y = np.random.normal(loc=0, scale=1, size=T)
    
    # first half
    e1 = .5
    w1 = np.random.choice([0, 1], p=[e1, 1 - e1], size=T//2)
  
    # first arm mean at T/2
    muhat0 = np.mean(y[:T//2][w1 == 0])

    # second arm mean at T/2 
    # drawn from is from its asymptotic sampling distribution N(0, 1/(T/4))
    muhat1 = np.random.normal(loc=0, scale=1/np.sqrt(T/4), size=1) 
  
    # select arm of interest more often if its point estimate is larger
    e2 = .9 if muhat0 > muhat1 else .1
    w2 = np.random.choice([0, 1], p=[e2, 1 - e2], size=T//2)
  
    # concatenate first and second halves
    w = np.hstack([w1, w2])
    e = np.array([e1]*(T//2) + [e2]*(T//2))
    
    # ---- estimates: sample mean -----
    avg_estimate[s] = np.mean(y[w == 0])
    avg_stderr[s] = np.std(y[w == 0]) / np.sqrt(np.sum(w == 0))
    avg_student[s] = avg_estimate[s] / avg_stderr[s]
    Tw[s] = np.sum(w == 0)
    
    # ---- estimates: ipw ---- 
    ipw_estimate[s] = np.mean(y * (w == 0) / e)
    ipw_stderr[s] = np.std(y * (w == 0) / e) / np.sqrt(T)
    ipw_student[s] = ipw_estimate[s] / ipw_stderr[s]
    
    # ---- estimates: aw (constant-allocation) ----
    scores = muhat1 + (w == 0)/e * (y - muhat1)
    lambda_alloc = 1 / (T - np.arange(1, T + 1) + 1)  # constant allocation rates
    
    # evaluation weights
    h2e = np.zeros(T)  # h^2/e
    h2e_sum = 0
    for t in range(T):
        h2e[t] = lambda_alloc[t] * (1 - h2e_sum)
        h2e_sum += h2e[t]
    evaluation_weights = np.sqrt(np.maximum(0., h2e * e))
        
    # statistics
    aw_estimate[s] = np.sum(evaluation_weights * scores, 0)  / np.sum(evaluation_weights, 0)
    aw_stderr[s] = np.sqrt(np.sum(evaluation_weights ** 2 * (scores - aw_estimate[s])** 2, 0)) / np.sum(evaluation_weights, 0)
    aw_student[s] = aw_estimate[s] / aw_stderr[s]


# In[ ]:


data = pd.DataFrame({
    "T":T,
    "Tw": Tw,
    "avg_estimate": avg_estimate,
    "avg_student": avg_student,
    "ipw_estimate": ipw_estimate,
    "ipw_student": ipw_student,
    "aw_estimate": aw_estimate,
    "aw_student": aw_student,
})

if on_sherlock():
    write_dir = get_sherlock_dir('adaptive-confidence-intervals', 'simulations', create=True)
else:
    write_dir = os.path.join(os.getcwd(), 'results')
filename = compose_filename('intro', 'pkl')
write_path = os.path.join(write_dir, filename)
print(f"Saving {write_path}")
data.to_pickle(write_path)


# In[ ]:


end_time = time()
print("Total time: {:1.1f} seconds.".format(end_time - begin_time))

