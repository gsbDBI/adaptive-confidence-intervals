"""
This script contains helper functions to compute evaluation weights. 
"""

from adaptive_CI.compute import *
import numpy as np


def twopoint_stable_var_ratio(probs, floor_start, floor_decay):
    """
    Compute lambda of two-point allocation rate weights

    INPUT:
        - probs: arm assignment probabilities of shape [T, K]
        - floor_start: assignment probability floor starting value
        - floor_decay: assignment probability floor decaying rate
        assignment probability floor = floor start * t ^ {-floor_decay}

    OUTPUT:
        - ratio: lambda of size [T]
    """
    # analytical expectation is E[e_t/(e_t+...+e_T)]
    T, K = probs.shape
    t = np.arange(T)[:, np.newaxis]
    t_plus = np.arange(1, T+1)[:, np.newaxis]
    l_t = floor_start * ((t_plus ** (1-floor_decay) - t **
                          (1-floor_decay))) / (1 - floor_decay)
    L_t = floor_start * ((T ** (1-floor_decay) - t **
                          (1-floor_decay))) / (1 - floor_decay)

    ratio_best = (1 - (K-1) * l_t) / (T - t - (K-1) * L_t)
    ratio_not_best = l_t / L_t
    # ratio_best = ratio_best * ratio_not_best[0,0] / ratio_best[0,0]
    ratio = probs * ratio_best + (1 - probs) * ratio_not_best
    # ratio = ratio / ratio[-1:, :]

    return ratio
