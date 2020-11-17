from adaptive_CI.compute import *
import numpy as np


def twopoint_stable_var_ratio(e, alpha):
    T, K = e.shape
    t = np.arange(1, T + 1)[:, np.newaxis]
    
    # bad arm, e small
    bad_lambda = (1 - alpha) / ((1 - alpha) + T*(t/T)**alpha - t)

    # good arm, e large 
    good_lambda = 1 / (1 + T - t)                        

    assert np.all(bad_lambda + 1e-7 >= good_lambda) # the 1e-7 is for numerical issues
        
    # weighted average of both
    lamb = (1 - e) * bad_lambda + e * good_lambda
        
    # Sometimes, due to numerical issues the lambdas end up very slightly above 1.
    # This clipping ensures that everyting is okay.
    assert np.all(lamb >= 0)
    assert np.all(lamb <= 1 + 1e-8)
    lamb = np.clip(lamb, 0, 1) 
    return lamb


def twopoint_stable_var_ratio_old(probs, floor_start, floor_decay):
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
    T, K = probs.shape
    t = np.arange(1, T + 1)[:, np.newaxis]
    l_t = floor_start * (((t + 1) ** (1-floor_decay) - t **
                          (1-floor_decay))) / (1 - floor_decay)
    L_t = floor_start * (((T + 1) ** (1-floor_decay) - t **
                          (1-floor_decay))) / (1 - floor_decay)

    ratio_best = (1 - (K-1) * l_t) / ((T + 1) - t - (K-1) * L_t)
    ratio_not_best = l_t / L_t
    ratio = probs * ratio_best + (1 - probs) * ratio_not_best

    return ratio