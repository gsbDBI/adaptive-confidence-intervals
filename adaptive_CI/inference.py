"""
This script contains helper functions to do inference including computing scores and statistics of arm values and contrasts.
"""

import numpy as np
from scipy.stats import norm
from confseq import boundaries
from adaptive_CI.compute import *
from adaptive_CI.inequalities import *



def aw_scores(rewards, arms, assignment_probs, muhat=None):
    """
    Compute AIPW scores. Return IPW scores if muhat is None.
    e[t] and mu[t, w] are functions of the history up to t-1.

    INPUT
        - rewards: observed rewards of shape [T]
        - arms: pulled arms of shape [T]
        - assignment_probs: probability of pulling arms of shape [T, K]
        - muhat: plug-in estimator for arms of shape [T, K]

    OUTPUT
        - scores: AIPW scores of shape [T, K]
    """
    T, K = assignment_probs.shape
    balwts = 1 / collect(assignment_probs, arms)
    scores = expand(balwts * rewards, arms, K)  # Y[t]*W[t]/e[t] term
    if muhat is not None:  # (1 - W[t]/e[t])mu[t,w] term
        scores += (1 - expand(balwts, arms, K)) * muhat
    return scores
    

def aw_contrast_stderr(score, evalwts, estimate):
    """
    Compute standard error of estimates of arm contrasts between the last arm and the remaining arms

    INPUT:
        - score: AIPW scores of shape [T, K]
        - evalwts: evaluation weights of shape [T]
        - estimate: weighted estimates of arm values of shape [K] 

    OUTPUT:
        - standard error: shape [K-1]
    """
    h_sum = evalwts.sum(0)
    diff = score - estimate
    numerator = h_sum[:-1] * evalwts[:, -1:] * diff[:, -1:] - h_sum[-1] * evalwts[:, :-1] * diff[:, :-1]
    numerator = np.sum(numerator ** 2, axis=0)
    denominator = h_sum[-1]**2 * h_sum[:-1]**2
    return np.sqrt(numerator / denominator)


    
def get_statistics(estimate, stderr, truth, ci_radius):
    bias = estimate - truth
    coverage = np.abs(bias) < ci_radius
    relerr = np.where(stderr > 0, bias / stderr, np.nan)
    mse = bias ** 2
    return np.stack((estimate, stderr, bias, coverage, relerr, mse, ci_radius, truth))

    
def evaluate_aipw_stats(score, evalwts, truth, alpha=.1):
    estimate = np.sum(evalwts * score, 0) / np.sum(evalwts, 0)
    stderr = np.sqrt(np.sum(evalwts ** 2 * (score - estimate)** 2, 0)) / np.sum(evalwts, 0)
    ci_radius =  norm.ppf(1 - alpha / 2) * stderr
    return get_statistics(estimate, stderr, truth, ci_radius)


def evaluate_sample_mean_naive_stats(outcomes, treatments, truth, K, weights=None, alpha=.1):
    estimate = np.empty(K)
    stderr = np.empty(K)
    ci_radius = np.empty(K)
    for w in range(K):
        y = outcomes[treatments == w]
        Tw = np.sum(treatments == w)
        estimate[w] = np.mean(y)
        stderr[w] = se = np.std(y) / np.sqrt(Tw)
        ci_radius[w] = norm.ppf(1 - alpha/2) * se
    return get_statistics(estimate, stderr, truth, ci_radius)  
    

def evaluate_sample_mean_naive_contrasts(outcomes, treatments, arm_truth, K, weights=None, alpha=.1):
    T = len(outcomes)
    arm_estimate = np.empty(K)
    arm_variances = np.empty(K)
    for w in range(K):
        y = outcomes[treatments == w]
        Tw = np.sum(treatments == w)
        arm_estimate[w] = np.mean(y)
        arm_variances[w] = np.var(y) / Tw
    
    estimate = arm_estimate[-1] - arm_estimate[:-1]
    stderr = np.sqrt(arm_variances[-1] + arm_variances[:-1])
    truth = arm_truth[-1] - arm_truth[:-1]
    ci_radius = norm.ppf(1 - alpha/2) * stderr
    return get_statistics(estimate, stderr, truth, ci_radius)
    

def evaluate_beta_bernoulli_stats(outcomes, treatments, truth, K, decay_rate, alpha=.1):
    T = len(outcomes)
    t_opt =  int((1/K) * np.sum(np.arange(1, T+1)**-decay_rate))
    estimate = np.empty(K)
    stderr = np.empty(K)
    ci_radius = np.empty(K)
    for w in range(K):
        y = outcomes[treatments == w]
        Tw = np.sum(treatments == w)
        estimate[w] = np.mean(y)
        stderr[w] = np.std(y) / np.sqrt(Tw)
        y_min, y_max = np.min(y), np.max(y)
        y_normalized = (y - y_min) / (y_max - y_min) # normalizing to [0, 1]
        ci_normalized = boundaries.bernoulli_confidence_interval(  
            num_successes=int(np.sum(y_normalized)), 
            num_trials=Tw, 
            t_opt=t_opt, 
            alpha=alpha / 2, 
            alpha_opt=alpha / 2)
        ci_denormalized = np.array(ci_normalized) * (y_max - y_min) + y_min
        ci_radius[w] = np.diff(ci_denormalized) / 2
    return get_statistics(estimate, stderr, truth, ci_radius)    


def evaluate_gamma_exponential_stats(outcomes, treatments, truth, K, decay_rate, c, expected_noise_variance, alpha=.1):
    T = len(outcomes)
    t_opt =  int((1/K) * np.sum(np.arange(1, T+1)**-decay_rate))
    v_opt = t_opt * expected_noise_variance
    estimate = np.empty(K)
    stderr = np.empty(K)
    ci_radius = np.empty(K)
    for w in range(K):
        y = outcomes[treatments == w]
        Tw = np.sum(treatments == w)
        estimate[w] = np.mean(y)
        stderr[w] = np.std(y) / np.sqrt(Tw)
        lagged_means = np.roll(np.cumsum(y) / np.arange(1, Tw + 1), 1)
        Vt = np.sum((y - lagged_means)**2)
        ci_radius[w] = (1 / Tw *
                        boundaries.gamma_exponential_mixture_bound(
                            v=Vt,
                            v_opt=v_opt, 
                            c=c, 
                            alpha=alpha / 2,
                            alpha_opt=alpha / 2))
    return get_statistics(estimate, stderr, truth, ci_radius) 
    

def evaluate_aipw_contrasts(scores, evalwts, arm_truth, alpha=.1):
    """
    Compute statistics of arm contrast estimations (last arm vs others)

    INPUT:
        - score: AIPW scores of shape [T, K]
        - evalwts: evaluation weights of shape [T]
        - truth: true arm values of shape [K]

    OUTPUT:
        - statistics of arm contrasts
    """
    arm_estimate = np.sum(evalwts * scores, 0) / np.sum(evalwts, 0)
    estimate = arm_estimate[-1] - arm_estimate[:-1]
    stderr = aw_contrast_stderr(scores, evalwts, arm_estimate)
    truth = arm_truth[-1] - arm_truth[:-1]
    ci_radius = norm.ppf(1 - alpha / 2) * stderr 
    return get_statistics(estimate, stderr, truth, ci_radius)


def evaluate_beta_bernoulli_contrasts(outcomes, treatments, arm_truth, K, decay_rate, alpha=.1):
    T = len(outcomes)
    t_opt =  int((1/K) * np.sum(np.arange(1, T+1)**-decay_rate))
    arm_estimate = np.empty(K)
    arm_variances = np.empty(K)
    arm_ci = np.empty((K, 2))
    ci_radius = np.empty(K-1)
    for w in range(K):
        y = outcomes[treatments == w]
        Tw = np.sum(treatments == w)
        arm_estimate[w] = mean = np.mean(y)
        arm_variances[w] = np.var(y) / Tw
        y_min, y_max = np.min(y), np.max(y)
        y_normalized = (y - y_min) / (y_max - y_min) # normalizing to [0, 1]
        try:
            ci_normalized = boundaries.bernoulli_confidence_interval(  
                num_successes=int(np.sum(y_normalized)), 
                num_trials=Tw, 
                t_opt=t_opt, 
                alpha=alpha / 4, 
                alpha_opt=alpha / 4)
        except Exception as e:
            print("Exception while running bb contrasts")
            print(int(np.sum(y_normalized)), Tw, t_opt, alpha/4)
            ci_normalized = (np.nan, np.nan)
        arm_ci[w] = np.array(ci_normalized) * (y_max - y_min) + y_min
    
    # Union bound
    for w in range(K-1):
        ci_union = (arm_ci[-1, 0] - arm_ci[w, 1], arm_ci[-1, 1] - arm_ci[w, 0])
        ci_radius[w] = np.ravel(np.diff(ci_union, 1) / 2)
    
    estimate = arm_estimate[-1] - arm_estimate[:-1]
    stderr = np.sqrt(arm_variances[-1] + arm_variances[:-1])
    truth = arm_truth[-1] - arm_truth[:-1]
    return get_statistics(estimate, stderr, truth, ci_radius)


def evaluate_gamma_exponential_contrasts(outcomes, treatments, arm_truth, K, decay_rate, c, 
                                             expected_noise_variance, alpha=.1):   
    T = len(outcomes)     
    t_opt =  int((1/K) * np.sum(np.arange(1, T+1)**-decay_rate))
    v_opt = t_opt * expected_noise_variance
    arm_estimate = np.empty(K)
    arm_variances = np.empty(K)
    arm_ci = np.empty((K, 2))
    ci_radius = np.empty(K-1)
    for w in range(K):
        y = outcomes[treatments == w]
        Tw = np.sum(treatments == w)
        arm_estimate[w] = mean = np.mean(y)
        arm_variances[w] = np.var(y) / Tw
        lagged_means = np.roll(np.cumsum(y) / np.arange(1, Tw + 1), 1)
        Vt = np.sum((y - lagged_means)**2)
        arm_ci_radius = (1 / Tw *
                            boundaries.gamma_exponential_mixture_bound(
                                v=Vt,
                                v_opt=v_opt, 
                                c=c, 
                                alpha=alpha / 4,  # note the alpha / 4 for a two-sided alpha interval
                                alpha_opt=alpha / 4))
        arm_ci[w] = (mean - arm_ci_radius, mean + arm_ci_radius)
    
    # Union bound
    for w in range(K-1):
        ci_union = (arm_ci[-1, 0] - arm_ci[w, 1], arm_ci[-1, 1] - arm_ci[w, 0])
        ci_radius[w] = np.ravel(np.diff(ci_union, 1) / 2)
    
    estimate = arm_estimate[-1] - arm_estimate[:-1]
    stderr = np.sqrt(arm_variances[-1] + arm_variances[:-1])
    truth = arm_truth[-1] - arm_truth[:-1]
    return get_statistics(estimate, stderr, truth, ci_radius)

    

def wdecorr_stats(arms, rewards, K, W_lambdas, truth, alpha=0.10):
    """
    Compute W-decorrelation estimates of arm values
    Adapted from Multi-armed Bandits.ipynb in https://github.com/yash-deshpande/decorrelating-linear-models.
    Source: Deshpande, Y., Mackey, L., Syrgkanis, V., & Taddy, M. (2017). Accurate inference for adaptive linear models. arXiv preprint arXiv:1712.06695.

    INPUT: 
        - rewards: observed rewards of shape [T]
        - arms: pulled arms of shape [T]
        - K: number of arms
        - W_lambdas: bias-variance tradeoff parameter lambda in W-decorrelation paper, of shape [T]
        - truth: true arm values of shape [K]

    OUTPUT:
        - W-decorrelation statistics of arm values: [estimate, S.E., bias, (1-alpha)-coverage, t-statistic, MSE, confidence_interval_radius, truth]
    """
    T = len(arms)

    # estimate OLS sample mean and sample variance
    arms_W = expand(np.ones(T), arms, K)
    arms_Y = expand(rewards, arms, K)
    samplemean = np.sum(arms_W * arms_Y, 0) / np.maximum(1, np.sum(arms_W, 0))
    samplevars = np.sum(arms_W * (arms_Y - samplemean) ** 2,
                        0) / (np.maximum(np.sum(arms_W, 0), 1))

    # latest parameter estimate vector
    beta = np.copy(samplemean)
    # Latest w_t vector
    w = np.zeros((K))
    # Latest matrix W_tX_t = w_1 x_1^T + ... + w_t x_t^T
    WX = np.zeros((K, K))
    # Latest vector of marginal variances reward_vars * (w_1**2 + ... + w_t**2)
    variances = np.zeros(K)

    for t in range(T):
        # x_t = e_{arm}
        arm = arms[t]
        # y_t = reward
        reward = rewards[t]
        # Update w_t = (1/(norm{x_t}^2+lambda_t)) (x_t - W_{t-1} X_{t-1} x_t)
        np.copyto(w, -WX[:, arm])
        w[arm] += 1
        w /= (1.0 + W_lambdas[:, t])
        # Update beta_t = beta_{t-1} + w_t (y_t - <beta_OLS, x_t>)
        beta += w * (reward - samplemean[arm])
        # Update W_tX_t = W_{t-1}X_{t-1} + w_t x_t^T
        WX[:, arm] += w
        # Update marginal variances
        variances += samplevars * w ** 2
    estimate = beta
    stderr = np.sqrt(variances)
    bias = estimate - truth
    tstat = bias / stderr
    tstat[stderr == 0] = np.nan
    quantile = norm.ppf(1 - alpha / 2)
    cover = (np.abs(tstat) < quantile).astype(np.float_)
    ci_r = quantile * stderr
    error = bias ** 2
    return np.stack((estimate, stderr, bias, cover, tstat, error, ci_r, truth))


def sample_mean(rewards, arms, K):
    """
    Compute F_{t} measured sample mean estimator

    INPUT: 
        - rewards: observed rewards of shape [T]
        - arms: pulled arms of shape [T]
        - K: number of arms

    Output:
        - estimate: F_{t} measured sample mean estimator of shape [T,K]
    """
    # return F_t measured sample mean
    T = len(arms)
    W = expand(np.ones(T), arms, K)
    Y = expand(rewards, arms, K)
    estimate = np.cumsum(W * Y, 0) / np.maximum(np.cumsum(W, 0), 1)
    return estimate
    
