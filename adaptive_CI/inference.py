"""
This script contains helper functions to do inference including computing scores and statistics of arm values and contrasts.
"""

import numpy as np
from scipy.stats import norm
from adaptive_CI.compute import *
from adaptive_CI.inequalities import *

__all__ = [
    "aw_scores",
    "aw_estimate",
    "aw_stderr",
    "aw_tstat",
    "aw_stats",
    "aw_contrasts",
    "wdecorr_stats",
    "naive_stats",
    "sample_mean",
    "finite_sample_inequality_stats",
    "finite_sample_inequality_contrast",
]


def aw_scores(rewards, arms, assignment_probs, muhat=None):
    """
    Compute leave-future-out AIPW scores. Return IPW scores if muhat is None.
    e[t] and mu[t, w] are fitted based on history up to t-1.

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


def aw_estimate(score, evalwts):
    """
    Compute weighted estimates of arm values

    INPUT:
        - score: AIPW scores of shape [T, K]
        - evalwts: evaluation weights of shape [T]

    OUTPUT:
        - estimates: shape [K]
    """
    return np.sum(evalwts * score, 0) / np.sum(evalwts, 0)


def aw_stderr(score, evalwts, estimate):
    """
    Compute standard error of estimates of arm values

    INPUT:
        - score: AIPW scores of shape [T, K]
        - evalwts: evaluation weights of shape [T]
        - estimate: weighted estimates of arm values of shape [K] 

    OUTPUT:
        - standard error: shape [K]
    """
    return np.sqrt(np.sum(evalwts ** 2 * (score - estimate)
                          ** 2, 0)) / np.sum(evalwts, 0)


def aw_tstat(estimate, stderr, truth):
    """
    Compute t-statistic of estimates of arm values:

    INPUT:
        - estimate: weighted estimates of arm values of shape [K]
        - stderr: standard error of estimators of arm values of shape [K]
        - truth: true arm values of shape [K]

    OUTPUT:
        - out: t-statistic of estimates of arm values of shape [K]
    """
    out = (estimate - truth) / stderr
    out[stderr == 0] = np.nan
    return out


def aw_stats(score, evalwts, truth, alpha=0.10):
    """
    Compute statistics of arm estimation

    INPUT:
        - score: AIPW scores of shape [T, K]
        - evalwts: evaluation weights of shape [T]
        - truth: true arm values of shape [K]

    OUTPUT:
        - statistics of arm estimates: [estimate, S.E., bias, (1-alpha)-coverage, t-statistic, MSE, confidence_interval_radius, truth]

    output: dim (len(stats), K)
    """
    estimate = aw_estimate(score, evalwts)
    bias = estimate - truth
    stderr = aw_stderr(score, evalwts, estimate)
    tstat = aw_tstat(estimate, stderr, truth)
    quantile = norm.ppf(1.0 - alpha / 2)
    cover = (np.abs(tstat) < quantile).astype(np.float_)
    error = bias ** 2
    ci_r = quantile * stderr
    return np.stack((estimate, stderr, bias, cover, tstat, error, ci_r, truth))


def aw_contrast_stderr(score, evalwts, estimate):
    """
    Compute standard error of estimates of arm contrasts between arm[0] and the remaining arms

    INPUT:
        - score: AIPW scores of shape [T, K]
        - evalwts: evaluation weights of shape [T]
        - estimate: weighted estimates of arm values of shape [K] 

    OUTPUT:
        - standard error: shape [K-1]
    """

    h_sum = evalwts.sum(0)
    T, K = np.shape(score)
    diff = score - estimate
    # numerator = h_sum[1:] * evalwts[:, :1] * diff[:, :1] - h_sum[0] * evalwts[:, 1:] * diff[:, 1:]
    numerator = h_sum[:-1] * evalwts[:, -1:] * diff[:, -1:] - h_sum[-1] * evalwts[:, :-1] * diff[:, :-1]
    numerator = np.sum(numerator ** 2, axis=0)
    #denominator = h_sum[0]**2 * h_sum[1:]**2
    denominator = h_sum[-1]**2 * h_sum[:-1]**2
    return np.sqrt(numerator / denominator)
    
    
    


def aw_contrasts(score, evalwts, truth, alpha=0.10):
    """
    Compute statistics of arm contrast estimations (last arm vs others)

    INPUT:
        - score: AIPW scores of shape [T, K]
        - evalwts: evaluation weights of shape [T]
        - truth: true arm values of shape [K]

    OUTPUT:
        - statistics of arm contrasts: [truth, estimate, bias, MSE, stderr, t-statistic, (1-alpha)-coverage, confidence_interval_radius]
    """
    estimate = aw_estimate(score, evalwts)
    contrast_estimate = estimate[-1] - estimate[:-1]
    contrast_truth = truth[-1] - truth[:-1]
    contrast_bias = contrast_estimate - contrast_truth
    contrast_mse = contrast_bias ** 2

    contrast_stderr = aw_contrast_stderr(score, evalwts, estimate)
    contrast_tstat = aw_tstat(
        contrast_estimate, contrast_stderr, contrast_truth)
    quantile = norm.ppf(1 - alpha / 2)
    contrast_cover = (np.abs(contrast_tstat) < quantile).astype(np.float_)
    ci_r = quantile * contrast_stderr 

    return np.stack((contrast_truth, contrast_estimate,  contrast_bias, contrast_mse,
                     contrast_stderr,  contrast_tstat, contrast_cover, ci_r))


def naive_stats(rewards, arms, truth, K, weights=None, alpha=0.10):
    """
    Compute naive sample mean estimator

    INPUT:
        - rewards: observed rewards of shape [T]
        - arms: pulled arms of shape [T]
        - truth: true arm values of shape [K]
        - K: number of arms
        - weights: weights applied to samples of shape [K]

    OUTPUT:
        - sample mean statistics of arm values: [estimate, S.E., bias, (1-alpha)-coverage, t-statistic, MSE, confidence_interval_radius, truth]
    """
    T = len(rewards)
    if weights is None:
        weights = np.ones(T)
    W = expand(weights, arms, K)
    Y = expand(rewards, arms, K)
    estimate = np.sum(W * Y, 0) / np.maximum(1, np.sum(W, 0))
    stderr = np.sqrt(np.sum(W ** 2 * (Y - estimate) ** 2, 0)) / \
        np.maximum(np.sum(W, 0), 1)
    tstat = (estimate - truth) / stderr
    tstat[stderr == 0] = np.nan
    bias = estimate - truth
    error = bias ** 2
    quantile = norm.ppf(1 - alpha / 2)
    cover = (np.abs(tstat) < quantile).astype(np.float_)
    ci_r = quantile * stderr
    out = np.stack((estimate, stderr, bias, cover, tstat, error, ci_r, truth))
    return out


def finite_sample_inequality_stats(rewards, arms, truth, K, inequality, delta=0.10):
    """
    Compute the population bernstein confidence interval, plugging in sample values.
    Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2827893/.

    INPUT:
        - rewards: observed rewards of shape [T]
        - arms: pulled arms of shape [T]
        - truth: true arm values of shape [K]
        - K: number of arms
        - inequality: bernstein, bennett or hoeffding

    OUTPUT:
        - sample mean statistics with population Bernstein confidence interval of arm values: [estimate, S.E., bias, (1-alpha)-coverage, t-statistic, MSE, confidence_interval_radius, truth]
    """
    Tw = np.maximum([np.sum(arms == w) for w in range(K)], 1)
    means = np.array([np.mean(rewards[arms == w]) for w in range(K)])
    var = np.array([np.var(rewards[arms == w]) for w in range(K)])
    M = np.array([np.max(rewards[arms == w] - means[w]) for w in range(K)])

    if inequality == 'bernstein':
        ci_radius = get_bernstein_radius(M / Tw, var / Tw, delta)
    elif inequality == 'bennett':
        ci_radius = np.array([get_bennett_radius(M[w] / Tw[w], var[w] / Tw[w], delta) for w in range(K)])
    elif inequality == 'hoeffding':
        ci_radius = np.array([get_hoeffding_radius(Tw[w], M[w] / Tw[w], var[w] / Tw[w], delta) for w in range(K)])
    else:
        raise ValueError(f'Cannot understand inequality type {inequality}')
        
    bias = means - truth
    cover = (np.abs(bias) < ci_radius).astype(np.float_)
    error = bias ** 2
    stderr = np.sqrt(var / Tw)
    tstat = bias / stderr  # Note this is not a t-stat, need to fix names
    out = np.stack((means, stderr, bias, cover, tstat, error, ci_radius, truth))
    return out



def finite_sample_inequality_contrast(rewards, arms, truth, K, inequality, delta=0.10):
    """
    Compute the population bernstein confidence interval, plugging in sample values.
    Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2827893/.

    INPUT:
        - rewards: observed rewards of shape [T]
        - arms: pulled arms of shape [T]
        - truth: true arm values of shape [K]
        - K: number of arms
        - inequality: bernstein, bennett or hoeffding

    OUTPUT:
        - sample mean statistics with population Bernstein confidence interval of arm values: [estimate, S.E., bias, (1-alpha)-coverage, t-statistic, MSE, confidence_interval_radius, truth]
    """
    Tw = np.maximum([np.sum(arms == w) for w in range(K)], 1)
    means = np.array([np.mean(rewards[arms == w]) for w in range(K)])
    var = np.array([np.var(rewards[arms == w]) for w in range(K)])
    
    # Treatment effect estimate
    estimate = means[-1] - means[:-1]
    
    # Inequality parameter: maximum proxy
    YW_ctr = np.column_stack([(rewards - means[w]) / Tw[w] * (arms == w) for w in range(K)])
    M = np.max(np.abs(YW_ctr[:, -1:] - YW_ctr[:, :-1]), 0) 
    
    # Inequality parameter: variance proxy
    v = var[-1] / Tw[-1] + var[:-1] / Tw[:-1]
    
    if inequality == 'bernstein':
        ci_radius = get_bernstein_radius(M, v, delta)
    elif inequality == 'bennett':
        ci_radius = np.array([get_bennett_radius(M[w], v[w], delta) for w in range(K - 1)])
    elif inequality == 'hoeffding':
        ci_radius = np.array([get_hoeffding_radius(Tw[w], M[w], v[w], delta) for w in range(K - 1)])
    else:
        raise ValueError(f'Cannot understand inequality type {inequality}')
        
    # Other statistics
    bias = estimate - (truth[-1] - truth[:-1])
    cover = (np.abs(bias) < ci_radius).astype(np.float_)
    sqerror = bias ** 2
    stderr = np.sqrt(v)
    contrast_truth = truth[-1] - truth[:-1]
    tstat = bias / stderr # note this is not really a t-stat, need to fix names
    
    # Store everything in the same order as aw_contrasts
    out = np.stack([
        contrast_truth,
        estimate, 
        bias,
        sqerror,
        stderr,
        tstat,
        cover,
        ci_radius,
    ])
    return out
    

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
