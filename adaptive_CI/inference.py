"""
This script contains helper functions to do inference including computing scores and statistics of arm values and contrasts.
"""

import numpy as np
from scipy.stats import norm
from adaptive_CI.compute import *

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
    "population_bernstein_stats",
    "empirical_bernstein_stats",
    "population_bernstein_contrast"
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
        - statistics of arm estimates: [estimate, S.E., bias, (1-alpha)-coverage, t-statistic, MSE, confidence_interval_width, truth]

    output: dim (len(stats), K)
    """
    estimate = aw_estimate(score, evalwts)
    bias = estimate - truth
    stderr = aw_stderr(score, evalwts, estimate)
    tstat = aw_tstat(estimate, stderr, truth)
    quantile = norm.ppf(1.0 - alpha / 2)
    cover = (np.abs(tstat) < quantile).astype(np.float_)
    error = bias ** 2
    ci_w = quantile * stderr * 2
    return np.stack((estimate, stderr, bias, cover, tstat, error, ci_w, truth))


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
    numerator = h_sum[1:] * evalwts[:, :1] * diff[:, :1] - \
        h_sum[0] * evalwts[:, 1:] * diff[:, 1:]
    numerator = np.sum(numerator ** 2, axis=0)
    denominator = h_sum[0]**2 * h_sum[1:]**2
    return np.sqrt(numerator / denominator)


def aw_contrasts(score, evalwts, truth, alpha=0.10):
    """
    Compute statistics of arm contrast estimations

    INPUT:
        - score: AIPW scores of shape [T, K]
        - evalwts: evaluation weights of shape [T]
        - truth: true arm values of shape [K]

    OUTPUT:
        - statistics of arm contrasts: [truth, estimate, bias, MSE, stderr, t-statistic, (1-alpha)-coverage, confidence_interval_width]
    """
    estimate = aw_estimate(score, evalwts)
    contrast_estimate = estimate[0] - estimate[1:]
    contrast_truth = truth[0] - truth[1:]
    contrast_bias = contrast_estimate - contrast_truth
    contrast_mse = contrast_bias ** 2

    contrast_stderr = aw_contrast_stderr(score, evalwts, estimate)
    contrast_tstat = aw_tstat(
        contrast_estimate, contrast_stderr, contrast_truth)
    quantile = norm.ppf(1 - alpha / 2)
    contrast_cover = (np.abs(contrast_tstat) < quantile).astype(np.float_)
    ci_w = quantile * contrast_stderr * 2

    return np.stack((contrast_truth, contrast_estimate,  contrast_bias, contrast_mse,
                     contrast_stderr,  contrast_tstat, contrast_cover, ci_w))


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
        - sample mean statistics of arm values: [estimate, S.E., bias, (1-alpha)-coverage, t-statistic, MSE, confidence_interval_width, truth]
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
    ci_w = quantile * stderr * 2
    out = np.stack((estimate, stderr, bias, cover, tstat, error, ci_w, truth))
    return out


def population_bernstein_stats(rewards, arms, truth, K, alpha=0.10):
    """
    Compute the population bernstein confidence interval, plugging in sample values.
    Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2827893/.

    INPUT:
        - rewards: observed rewards of shape [T]
        - arms: pulled arms of shape [T]
        - truth: true arm values of shape [K]
        - K: number of arms

    OUTPUT:
        - sample mean statistics with population Bernstein confidence interval of arm values: [estimate, S.E., bias, (1-alpha)-coverage, t-statistic, MSE, confidence_interval_width, truth]
    """
    T = len(rewards)
    W = expand(np.ones(T), arms, K)
    Tw = np.sum(W, 0)
    Y = expand(rewards, arms, K)
    estimate = np.sum(W * Y, 0) / np.maximum(1, Tw)
    
    M = np.max(np.abs(Y - estimate) * W, axis=0) / np.maximum(1, Tw)
    v = np.sum(W * (Y - estimate) ** 2, axis=0) / np.maximum(1, Tw) ** 2
    ci_radius = 1/3 * np.log(2/alpha) * M + np.sqrt(1/9 * np.log(1/alpha)**2 * M ** 2 + 2 * v * np.log(2/alpha))

    bias = estimate - truth
    cover = (np.abs(bias) < ci_radius).astype(np.float_)

    error = bias ** 2
    stderr = np.sqrt(np.sum(W ** 2 * (Y - estimate) ** 2, axis = 0)) / np.maximum(1, Tw)
    out = np.stack((estimate, stderr, bias, cover, [None] * K, error, ci_radius * 2, truth))
    return out

    

def population_bernstein_contrast(rewards, arms, truth, K, alpha=0.10):
    # Means and variances of rewards for each arm
    Tw = np.maximum([np.sum(arms == w) for w in range(K)], 1)
    means = np.array([np.mean(rewards[arms == w]) for w in range(K)])
    var = np.array([np.var(rewards[arms == w]) for w in range(K)])

    # Treatment effect estimate
    estimate = means[1:] - means[0]

    # Bernstein parameter: variance
    v = var[0] / Tw[0] + var[1:] / Tw[1:]

    # Bernstein parameter: maximum proxy
    YW_ctr = np.column_stack([(rewards - means[w]) / Tw[w] * (arms == w) for w in range(K)])
    M = np.max(np.abs(YW_ctr[:, 0:1] - YW_ctr[:, 1:]), 0)   
    
    # Bernstein confidence interval
    ci_radius = 1/3 * np.log(2/alpha) * M + np.sqrt(1/9 * np.log(1/alpha)**2 * M ** 2 + 2 * v * np.log(2/alpha))

    # Other statistics
    bias = estimate - (truth[1:] - truth[0])
    cover = (np.abs(bias) < ci_radius).astype(np.float_)
    sqerror = bias ** 2
    stderr = np.sqrt(v)
    contrast_truth = truth[1:] - truth[0]
    
    # Store everything in the same order as aw_contrasts
    out = np.stack([
        contrast_truth,
        estimate, 
        bias,
        sqerror,
        stderr,
        [np.nan] * (K - 1),
        cover,
        2 * ci_radius,
    ])
    return out


def empirical_bernstein_stats(rewards, arms, truth, K, R, alpha=0.10):
    """
    Compute the empirical bernstein confidence interval, by Audibert et al. [2007], Mnih et al. [2008], Maurer and Pontil [2009]. 

    INPUT:
        - rewards: observed rewards of shape [T]
        - arms: pulled arms of shape [T]
        - truth: true arm values of shape [K]
        - K: number of arms

    OUTPUT:
        - sample mean statistics with population Bernstein confidence interval of arm values: [estimate, S.E., bias, (1-alpha)-coverage, t-statistic, MSE, confidence_interval_width, truth]
    """
    T = len(rewards)
    W = expand(np.ones(T), arms, K)
    Tw = np.sum(W, 0)
    Y = expand(rewards, arms, K)
    estimate = np.sum(W * Y, 0) / np.maximum(1, Tw)
    stderr = np.sqrt(np.sum(W ** 2 * (Y - estimate) ** 2, 0)
                     ) / np.maximum(1, Tw)

    v = np.sum(W * (Y - estimate) ** 2, axis=0) / np.maximum(1, Tw)
    empirical_bernstein = np.sqrt(v) * np.sqrt(2 * np.log(3 / alpha) / Tw) + 3 * R * np.log(3 / alpha) / Tw
    ci_r = empirical_bernstein 

    bias = estimate - truth
    error = bias ** 2
    cover = (np.abs(bias) < ci_r).astype(np.float_)

    out = np.stack((estimate, stderr, bias, cover,
                    [None] * K, error, ci_r * 2, truth))
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
        - W-decorrelation statistics of arm values: [estimate, S.E., bias, (1-alpha)-coverage, t-statistic, MSE, confidence_interval_width, truth]
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
    ci_w = quantile * stderr * 2
    error = bias ** 2
    return np.stack((estimate, stderr, bias, cover, tstat, error, ci_w, truth))


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
