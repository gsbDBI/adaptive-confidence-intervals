import numpy as np

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
    "aw_conditions",
]





def aw_scores(rewards, arms, assignment_probs, muhat=None):
    T, K = assignment_probs.shape
    balwts = 1 / collect(assignment_probs, arms)
    scores = expand(balwts * rewards, arms, K)  # Y[t]*W[t]/e[t] term
    if muhat is not None:  # (1 - W[t]/e[t])mu[t,w] term
        scores += (1 - expand(balwts, arms, K)) * muhat
    return scores


def aw_estimate(score, evalwts):
    return np.sum(evalwts * score, 0) / np.sum(evalwts, 0)


def aw_stderr(score, evalwts, estimate):
    return np.sqrt(np.sum(evalwts ** 2 * (score - estimate)
                          ** 2, 0)) / np.sum(evalwts, 0)


def aw_tstat(estimate, stderr, truth):
    out = (estimate - truth) / stderr
    out[stderr == 0] = np.nan
    return out


def aw_stats(score, evalwts, truth):
    """
    output: dim (len(stats), K)
    """
    estimate = aw_estimate(score, evalwts)
    bias = estimate - truth
    stderr = aw_stderr(score, evalwts, estimate)
    tstat = aw_tstat(estimate, stderr, truth)
    cover = (np.abs(tstat) < 1.96).astype(np.float_)
    error = bias ** 2
    return np.stack((estimate, stderr, bias, cover, tstat, error, truth))



def aw_contrast_stderr(score, evalwts, estimate):
    h_sum = evalwts.sum(0)
    T, K = np.shape(score)
    diff = score - estimate
    numerator = h_sum[1:]* evalwts[:, :1] * diff[:, :1] - h_sum[0] * evalwts[:, 1:] * diff[:, 1:]
    numerator = np.sum(numerator ** 2, axis=0)
    denominator = h_sum[0]**2 * h_sum[1:]**2
    return np.sqrt(numerator / denominator)



def aw_contrasts(score, evalwts, truth):
    """
    output: dim (len(stats), K-1)
    estimate contrasts between arm 0 and the rest arms
    """
    estimate = aw_estimate(score, evalwts)
    contrast_estimate = estimate[0] - estimate[1:]
    contrast_truth = truth[0] - truth[1:]
    contrast_bias = contrast_estimate - contrast_truth
    contrast_mse = contrast_bias ** 2


    contrast_stderr = aw_contrast_stderr(score, evalwts, estimate)
    contrast_tstat = aw_tstat(contrast_estimate, contrast_stderr, contrast_truth)
    contrast_cover = (np.abs(contrast_tstat) < 1.96).astype(np.float_)

    return np.stack((contrast_truth, contrast_estimate,  contrast_bias, contrast_mse, 
        contrast_stderr,  contrast_tstat, contrast_cover)) 

def naive_stats(rewards, arms, truth, K, weights=None):
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
    cover = (np.abs(tstat) < 1.96).astype(np.float_)
    out = np.stack((estimate, stderr, bias, cover, tstat, error, truth))
    return out


def wdecorr_stats(arms, rewards, K, W_lambdas, truth):
    """
    Adapted from Multi-armed Bandits.ipynb in https://github.com/yash-deshpande/decorrelating-linear-models.
    Source: Deshpande, Y., Mackey, L., Syrgkanis, V., & Taddy, M. (2017). Accurate inference for adaptive linear models. arXiv preprint arXiv:1712.06695.
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
        w /= (1.0 + W_lambdas[t])
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
    cover = (np.abs(tstat) < 1.96).astype(np.float_)
    error = bias ** 2
    return np.stack((estimate, stderr, bias, cover, tstat, error, truth))


def sample_mean(rewards, arms, K):
    # returb F_t measured sample mean
    T = len(arms)
    W = expand(np.ones(T), arms, K)
    Y = expand(rewards, arms, K)
    estimate = np.cumsum(W * Y, 0) / np.maximum(np.cumsum(W, 0), 1)
    return estimate


def aw_conditions(wts, probs):
    h2e = wts ** 2 / probs
    sumh4e3 = np.sum(wts ** 4 / probs ** 3, 0)
    sumh = np.sum(wts, 0)
    maxh2e = np.amax(h2e, 0)
    sumh2e = np.sum(h2e, 0)

    cond51 = maxh2e / sumh2e  # [K]
    cond52 = sumh ** 2 / sumh2e  # [K]
    cond6num = sumh2e  # [K]
    cond7num = sumh4e3  # [K]
    return np.stack([cond51, cond52, cond6num, cond7num])



