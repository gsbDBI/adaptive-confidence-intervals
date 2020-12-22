"""
This script contains helper functions to do inference including computing scores and statistics of arm values and contrasts.
"""

import numpy as np
from scipy.stats import norm
from confseq import boundaries
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
    "beta_bernoulli_stats",
    "gamma_exponential_stats",
]

        

def beta_bernoulli_stats(outcomes, treatments, truth, K, decay_rate, alpha=.1):
    T = len(outcomes)
    means = np.empty(K)
    stderr = np.empty(K)
    confidence_interval = np.empty((K, 2))
    coverage = np.empty(K)
    ci_radius = np.empty(K)
    
    for w in range(K):
        y = outcomes[treatments == w]
        y_min, y_max = np.min(y), np.max(y)
        y_normalized = (y - y_min) / (y_max - y_min) # normalizing to [0, 1]
        t_opt = int((1/K) * np.sum(np.arange(1, T+1)**-decay_rate))
        num_successes = np.sum(y_normalized)
        num_trials = np.sum(treatments == w)

        try:
            ci_normalized = boundaries.bernoulli_confidence_interval(  
                num_successes=int(num_successes), 
                num_trials=num_trials, 
                t_opt=t_opt, 
                alpha=alpha / 2, 
                alpha_opt=alpha / 2)
        except Exception as e:
            print("Error while computing bernoulli confidence interval.")
            print("num_successes: ", int(num_successes))
            print("num_trials", num_trials)
            print("t_opt", t_opt)
            print(e)
            ci_normalized = (np.nan, np.nan)
                
        means[w] = np.mean(y)
        stderr[w] = np.std(y) / np.sqrt(len(y))
        confidence_interval = np.array(ci_normalized) * (y_max - y_min) + y_min
        ci_radius[w] = np.diff(confidence_interval) / 2
        coverage[w] = float(confidence_interval[0] < truth[w] < confidence_interval[1])

    relerror = (means - truth) / stderr
    relerror[stderr == 0] = np.nan
    bias = means - truth
    error = bias ** 2
    out = np.stack((means, stderr, bias, coverage, relerror, error, ci_radius, truth))
    return out


def gamma_exponential_stats(outcomes, treatments, truth, K, decay_rate, c, expected_noise_variance, alpha=.1):
    T = len(outcomes)
    means = np.empty(K)
    stderr = np.empty(K)
    confidence_interval = np.empty((K, 2))
    coverage = np.empty(K)
    ci_radius = np.empty(K)
    t_opt = int((1/K) * np.sum(np.arange(1, T+1)**-decay_rate))
    v_opt = t_opt * expected_noise_variance

    for w in range(K):
        selector = treatments == w
        Tw = np.sum(selector)
        y = outcomes[selector]
        lagged_means = np.roll(np.cumsum(y) / np.arange(1, Tw + 1), 1)
        Vt = np.sum((y - lagged_means)**2)
        ci_radius[w] = (1 / Tw *
                        boundaries.gamma_exponential_mixture_bound(
                            v=Vt,
                            v_opt=v_opt, 
                            c=c, 
                            alpha=alpha / 2,
                            alpha_opt=alpha / 2))

        means[w] = np.mean(y)
        stderr[w] = np.std(y) / np.sqrt(Tw)
        confidence_interval = (means[w] - ci_radius[w], means[w] + ci_radius[w])
        coverage[w] = float(confidence_interval[0] < truth[w] < confidence_interval[1])

    relerror = (means - truth) / stderr
    relerror[stderr == 0] = np.nan
    bias = means - truth
    error = bias ** 2
    out = np.stack((means, stderr, bias, coverage, relerror, error, ci_radius, truth))
    return out




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
    
    
# 
# def form_predictions(outcomes, selector, default_prediction):
#     means = np.cumsum(outcomes * selector) / np.maximum(1, np.cumsum(selector))
#     lagged_means = np.roll(means, 1)
#     lagged_means[0] = default_prediction
#     return lagged_means
# 
# 
# def estimate_average_treatment_effect_gamma_mixture(outcomes, treatments, propensity,
#                                       support_bounds, expected_outcome_noise,
#                                       p_min, # <--------- CHANGED
#                                       optimized_t, y1_predictions=None,
#                                       y0_predictions=None, coverage_alpha=0.05,
#                                       alpha_opt=0.05):
#     support_center = (support_bounds[0] + support_bounds[1]) / 2.0
#     v_opt = (optimized_t * expected_outcome_noise
#              * (1 / propensity + 1 / (1 - propensity)))
#     if y1_predictions is None:
#         y1_predictions = form_predictions(outcomes, treatments == 1,
#                                           support_center)
#     if y0_predictions is None:
#         y0_predictions = form_predictions(outcomes, treatments == 0,
#                                           support_center)
# 
#     tau_hat = y1_predictions - y0_predictions
#     weights = (treatments - propensity) / (propensity * (1 - propensity))
#     predictions = np.where(treatments == 1, y1_predictions, y0_predictions)
#     Xt = tau_hat + weights * (outcomes - predictions)
#     St = np.cumsum(Xt)
#     Vt = np.cumsum((Xt - tau_hat)**2)
#     #    p_min = min(propensity, 1 - propensity)  # <--------- CHANGED
#     support_diameter = support_bounds[1] - support_bounds[0]
#     c = 2 * support_diameter / p_min
#     t_array = np.arange(1.0, len(outcomes) + 1.0)
#     p_value = np.exp(-boundaries.gamma_exponential_log_mixture(
#              St, Vt, v_opt, c, alpha_opt=alpha_opt / 2))
#     confidence_radius = (
#         1.0 / t_array * boundaries.gamma_exponential_mixture_bound(
#             Vt, coverage_alpha / 2, v_opt, c, alpha_opt=alpha_opt / 2))
#     return pd.DataFrame(collections.OrderedDict([
#         ('t', t_array),
#         ('point_estimate', St / t_array),
#         ('upper_confidence_bound', np.minimum(
#             support_diameter, St / t_array + confidence_radius)),
#         ('lower_confidence_bound', np.maximum(
#             -support_diameter, St / t_array - confidence_radius)),
#         ('p_value', np.minimum(p_value, 1.0))]))
# 
# 
# 
# def howard_stats(outcomes, treatments, propensity, truth, K, decay_rate, delta=0.10):
#     """
#     Compute the population bernstein confidence interval, plugging in sample values.
#     Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2827893/.
# 
#     INPUT:
#         - rewards: observed rewards of shape [T]
#         - arms: pulled arms of shape [T]
#         - truth: true arm values of shape [K]
#         - K: number of arms
#         - inequality: bernstein, bennett or hoeffding
# 
#     OUTPUT:
#         - sample mean statistics with population Bernstein confidence interval of arm values: [estimate, S.E., bias, (1-alpha)-coverage, t-statistic, MSE, confidence_interval_radius, truth]
#     """
#     means = np.empty(K)
#     ci_radius = np.empty(K)
#     T = len(outcomes)
#     stderr = np.empty(K)
# 
#     for w in range(K):
# 
#         optimized_t = int(np.sum(np.arange(1, T+1)**-decay_rate))
#         sigma = outcomes[treatments == w].std()
# 
#         results = estimate_value_normal_mixture(
#             outcomes=outcomes,
#             treatments=treatments,
#             propensity=propensity,
#             arm_index=w,
#             expected_outcome_noise=sigma,
#             coverage_alpha=delta,
#             alpha_opt=delta,
#             optimized_t=optimized_t
#         )
# 
#         Tw = int(np.sum(treatments == w))
#         result = results.iloc[Tw]
#         means[w] = result['point_estimate']
#         ci_radius[w] = (result['upper_confidence_bound'] - result['lower_confidence_bound'])/2
#         stderr[w] = outcomes[treatments == w].std() / np.sqrt(Tw)
# 
#     bias = means - truth
#     cover = (np.abs(bias) < ci_radius).astype(np.float_)
#     error = bias ** 2
# 
#     tstat = bias / stderr  # Note this is not a t-stat, need to fix names
#     out = np.stack((means, stderr, bias, cover, tstat, error, ci_radius, truth))
#     return out
# 
# 
# 
# 
# def estimate_average_treatment_effect_normal_mixture(
#         outcomes, treatments, propensity,
#         expected_outcome_noise,
#         optimized_t, 
#         y1_predictions=None,
#         y0_predictions=None, 
#         coverage_alpha=0.05,
#         alpha_opt=0.05):
# 
#     if y1_predictions is None:
#         y1_predictions = form_predictions(outcomes, treatments == 1, 0)
#     if y0_predictions is None:
#         y0_predictions = form_predictions(outcomes, treatments == 0, 0)
# 
#     # ---- begin change -----    
#     v_opt = expected_outcome_noise * np.sum(1 / propensity[:optimized_t] + 1 / (1 - propensity[:optimized_t]))
#     # --- end change ----
# 
#     tau_hat = y1_predictions - y0_predictions
#     weights = (treatments - propensity) / (propensity * (1 - propensity))
#     predictions = np.where(treatments == 1, y1_predictions, y0_predictions)
#     Xt = tau_hat + weights * (outcomes - predictions)
#     St = np.cumsum(Xt)
#     Vt = np.cumsum((Xt - tau_hat)**2)
# 
#     t_array = np.arange(1.0, len(outcomes) + 1.0)
# 
#     # ---- begin change ----
#     p_value = np.exp(-boundaries.normal_log_mixture(
#              St, Vt, v_opt, alpha_opt=alpha_opt, is_one_sided = False))
#     confidence_radius = (
#         1.0 / t_array * boundaries.normal_mixture_bound(
#             Vt, coverage_alpha, v_opt, alpha_opt=alpha_opt, is_one_sided = False))
#     # ---- end change -----
# 
#     return pd.DataFrame(collections.OrderedDict([
#         ('t', t_array),
#         ('point_estimate', St / t_array),
#         ('upper_confidence_bound', St / t_array + confidence_radius),
#         ('lower_confidence_bound', St / t_array - confidence_radius),
#         ('p_value', np.minimum(p_value, 1.0))]))
# 
