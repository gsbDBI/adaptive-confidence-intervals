from itertools import product

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, Lasso
from sklearn.utils.testing import ignore_warnings

from adaptive_CI.compute import expand


def preprocess(x, w, K):
    t = len(x)
    w_dummy = expand(np.ones(t), w, K)[:, 1:]
    kx = x.shape[1]
    kw = w_dummy.shape[1]
    xw = np.column_stack([x[:, i] * w_dummy[:, j]
                          for i, j in product(range(kx), range(kw))])
    return np.column_stack([x, w, xw])


@ignore_warnings(category=ConvergenceWarning)
def fit_bs_lasso(x, w, y, K, num_bs):
    t = len(x)
    prepxw = preprocess(x, w, K)

    # Fit once with cv
    lasso1 = LassoCV(cv=3).fit(prepxw, y)
    alpha = lasso1.alpha_

    # Fit num_bs times with optimal cv
    fitted_models = []
    for i in range(num_bs):
        idx = np.random.randint(0, t, size=t)
        xwbs = prepxw[idx]
        ybs = y[idx]
        lasso = Lasso(alpha=alpha).fit(xwbs, ybs)
        fitted_models.append(lasso)
    return fitted_models


def predict_bs_lasso(fitted_models, x_new, K):
    assert x_new.ndim == 2
    t_new = len(x_new)
    # Predict rewards for each arm
    ypreds = np.empty((len(fitted_models), t_new, K))
    for wval in range(K):
        wp = np.full(t_new, fill_value=wval)
        xwp = preprocess(x_new, wp, K)
        for i, m in enumerate(fitted_models):
            ypreds[i, :, wval] = m.predict(xwp)
    return ypreds


def ts_probs(mu, sigma, nts):
    t, K = mu.shape
    draws = np.stack([np.random.normal(mu, sigma) for _ in range(nts)], axis=2)
    best = np.argmax(draws, 1)
    nselected = np.apply_along_axis(
        lambda z: np.bincount(
            z, minlength=K), 1, best)
    return nselected / nts
