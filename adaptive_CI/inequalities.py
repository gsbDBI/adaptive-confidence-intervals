import numpy as np
from scipy.optimize import root_scalar
from warnings import catch_warnings, simplefilter

__all__ = [
    'get_bernstein_radius',
    'get_bennett_radius',
    'get_hoeffding_radius',
]


""" Bernstein """

def get_bernstein_radius(M, v_sum, delta):
    return 1/3 * np.log(2/delta) * M + np.sqrt(1/9 * np.log(2/delta)**2 * M ** 2 + 2 * v_sum * np.log(2/delta))
    

""" Bennett """

def theta(x):
    return (1 + x) * np.log(1 + x) - x

def bennett_rhs(x, M, v_sum):
    return 2 * np.exp( - v_sum / M ** 2 * theta(M * x / v_sum))
    
def get_bennett_radius(M, v_sum, delta):
    """
    Radius B of pointwise confidence interval around based on the Bennett inequality,
    i.e., B such that P(|sum[i to n] X[i]| > B) < delta. (Note: two-sided)
    
    n: total number of observations
    M: proxy for max P(|X[i] - EX[i]| < M) = 1 for all i
    v: proxy for sum of conditional variances sum[i to n] E[X[i]^2|F[i-1]]
    delta: significance level
    
    Reference: Wainwright textbook, exercise 2.7, page 51.
    """
    xstar = root_scalar(lambda x: bennett_rhs(x, M, v_sum) - delta,
                        method="brentq",
                        bracket=[0, 1 / M])  # TODO
    return xstar.root
    

""" Hoeffding """
    
def hoeffding_rhs(x, n, M, v_sum):
    # normalize parameters
    v_sum = v_sum / M ** 2
    x = x / M
    if x >= n:
        return 0
    # bound
    a = (v_sum / (x + v_sum)) ** (x + v_sum)
    b = (n / (n - x)) ** (n - x)
    return 2 * (a * b) ** (n / (n + v_sum))
    
def get_hoeffding_radius(n, M, v_sum, delta):
    """
    Radius B of pointwise confidence interval around based on the Hoeffding inequality,
    i.e., B such that P(|sum[i to n] X[i]| > B) < delta. (Note: two-sided)
    
    n: total number of observations
    M: proxy for max P(|X[i] - EX[i]| < M) = 1 for all i
    v: proxy for sum of conditional variances sum[i to n] E[X[i]^2|F[i-1]]
    delta: significance level
    """
    with catch_warnings() as w:
        simplefilter("ignore", RuntimeWarning)
        xstar = root_scalar(lambda x: hoeffding_rhs(x, n, M, v_sum) - delta, 
                            method="brentq",
                            bracket=[0, 1 / M])
    return xstar.root
