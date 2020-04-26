import numpy as np

__all__ = ["groupsum",
           "collect",
           "expand",
           "draw",
           "apply_floor",
           "stick_breaking"]


def collect(arr, idx):
    """
    Collect values of specific indices in array. _collect_ and _expand_ are inverse function of each other.

    INPUT:
        - arr: array of shape [T, K]
        - idx: indices of shape [T]

    OUTPUT:
        - out: collected values of shape [T]
    """
    out = np.empty(len(idx), dtype=arr.dtype)
    for i, j in enumerate(idx):
        out[i] = arr[i, j]
    return out


def expand(values, idx, num_cols):
    """
    Expand values to the specific indices in new array. _collect_ and _expand_ are inverse function of each other.

    INPUT: 
        - arr: array of shape [T]
        - idx: indices of shape [T]
        - num_cols: number of columns (K) of expanded arrays

    OUTPUT:
        - out: expanded values of shape [T, K]
    """
    out = np.zeros((len(idx), num_cols), dtype=values.dtype)
    for i, (j, v) in enumerate(zip(idx, values)):
        out[i, j] = v
    return out


def groupsum(array, group, K):
    """
    Compute summation within groups.

    INPUT:
        - array: values to be summed up
        - group: group ids
        - K: number of groups

    OUTPUT:
        - out: sums within groups of shape [K]
    """
    out = np.zeros(K, dtype=array.dtype)
    for a, g in zip(array, group):
        out[g] += a
    return out


def draw(p):
    """
    Draw samples based on probability p.
    """
    return np.searchsorted(np.cumsum(p), np.random.random(), side="right")


def apply_floor(a, amin):
    """
    Apply assignment probability floor.

    INPUT:
        - a: assignmented probabilities of shape [K]
        - amin: assignment probability floor

    OUTPUT:
        - assignmented probabilities of shape [K] after applying floor 
    """
    new = np.maximum(a, amin)
    total_slack = np.sum(new) - 1
    individual_slack = new - amin
    c = total_slack / np.sum(individual_slack)
    return new - c * individual_slack


def stick_breaking(Z):
    """
    Stick breaking algorithm in stable-var weights calculation

    Input:
        - Z: input array of shape [T, K]

    Output:
        - weights: stick_breaking weights of shape [T, K]
    """
    T, K = Z.shape
    weights = np.zeros((T, K))
    weight_sum = np.zeros(K)
    for t in range(T):
        weights[t] = Z[t] * (1 - weight_sum)
        weight_sum += weights[t]
    return weights
