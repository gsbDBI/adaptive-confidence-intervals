import numpy as np

__all__ = ["groupsum",
           "collect",
           "expand",
           "draw",
           "apply_floor",
           "stick_breaking"]


def collect(arr, idx):
    out = np.empty(len(idx), dtype=arr.dtype)
    for i, j in enumerate(idx):
        out[i] = arr[i, j]
    return out


def expand(values, idx, num_cols):
    out = np.zeros((len(idx), num_cols), dtype=values.dtype)
    for i, (j, v) in enumerate(zip(idx, values)):
        out[i, j] = v
    return out


def groupsum(array, group, K):
    out = np.zeros(K, dtype=array.dtype)
    for a, g in zip(array, group):
        out[g] += a
    return out


def draw(p):
    return np.searchsorted(np.cumsum(p), np.random.random(), side="right")


def apply_floor(a, amin):
    new = np.maximum(a, amin)
    total_slack = np.sum(new) - 1
    individual_slack = new - amin
    c = total_slack / np.sum(individual_slack)
    return new - c * individual_slack


def stick_breaking(Z):
    T, K = Z.shape
    weights = np.zeros((T, K))
    weight_sum = np.zeros(K)
    for t in range(T):
        weights[t] = Z[t] * (1 - weight_sum)
        weight_sum += weights[t]
    return weights
