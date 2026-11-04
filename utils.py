import numpy as np


def logsumexp(arr):
    arr = np.asarray(arr)
    m = np.max(arr)
    if np.isneginf(m):
        return -np.inf
    return m + np.log(np.sum(np.exp(arr - m)))