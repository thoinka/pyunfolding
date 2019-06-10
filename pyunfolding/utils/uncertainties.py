import numpy as np


ONE_SIGMA = 0.682689492137085897


def error_central(x, p=ONE_SIGMA):
    lower = np.percentile(x, 100.0 * (1.0 - p) / 2.0, axis=0)
    upper = np.percentile(x, 100.0 * (1.0 + p) / 2.0, axis=0)
    return lower, upper


def error_shortest(x, p=ONE_SIGMA):
    lower = np.zeros(x.shape[1])
    upper = np.zeros(x.shape[1])
    N = int(p * len(x))
    for i in range(x.shape[1]):
        x_sort = np.sort(x[:, i])
        x_diff = np.abs(x_sort[N:] - x_sort[:-N])
        shortest = np.argmin(x_diff)
        lower[i], upper[i] = x_sort[shortest], x_sort[shortest + N]
    return lower, upper


def error_best(x, f, p=ONE_SIGMA):
    N = int(p * len(x))
    best = np.argsort(f)[:N]
    return np.min(x[best, :], axis=0), np.max(x[best, :], axis=0)