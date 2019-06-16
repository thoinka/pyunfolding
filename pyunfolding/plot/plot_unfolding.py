from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np


def cov2corr(cov):
    '''Convert covariance matrix to correlation matrix.
    '''
    std = np.sqrt(cov.diagonal())
    corr = cov / np.outer(std, std)
    return corr


def plot_unfolding_result(result,
                          ax=None,
                          truth=None,
                          exclude_edges=True,
                          correlations=True):
    if exclude_edges:
        sl = slice(1, -1, None)
    else:
        sl = slice(None, None, None)
    x = result.binning_y.bmids[0]
    dx = result.binning_y.bdiff[0]
    y = result.f
    y_err = result.f_err
    if ax is None:
        fig, ax = plt.subplots(1)
    if correlations:
        cov = result.cov
        corr = cov2corr(cov)
        corr_adj = corr.diagonal(1)
        cmap = get_cmap('coolwarm')
        for i in range(len(y))[sl][:-1]:
            color = cmap(corr_adj[i] * 0.5 + 0.5)
            ax.fill_between(x[i:i+2], y[i:i+2] - y_err[0,i:i+2],
                            y[i:i+2] + y_err[1,i:i+2],
                            step='mid', color=color, lw=0)
    else:
        ax.fill_between(x[sl], y[sl] - y_err[0,sl], y[sl] + y_err[1,sl], step='mid', color='k', alpha=0.1, lw=0)
    ax.errorbar(x[sl], y[sl], 0.0, 0.5 * dx[sl], ls='', color='k', label='Unfolding result')
    if truth is not None:
        ax.plot(x[sl], truth[sl], drawstyle='steps-mid', label='Truth', color='k', alpha=0.5)
    ax.set_xlabel('Target Variable $y$')
    ax.set_ylabel('Counts per Bin')
    return ax