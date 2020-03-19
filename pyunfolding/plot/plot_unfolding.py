from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize
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
    try:
        x = result.binning_y.bmids[0]
        dx = result.binning_y.bdiff[0]
        target_var = True
    except AttributeError:
        x = np.arange(len(result.f)) + 1
        dx = np.ones(len(result.f))
        target_var = False
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
                            step='mid', facecolor=color, lw=2,
                            edgecolor=plt.rcParams['axes.facecolor'])
    else:
        ax.fill_between(x[sl], y[sl] - y_err[0,sl], y[sl] + y_err[1,sl], step='mid', color='k', alpha=0.1, lw=0)
    ax.errorbar(x[sl], y[sl], 0.0, 0.5 * dx[sl], ls='', lw=2,
                color=plt.rcParams['lines.color'], label='Unfolding result')
    if truth is not None:
        ax.plot(x[sl], truth[sl], drawstyle='steps-mid', label='Truth',
                color=plt.rcParams['lines.color'], alpha=0.5)
    if target_var:
        ax.set_xlabel('Target Variable $y$')
    else:
        ax.set_xlabel('Target Variable Bin $i$')
    ax.set_ylabel('Counts per Bin')
    mappable = ScalarMappable(norm=Normalize(-1.0, 1.0), cmap=get_cmap('coolwarm'))
    mappable.set_array([])
    cbar = plt.colorbar(mappable=mappable, ax=ax, pad=0, ticks=np.linspace(-1.0, 1.0, 3))
    cbar.set_ticklabels(['–1', '±0', '+1'])
    cbar.set_label('Bin-to-Bin Correlation')