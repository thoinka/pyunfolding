from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import numpy as np
from itertools import product
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde


def hex2int(txt):
    h = txt.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2 ,4))


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return LinearSegmentedColormap('CustomMap', cdict)


cmap_colors = ["#ffffff", "7cc149"]
custom_greens = make_colormap([hex2int(c) for c in cmap_colors])


def corner_plot(X, n_bins=20, hist2d_kw=dict(), kde=False, best_fit=None, scatter=False):
    """Corner-style plot
    """
    n_corners = X.shape[1]
    fig, ax = plt.subplots(n_corners, n_corners, figsize=(n_corners * 2, n_corners * 2))
    for i, j in product(range(n_corners), range(n_corners)):
        if i < j:
            ax[i,j].set_visible(False)
        elif i == j:
            if kde:
                kde = gaussian_kde(X[:,i].T)
                t = np.linspace(np.min(X[:,i]), np.max(X[:,i]), 100)
                y = kde(t)
                cs = np.cumsum(y)
                cs /= np.max(cs)
                one_s = (cs > 0.15) & (cs < 0.85)
                two_s = (cs > 0.025) & (cs < 0.975)
                ax[i,j].fill_between(t[two_s], y[two_s], facecolor="#7cc149", alpha=0.2)
                ax[i,j].fill_between(t[one_s], y[one_s], facecolor="#7cc149")
                ax[i,j].plot(t, y, color="k")
                #ax[i,j].axhline(0.0, color="k")
                ax[i,j].set_xlim([np.min(X[:,i]), np.max(X[:,i])])
                ax[i,j].set_ylim([0,None])
            else:
                H, b, _ = ax[i,j].hist(X[:,i], bins="auto", density=True, histtype="step", color="k")
                bmid = (b[1:] + b[:-1]) * 0.5
                perc_s1 = [np.percentile(X[:,i], 15), np.percentile(X[:,i], 85)] 
                perc_s2 = [np.percentile(X[:,i], 2.5), np.percentile(X[:,i], 99.75)] 
                ax[i,j].fill_between(bmid, np.zeros_like(H), H, step="mid", facecolor="#7cc149",
                                     where=(bmid >= perc_s1[0]) & (bmid <= perc_s1[1]))
                ax[i,j].fill_between(bmid, np.zeros_like(H), H, step="mid", facecolor="#7cc149", alpha=0.2, where=(bmid >= perc_s2[0]) & (bmid <= perc_s2[1]))
            if best_fit is not None:
                ax[i,j].axvline(best_fit[i], color="k", linestyle=":")
            ax[i,j].set_frame_on(False)
        else:
            if kde:
                kde = gaussian_kde(X[:,[j,i]].T)
                mins = np.min(X[:,[j,i]], axis=0)
                maxs = np.max(X[:,[j,i]], axis=0)
                t_0 = np.linspace(mins[0], maxs[0], 100)
                t_1 = np.linspace(mins[1], maxs[1], 100)
                mx, my = np.meshgrid(t_0, t_1)
                Z = kde(np.vstack((mx.flatten(),
                                   my.flatten()))).reshape(100,100)
                z0 = np.max(Z)
                z1 = z0 / np.e
                z2 = z1 / np.e ** 2
                ax[i,j].legend([], title="%.3f" % np.corrcoef(X[:,i],X[:,j])[0,1], frameon=False, loc="upper left")
                ax[i,j].contourf(t_0, t_1, Z, levels=np.linspace(0.0, z0, 20), cmap=custom_greens)
                if scatter:
                    Z_pnts = kde(X[:,[j,i]].T)
                    sel = Z_pnts < z2
                    ax[i,j].scatter(X[sel,j], X[sel,i], s=2, color="k")
                ax[i,j].contour(t_0, t_1, Z, levels=[z2, z1, z0], colors=["k", "k"], linestyles=["--", "-"])
            else:
                H, bx, by, _ = ax[i,j].hist2d(X[:,j], X[:,i], bins=n_bins,
                                              cmap=custom_greens, **hist2d_kw)
                z0 = np.max(H)
                z1 = z0 / np.e
                z2 = z1 / np.e ** 2
                ax[i,j].contour((bx[1:] + bx[:-1]) * 0.5, (by[1:] + by[:-1]) * 0.5, H.T, levels=[z2, z1, z0], colors=["k", "k"], linestyles=["--", "-"])
                ax[i,j].legend([], title="%.3f" % np.corrcoef(X[:,i],X[:,j])[0,1], frameon=False, loc="upper left")
            if best_fit is not None:
                ax[i,j].axvline(best_fit[j], color="k", linestyle=":")
                ax[i,j].axhline(best_fit[i], color="k", linestyle=":")
        ax[i,j].xaxis.set_visible(False)
        ax[i,j].yaxis.set_visible(False)
        if j == 0:
            if i != 0:
                ax[i,j].yaxis.set_visible(True)
            if i != 0:
                ax[i,j].set_ylabel("Bin {}".format(i))
        if i == n_corners - 1:
            ax[i,j].xaxis.set_visible(True)
            ax[i,j].set_xlabel("Bin {}".format(j + 1))
            
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    return ax