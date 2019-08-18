import numpy as np


__all__ = ["calc_bmids",
           "calc_bdiff",
           "digitize_uni",
           "bin_edges",
           "equidistant_bins",
           "equal_bins",
           "random_bins",
           "random_equal_bins"]


def calc_bmids(bin_edges, underflow=True, overflow=True):
    """Calculates the centers of bins from bin edges.
    """
    bmids = (bin_edges[1:] + bin_edges[:-1]) * 0.5
    if underflow:
    	underflow_bmid = 2 * bin_edges[0] - bin_edges[1]
    	bmids = np.r_[underflow_bmid, bmids]
    if overflow:
    	overflow_bmid =  2 * bin_edges[-1] - bin_edges[-2]
    	bmids = np.r_[bmids, overflow]
    return bmids


def calc_bdiff(bin_edges, underflow=True, overflow=True):
    bdiff = np.diff(bin_edges)
    if underflow:
    	bdiff = np.r_[bdiff[0], bdiff]
    if overflow:
    	bdiff = np.r_[bdiff, bdiff[-1]]
    return bdiff


def digitize_uni(x, bins, underflow=True, overflow=True):
    idx = np.digitize(x, bins)
    selection = np.ones(len(idx), dtype=bool)
    if not overflow:
        selection[idx == len(bins)] = False
    if not underflow:
        selection[idx == 0] = False
        idx -= 1
    idx[~selection] = -1
    return idx


def bin_edges(X):
    X_sort = np.sort(X)
    return (X_sort[1:] + X_sort[:-1]) * 0.5


def equidistant_bins(X, Xmin, Xmax, n_bins, **kwargs):
    return np.linspace(Xmin, Xmax, n_bins - 1)


def equal_bins(X, Xmin, Xmax, n_bins, **kwargs):
    edges = bin_edges(X[(X > Xmin) & (X < Xmax)])
    idx = np.linspace(0, len(edges) - 1, n_bins - 1).astype(int)
    return edges[idx]


def random_bins(X, Xmin, Xmax, n_bins, rnd, **kwargs):
    edges = np.sort(rnd.uniform(Xmin, Xmax, n_bins - 3))
    return np.r_[Xmin, edges, Xmax]


def random_equal_bins(X, Xmin, Xmax, n_bins, rnd, **kwargs):
    edges = bin_edges(X[(X > Xmin) & (X < Xmax)])
    idx = np.sort(rnd.choice(len(edges) - 2, n_bins - 3,
                                   replace=True) + 1)
    return np.r_[Xmin, edges[idx], Xmax]