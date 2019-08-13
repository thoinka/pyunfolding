import numpy as np
from itertools import product
from sklearn.tree import DecisionTreeClassifier
from ..utils import calc_bmids, calc_bdiff
from .base import Binning

import warnings


def _bin_edges(X):
    X_sort = np.sort(X)
    return (X_sort[1:] + X_sort[:-1]) * 0.5


def _equidistant_bins(X, Xmin, Xmax, n_bins):
    return np.linspace(Xmin, Xmax, n_bins - 1)


def _equal_bins(X, Xmin, Xmax, n_bins):
    bin_edges = _bin_edges(X[(X > Xmin) & (X < Xmax)])
    idx = np.linspace(0, len(bin_edges) - 1, n_bins - 1).astype(int)
    return bin_edges[idx]


def _random_bins(X, Xmin, Xmax, n_bins):
    rand_edges = np.sort(np.random.uniform(Xmin, Xmax, n_bins - 3))
    return np.r_[Xmin, rand_edges, Xmax]


def _random_equal_bins(X, Xmin, Xmax, n_bins):
    bin_edges = _bin_edges(X[(X > Xmin) & (X < Xmax)])
    idx = np.sort(np.random.choice(len(bin_edges) - 2, n_bins - 3,
                                   replace=True) + 1)
    return np.r_[Xmin, bin_edges[idx], Xmax]


__binning_schemes__ = {
    'equidistant': _equidistant_bins,
    'equal': _equal_bins,
    'random': _random_bins,
    'random-equal': _random_equal_bins
}


class GridBinning(Binning):
    """Multidimensional Binning on regular grids. Bin edges can either be
    chosen manually or using a specific scheme.

    Parameters
    ----------
    bins : either list of `numpy array` or list of `int`
        Either list of bin edges or list of number of bins in each dimension.
    scheme : `str`
        Specifies the method that picks the positions of the bin edges. The
        following options are available:
            * ``equidistant`` will pick equidistant bin edges starting from
              `pmin` and ending in `pmax`
            * ``equal`` will pick bins that contain an equal number of samples
              in it.
            * ``random`` will pick completely random bin edges in between
              `pmin` and `pmax`
            * ``random-equal`` will pick random bin edges according quantiles.
    pmin, pmax : `float`
        Quantile of the lowest and highest bin edge (in percent!)

    Attributes
    ----------
    pmin, pmax : `float`
        Quantile of the lowest and highest bin edge (in percent!)
    scheme : `function` 
        binning scheme function with signature ``f(X, Xmin, Xmax, n_bins)``.
    n_bins : `int`
        Number of bins, *including* under- and overflow bins.
    bins : list of `numpy array`
        Either list of bin edges in each dimension.
    bmids : numpy array
        Mid points of each bin. For under- and overflow bins the bin mids are
        extrapolated (and meaningless).
    bdiff : numpy array
        Size of each bin. For under- and overflow bins the bin diffs are
        extrapolated (and meaningless).
    """

    def __init__(self, bins, scheme='equidistant', pmin=0.0, pmax=100.0):
        super(GridBinning, self).__init__(bins)
        self.pmin = pmin
        self.pmax = pmax
        self.scheme = __binning_schemes__[scheme]

    def fit(self, X, *args, **kwargs):
        super(GridBinning, self).fit(X)
        if len(X.shape) < 2:
            X = X.reshape(-1, 1)
        if type(self.bins) == int:
            self.bins = [self.scheme(f,
                                     np.percentile(f, self.pmin),
                                     np.percentile(f, self.pmax), self.bins)
                         for f in X.T]
        if type(self.bins[0]) == int:
            self.bins = [self.scheme(np.percentile(f, self.pmin),
                                     np.percentile(f, self.pmax), n)
                         for f, n in zip(X.T, self.bins)]
        for i in range(len(self.bins)):
            self.bins[i][-1] += np.finfo("float64").resolution
        self.n_bins = np.product([len(b) + 1 for b in self.bins])
        self.bmids = np.array([calc_bmids(b) for b in self.bins])
        self.bdiff = np.array([calc_bdiff(b) for b in self.bins])

    def digitize(self, X):
        super(GridBinning, self).digitize(X)
        if len(X.shape) < 2:
            X = X.reshape(-1, 1)
        n_dim = X.shape[1]
        B = list(product(*[range(len(self.bins[i]) + 1)
                           for i in range(n_dim)]))
        D = {b: i for i, b in enumerate(B)}
        c = np.array([np.digitize(X[:, i], self.bins[i])
                      for i in range(n_dim)]).T
        bin_assoc = [D[tuple(k)] for k in c]
        return np.array(bin_assoc)