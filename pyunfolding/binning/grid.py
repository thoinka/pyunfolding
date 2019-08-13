import numpy as np
from itertools import product
from sklearn.tree import DecisionTreeClassifier
from ..utils import (calc_bmids,
                     calc_bdiff,
                     equidistant_bins,
                     equal_bins,
                     random_bins,
                     random_equal_bins,
                     digitize_uni)
from .base import Binning

import warnings


__binning_schemes__ = {
    'equidistant': equidistant_bins,
    'equal': equal_bins,
    'random': random_bins,
    'random-equal': random_equal_bins
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
    underflow, overflow : `bool`
        Whether or not to include an over- or underflow bin.

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

    def __init__(self,
                 bins,
                 scheme='equidistant',
                 pmin=0.0,
                 pmax=100.0,
                 underflow=True,
                 overflow=True):
        super(GridBinning, self).__init__(bins)
        self.pmin = pmin
        self.pmax = pmax
        self.scheme = __binning_schemes__[scheme]
        self.underflow = underflow
        self.overflow = overflow

    def fit(self, X, *args, **kwargs):
        super(GridBinning, self).fit(X)
        if len(X.shape) < 2:
            X = X.reshape(-1, 1)
        add_bins = 0
        if not self.underflow:
            add_bins += 1
        if not self.overflow:
            add_bins += 1
        if type(self.bins) == int:
            self.bins = [self.scheme(f,
                                     np.percentile(f, self.pmin),
                                     np.percentile(f, self.pmax),
                                     self.bins + add_bins)
                         for f in X.T]
        if type(self.bins[0]) == int:
            self.bins = [self.scheme(np.percentile(f, self.pmin),
                                     np.percentile(f, self.pmax),
                                     n + add_bins)
                         for f, n in zip(X.T, self.bins)]
        for i in range(len(self.bins)):
            self.bins[i][-1] += np.finfo("float64").resolution
        self.n_bins = np.product([len(b) + 1 - add_bins for b in self.bins])
        self.bmids = np.array([calc_bmids(b, self.underflow, self.overflow)
                               for b in self.bins])
        self.bdiff = np.array([calc_bdiff(b, self.underflow, self.overflow)
                               for b in self.bins])


    def digitize(self, X):
        super(GridBinning, self).digitize(X)
        if len(X.shape) < 2:
            X = X.reshape(-1, 1)
        n_dim = X.shape[1]
        B = list(product(*[range(len(self.bins[i]) + 1)
                           for i in range(n_dim)]))
        D = {b: i for i, b in enumerate(B)}
        c = np.array([digitize_uni(X[:, i], self.bins[i], self.underflow,
                                   self.overflow)
                      for i in range(n_dim)]).T
        bin_assoc = [D[tuple(k)] for k in c]
        return np.array(bin_assoc)