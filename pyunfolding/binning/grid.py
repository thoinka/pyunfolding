import numpy as np
from itertools import product
from sklearn.tree import DecisionTreeClassifier
from ..utils import calc_bmids, calc_bdiff
from .base import Binning

import warnings


class GridBinning(Binning):
    """Gridded Binning.

    Attributes
    ----------
    bins : {list(numpy array) or list(int)}
        Either list of bin edges or list of number of bins in each dimension.
    n_bins : int
        Number of bins.
    name : str
        Name of object.
    """
    name = "EquidistantBinning"

    def __init__(self, bins, pmin=0.0, pmax=100.0):
        super(GridBinning, self).__init__(bins)
        self.pmin = pmin
        self.pmax = pmax

    def fit(self, X, *args, **kwargs):
        super(GridBinning, self).fit(X)
        if len(X.shape) < 2:
            X = X.reshape(-1, 1)
        if type(self.bins) == int:
            self.bins = [np.linspace(
                            np.percentile(f, self.pmin),
                            np.percentile(f, self.pmax), self.bins - 1)
                         for f in X.T]
        if type(self.bins[0]) == int:
            self.bins = [np.linspace(np.percentile(f, self.pmin),
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