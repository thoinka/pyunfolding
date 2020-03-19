import numpy as np
from itertools import product
from sklearn.tree import DecisionTreeClassifier
from ..utils import binning
from .base import Binning

import warnings


class GridBinning(Binning):
    """Multidimensional Binning on regular grids. Bin edges can either be
    chosen manually or using a specific scheme.

    Parameters
    ----------
    bins : either list of `numpy array` or list of `int`
        Either list of bin edges or list of number of bins in each dimension.

    scheme : `str` or callable
        Specifies the method that picks the positions of the bin edges. The
        following options are available:
            * ``equidistant`` will pick equidistant bin edges starting from
              `pmin` and ending in `pmax`
            * ``equal`` will pick bins that contain an equal number of samples
              in it.
            * ``random`` will pick completely random bin edges in between
              `pmin` and `pmax`
            * ``random-equal`` will pick random bin edges according quantiles.
            * callable : Any function with signature
              `f(X, Xmin, Xmax, n_bins, **kwargs) -> 1dim array` can be used
    
    pmin, pmax : `float` (default=0.0, 1.0)
        Quantile of the lowest and highest bin edge (in percent!)
    
    underflow, overflow : `bool` (default=True, True)
        Whether or not to include an over- or underflow bin. As SVDUnfolding
        and BayesianUnfolding have no way to account for the discontintuity
        caused by under- and overflow bins in the spectrum, it is recommended
        to set to `False` for the target variable.
    
    random_seed : `int` or None
        Random seed. If None, numpy will choose its own random seed.

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
    __binning_schemes__ = {
        'equidistant':  binning.equidistant_bins,
        'equal':        binning.equal_bins,
        'random':       binning.random_bins,
        'random-equal': binning.random_equal_bins
    }

    def __init__(self,
                 bins,
                 scheme='equidistant',
                 pmin=0.0,
                 pmax=100.0,
                 underflow=True,
                 overflow=True,
                 random_seed=None):
        super(GridBinning, self).__init__(bins)
        if type(bins) == int:
            self.bins = [bins]
        if not hasattr(pmin, '__getitem__'):
            self.pmin = pmin * np.ones(len(self.bins))
        else:
            self.pmin = pmin

        if not hasattr(pmax, '__getitem__'):
            self.pmax = pmax * np.ones(len(self.bins))
        else:
            self.pmax = pmax

        if not hasattr(scheme, '__getitem__') or type(scheme) is str:
            scheme = [scheme] * len(self.bins)
        self.scheme = []
        for s in scheme:
            if callable(s):
                self.scheme.append(s)
            else:
                try:
                    self.scheme = self.__binning_schemes__[s]
                except KeyError:
                    raise ValueError('Method `{}` not supported! Available Methods are {}'.format(s, list(self.__binning_schemes__.keys())))
        if not hasattr(underflow, '__getitem__'):
            self.underflow = [underflow] * len(self.bins)
        if not hasattr(overflow, '__getitem__'):
            self.overflow = [overflow] * len(self.bins)

        self.RandomState = np.random.RandomState(random_seed)

    def _fit1dim(self, i, x):
        X_min, X_max = np.percentile(x, [self.pmin[i], self.pmax[i]])
        return self.scheme(x, X_min, X_max,
                           self.bins[i] + self._add_bins[i],
                           rnd=self.RandomState)

    def fit(self, X, *args, **kwargs):
        super(GridBinning, self).fit(X)
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        assert X.shape[1] == len(self.bins)

        bins_fitted = []

        self._add_bins = np.zeros(len(self.bins), dtype=int) 
        for i in range(len(self.bins)):
            if not self.underflow[i]:
                self._add_bins[i] += 1
            if not self.overflow[i]:
                self._add_bins[i] += 1
            if type(self.bins[i]) == int:
                bins_fitted.append(self._fit1dim(i, X[:,i]))
            else:
                bins_fitted.append(self.bins[i])
        for i in range(len(self.bins)):
            bins_fitted[i][-1] += np.finfo("float64").resolution
        self.n_bins = np.product([len(bins_fitted[i]) + 1 - self._add_bins[i]
                                  for i in range(len(bins_fitted))])
        self.bins = bins_fitted
        self.bmids = np.array([binning.calc_bmids(self.bins[i],
                                                  self.underflow[i],
                                                  self.overflow[i])
                               for i in range(len(self.bins))])
        self.bdiff = np.array([binning.calc_bdiff(self.bins[i],
                                                  self.underflow[i],
                                                  self.overflow[i])
                               for i in range(len(self.bins))])

    def digitize(self, X):
        super(GridBinning, self).digitize(X)
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_dim = X.shape[1]
        # This is kind of difficult to understand I see. This creates a list
        # Of all possible combinations of numbers from 0 to n_bins for
        # each dimension as a tuple.
        # So for n_bins = 3 and n_dim = 2, the output is [(0,0), (0,1), (0,2),
        # (1,0), (1,1), ...]
        B = list(product(*[range(len(self.bins[i]) + 1)
                           for i in range(n_dim)]))
        # This then creates a dictionary that assigns every combination of
        # one-dimensional digitizations for every bin to one integer,
        # effectively flattening digitze.
        D = {b: i for i, b in enumerate(B)}
        # Then each dimension is digitized seperately
        c = np.array([binning.digitize_uni(X[:, i],
                                           self.bins[i],
                                           self.underflow[i],
                                           self.overflow[i])
                      for i in range(n_dim)]).T
        # Finally, the bin association is determined using the dictionary from
        # before.
        # In case over- and underflow binning is disabled, samples can get
        # "lost". This is handled by assigning -1 to their digitzation.
        # This has to be handled in some way.
        bin_assoc = -np.ones(len(X), dtype=int)
        inrange = (c >= 0).all(axis=-1)
        bin_assoc[inrange] = [D[tuple(k)] for k in c if (k >= 0).all()]

        return np.array(bin_assoc)