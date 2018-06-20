import numpy as np
from itertools import product
from sklearn.tree import DecisionTreeClassifier

import warnings


class Binning(object):
    """Binning base class, outlining what is required of a
    descretization class. Never use this directly, instead use derived
    class objects. The least what a discretization class object should be
    capable of is:

    Attributes
    ----------
    bins : {list(int), list(arrays)}
        Depends on method.
    fitted : bool
        Whether or not the binning has been fitted.
    n_dim : int
        Number of dimensions of the binned space.
    name : str
        Name of the object.
    """
    name = "Binning"

    def __init__(self, bins, *args, **kwargs):
        self.bins = bins
        self.fitted = False

    def fit(self, X, *args, **kwargs):
        """Fit method. Uses the samples X to fit the binning routine. 

        Parameters
        ----------
        X : numpy array, shape=(n_samples, n_dim)
            Training samples.
        """
        self.fitted = True
        if len(X.shape) < 2:
            self.n_dim = 1
        else:
            self.n_dim = X.shape[1]

    def digitize(self, X, *args, **kwargs):
        """Digitze method. Assigns the associated bin to each of the samples in
        X.

        Parameters
        ----------
        X : numpy array, shape=(n_samples, n_dim)
            Samples to digitize.

        Returns
        -------
        idx : numpy array, shape=(n_samples,)
            Bin number assigned to each of the samples.

        Raises
        ------
        RuntimeError
        """
        if not self.fitted:
            raise RuntimeError("Binning Object not fitted: Call fit first!")

    def histogram(self, X, return_xvals=False, *args, **kwargs):
        """Histogram method. Calculates a count vector for the samples X.

        Parameters
        ----------
        X : numpy array, shape=(n_samples, n_dim)
            Samples to histogram.
        return_xvals : bool, optional, default=False
            Returns weighted midpoint for each of the bins.

        Returns
        -------
        H : numpy array, shape=(n_bins,)
            The histogrammed vector

        xvals : numpy array, shape=(n_bins,)
                The weighted midpoints for each bin. Only returned when
                return_xvals is True.

        Raises
        ------
        RuntimeError
        """
        if not self.fitted:
            raise RuntimeError("Binning Object not fitted: Call fit first!")
        if len(X.shape) < 2:
            X = X.reshape(-1, 1)
        cnts = self.digitize(X)
        H = np.bincount(cnts, minlength=self.n_bins)
        if return_xvals:
            xvals = [np.mean(X[cnts == b], axis=0) for b in range(self.n_bins)]
            return H, np.array(xvals)
        return H


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

    def __init__(self, bins):
        super(GridBinning, self).__init__(bins)

    def fit(self, X):
        super(GridBinning, self).fit(X)
        if len(X.shape) < 2:
            X = X.reshape(-1, 1)
        if type(self.bins) == int:
            self.bins = [np.linspace(np.min(f), np.max(f), self.bins - 1)
                         for f in X.T]
        if type(self.bins[0]) == int:
            self.bins = [np.linspace(np.min(f), np.max(f), n)
                         for f, n in zip(X.T, self.bins)]
        for i in range(len(self.bins)):
            self.bins[i][-1] += np.finfo("float64").resolution
        self.n_bins = np.product([len(b) + 1 for b in self.bins])

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


class TreeBinning(Binning):
    """Tree based binning.

    Attributes
    ----------
    dtc : sklearn.tree.DecisionTreeClassifier
        Decision tree trained to bin the attribute space.
    leaves : np.array(n_bins)
        Indices of the leaves
    n_bins : int
        Number of bins.
    """
    name = "TreeBinning"

    def __init__(self, bins, **kwargs):
        super(TreeBinning, self).__init__(bins)
        self.dtc = DecisionTreeClassifier(max_leaf_nodes=self.bins, **kwargs)

    def fit(self, X, y=None):
        super(TreeBinning, self).fit(X)
        if y is None:
            y = np.random.rand(len(X)) > 0.5
        if len(X.shape) < 2:
            X = X.reshape(-1, 1)
        self.dtc.fit(X, y)
        self.leaves = np.unique(self.dtc.apply(X))
        self.n_bins = len(self.leaves)

    def digitize(self, X):
        super(TreeBinning, self).digitize(X)
        if len(X.shape) < 2:
            X = X.reshape(-1, 1)
        idx = self.dtc.apply(X)
        p = np.zeros(len(idx), dtype=int)
        for i, l in enumerate(self.leaves):
            p[idx == l] = i
        return p
