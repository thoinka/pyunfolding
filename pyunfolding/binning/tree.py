import numpy as np
from itertools import product
from sklearn.tree import DecisionTreeClassifier
from .base import Binning


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

    def fit(self, X, y=None, *args, **kwargs):
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