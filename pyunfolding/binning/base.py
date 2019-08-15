import numpy as np
from itertools import product


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

    def __init__(self, bins, *args, **kwargs):
        self.bins = bins
        self.fitted = False
        self.rep = 1

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
        weights : numpy array, shape=(n_samples,)
            Weights for each sample.

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

    def histogram(self, X, weights=None, return_xvals=False, *args, **kwargs):
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
        H = np.bincount(cnts[cnts >= 0], weights=weights,
                        minlength=self.n_bins)
        if return_xvals:
            xvals = [np.mean(X[cnts == b], axis=0) for b in range(self.n_bins)]
            return H, np.array(xvals)
        return H