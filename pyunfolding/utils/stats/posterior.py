import numpy as np
from ...plot import corner_plot
from .intervals import (ONE_SIGMA,
                        error_central,
                        error_best,
                        error_feldman_cousins,
                        error_shortest)


class Posterior:
    '''Helper class to calculate values and uncertainty regions from posterior
    samples.

    Parameters
    ----------
    X : numpy.array, shape=(n_samples, n_dim)
        Posterior sample.

    F : numpy.array, shape=(n_samples,)
        Function values of posterior sample if available.
    
    Attributes
    ----------
    X : numpy.array, shape=(n_samples, n_dim)
        Posterior sample.

    F : numpy.array, shape=(n_samples,)
        Function values of posterior sample if available.
    '''
    def __init__(self, X, F=None):
        if F is not None:
            assert len(X) == len(F)
        self.len = len(X)
        self.X = X
        self.F = F
        self.n_samples, self.n_dim = X.shape

    def __len__(self):
        return self.len
        
    def error(self, method='central', best_fit=None, p=ONE_SIGMA):
        '''Calculates error from given sample.

        Parameters
        ----------
        method : str
            Method used to calculate uncertainty regions. Either
            
            * `central`: Central interval, i.e. (1-p) / 2 and (1+p) / 2
                         quantiles.
            * `feldman-cousins`: Feldman-cousins interval. _Note_: Requires
                                 `best_fit`
            * `shortest`: Shortest interval containing p.
            * `best`: Interval of best function values _Note_: Requires
                      function values.
            * `std`: Simply the standard deviation.

        best_fit : numpy.array, shape=(n_dim,)
            Best fit in the sample. Required by method `feldman-cousins`.

        p : float, default=0.68...
            Quantile covered by ncertainty interval

        Returns
        -------
        lower, upper : numpy.array, shape=(2, n_dim)
            lower and upper end of the uncertainty region for every dimension.
        '''
        if method == "central":
            lower, upper = error_central(self.X)
        elif method == "feldman-cousins":
            if best_fit is None:
                raise ValueError('Method {} requires best_fit.'.format(method))
            lower, upper = error_feldman_cousins(self.X, best_fit)
        elif method == "shortest":
            lower, upper = error_shortest(self.X)
        elif method == "best":
            if self.F is None:
                raise ValueError('Method {} requires function values.'.format(method))
            lower, upper = error_best(self.X, self.F)
        elif method == "std":
            std = np.std(self.X, axis=0)
            value = np.mean(self.X, axis=0)
            lower, upper = value - std, value + std
        else:
            raise ValueError("error_method {} not supported!"\
                             .format(error_method))
        return lower, upper
    
    def plot(self, n_bins=20, hist2d_kw=dict(), kde=False, best_fit=None,
                scatter=False, color_correlation=True, color='#aaaaaa',
                colormap='coolwarm', **kwargs):
        '''Plot all marginal and two-dimensional common distributions of the
        sample contained in this object.

        Parameters
        ----------
        n_bins : int, optional (default=20)
            Number of bins for the two-dimensional histogrmas

        hist2d_kw : dict, optional
            Keywords for the `pyplot.hist2d` method.

        kde : bool, optional (default=False)
            Whether to smooth out the two-dimensional marginal distributions
            using a kernel density estimations (this is usually very, very
            slow).

        best_fit : numpy.array, optional (default=None)
            Best fit, will be contained in the plot with a dashed line, if
            provided.

        scatter : bool, optional (default=False)
            Whether or not to add scatter points.

        color_correlation : bool, optional (default=False)
            Whether to encode the correlation of each two-dimensional marginal
            distribution in the color of the density map

        color : str, optional (default='#aaaaaa')
            Color of the one-dimensional marginal distributions (or the rest
            of the plot in case `color_correlation` is `False`)

        colormap : str. optional (default='coolwarm')
            Colormap used for the correlation coloring.

        Returns
        -------
        matplotlib.pyplot.axis object
        '''
        return corner_plot(self.X, n_bins, hist2d_kw, kde, best_fit,
                           scatter, color_correlation, color,
                           colormap, **kwargs)

    def value(self, method='median'):
        '''Calculates value from given sample.

        Parameters
        ----------
        method : str
            Method used to calculate values. Either
            
            * `median`: 50% quantile
            * `best`: value corresponding to the best function value. __Note__:
                      Requires function values.
            * `mean`: Mean of sample.
        '''
        if method == "median":
            value = np.median(self.X, axis=0)
        elif method == "best":
            if self.F is None:
                raise ValueError('Method {} requires function values.'.format(method))
            value = self.X[np.argmin(self.F)]
        elif method == "mean":
            value = np.mean(self.X, axis=0)
        else:
            raise ValueError("value_method {} not supported!"\
                             .format(value_method))
        return value