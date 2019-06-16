import numpy as np
from ..plot import corner_plot


ONE_SIGMA = 0.682689492137085897


def cov2corr(cov):
    '''Convert covariance matrix to correlation matrix.
    '''
    std = np.sqrt(cov.diagonal())
    corr = cov / np.outer(std, std)
    return corr


def error_central(x, p=ONE_SIGMA, **kwargs):
    lower = np.percentile(x, 100.0 * (1.0 - p) / 2.0, axis=0)
    upper = np.percentile(x, 100.0 * (1.0 + p) / 2.0, axis=0)
    return lower, upper


def error_shortest(x, p=ONE_SIGMA, **kwargs):
    lower = np.zeros(x.shape[1])
    upper = np.zeros(x.shape[1])
    N = int(p * len(x))
    for i in range(x.shape[1]):
        x_sort = np.sort(x[:, i])
        x_diff = np.abs(x_sort[N:] - x_sort[:-N])
        shortest = np.argmin(x_diff)
        lower[i], upper[i] = x_sort[shortest], x_sort[shortest + N]
    return lower, upper


def error_best(x, f, p=ONE_SIGMA, **kwargs):
    N = int(p * len(x))
    best = np.argsort(f)[:N]
    return np.min(x[best, :], axis=0), np.max(x[best, :], axis=0)


def error_feldman_cousins(x, best_fit, p=ONE_SIGMA, **kwargs):
    diff = np.abs(x - best_fit)
    n_events = int(len(x) * p) + 1
    sigma_vec_best = np.zeros((2, len(best_fit)))
    for i in range(len(best_fit)):
        order = np.argsort(diff[:, i])
        select = order[:n_events]
        selected_sample = x[select, i]
        sigma_vec_best[0, i] = np.min(selected_sample, axis=0)
        sigma_vec_best[1, i] = np.max(selected_sample, axis=0)
    return sigma_vec_best


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
        self.X = X
        self.F = F
        self.n_samples, self.n_dim = X.shape
        
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
            lower, upper = value - std * 0.5, value + std * 0.5
        else:
            raise ValueError("error_method {} not supported!"\
                             .format(error_method))
        return lower, upper
    
    def plot(self, **kwargs):
        return corner_plot(self.X, **kwargs)

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