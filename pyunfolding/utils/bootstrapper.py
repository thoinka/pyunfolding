import numpy as np
from .uncertainties import Posterior
from .unfoldingresult import UnfoldingResult


class Bootstrapper:
    '''Helps wrapping an unfolding pipeline into a bootstrapping scheme in order
    to estimate the uncertainties.

    Parameters
    ----------
    unfolding : pu.UnfoldingBase object
        The unfolding object.
    n_boostraps : int
        Number of bootstraps.

    Attributes
    ----------
    unfolding : pu.UnfoldingBase object
        The unfolding object.
    n_boostraps : int
        Number of bootstraps.
    '''

    def __init__(self, unfolding, n_bootstraps=100):
        self.unfolding = unfolding
        self.n_bootstraps = n_bootstraps

    def g(self, X, weights=None):
        '''Returns an observable vector :math:`\mathbf{g}` given a sample
        of observables `X`.

        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_observables)
            Observable sample.
        weights : numpy.array, shape=(n_samples,)
            Weights corresponding to the sample.

        Returns
        -------
        g : numpy.array, shape=(n_bins_X,)
            Observable vector
        '''
        if self.is_fitted:
            return self.unfolding.g(X, weights=weights)
        raise RuntimeError('Unfolding not yet fitted! Use `fit` method first.')

    def f(self, y, weights=None):
        '''Returns a result vector :math:`\mathbf{f}` given a sample
        of target variable values `y`.

        Parameters
        ----------
        y : numpy.array, shape=(n_samples,)
            Target variable sample.
        weights : numpy.array, shape=(n_samples,)
            Weights corresponding to the sample.

        Returns
        -------
        f : numpy.array, shape=(n_bins_X,)
             Result vector.
        '''
        if self.is_fitted:
            return self.unfolding.f(y, weights=weights)
        raise RuntimeError('Unfolding not yet fitted! Use `fit` method first.')

    def fit(self, X_train, y_train, **kwargs):
        '''Fit routine, calls the fit routine of `Bootstrapper.unfolding`

        Parameters
        ----------
        X_train : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        y_train : numpy.array, shape=(n_samples,)
            Target variable sample.
        kwargs : dict
            Additional keywords for the fit routine.
        '''
        self.unfolding.fit(X_train, y_train, **kwargs)
    
    def predict(self, X, x0=None,
                error_method='central',
                value_method='median',
                return_sample=True,
                **kwargs):
        '''Calculates the unfolding in a bootstrapping scheme.

        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        x0 : numpy.array, shape=(n_bins_y)
            Initial value for the unfolding.
        error_method : str
            Method used to estimate the uncertainty region of each bin.
        value_method : str
            Method used to estimate the best fit value for each bin.
        return_sample : bool
            Whether or not to return the full bootstrapped sample.
        '''
        fs = []
        for _ in range(self.n_bootstraps):
            bs = np.random.choice(len(X), len(X), replace=True)
            result = self.unfolding.predict(X=X[bs], x0=x0, **kwargs)
            fs.append(result.f)
        fs = np.array(fs)

        ppdf = Posterior(fs)
        lower, upper = ppdf.error(error_method)
        value = ppdf.value(value_method)

        error = np.vstack((value - lower, upper - value))
        result = dict(
            f=value,
            f_err=error,
            success=True,
            cov=np.cov(fs.T),
            binning_y=self.unfolding.model.binning_y
        )
        if return_sample:
            result.update(sample=ppdf)

        return UnfoldingResult(**result)