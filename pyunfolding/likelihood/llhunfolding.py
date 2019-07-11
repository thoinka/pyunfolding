from . import llh
from ..model import Unfolding

import numpy as np


class LLHUnfolding:
    '''Unfolding based on a maximum likelihood fit. Multiple likelihoods can
    be defined using several predefined likelihood terms in the sub-module
    `pyunfolding.likelihood.llh`.

    Parameters
    ----------
    binning_X : pyunfolding.binning.Binning object
        The binning of the observable space.
    binning_y : pyunfolding.binning.Binning object
        The binning of the target variable.
    likelihood : list(pyunfolding.likelihood.llh.LikelihoodTerm)
        List of LikelihoodTerms that describe the likelihood.

    Attributes
    ----------
    model : pyunfolding.model.Unfolding object
        Unfolding model.
    llh : pyunfolding.likelihood.llh.Likelihood object
        The likelihood.
    is_fitted : bool
        Whether the object has already been fitted.
    n_bins_X : int
        Number of bins in the observable space.
    n_bins_y : int
        Number of bins in the target space.
    '''
    def __init__(self, binning_X, binning_y, likelihood=None):
        self.model = Unfolding(binning_X, binning_y)
        
        if likelihood is None:
            self._llh = [llh.LeastSquares()]
        else:
            self._llh = likelihood
        self.is_fitted = False
    
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
            return self.model.binning_X.histogram(X, weights=weights)
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
            return self.model.binning_y.histogram(y, weights=weights)
        raise RuntimeError('Unfolding not yet fitted! Use `fit` method first.')

    def fit(self, X_train, y_train, *args, **kwargs):
        '''Fit routine.

        Parameters
        ----------
        X_train : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        y_train : numpy.array, shape=(n_samples,)
            Target variable sample.
        '''
        self.model.fit(X_train, y_train, *args, **kwargs)
        self.llh = llh.Likelihood(self.model, self._llh)
        self.is_fitted = True
        self.n_bins_y = self.model.binning_y.n_bins
        self.n_bins_X = self.model.binning_X.n_bins
        
    def predict(self, X, x0=None, solver_method='mcmc', **kwargs):
        '''Calculates an estimate for the unfolding by maximizing the likelihood function (or minimizing the log-likelihood).

        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        x0 : numpy.array, shape=(n_bins_y)
            Initial value for the unfolding.
        solver_method : str
            Method for maxmizing the likelihood and estimating error contours.
            Supported values:
            * `minimizer`: Uses `scipy.optimize.minimize` to minimize the
                           negative log-likelihood. See the documentation of
                           `scipy.optimize.minimize` for further information
                           on possible parameters.
            * `newton`: Uses the newtons method to minimize the negative
                        log-likelihood.
            * `mcmc`: Uses a Markov Chain Monte Carlo approach to estimate the
                      minimum and error contours of the negative
                      log-likelihood.
        kwargs : dict
            Keywords for the solver.
        '''
        if self.is_fitted:
            if x0 is None:
                x0 = len(X) * np.ones(self.n_bins_y) / self.n_bins_y
            result = self.llh.solve(x0, X, solver_method=solver_method,
                                    **kwargs)
            result.update(binning_y=self.model.binning_y)
            return result
        raise RuntimeError('Unfolding not yet fitted! Use `fit` method first.')