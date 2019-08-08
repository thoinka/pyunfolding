from . import llh
from ..model import Unfolding
from ..base import UnfoldingBase

import numpy as np


class LLHUnfolding(UnfoldingBase):
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
        super(LLHUnfolding, self).__init__()
        self.model = Unfolding(binning_X, binning_y)
        
        if likelihood is None:
            self._llh = [llh.LeastSquares()]
        else:
            self._llh = likelihood
    
    def fit(self, X_train, y_train, *args, **kwargs):
        '''Fit routine.

        Parameters
        ----------
        X_train : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        y_train : numpy.array, shape=(n_samples,)
            Target variable sample.
        '''
        X_train, y_train = super(LLHUnfolding, self).fit(X_train, y_train)
        self.model.fit(X_train, y_train, *args, **kwargs)
        self.llh = llh.Likelihood(self.model, self._llh)
        self.is_fitted = True
        self.n_bins_y = self.model.binning_y.n_bins
        self.n_bins_X = self.model.binning_X.n_bins
    
    def predict(self,
                X,
                x0=None,
                minimizer=True,
                method='trust-exact',
                minimizer_options={},
                mcmc=False,
                mcmc_options={}):
        '''Calculates an estimate for the unfolding by maximizing the likelihood function (or minimizing the log-likelihood).

        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        x0 : None or numpy.array, shape=(n_bins_y)
            Initial value for the unfolding.
        minimizer : bool, default=True
            Whether to use a minimizer
        method : str
            Which minimization algorithm to employ. Options are:
            * `nelder-mead (from `scipy.optimize.minimize`)
            * `powell` (from `scipy.optimize.minimize`) 
            * `cg` (from `scipy.optimize.minimize`) 
            * `bfgs` (from `scipy.optimize.minimize`) 
            * `newton-cg` (from `scipy.optimize.minimize`) 
            * `dogleg` (from `scipy.optimize.minimize`) 
            * `trust-ncg` (from `scipy.optimize.minimize`) 
            * `trust-krylov` (from `scipy.optimize.minimize`) 
            * `trust-exact` (from `scipy.optimize.minimize`) 
            * `l-bfgs-g` (from `scipy.optimize.minimize`) 
            * `tnc` (from `scipy.optimize.minimize`) 
            * `cobyla` (from `scipy.optimize.minimize`) 
            * `slsqp` (from `scipy.optimize.minimize`) 
            * `trust-const` (from `scipy.optimize.minimize`) 
            * `adam` (from `pyunfolding.utils.minimization`)
            * `adadelta` (from `pyunfolding.utils.minimization`)
            * `momentum` (from `pyunfolding.utils.minimization`)
            * `rmsprop` (from `pyunfolding.utils.minimization`)
        minimizer_options : dict
            additional options for the minimizer, see scipy.optimize.minimize
            for more information.
        mcmc : bool, default=False
            Whether to use a MCMC to estimate uncertainties.
        mcmc_options : dict
            additional options for the MCMC, see
            pyunfolding.likelihood.solution.mcmc` for more information
        '''
        X = super(LLHUnfolding, self).predict(X)
        if self.is_fitted:
            if x0 is None:
                x0 = len(X) * np.ones(self.n_bins_y) / self.n_bins_y

            step_size_init = None

            if minimizer:
                result = self.llh.solve(x0, X, solver_method='minimizer',
                                        method=method,
                                        **minimizer_options)
                smear = np.random.multivariate_normal(np.zeros(self.n_bins_y),
                                                      result.cov)
                x0 = result.f + smear
                step_size_init = np.mean(np.sqrt(result.cov.diagonal())) / 4.0
            if mcmc:
                try:
                    step_size_init = mcmc_options['step_size_init']
                except:
                    pass
                result = self.llh.solve(x0, X, solver_method='mcmc',
                                        step_size_init=step_size_init,
                                        **mcmc_options)
            result.update(binning_y=self.model.binning_y)
            return result
        raise RuntimeError('Unfolding not yet fitted! Use `fit` method first.')