import numpy as np
from ..model import Unfolding
from ..utils import diff0_matrix, diff2_matrix, diff1_matrix
from ..utils import UnfoldingResult
from ..base import UnfoldingBase

from numba import jit


@jit(nopython=True)
def _analytical_solution(A, g, tau, Sigma, C_matrix):
    lenf = A.shape[1]
    iSigma = np.linalg.pinv(Sigma)
    B = np.linalg.pinv(A.T @ iSigma @ A
                       + tau * C_matrix.T @ C_matrix) @ A.T @ iSigma
    f_est = B @ g
    cov = B @ np.diag(g) @ B.T
    
    return f_est, cov


class AnalyticalUnfolding(UnfoldingBase):
    '''Analytical solution to the unfolding problem in the case of a Weighted
    Least Squares Likelihood with Tikhonov regularization.

    Parameters
    ----------
    binning_X : pyunfolding.binning.Binning object
        The binning of the observable space.
    binning_y : pyunfolding.binning.Binning object
        The binning of the target variable.
    exclude_edges : bool
        Whether or not to exclude the edges, i.e. the under- and overflow bin
        in the regularization.
    C : string or numpy.array
        Regularization matrix
    Sigma : None, string or numpy.array
        Weight matrix.
    
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
    def __init__(self, binning_X, binning_y, exclude_edges=True, C='diff2',
                 Sigma=None):
        super(AnalyticalUnfolding, self).__init__()
        self.model = Unfolding(binning_X, binning_y)
        self.Sigma = Sigma
        self.exclude_edges = exclude_edges
        self.C = C

    def fit(self, X_train, y_train, *args, **kwargs):
        '''Fit routine.

        Parameters
        ----------
        X_train : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        y_train : numpy.array, shape=(n_samples,)
            Target variable sample.
        '''
        X_train, y_train = super(AnalyticalUnfolding, self).fit(X_train,
                                                                y_train)
        self.model.fit(X_train, y_train, *args, **kwargs)
        self.is_fitted = True
        self.n_bins_y = self.model.binning_y.n_bins
        self.n_bins_X = self.model.binning_X.n_bins
        if self.C == "diff1":
            c_gen = diff1_matrix
        elif self.C == 'diff2':
            c_gen = diff2_matrix
        elif self.C == 'diff0':
            self.C = diff0_matrix

        if self.exclude_edges:
            self.C = np.zeros((self.n_bins_y, self.n_bins_y))
            self.C[1:-1, 1:-1] = c_gen(self.n_bins_y - 2)
        else:
            self.C = c_gen(self.n_bins_y)
        
    def predict(self, X, tau=0.0, **kwargs):
        '''Calculates an estimate for the unfolding by maximizing the likelihood
        function (or minimizing the log-likelihood).

        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        tau : float
            Regularization strength.
        '''
        X = super(AnalyticalUnfolding, self).predict(X)
        if self.is_fitted:
            g = self.g(X).astype('float64')
            if type(self.Sigma) is str:
                if self.Sigma == 'poisson':
                    self.Sigma = np.diag(g)
            if self.Sigma is None:
                self.Sigma = np.eye(self.model.A.shape[0]).astype('float64')
            f_est, cov = _analytical_solution(self.model.A,
                                              g,
                                              tau,
                                              self.Sigma,
                                              self.C.astype('float64'))
            f_err = np.vstack((np.sqrt(cov.diagonal()),
                               np.sqrt(cov.diagonal())))
            result = UnfoldingResult(f=f_est, f_err=f_err, cov=cov,
                                     success=True)
            result.update(binning_y=self.model.binning_y)
            return result
        raise RuntimeError('Unfolding not yet fitted! Use `fit` method first.')