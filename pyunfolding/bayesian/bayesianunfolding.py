from ..model import Unfolding
from ..utils import UnfoldingResult
from ..base import UnfoldingBase
from ..utils import num_gradient
import numpy as np


def _ibu(A, g, f0=None, n_iterations=5, alpha=1.0):
    if f0 is None:
        f_est = [np.ones(A.shape[1]) * np.sum(g) / A.shape[1]]
    else:
        f_est = [f0]
    for _ in range(n_iterations):
        f_est_new = g @ (A * f_est[-1] / (A @ f_est[-1]).reshape(-1,1))
        df = f_est_new - f_est[-1]
        f_est.append(f_est[-1] + alpha * df)
    return np.array(f_est)[-1]


class BayesianUnfolding(UnfoldingBase):
    '''Iterative Bayesian Unfolding, i.e. estimating :math:`\mathbf{f}` by
    :math:\hat f(j) = \sum_i B(j|i) g(i)`. To shake of the inherent bias of the
    matrix B, it is estimated iteratively using the last estimation of f for
    reweighting.

    Parameters
    ----------
    binning_X : pyunfolding.binning.Binning object
        The binning of the observable space.
    binning_y : pyunfolding.binning.Binning object
        The binning of the target variable.

    Attributes
    ----------
    model : pyunfolding.model.Unfolding object
        Unfolding model.
    is_fitted : bool
        Whether or not the unfolding has been fitted.
    '''
    def __init__(self, binning_X, binning_y):
        super(BayesianUnfolding, self).__init__()
        self.model = Unfolding(binning_X, binning_y)
    
    def fit(self, X_train, y_train):
        '''Fit routine.

        Parameters
        ----------
        X_train : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        y_train : numpy.array, shape=(n_samples,)
            Target variable sample.
        '''
        X_train, y_train = super(BayesianUnfolding, self).fit(X_train, y_train)
        self.model.fit(X_train, y_train)
        self.n_bins_X = self.model.binning_X.n_bins
        self.n_bins_y = self.model.binning_y.n_bins
        self.is_fitted = True
        
    def predict(self, X, x0=None, n_iterations=5, alpha=1.0, eps=1e-3):
        '''Calculates an estimate for the unfolding.
        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        x0 : numpy.array, shape=(n_bins_y)
            Initial value for the unfolding.
        n_iterations : int
            Number of iterations.
        alpha : float
            Step size, alpha=1.0 means unaltered iterative bayesian unfolding.
        eps : float
            Epsilon used for estimating uncertainties.
        '''
        X = super(BayesianUnfolding, self).predict(X)
        g = self.g(X)
        ibu_func = lambda g_: _ibu(self.model.A, g_,
                                   f0=x0, n_iterations=n_iterations,
                                   alpha=alpha)
        f = ibu_func(g)
        B = num_gradient(ibu_func, g, eps)
        cov = B.T @ np.diag(g) @ B
        return UnfoldingResult(f=f,
                               f_err=np.vstack((np.sqrt(cov.diagonal()),
                                                np.sqrt(cov.diagonal()))),
                               cov=cov,
                               success=True,
                               binning_y=self.model.binning_y)