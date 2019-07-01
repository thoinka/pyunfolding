from ..model import Unfolding
from ..utils import UnfoldingResult
from ..base import UnfoldingBase
import numpy as np


def _ibu(A, g, f0=None, n_iterations=5):
    if f0 is None:
        f_est = [np.ones(A.shape[1]) * np.sum(g) / A.shape[1]]
    else:
        f_est = [f0]
    for _ in range(n_iterations):
        f_est_new = g @ (A * f_est[-1] / (A @ f_est[-1]).reshape(-1,1))
        f_est.append(f_est_new)
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
    
    def g(self, X):
        if self.is_fitted:
            return self.model.binning_X.histogram(X)
        raise RuntimeError('Unfolding not yet fitted! Use `fit` method first.')

    def f(self, y):
        if self.is_fitted:
            return self.model.binning_y.histogram(y)
        raise RuntimeError('Unfolding not yet fitted! Use `fit` method first.')
    
    def fit(self, X_train, y_train):
        X_train, y_train = super(BayesianUnfolding, self).fit(X_train, y_train)
        self.model.fit(X_train, y_train)
        self.n_bins_X = self.model.binning_X.n_bins
        self.n_bins_y = self.model.binning_y.n_bins
        self.is_fitted = True
        
    def predict(self, X, x0=None, n_iterations=5):
        X = super(BayesianUnfolding, self).predict(X)
        f = _ibu(self.model.A, self.g(X), f0=x0, n_iterations=n_iterations)
        return UnfoldingResult(f=f,
                               f_err=np.zeros((2,len(f))),
                               cov=np.eye(len(f)) * 1e-8,
                               success=True,
                               binning_y=self.model.binning_y)