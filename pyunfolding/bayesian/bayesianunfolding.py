from ..model import Unfolding
from ..utils import UnfoldingResult
import numpy as np


def _ibu(A, g, f0=None, n_iterations=5):
    if f0 is None:
        f_est = [np.ones(A.shape[1]) * np.sum(g) / A.shape[1]]
    else:
        f_est = [f0]
    for _ in range(n_iterations):
        B = (A * f_est[-1] / (A @ f_est[-1]).reshape(-1,1))
        f_est_new = g @ B
        f_est.append(f_est_new)
    return np.array(f_est)[-1]


class BayesianUnfolding:
    
    def __init__(self, binning_X, binning_y):
        self.binning_X = binning_X
        self.binning_y = binning_y
        self.is_fitted = False
    
    def g(self, X):
        if self.is_fitted:
            return self.binning_X.histogram(X)
        raise SyntaxError('Unfolding not yet fitted! Use `fit` method first.')

    def f(self, y):
        if self.is_fitted:
            return self.binning_y.histogram(y)
        raise SyntaxError('Unfolding not yet fitted! Use `fit` method first.')
    
    def fit(self, X_train, y_train):
        self.binning_X.fit(X_train)
        self.binning_y.fit(y_train)
        self.model = Unfolding(self.binning_X, self.binning_y)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
    def predict(self, X, x0=None, n_iterations=5):
        if self.is_fitted:
        	f = _ibu(self.model.A, self.g(X), f0=x0, n_iterations=n_iterations)
        	return UnfoldingResult(f=f,
        		                   f_err=np.nan,
        		                   success=True)
        raise SyntaxError('Unfolding not yet fitted! Use `fit` method first.')