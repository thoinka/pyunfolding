from ..model import Unfolding
from ..utils import UnfoldingResult
import numpy as np


class SVDUnfolding:
    
    def __init__(self, binning_X, binning_y):
        self.model = Unfolding(binning_X, binning_y)
        self.is_fitted = False
    
    def g(self, X):
        if self.is_fitted:
            return self.model.binning_X.histogram(X)
        raise SyntaxError('Unfolding not yet fitted! Use `fit` method first.')

    def f(self, y):
        if self.is_fitted:
            return self.model.binning_y.histogram(y)
        raise SyntaxError('Unfolding not yet fitted! Use `fit` method first.')
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.U, self.S, self.V = np.linalg.svd(self.model.A)
        
    def predict(self, X, x0=None, sig_level=1.0):
        if self.is_fitted:
            g = self.g(X)
            R = np.eye(100)
            h = np.abs(self.U.T @ g) / (self.U.T @ np.diag(g) @ self.U).diagonal()
            R[h < sig_level,:] = 0.0
            A_pinv_reg = self.V.T @ np.linalg.pinv(np.pad(np.diag(self.S), ((0, 80), (0, 0)), 'constant')) @ R @ self.U.T
            f = A_pinv_reg @ g
            cov = A_pinv_reg @ np.diag(g) @ A_pinv_reg.T
            return UnfoldingResult(f=f,
                                   f_err=np.vstack((np.sqrt(cov.diagonal()),
                                                    np.sqrt(cov.diagonal()))),
                                   cov=cov,
                                   success=True)
        raise SyntaxError('Unfolding not yet fitted! Use `fit` method first.')