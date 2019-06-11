from . import llh
from ..model import Unfolding


class LLHUnfolding:
    
    def __init__(self, binning_X, binning_y, likelihood=None):
        self.binning_X = binning_X
        self.binning_y = binning_y
        
        if likelihood is None:
            self._llh = [llh.LeastSquares()]
        else:
            self._llh = likelihood
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
        self.llh = llh.Likelihood(self.model, self._llh)
        self.is_fitted = True
        
    def predict(self, X, x0=None, solver_method='mcmc', **kwargs):
        if self.is_fitted:
            return self.llh.solve(x0, X, solver_method=solver_method, **kwargs)
        raise SyntaxError('Unfolding not yet fitted! Use `fit` method first.')