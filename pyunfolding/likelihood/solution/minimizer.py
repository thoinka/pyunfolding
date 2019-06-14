import numpy as np
from scipy.optimize import minimize
from ...utils import UnfoldingResult
from .base import SolutionBase


class Minimizer(SolutionBase):

    def __init__(self, likelihood):
        self.likelihood = likelihood

    def solve(self, x0, X, method=None,
              gradient=False, hessian=False, *args, **kwargs):
        self.gradient = gradient
        self.hessian = hessian
        g = self.likelihood.model.predict_g(X)

        def F(p): return self.likelihood(p, g)
        params = {}
        if self.gradient:
            params.update(jac=lambda p: self.likelihood.grad(p, g))
            if method is not None:
                params.update(method=method)
        if self.hessian:
            params.update(hess=lambda p: self.likelihood.hess(p, g))
            if method is None:
                params.update(method="trust-ncg")
            else:
                params.update(method=method)
        result = minimize(F, x0=x0, **kwargs)
        try:
            error = np.sqrt(np.linalg.pinv(
                self.likelihood.hess(result.x, g)).diagonal())
        except:
            print('Analytical Hessian unavailable, will use numerical Hessian instead.')
            if hasattr(result, "hess_inv"):
                error = np.sqrt(np.abs(result.hess_inv)).diagonal()
            elif hasattr(result, "hess"):
                error = np.sqrt(np.linalg.inv(result.hess).diagonal())
            else:
                print('No estimate for Hessian available.')
                error = np.nan
        return UnfoldingResult(f=result.x,
                               f_err=np.vstack((0.5 * error, 0.5 * error)),
                               success=result.success,
                               fun=result.fun,
                               jac=result.jac)