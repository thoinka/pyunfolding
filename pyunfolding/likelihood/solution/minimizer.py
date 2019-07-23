import numpy as np
from scipy.optimize import minimize
from ...utils import UnfoldingResult, minimization
from .base import SolutionBase


__gd_minimizer__ = [
    'adam',
    'adadelta',
    'momentum',
    'rmsprop'
]

# List of available scipy optimizers and whether or not they
# support gradient or hessian information.
__scipy_minimizer_opt__ = {
    'nelder-mead':  {'grad': False, 'hess': False},
    'powell':       {'grad': False, 'hess': False},
    'cg':           {'grad': True,  'hess': False},
    'bfgs':         {'grad': True,  'hess': False},
    'newton-cg':    {'grad': True,  'hess': True },
    'dogleg':       {'grad': True,  'hess': True },
    'trust-ncg':    {'grad': True,  'hess': True },
    'trust-krylov': {'grad': True,  'hess': True },
    'trust-exact':  {'grad': True,  'hess': True },
    'l-bfgs-g':     {'grad': True,  'hess': False},
    'tnc':          {'grad': True,  'hess': False},
    'cobyla':       {'grad': False, 'hess': False},
    'slsqp':        {'grad': True,  'hess': False},
    'trust-constr': {'grad': True,  'hess': True },
}


class Minimizer(SolutionBase):

    def __init__(self, likelihood):
        self.likelihood = likelihood

    def solve(self, x0, X, method=None, *args, **kwargs):
        g = self.likelihood.model.predict_g(X)

        def F(p): return self.likelihood(p, g)
        if method in __gd_minimizer__:
            def G(p): return self.likelihood.grad(p, g)
            if method == 'adam':
                _minimizer = minimization.adam_minimizer
            elif method == 'momentum':
                _minimizer = minimization.momentum_minimizer
            elif method == 'rmsprop':
                _minimizer = minimization.rmsprop_minimizer
            elif method == 'adadelta':
                _minimizer = minimization.adadelta_minimizer
            result = _minimizer(F, G, x0, **kwargs)
            Hinv = np.linalg.pinv(self.likelihood.hess(result.x, g))
            error = np.sqrt(Hinv.diagonal())
        else:
            params = {}
            if method is None:
                method = 'bfgs'
            if __scipy_minimizer_opt__[method.lower()]['grad']:
                params.update(jac=lambda p: self.likelihood.grad(p, g))
            if __scipy_minimizer_opt__[method.lower()]['hess']:
                params.update(jac=lambda p: self.likelihood.hess(p, g))
            result = minimize(F, x0=x0, **kwargs)
            try:
                Hinv = np.linalg.pinv(self.likelihood.hess(result.x, g))
                error = np.sqrt(Hinv.diagonal())
            except:
                print('Analytical Hessian unavailable, will use numerical Hessian instead.')
                if hasattr(result, "hess_inv"):
                    Hinv = result.hess_inv
                    error = np.sqrt(Hinv.diagonal())
                elif hasattr(result, "hess"):
                    Hinv = np.linalg.pinv(result.hess)
                    error = np.sqrt(Hinv.diagonal())
                else:
                    print('No estimate for Hessian available.')
                    error = np.nan
        return UnfoldingResult(f=result.x,
                               f_err=np.vstack((0.5 * error, 0.5 * error)),
                               success=result.success,
                               fun=result.fun,
                               jac=result.jac,
                               cov=Hinv)