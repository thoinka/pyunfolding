import numpy as np
from scipy.optimize import minimize
from ...utils import UnfoldingResult, minimization as mini
from .base import SolutionBase
from warnings import warn
from ...exceptions import FailedMinimizationWarning

# List of available scipy optimizers and whether or not they
# support gradient or hessian information.
__scipy_minimizer__ = {
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

# List of available gradient descent optimizers from this module
__gd_minimizer__ = {
    'adam':     {'call': mini.adam_minimizer,     'grad': True, 'hess': False},
    'momentum': {'call': mini.momentum_minimizer, 'grad': True, 'hess': False},
    'rmsprop':  {'call': mini.rmsprop_minimizer,  'grad': True, 'hess': False},
    'adadelta': {'call': mini.adadelta_minimizer, 'grad': True, 'hess': False},
    'newton':   {'call': mini.newton_minimizer,   'grad': True, 'hess': True},
}


class Minimizer(SolutionBase):

    def __init__(self, likelihood):
        self.likelihood = likelihood

    def solve(self, x0, X, method=None, *args, **kwargs):
        g = self.likelihood.model.predict_g(X)

        def F(p): return self.likelihood(p, g)
        def G(p): return self.likelihood.grad(p, g)
        def H(p): return self.likelihood.hess(p, g)

        is_gd = method.lower() in __gd_minimizer__.keys()
        is_scipy = method.lower() in __scipy_minimizer__.keys()

        if is_gd:  
            _minimizer = __gd_minimizer__[method.lower()]['call']
            params = {}
            if __gd_minimizer__[method.lower()]['grad']:
                params['grad'] = G
            if __gd_minimizer__[method.lower()]['hess']:
                params['hess'] = H
            result = _minimizer(F, x0=x0, **params, **kwargs)
            Hinv = np.linalg.pinv(H(result.x))
            error = np.sqrt(Hinv.diagonal())

        elif is_scipy:
            params = {}
            if method is None:
                method = 'bfgs'
            if __scipy_minimizer__[method.lower()]['grad']:
                params.update(jac=G)
            if __scipy_minimizer__[method.lower()]['hess']:
                params.update(hess=H)
            result = minimize(F, x0=x0, method=method, **params, **kwargs)
            try:
                Hinv = np.linalg.pinv(H(result.x))
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

        else:
            raise NotImplementedError(
                      'Method {} is not available'.format(method))
        try:
            jac = result.jac
        except:
            jac = G(result.x)
        if not result.success:
            warn(FailedMinimizationWarning('Minimization not successful.'))
        return UnfoldingResult(f=result.x,
                               f_err=np.vstack((error, error)),
                               success=result.success,
                               fun=result.fun,
                               jac=jac,
                               cov=Hinv)