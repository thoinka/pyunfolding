import numpy as np
from scipy.optimize import minimize
from collections import namedtuple
from tqdm import tqdm


class SolverResult(dict):
	def __init__(self, f, f_err, success, *args, **kwargs):
		self["f"] = f
		self["f_err"] = f_err
		self["success"] = success
		for key, val in kwargs:
			self[key] = val

	def __getattr__(self, key):
		return self[key]

class Solver:
    def __init__(self, likelihood, *args, **kwargs):
        self.likelihood = likelihood

    def solve(self, *args, **kwargs):
        raise NotImplementedError("Solve routine needs to be implemented.")


class Minimizer(Solver):
    name = "Minimizer"

    def __init__(self, likelihood, gradient=False, hessian=False):
        self.likelihood = likelihood
        self.gradient = gradient
        self.hessian = hessian

    def solve(self, x0, X, method=None, *args, **kwargs):
        g = self.likelihood.model.binning_X.histogram(X)

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
        if hasattr(result, "hess_inv"):
        	error = np.sqrt(np.abs(result.hess_inv)).diagonal()
        elif hasattr(result, "hess"):
        	error = 1.0 / np.sqrt(np.abs(result.hess_inv)).diagonal()
        return SolverResult(f=result.x, f_err=error, success=result.success)


class MCMCSolver(Solver):
	name = "Solver"

	def _steps(self, x0, g, scale, n_iter, verbose=False):
		x = [x0]
		f = [self.likelihood(x[-1], g)]
		acc = 0
		if verbose:
			iterator = tqdm(range(n_iter))
		else:
			iterator = range(n_iter)
		for i in iterator:
			x_new = x[-1] + np.random.randn(len(x0)) * scale
			f_new = self.likelihood(x_new, g)
			if np.log(np.random.rand()) < f[-1] - f_new:
				x.append(x_new)
				f.append(f_new)
				acc += 1
		return np.array(x), np.array(f), acc / float(n_iter)

	def solve(self,
		      x0,
		      X,
		      n_iter=100000,
		      verbose=True,
		      pass_samples=False,
		      burnin=10000):
		g = self.likelihood.model.binning_X.histogram(X)

		scale = 1.0
		i = 0
		while True:
			if verbose:
				print("Burnin Attempt %i" % i)
			x, f, acc = self._steps(x0, g, scale, 1000, verbose)
			x0 = x[np.argmin(f)]
			if acc > 0.7:
				scale *= 2.0
			elif acc < 0.3:
				scale *= 0.5
			else:
				break
			i += 1
			if i > 10:
				if verbose:
					print("Shit aint working, aborting ... %f" % scale)
				break

		x, fvals, acc = self._steps(x0, g, scale, burnin, verbose)
		x, fvals, acc = self._steps(x[np.argmin(f)], g, scale, n_iter, verbose)
		error = np.vstack((np.percentile(x, 15, axis=0),
			               np.percentile(x, 85, axis=0)))
		median = np.median(x, axis=0)
		error -= median
		error = np.abs(error)

		result = SolverResult(f=np.median(x, axis=0),
			                  f_err=error,
			                  success=True)
		if pass_samples:
			result.update(samples_x=x, samples_f=fvals)
		return result
