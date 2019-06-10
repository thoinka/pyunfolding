import numpy as np
from scipy.optimize import minimize
from collections import namedtuple
from tqdm import tqdm

ONE_SIGMA = 0.682689492137085897


def error_central(x, p=ONE_SIGMA):
    lower = np.percentile(x, 100.0 * (1.0 - p) / 2.0, axis=0)
    upper = np.percentile(x, 100.0 * (1.0 + p) / 2.0, axis=0)
    return lower, upper


def error_shortest(x, p=ONE_SIGMA):
    lower = np.zeros(x.shape[1])
    upper = np.zeros(x.shape[1])
    N = int(p * len(x))
    for i in range(x.shape[1]):
        x_sort = np.sort(x[:, i])
        x_diff = np.abs(x_sort[N:] - x_sort[:-N])
        shortest = np.argmin(x_diff)
        lower[i], upper[i] = x_sort[shortest], x_sort[shortest + N]
    return lower, upper


def error_best(x, f, p=ONE_SIGMA):
    N = int(p * len(x))
    best = np.argsort(f)[:N]
    return np.min(x[best, :], axis=0), np.max(x[best, :], axis=0)


class SolverResult:
    def __init__(self, f, f_err, success, *args, **kwargs):
        self.f = f
        self.f_err = f_err
        self.success = success
        for key, val in kwargs:
            self.__dict__[key] = val

    def __str__(self):
        if self.success:
            s = "Successful unfolding:\n"
        else:
            s = "Unsuccessful unfolding:\n"
        for i, (f, ferr1, ferr2) in enumerate(zip(self.f, self.f_err[0], self.f_err[1])):
            s += "Var {}:\t{:.2f} +{:.2f} -{:.2f}\n".format(i + 1,
                                                            f, ferr1, ferr2)
        try:
            s += "Error-Message: {}".format(self.error)
        except:
            s += "No error message was left."
        return s

    def __repr__(self):
        return self.__str__()

    def update(self, **kwargs):
        self.__dict__.update(kwargs)


class Solver:
    def __init__(self, likelihood, *args, **kwargs):
        self.likelihood = likelihood

    def solve(self, *args, **kwargs):
        raise NotImplementedError("Solve routine needs to be implemented.")


class NewtonMinimizer(Solver):
    name = "NewtonMinimizer"

    def solve(self, x0, X, max_iter=100, tol=1e-6, beta=0.5, *args, **kwargs):
        g = self.likelihood.model.binning_X.histogram(X)

        def F(p): return self.likelihood(p, g)

        def G(p): return self.likelihood.grad(p, g)

        def H(p): return self.likelihood.hess(p, g)
        x = [x0]
        f = [F(x0)]
        alpha = 1.0
        success = False
        for i in range(max_iter):
            hess = H(x[-1])
            grad = G(x[-1])
            if np.linalg.norm(grad) < tol:
                success = True
                break
            x_new = x[-1] - alpha * \
                np.dot(np.linalg.inv(hess + 1e-8 * np.eye(len(hess))), grad)
            f_new = F(x_new)

            if f_new < f[-1]:
                x.append(x_new)
                f.append(f_new)
                alpha = 1.0
            else:
                alpha *= beta

        std = np.sqrt(np.linalg.pinv(H(x[-1])).diagonal())
        return SolverResult(f=x[-1],
                            f_err=np.vstack((0.5 * std, 0.5 * std)),
                            success=success)


class Minimizer(Solver):
    name = "Minimizer"

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
        return SolverResult(f=result.x,
                            f_err=np.vstack((0.5 * error, 0.5 * error)),
                            success=result.success)


class MCMCSolver(Solver):
    name = "Solver"

    def _steps(self, x0, g, scale, n_iter, verbose=False):
        x = np.zeros((n_iter, len(x0)))
        x[0,:] = x0
        f = np.zeros(n_iter)
        f[0] = self.likelihood(x[0,:], g)
        acc = 0
        if verbose:
            iterator = tqdm(range(1, n_iter), ascii=True)
        else:
            iterator = range(1, n_iter)
        for i in iterator:
            x_new = x[i - 1] + np.random.randn(len(x0)) * scale
            f_new = self.likelihood(x_new, g)
            if np.log(np.random.rand()) < f[i - 1] - f_new:
                x[i,:] = x_new
                f[i] = f_new
                acc += 1
            else:
            	x[i,:] = x[i - 1,:]
            	f[i] = f[i - 1]
        return x, f, acc / float(n_iter)

    def solve(self,
              x0,
              X,
              n_iter=100000,
              verbose=True,
              pass_samples=False,
              burnin=10000,
              error_method="central",
              value_method="median"):
        """Solving Routine.
        
        Parameters
        ----------
        x0 : numpy array, shape=(len(f),)
            Initial value for minimization.
        X : numpy array, shape=(n_samples,n_obs)
            Measured observables.
        n_iter : int, optional, default=100000
            Number of iterations.
        verbose : bool, optional, default=True
            Whether or not to use verbose mode.
        pass_samples : bool, optional, default=False
            Whether or not to pass the samples generated.
        burnin : int, optional, default=10000
            Number of samples to withdraw as burn-in.
        error_method : str, optional, default="central"
            Method to calculate errorbars, options:
            	* `central`: Central interval, defined by the quantiles 15% and
            	             85%
            	* `shortest`: Shortest interval containing 68% of the samples.
            	* `best`: Shortest interval containing 68% of the samples.
            	* `std`: Interval spanned by the standard deviation.
        value_method : str, optional, default="median"
            Method to calculate the value
            	* `median`: Use the median of the samples.
            	* `best`: Use the best fit among the samples.
            	* `mean`: Use the mean of the samples.
        
        Returns
        -------
        result : SolverResult object
            The result of the solution.
        """
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
            if i > 50:
                if verbose:
                    print("Shit aint working, aborting ... %f" % scale)
                break

        x, fvals, acc = self._steps(x0, g, scale, burnin, verbose)
        x, fvals, acc = self._steps(x[np.argmin(f)], g, scale, n_iter, verbose)

        if verbose:
        	print("Acceptance rate: {}".format(acc))

        if value_method == "median":
            value = np.median(x, axis=0)
        elif value_method == "best":
            value = x[np.argmin(fvals)]
        elif value_method == "mean":
            value = np.mean(x, axis=0)
        else:
            raise ValueError("value_method {} not supported!"\
                             .format(value_method))

        if error_method == "central":
            lower, upper = error_central(x)
        elif error_method == "shortest":
            lower, upper = error_shortest(x)
        elif error_method == "best":
            lower, upper = error_best(x, fvals)
        elif error_method == "std":
            std = np.std(x, axis=0)
            lower, upper = value - std * 0.5, value + std * 0.5
        else:
            raise ValueError("error_method {} not supported!"\
                             .format(error_method))

        error = np.vstack((value - lower, upper - value))

        result = SolverResult(f=value,
                              f_err=error,
                              success=True)
        if pass_samples:
            result.update(samples_x=x, samples_f=fvals)
        return result
