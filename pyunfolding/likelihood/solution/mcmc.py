import numpy as np
from collections import namedtuple
from tqdm import tqdm
from ...utils import error_central, error_shortest, error_best
from ...utils import UnfoldingResult
from ...utils import Posterior
from .base import SolutionBase


class MCMC(SolutionBase):

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
            x, f, acc = self._steps(x0, g, scale, 1000, verbose=False)
            x0 = x[np.argmin(f)]
            print("Acceptance rate: {}".format(acc))
            if acc > 0.5:
                scale *= 1.5
            elif acc < 0.15:
                scale *= 0.75
            else:
                break
            i += 1
            if i > 50:
                if verbose:
                    print("Maximum number of burnin attempts unsuccessful... Aborting. %f" % scale)
                break

        x, fvals, acc = self._steps(x0, g, scale, burnin, verbose=False)
        x, fvals, acc = self._steps(x[np.argmin(f)], g, scale, n_iter, verbose)

        if verbose:
        	print("Acceptance rate: {}".format(acc))

        ppdf = Posterior(x, fvals)
        value = ppdf.value(value_method)
        lower, upper = ppdf.error(error_method, best_fit=ppdf.value('best'))

        error = np.vstack((value - lower, upper - value))

        result = UnfoldingResult(f=value,
                                 f_err=error,
                                 success=True,
                                 cov=np.cov(x.T))
        if pass_samples:
            result.update(sample=ppdf)
        return result
