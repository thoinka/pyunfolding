import numpy as np
from collections import namedtuple
from tqdm import tqdm
from ...utils import error_central, error_shortest, error_best
from ...utils import UnfoldingResult
from ...utils import Posterior
from .base import SolutionBase
from joblib import Parallel, delayed

from warnings import warn
from ...exceptions import FailedMCMCWarning


# Default values for the burnin procedure. These are rather random values, but
# they turned out to work reasonably well.
MCMC_BURNIN_MIN_ACC = 0.15
MCMC_BURNIN_MAX_ACC = 0.40
MCMC_BURNIN_SCALING = 0.75
MCMC_BURNIN_NSTEPS = 1000
MCMC_BURNIN_NATTEMPTS = 50


class MCMC(SolutionBase):

    def _steps(self, x0, g, scale, n_steps, rnd, verbose=False):
        if n_steps == 0:
            return x0.reshape(1,-1), np.array(self.likelihood(x0, g)), 0.0
        x = np.zeros((n_steps, len(x0)))
        x[0,:] = x0
        f = np.zeros(n_steps)
        f[0] = self.likelihood(x[0,:], g)
        acc = 0
        if verbose:
            iterator = tqdm(range(1, n_steps), ascii=True)
        else:
            iterator = range(1, n_steps)
        for i in iterator:
            x_new = x[i - 1] + rnd.randn(len(x0)) * scale
            f_new = self.likelihood(x_new, g)
            if np.log(rnd.rand()) < f[i - 1] - f_new:
                x[i,:] = x_new
                f[i] = f_new
                acc += 1
            else:
            	x[i,:] = x[i - 1,:]
            	f[i] = f[i - 1]
        return x, f, acc / float(n_steps)

    def _perform_mcmc(self,
                      x0,
                      g,
                      scale,
                      n_steps,
                      verbose,
                      n_burnin,
                      random_seed):
        rnd = np.random.RandomState(random_seed)
        x, fvals, acc = self._steps(x0, g, scale, n_burnin, rnd, False)
        x, fvals, acc = self._steps(x[-1], g, scale, n_steps, rnd, verbose)

        if verbose:
            print("Acceptance rate: {}".format(acc))

        return {'x': x, 'f': fvals, 'acc': acc}

    def solve(self,
              x0,
              X,
              n_steps=100000,
              verbose=True,
              return_samples=True,
              n_burnin=10000,
              step_size_init=1.0,
              n_jobs=None,
              fisher_info=False,
              error_method="shortest",
              value_method="best",
              random_seed=None):
        """Solving Routine.
        
        Parameters
        ----------
        x0 : numpy array, shape=(len(f), n_jobs)
            Initial value for minimization.
        X : numpy array, shape=(n_samples,n_obs)
            Measured observables.
        n_steps : int, optional, default=100000
            Number of iterations.
        verbose : bool, optional, default=True
            Whether or not to use verbose mode.
        return_samples : bool, optional, default=True
            Whether or not to pass the samples generated.
        n_burnin : int, optional, default=10000
            Number of samples to withdraw as burn-in.
        fisher_info : bool
            Estimate covariance using the Fisher matrix
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
        if x0.ndim == 1:
            n_jobs = 1
        else:
            n_jobs = x0.shape[0]

        if random_seed is None:
            random_seed = np.random.randint(np.iinfo('uint32').max)

        g = self.likelihood.model.binning_X.histogram(X)

        success = True

        n_steps_per_job = [n_steps // n_jobs] * (n_jobs - 1)
        n_steps_per_job.append(n_steps - sum(n_steps_per_job))

        scale = step_size_init

        i_burnin = 0
        x0_n_burnin = x0[0,:]
        while True:
            if verbose:
                print("Burnin Attempt %i" % i_burnin)
            burnin_sample = self._perform_mcmc(x0_n_burnin,
                                               g,
                                               scale,
                                               MCMC_BURNIN_NSTEPS,
                                               False,
                                               0,
                                               random_seed + i_burnin)
            x, f, acc = burnin_sample['x'], burnin_sample['f'], burnin_sample['acc']
            x0_n_burnin = x[-1]
            if verbose:
                print("Acceptance rate: {}".format(acc))
            if acc > MCMC_BURNIN_MAX_ACC:
                scale *= 1.0 / MCMC_BURNIN_SCALING
            elif acc < MCMC_BURNIN_MIN_ACC:
                scale *= MCMC_BURNIN_SCALING
            else:
                break
            i_burnin += 1
            if i_burnin > MCMC_BURNIN_NATTEMPTS:
                warn(FailedMCMCWarning('Maximum number of burn-in attempts exceeded.'))
                success = False
                error = 'MCMC Burn-in failed after {} attempts. Final acceptance was {}.'.format(MCMC_BURNIN_NATTEMPTS, acc)
                break
        
        # Calculate multiple separate mcmcs using joblib
        ppdfs = Parallel(n_jobs=n_jobs)([
            delayed(self._perform_mcmc)(
                    x0[i,:],
                    g,
                    scale,
                    n_steps_per_job[i],
                    verbose,
                    n_burnin,
                    random_seed + i + i_burnin
                ) for i in range(n_jobs)
        ])

        # Concatenate results
        x = np.concatenate([p['x'] for p in ppdfs])
        f = np.concatenate([p['f'] for p in ppdfs])

        ppdf = Posterior(x, f)

        value = ppdf.value(value_method)
        lower, upper = ppdf.error(error_method, best_fit=ppdf.value('best'))

        error = np.vstack((value - lower,
                           upper - value))
    
        cov = np.cov(x.T)

        result = UnfoldingResult(f=value,
                                 f_err=error,
                                 success=success,
                                 cov=cov)
        if return_samples:
            result.update(sample=ppdf)
        try:
            result.update(error=error)
        except:
            pass
        if fisher_info:
            fi = np.mean([self.likelihood.hess(x_, g) for x_ in x], axis=0)
            result.update(fisher_matrix=fi)
        return result
