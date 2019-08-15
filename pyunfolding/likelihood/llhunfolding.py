from . import llh
from ..model import Unfolding
from ..base import UnfoldingBase
from ..exceptions import NotFittedError

import numpy as np
from joblib import cpu_count


class LLHUnfolding(UnfoldingBase):
    r'''Unfolding based on a maximum likelihood fit. Multiple likelihoods can
    be defined using several predefined (log) likelihood terms in the 
    sub-module ``pyunfolding.likelihood.llh``.

    .. math::
        \hat{\mathbf{f}}(\mathbf{g}) = \mathrm{argmax}_{\mathbf{f}}\log \mathcal{L}(\mathbf{f}|\mathbf{g})

    The uncertainty of the estimate is either approximated using the inverse
    Hessian of the log likelihood at the best fit,

    .. math::
        \mathrm{cov}(\hat{\mathbf{f}})_{ij} = \frac{\partial^2 \log \mathcal{L}(\mathbf{f}|\mathbf{g})}{\partial f_i \partial f_j}

    or using a Markov Chain Monte Carlo scheme that returns a sample of the
    posterior pdf that enables multiple ways of estimating the covariance and
    uncertainty intervals.

    Parameters
    ----------
    binning_X : ``pyunfolding.binning.Binning`` object
        The binning of the observable space. The binning object does not need
        to be fitted prior to initializing this object. In case it is not yet
        fitted, it will be fitted when this object is fitted.

    binning_y : ``pyunfolding.binning.Binning`` object
        The binning of the target variable. The binning object does not need
        to be fitted prior to initializing this object. In case it is not yet
        fitted, it will be fitted when this object is fitted.

    likelihood : List of ``pyunfolding.likelihood.llh.LikelihoodTerm`` objects
        List of likelihood terms as defined in the `likelihood.llh` submodule.
        The final likelihood used to fit the unfolding estimate is the sum
        of all these terms.

    Attributes
    ----------
    model : ``pyunfolding.model.Unfolding`` object
        Unfolding model that maps the detection process in some shape or form.

    llh : ``pyunfolding.likelihood.llh.Likelihood`` object
        The likelihood object that contains the sum of all involved likelihood
        terms.

    is_fitted : `bool`
        Whether the object has already been fitted.

    n_bins_X : `int`
        Number of bins in the observable space.
        
    n_bins_y : `int`
        Number of bins in the target space.

    Raises
    ------
    NotFittedError
        Raised when ``predict`` is called before ``fit`` was.

    Examples
    --------
    >>> from pyunfolding import LLHUnfolding
    >>> from pyunfolding.binning import GridBinning
    >>> from pyunfolding.datasets.spectra import gen_linear
    >>> from pyunfolding.datasets.observables import gen_gaussian_smearing

    >>> df_train = gen_gaussian_smearing(gen_linear(0.0, 100000))
    >>> df_test = gen_gaussian_smearing(gen_linear(-1.0, 1000))
    >>> unfolding = LLHUnfolding(binning_x=GridBinning(20),
    ...                          binning_y=GridBinning(10))
    >>> unfolding.fit(df_train.X, df_train.y)
    >>> unfolding.predict(df_test.y)
    Successful unfolding:
    Var 1:  10.10 +3.96 -3.96
    Var 2:  179.91 +21.08 -21.08
    Var 3:  210.56 +48.53 -48.53
    Var 4:  -42.18 +57.41 -57.41
    Var 5:  379.33 +48.66 -48.66
    Var 6:  64.15 +36.06 -36.06
    Var 7:  -39.61 +23.71 -23.71
    Var 8:  223.65 +12.94 -12.94
    Var 9:  19.58 +5.45 -5.45
    Var 10: -5.49 +3.96 -3.96
    No error message was left.

    Notes
    -----
    This approach is based on work by V. Blobel with additions by M. Börner

    References
    ----------
    .. [1] V. Blobel and E. Lohrmann, "Statistische Methoden der Datenanalyse"
    .. [2] V. Blobel, "Unfolding Methods in High-Energy Physics Experiments", DESY-84-118, 40, 1984
    '''
    def __init__(self, binning_X, binning_y, likelihood=None):
        super(LLHUnfolding, self).__init__()
        self.model = Unfolding(binning_X, binning_y)
        
        if likelihood is None:
            self._llh = [llh.LeastSquares()]
        else:
            self._llh = likelihood
    
    def fit(self, X_train, y_train, *args, **kwargs):
        '''Fit routine. Fits unfolding model, inluding binning.

        Parameters
        ----------
        X_train : numpy.array, shape=(n_samples, n_obervables)
            Observables from the training sample.
        y_train : numpy.array, shape=(n_samples,)
            Target variable from the training sample.
        '''
        X_train, y_train = super(LLHUnfolding, self).fit(X_train, y_train)
        self.model.fit(X_train, y_train, *args, **kwargs)
        self.llh = llh.Likelihood(self.model, self._llh)
        self.is_fitted = True
        self.n_bins_y = self.model.binning_y.n_bins
        self.n_bins_X = self.model.binning_X.n_bins
    
    def predict(self,
                X,
                x0=None,
                minimizer=True,
                method='trust-exact',
                minimizer_options={},
                mcmc=False,
                mcmc_options={},
                random_seed=None):
        '''Calculates an estimate for the unfolding by maximizing the likelihood function (or minimizing the log-likelihood).

        Parameters
        ----------
        X : ``numpy.array``, shape=(n_samples, n_obervables)
            Observable sample.
        x0 : `None` or ``numpy.array``, shape=(n_bins_y)
            Initial value for the unfolding.
        minimizer : `bool` (default=True)
            Whether to use a minimizer
        method : str
            Which minimization algorithm to employ.
            From ``scipy.optimize.minimize`:
                
                * ``nelder-mead``
                * ``powell``
                * ``cg``
                * ``bfgs``
                * ``newton-cg``
                * ``dogleg``
                * ``trust-ncg``
                * ``trust-krylov``
                * ``trust-exact``
                * ``l-bfgs-g``
                * ``tnc``
                * ``cobyla``
                * ``slsqp``
                * ``trust-const``

            From ``pyunfolding.utils.minimization``:

                * ``adam``
                * ``adadelta``
                * ``momentum``
                * ``rmsprop``
        
        minimizer_options : `dict`
            additional options for the minimizer, see scipy.optimize.minimize
            for more information.
        mcmc : bool (default=False)
            Whether to use a MCMC to estimate uncertainties.
        mcmc_options : dict
            additional options for the MCMC. Available options:
            
            * ``n_steps`` : int (default=100000)
              Number of iterations
            * ``verbose`` : bool (default=True)
              Whether or not to print status messages
            * ``n_burnin`` : int (default=10000)
              Number of burnin steps
            * ``step_size_init`` : float (default=1.0)
              Initial step size
            * ``n_jobs`` : int
              Number of jobs.
            * ``error_method`` : str (default='shortest')
              Method to estimate uncertainty intervals.
              Available methods:
                  * ``central``: Central interval, defined by the quantiles 15%
                    and 85%
                  * ``shortest``: Shortest interval containing 68% of the
                    samples.
                  * ``std``: Interval spanned by the standard deviation.`
            * ``value_method`` : str, default='best'
              Method to estimate values.
              Available methods:
                  * ``median``: Use the median of the samples.
                  * ``best``: Use the best fit among the samples.
                  * ``mean``: Use the mean of the samples.
        '''
        X = super(LLHUnfolding, self).predict(X)
        if self.is_fitted:
            if x0 is None:
                x0 = len(X) * np.ones(self.n_bins_y) / self.n_bins_y

            step_size_init = None

            try:
                avail_cpus = cpu_count()
                if mcmc_options['n_jobs'] >= 1:
                    n_jobs = mcmc_options['n_jobs']
                elif -mcmc_options['n_jobs'] <= avail_cpus:
                    n_jobs = avail_cpus + 1 + mcmc_options['n_jobs']
                else:
                    raise ValueError('n_jobs must be > {} on this system.'.format(-avail_cpus))
            except KeyError:
                n_jobs = 1

            if minimizer:
                result = self.llh.solve(x0, X, solver_method='minimizer',
                                        method=method,
                                        **minimizer_options)
                if random_seed is not None:
                    mcmc_options['random_seed'] = random_seed
                else:
                    random_seed = np.random.randint(np.iinfo('uint32').max)

                rnd = np.random.RandomState(random_seed - 1)
                
                smear = rnd.multivariate_normal(np.zeros(self.n_bins_y),
                                                result.cov, size=n_jobs)
                x0 = result.f.reshape(1, -1) + smear
                step_size_init = np.mean(np.sqrt(result.cov.diagonal())) / 4.0
            if mcmc:
                try:
                    step_size_init = mcmc_options['step_size_init']
                except:
                    pass

                result = self.llh.solve(x0, X, solver_method='mcmc',
                                        step_size_init=step_size_init,
                                        **mcmc_options)
            result.update(binning_y=self.model.binning_y)
            return result
        raise NotFittedError('LLHUnfolding not yet fitted! Use `fit` method first.')