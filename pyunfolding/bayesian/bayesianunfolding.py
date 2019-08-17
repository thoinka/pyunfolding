from ..model import Unfolding
from ..utils import UnfoldingResult
from ..base import UnfoldingBase
from ..utils.minimization import num_gradient
import numpy as np

from numba import jit


@jit("float64[:](float64[:], float64[:], float64[:], int64, float64)",
    error_model='numpy')
def _ibu(A, g, f0, n_iterations=5, alpha=1.0):
    """Iterative Bayesian Unfolding helper function.
    
    Parameters
    ----------
    A : numpy array, shape=(n_bins_obs, n_bins_tgt)
        Migration matrix

    g : numpy array, shape=(n_bins_obs,)
        Observable vector

    n_iterations : int, optional (default=5)
        Number of iterations. The more, the less regularized the result will
        be.
    
    alpha : float, optional (default=1.0)
            Step size. Instead of simply using the next proposed iteration as
            the next step, only a fraction `alpha` of the difference between 
            the next iteration and the current iteration is added to the
            current iteration. That slows down convergence and can be useful
            to determine a more accurate regularization strength.
    
    Returns
    -------
    f_est : numpy array, shape=(n_bins_tgt)
        The unfolding result.
    """
    f_est = f0
    for _ in range(n_iterations):
        f_est_new = np.dot(g, (A * f_est / (A @ f_est).reshape(-1,1)))
        df = f_est_new - f_est
        f_est = f_est + alpha * df
    return f_est


class BayesianUnfolding(UnfoldingBase):
    r'''Iterative Bayesian Unfolding, i.e. estimating :math:`\mathbf{f}` by
    .. math::
        \hat f(j) = \sum_i B(j|i) g(i)

    To shake of the inherent bias of the matrix B, it is estimated iteratively
    using the last estimation of f for reweighting.
    Increasing the number of iterations will eventually bring the result
    closer to what a simple matrix inversion would result in. So limiting
    the number of iterations will act as a regularization when a flat
    spectrum is used as a first iteration.
    To adjust this regularization in a more subtle way, an additional parameter
    `alpha` is available that enables fractional iterations.
    The uncertainty is estimating using a very simple numerical gradient, i.e.
    it is estimated as

    .. math::
        \mathrm{cov}(\hat{\mathbf{f}}) = \left( \frac{\partial \hat{\mathbf{f}}}{\partial \mathbf{g}}\right ) \cdot \mathrm{diag}(\mathbf{g}) \cdot \left( \frac{\partial \hat{\mathbf{f}}}{\partial \mathbf{g}}\right )^\top

    whereas the gradients are calculated numerically.

    Parameters
    ----------
    binning_X : pyunfolding.binning.Binning object
        The binning of the observable space.

    binning_y : pyunfolding.binning.Binning object
        The binning of the target variable.

    Attributes
    ----------
    model : pyunfolding.model.Unfolding object
        Unfolding model.

    is_fitted : bool
        Whether or not the unfolding has been fitted.

    n_bins_X : `int`
        Number of bins in the observable space.

    n_bins_y : `int`
        Number of bins in the target space.
    '''
    def __init__(self, binning_X, binning_y):
        super(BayesianUnfolding, self).__init__()
        self.model = Unfolding(binning_X, binning_y)
    
    def fit(self, X_train, y_train):
        '''Fit routine.

        Parameters
        ----------
        X_train : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.

        y_train : numpy.array, shape=(n_samples,)
            Target variable sample.
        '''
        X_train, y_train = super(BayesianUnfolding, self).fit(X_train, y_train)
        self.model.fit(X_train, y_train)
        self.n_bins_X = self.model.binning_X.n_bins
        self.n_bins_y = self.model.binning_y.n_bins
        self.is_fitted = True
        
    def predict(self, X, x0=None, n_iterations=5, alpha=1.0, eps=None):
        '''Calculates an estimate for the unfolding.

        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.

        x0 : numpy.array, shape=(n_bins_y)
            Initial value for the unfolding. This is preferably a very smooth
            vector to make sure that the regularization works as intended.

        n_iterations : int, optional (default=5)
            Number of iterations. The more iterations the `less` regularized
            the result becomes. For the limit of large `n_iterations` the
            result is equal to :math:`\hat{\mathbf{f}}A^+\mathbf{g}`.

        alpha : float, optional (default=1.0)
            Step size. Instead of simply using the next proposed iteration as
            the next step, only a fraction `alpha` of the difference between 
            the next iteration and the current iteration is added to the
            current iteration. That slows down convergence and can be useful
            to determine a more accurate regularization strength.

        eps : float, optional (default=None)
            Epsilon used for estimating uncertainties. To estimate the errors
            the derivative :math:`\frac{\partial\mathbf{f}}{\partial\mathbf{g}}´ is calculated using a finite differences approach with
            stepsize `eps`. If `None`, the value of eps is chosen automatically
            based on the size of the elements in g.

        Returns
        -------
        result : ``pyunfolding.utils.UnfoldingResult`` object
            The result of the unfolding, see documentation for 
            `UnfoldingResult`.
        '''
        X = super(BayesianUnfolding, self).predict(X)
        g = self.g(X).astype('float64')
        if eps is None:
            g_max = np.max(g)
            eps = g_max * 1e-6
        if x0 is None:
            x0 = np.ones(self.model.A.shape[1]) * np.sum(g) / self.model.A.shape[1]
        ibu_func = lambda g_: _ibu(self.model.A, g_,
                                   f0=x0, n_iterations=n_iterations,
                                   alpha=alpha)
        f = ibu_func(g)
        B = num_gradient(ibu_func, g, eps)
        cov = B.T @ np.diag(g) @ B
        return UnfoldingResult(f=f,
                               f_err=np.vstack((np.sqrt(cov.diagonal()),
                                                np.sqrt(cov.diagonal()))),
                               cov=cov,
                               success=True,
                               binning_y=self.model.binning_y)