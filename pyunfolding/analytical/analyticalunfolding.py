import numpy as np
from ..model import Unfolding
from ..utils import diff2_matrix, diff1_matrix
from ..utils import UnfoldingResult


def analytical_solution(A, g, tau=0.0, Sigma=None, C_matrix=None):
	'''Analytical solution of the unfolding problem given a
	Tikhonov-regularized Least-Squares Likelihood approach:

	:math:`\mathcal{L}(\mathbf{f}|\mathbf{g}) = (\mathbf{g}-\mathrm{A}\mathbf{f})^\top \Sigma^{-1} (\mathbf{g}-\mathrm{A}\mathbf{f}) + \frac{\tau}{2} \mathbf{f}^\top \Gamma^\top \Gamma \mathbf{f}
	'''
	lenf = A.shape[1]

	if type(C_matrix) == str:
		if C_matrix == 'diff2':
			C_matrix = diff2_matrix(lenf)
		elif C_matrix == 'diff1':
			C_matrix = diff1_matrix(lenf)
	elif C_matrix is None:
		C_matrix = np.zeros((lenf, lenf))

	if Sigma is None:
		Sigma = np.eye(len(g))

	iSigma = np.linalg.pinv(Sigma)

	B = np.linalg.pinv(A.T @ iSigma @ A
	                   + 0.5 * tau * C_matrix.T @ C_matrix) @ A.T @ iSigma

	f_est = B @ g
	cov = B @ np.diag(g) @ B.T
	f_err = np.sqrt(np.vstack([cov.diagonal(), cov.diagonal()]))

	return UnfoldingResult(f=f_est, f_err=f_err, cov=cov, success=True)


class AnalyticalUnfolding:
    '''Analytical solution to the unfolding problem in the case of a Weighted
    Least Squares Likelihood with Tikhonov regularization.

    Parameters
    ----------
    binning_X : pyunfolding.binning.Binning object
        The binning of the observable space.
    binning_y : pyunfolding.binning.Binning object
        The binning of the target variable.
    exclude_edges : bool
        Whether or not to exclude the edges, i.e. the under- and overflow bin
        in the regularization.
    C : string or numpy.array
        Regularization matrix
    Sigma : None, string or numpy.array
        Weight matrix.
    
    Attributes
    ----------
    model : pyunfolding.model.Unfolding object
        Unfolding model.
    llh : pyunfolding.likelihood.llh.Likelihood object
        The likelihood.
    is_fitted : bool
        Whether the object has already been fitted.
    n_bins_X : int
        Number of bins in the observable space.
    n_bins_y : int
        Number of bins in the target space.
    '''
    def __init__(self, binning_X, binning_y, exclude_edges=True, C='diff2',
                 Sigma=None):
        self.model = Unfolding(binning_X, binning_y)
        self.Sigma = Sigma
        self.is_fitted = False
        self.exclude_edges = exclude_edges
        self.C = C
    
    def g(self, X, weights=None):
        '''Returns an observable vector :math:`\mathbf{g}` given a sample
        of observables `X`.

        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_observables)
            Observable sample.
        weights : numpy.array, shape=(n_samples,)
            Weights corresponding to the sample.

        Returns
        -------
        g : numpy.array, shape=(n_bins_X,)
            Observable vector
        '''
        if self.is_fitted:
            return self.model.binning_X.histogram(X, weights=weights)
        raise RuntimeError('Unfolding not yet fitted! Use `fit` method first.')

    def f(self, y, weights=None):
        '''Returns a result vector :math:`\mathbf{f}` given a sample
        of target variable values `y`.

        Parameters
        ----------
        y : numpy.array, shape=(n_samples,)
            Target variable sample.
        weights : numpy.array, shape=(n_samples,)
            Weights corresponding to the sample.

        Returns
        -------
        f : numpy.array, shape=(n_bins_X,)
             Result vector.
        '''
        if self.is_fitted:
            return self.model.binning_y.histogram(y, weights=weights)
        raise RuntimeError('Unfolding not yet fitted! Use `fit` method first.')

    def fit(self, X_train, y_train, *args, **kwargs):
        '''Fit routine.

        Parameters
        ----------
        X_train : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        y_train : numpy.array, shape=(n_samples,)
            Target variable sample.
        '''
        self.model.fit(X_train, y_train, *args, **kwargs)
        self.is_fitted = True
        self.n_bins_y = self.model.binning_y.n_bins
        self.n_bins_X = self.model.binning_X.n_bins
        if self.C == "diff1":
            c_gen = diff1_matrix
        elif self.C == 'diff2':
            c_gen = diff2_matrix

        if self.exclude_edges:
            self.C = np.zeros((self.n_bins_y, self.n_bins_y))
            self.C[1:-1, 1:-1] = c_gen(self.n_bins_y - 2)
        else:
            self.C = c_gen(self.n_bins_y)
        
    def predict(self, X, tau=0.0, **kwargs):
        '''Calculates an estimate for the unfolding by maximizing the likelihood function (or minimizing the log-likelihood).

        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        tau : float
            Regularization strength.
        '''
        if self.is_fitted:
            g = self.g(X)
            if self.Sigma == 'poisson':
                self.Sigma = np.diag(g)
            result = analytical_solution(self.model.A, g, tau, self.Sigma,
                                         self.C)
            result.update(binning_y=self.model.binning_y)
            return result
        raise RuntimeError('Unfolding not yet fitted! Use `fit` method first.')