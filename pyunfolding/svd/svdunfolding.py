from ..model import Unfolding
from ..utils import UnfoldingResult
import numpy as np


class SVDUnfolding:
    """Unfolding performed by performing a singular value decomposition of the migration matrix :math:`A`:

    :math:`\mathbf{g} = \mathrm{A}\mathbf{f} = \mathrm{U}\mathrm{S}\mathrm{V}\mathbf{f}`

    The pseudoinverse of :math:`A` is then regularized by cutting off statistically insignficant coefficients of :math:`\mathbf{g}`. The statistical significance of each coefficient is determined by assuming Poissonian statistics:

    :math:`\Sigma_{\mathbf{g}} = `U^\top \mathrm{diag}(\mahtbf{g}) U`

    Every coefficient with a value smaller than `sig_level` is then cut off when calculating the pseudo inverse.
    
    :math:`A^+_\mathrm{reg} = \mathrm{V}^\top \mathrm{S}^+\mathrm{U}^\top \mathrm{diag}(\mathbf{g} > s).

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
    self.U, self.S, self.V : numpy.arrays
        Matrices that form the singular value decomposition.
    """
    def __init__(self, binning_X, binning_y):
        self.model = Unfolding(binning_X, binning_y)
        self.is_fitted = False
    
    def g(self, X):
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
            return self.model.binning_X.histogram(X)
        raise SyntaxError('Unfolding not yet fitted! Use `fit` method first.')

    def f(self, y):
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
            return self.model.binning_y.histogram(y)
        raise SyntaxError('Unfolding not yet fitted! Use `fit` method first.')
    
    def fit(self, X_train, y_train):
        '''Fit routine.

        Parameters
        ----------
        X_train : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        y_train : numpy.array, shape=(n_samples,)
            Target variable sample.
        '''
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.U, self.S, self.V = np.linalg.svd(self.model.A)
        
    def predict(self, X, sig_level=1.0):
        '''Calculates an estimate for the unfolding.

        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        sig_level : float, default=1.0
            significance level demanded from all svd coefficients.
        '''
        if self.is_fitted:
            g = self.g(X)
            R = np.eye(100)
            h = np.abs(self.U.T @ g) / (self.U.T @ np.diag(g) @ self.U).diagonal()
            R[h < sig_level,:] = 0.0
            A_pinv_reg = self.V.T @ np.linalg.pinv(np.pad(np.diag(self.S), ((0, 80), (0, 0)), 'constant')) @ R @ self.U.T
            f = A_pinv_reg @ g
            cov = A_pinv_reg @ np.diag(g) @ A_pinv_reg.T
            return UnfoldingResult(f=f,
                                   f_err=np.vstack((np.sqrt(cov.diagonal()),
                                                    np.sqrt(cov.diagonal()))),
                                   cov=cov,
                                   success=True)
        raise SyntaxError('Unfolding not yet fitted! Use `fit` method first.')