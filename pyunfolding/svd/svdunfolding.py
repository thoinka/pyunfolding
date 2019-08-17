from ..model import Unfolding
from ..utils import UnfoldingResult
from ..base import UnfoldingBase
import numpy as np
from ..exceptions import NotFittedError


def _gaussian_cutoff(x, width):
    return np.exp(-x ** 2 / (2.0 * width ** 2))


def _exponential_cutoff(x, width):
    return np.exp(-x / width)


def _sigmoid_cutoff(x, loc, width):
    return 1.0 / (1.0 + np.exp(width * (x - loc)))


class SVDUnfolding(UnfoldingBase):
    r"""Unfolding performed by performing a singular value decomposition of the migration matrix :math:`A`:

    :math:`\mathbf{g} = \mathrm{A}\mathbf{f} = \mathrm{U}\mathrm{S}\mathrm{V}\mathbf{f}`

    The pseudoinverse of :math:`A` is then regularized by cutting off statistically insignficant coefficients of :math:`\mathbf{g}`. The statistical significance of each coefficient is determined by assuming Poissonian statistics:

    :math:`\Sigma_{\mathbf{g}} = `U^\top \mathrm{diag}(\mahtbf{g}) U`

    Every coefficient with a value smaller than `sig_level` is then cut off when calculating the pseudo inverse.
    
    :math:`A^+_\mathrm{reg} = \mathrm{V}^\top \mathrm{S}^+\mathrm{U}^\top \mathrm{diag}(\mathbf{g} > s).

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

    exclude_eges : `bool`
        Whether or not to exclude the edges during regularization. The way this
        is performed is by setting the elements of A to identity that are
        corresponding to the edges before performing the SVD.

    Attributes
    ----------
    model : pyunfolding.model.Unfolding object
        Unfolding model.

    is_fitted : bool
        Whether or not the unfolding has been fitted.
        ›
    self.U, self.S, self.V : numpy.arrays
        Matrices that form the singular value decomposition.
    """

    __cutoffs__ = {
        'gaussian' : _gaussian_cutoff,
        'exponential': _exponential_cutoff,
        'sigmoid': _sigmoid_cutoff
    }

    def __init__(self, binning_X, binning_y, exclude_edges=False):
        super(SVDUnfolding, self).__init__()
        self.model = Unfolding(binning_X, binning_y)
        self.exclude_edges = exclude_edges
        
    def fit(self, X_train, y_train):
        '''Fit routine.

        Parameters
        ----------
        X_train : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        y_train : numpy.array, shape=(n_samples,)
            Target variable sample.
        '''
        X_train, y_train = super(SVDUnfolding, self).fit(X_train, y_train)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        if self.exclude_edges:
            A_ = np.copy(self.model.A)
            A_[:, 0] = 0.0
            A_[:, -1] = 0.0
            A_[0, 0] = 1.0
            A_[-1, -1] = 1.0
            self.U, self.S, self.V = np.linalg.svd(A_)
        else:
            self.U, self.S, self.V = np.linalg.svd(self.model.A)
        
    def predict(self, X, mode='gaussian', **kwargs):
        '''Calculates an estimate for the unfolding.

        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        mode : str
            Shape used to cut off the singular value. Possible values:
            * `gaussian`: Gaussian with mean 0 and spread `width`
            * `exponential`: Exponential with decay length `width`
            * `sigmoid`: Sigmoid function with location `loc` and steepness `width`.
        kwargs : dict
            Keywords required by mode.

        '''
        X = super(SVDUnfolding, self).predict(X)
        if self.is_fitted:
            g = self.g(X)
            if mode is None:
                vec_r = np.ones(len(g))
            elif type(mode) is str:
                try:
                    vec_r = self.__cutoffs__[mode](np.arange(len(g)), **kwargs)
                except KeyError:
                    raise ValueError('Cutoff mode {} not available! Available cutoff modes: {}'.format(mode, list(self.__cutoffs__.keys())))
            R = np.diag(vec_r)
            h = np.abs(self.U.T @ g) / (self.U.T @ np.diag(g) @ self.U).diagonal()
            I_pinv = np.zeros((self.model.A.shape[1], self.model.A.shape[0]))
            I_pinv[np.arange(len(self.S)), np.arange(len(self.S))] = 1.0 / self.S
            A_pinv_reg = self.V.T @ I_pinv @ R @ self.U.T
            f = A_pinv_reg @ g
            cov = A_pinv_reg @ np.diag(g) @ A_pinv_reg.T
            return UnfoldingResult(f=f,
                                   f_err=np.vstack((np.sqrt(cov.diagonal()),
                                                    np.sqrt(cov.diagonal()))),
                                   cov=cov,
                                   success=True,
                                   binning_y=self.model.binning_y)
        raise NotFittedError('Unfolding not yet fitted! Use `fit` method first.')