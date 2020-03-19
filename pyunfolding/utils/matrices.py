import numpy as np
from warnings import warn
from ..exceptions import InvalidCovarianceWarning


__all__ = ["diff2_matrix",
           "diff1_matrix",
           "diff0_matrix",
           "check_covariance"]


def diff2_matrix(n, padding=0):
    r'''Matrix that represents the second derivative of a vector in the sense
    of :math:`(\mathrm{C}\mathbf{x})_n = -x_{n+1} + 2x_n - x_{n-1}`

    Parameters
    ----------
    n : int
        Size of the resulting matrix.

    padding : int
        Padding of the matrix in case edges are ignored.

    Returns
    -------
    C : numpy.array, shape=(n, n)
        Regularization matrix.
    '''
    n_ = n - 2 * padding
    M = 2.0 * np.eye(n_) - np.roll(np.eye(n_), 1) - np.roll(np.eye(n_), -1)
    return np.pad(M, padding, mode='constant')


def diff1_matrix(n, padding=0):
    r'''Matrix that represents the first derivative of a vector in the sense
    of :math:`(\mathrm{C}\mathbf{x})_n = x_{n+1} - x_{n}`

    Parameters
    ----------
    n : int
        Size of the resulting matrix.

    padding : int
        Padding of the matrix in case edges are ignored.

    Returns
    -------
    C : numpy.array, shape=(n, n)
        Regularization matrix.
    '''
    n_ = n - padding
    M = np.eye(n_) - np.eye(n_, k=1)
    return np.pad(M, padding, mode='constant')


def diff0_matrix(n, padding=0):
    r'''Identity matrix.

    Parameters
    ----------
    n : int
        Size of the resulting matrix.

    padding : int
        Padding of the matrix in case edges are ignored.

    Returns
    -------
    C : numpy.array, shape=(n, n)
        Regularization matrix.
    '''
    n_ = n - padding
    M = np.eye(n_)
    return np.pad(M, padding, mode='constant')


def _check_symmetry(A, rtol=1e-05, atol=1e-08):
    return np.allclose(A, A.T, rtol=rtol, atol=atol)


def _check_posdef(A, rtol=1e-05, atol=1e-08):
    try:
        u, s, v = np.linalg.svd(A)
    except:
        return False
    return np.allclose(A, u @ np.diag(s) @ u.T, rtol=rtol, atol=atol)


def check_covariance(cov, rtol=1e-5, atol=1e-8):
    '''Checks whether a matrix is a proper covariance matrix i.e. whether
    it is both positive semi-definite and symmetrical (up to some margin
    of error).

    Parameters
    ----------
    cov : numpy.array, shape=(n,n)
        Covariance matrix (hopefully)

    rtol : float
        Relative tolerance

    atol : float
        Absolute tolerance

    Returns
    -------
    result : bool
        Whether or not the provided covariance matrix is a covariance matrix.

    Raises
    ------
    InvalidCovarianceWarning
        Raised when this returns `False`.
    '''
    if not _check_posdef(cov, rtol, atol):
        warn(InvalidCovarianceWarning('Covariance not positive semidefinite.'))
        return False

    if not _check_symmetry(cov, rtol, atol):
        warn(InvalidCovarianceWarning('Covariance not symmetrical.'))
        return False

    return True
        
