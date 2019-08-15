import numpy as np
from warnings import warn
from ..exceptions import InvalidCovarianceWarning


def diff2_matrix(n, padding=0):
    n_ = n - 2 * padding
    M = 2.0 * np.eye(n_) - np.roll(np.eye(n_), 1) - np.roll(np.eye(n_), -1)
    return np.pad(M, padding, mode='constant')


def diff1_matrix(n, padding=0):
    n_ = n - padding
    M = np.eye(n_) - np.eye(n_, k=1)
    return np.pad(M, padding, mode='constant')


def diff0_matrix(n, padding=0):
    n_ = n - padding
    M = np.eye(n_)
    return np.pad(M, padding, mode='constant')


def check_symmetry(A, rtol=1e-05, atol=1e-08):
    return np.allclose(A, A.T, rtol=rtol, atol=atol)


def check_posdef(A, rtol=1e-05, atol=1e-08):
    try:
        u, s, v = np.linalg.svd(A)
    except:
        return False
    return np.allclose(A, u @ np.diag(s) @ u.T, rtol=rtol, atol=atol)


def check_covariance(cov):
    if not check_posdef(cov):
        warn(InvalidCovarianceWarning('Covariance not positive semidefinite.'))

    if not check_symmetry(cov):
        warn(InvalidCovarianceWarning('Covariance not symmetrical.'))
        
