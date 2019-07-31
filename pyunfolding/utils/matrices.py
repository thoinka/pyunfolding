import numpy as np
import warnings


def diff2_matrix(n, padding=0):
    n_ = n - 2 * padding
    M = 2.0 * np.eye(n_) - np.roll(np.eye(n_), 1) - np.roll(np.eye(n_), -1)
    return np.pad(M, padding, mode='constant')


def diff1_matrix(n, padding=0):
    n_ = n - padding
    M = np.eye(n_) - np.eye(n_, k=1)
    return np.pad(M, padding, mode='constant')


def check_symmetry(A, rtol=1e-05, atol=1e-08):
    return np.allclose(A, A.T, rtol=rtol, atol=atol)


def check_posdef(A, atol=1e-14):
    return (np.linalg.eig(A)[0] + atol>= 0.0).all()


def check_covariance(cov):
    if not check_posdef(cov):
        warnings.warn('Warning: Covariance matrix is not positive-semidefinite!'
                      ' Do not trust the uncertainty estimation!')
    if not check_symmetry(cov):
        warnings.warn('Warning: Covariance matrix is not symmetrical!'
                      ' Do not trust the uncertainty estimation!')

