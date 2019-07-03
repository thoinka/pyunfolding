import numpy as np
import warnings


def diff2_matrix(n):
	return 2.0 * np.eye(n) - np.roll(np.eye(n), 1) - np.roll(np.eye(n), -1)


def diff1_matrix(n):
	return np.eye(n) - np.eye(n, k=1)


def check_symmetry(A, rtol=1e-05, atol=1e-08):
    return np.allclose(A, A.T, rtol=rtol, atol=atol)


def check_posdef(A):
    return (np.linalg.eig(A)[0] > 0).all()


def check_covariance(cov):
    if not check_posdef(cov):
        warnings.warn('Warning: Covariance matrix is not positive-semidefinite!'
                      ' Do not trust the uncertainty estimation!')
    if not check_symmetry(cov):
        warnings.warn('Warning: Covariance matrix is not symmetrical!'
                      ' Do not trust the uncertainty estimation!')

