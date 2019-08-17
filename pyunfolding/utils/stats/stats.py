import numpy as np
from scipy.stats.distributions import chi2


def cov2corr(cov):
    '''Convert covariance matrix to correlation matrix.

    Parameters
    ----------
    cov : numpy array
        Covariance matrix

    Returns
    -------
    corr : numpy array
        Correlation matrix
    '''
    std = np.sqrt(cov.diagonal())
    corr = cov / np.outer(std, std)
    return corr


def isqrtm_semipos(M):
    r"""Calculates the inverse matrix squareroot :math:`\mathrm{M}^{-1/2}` for
    positive semi-definite matrices `M`.

    Parameters
    ----------
    M : `numpy array`
        Square, positive semi-definite matrix.

    Returns
    -------
    M_isqrt : `numpy array`
        Inverse squareroot of matrix.
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError('Matrix must be square!')
    U, S, V = np.linalg.svd(M)
    if not np.allclose(U, V.T):
        raise ValueError('Matrix must be positive semi-definite!')
    return U @ np.diag(1.0 / np.sqrt(S)) @ V


def normalized_projection(X, mu, cov):
    r"""Projects a set of random samples `X` so that the resulting random
    variable is normalized, i.e. all components have mean 0 and variance 1 in
    case `cov` is a good estimate of the covariance of `X` and `mu` is a good
    estimate of the mean of `X`:

    .. math::
        \vec x' = \Sigma^{-1/2} \cdot (\vec x - \vec \mu)

    The norm of this projection is the Mahalanobis distance.

    Parameters
    ----------
    X : numpy array, shape=(n_samples, n_dim) or (n_dim,)
        Sample to calculate the Mahalanobis distance for.

    mu : numpy array, shape=(n_dim)
        Mean of the Gaussian distribution

    cov : numpy array, shape=(n_dim, n_dim)
        Covariance matrix of the Gaussian distribution

    Returns
    -------
    projections : numpy array, shape=(n_samples, n_dim)
        The normalized projections.
    """
    return (X - mu) @ isqrtm(cov)



def mahalanobis(X, mu, cov):
    r"""Calculates the Mahalanobis distance of `X`, given `cov` and `mu` as
    parameters of the underlying multivariate Gaussian distribution.

    Parameters
    ----------
    X : numpy array, shape=(n_samples, n_dim) or (n_dim,)
        Sample to calculate the Mahalanobis distance for.

    mu : numpy array, shape=(n_dim)
        Mean of the Gaussian distribution

    cov : numpy array, shape=(n_dim, n_dim)
        Covariance matrix of the Gaussian distribution

    Returns
    -------
    mahalanobis_distances : numpy array, shape=(n_samples)
        The Mahalanobis distances.
    """
    if X.ndim == 1:
        X = X.reshape(1,-1)
    X_cent = X - mu
    n = X.shape[1]
    icov = np.linalg.pinv(cov)
    return (X_cent.reshape(-1,1,n) @ icov @ X_cent.reshape(-1,n,1)).flatten()



def gaussian_pvalue(X, mu, cov, ndof=None):
    r"""calculates p-value for the assumptions of `x` originating from a
    multivariate Gaussian pdf with mean `mu` and covariance `cov`.
    It exploits the fact that the mahalobonis distance of `x`

    .. math::
        d^2 = (\vec x - \vec \mu)^\top \Sigma^{-1} (\vec x - \vec \mu)

    is :math:`\chi^2`-distributed with :math:`n_\mathrm{dof} = dim(\vec x)`,
    then the pvalue is :math:`\mathrm{cdf}_{\chi^2}(d^2)`.

    Parameters
    ----------
    X : numpy array, shape=(n_samples, n_dim) or (n_dim,)
        Sample to calculate the Mahalanobis distance for.

    mu : numpy array, shape=(n_dim)
        Mean of the Gaussian distribution

    cov : numpy array, shape=(n_dim, n_dim)
        Covariance matrix of the Gaussian distribution

    ndof : float
        Number of degrees of freedom for the chi2 distribution. If `None`,
        `n_dim` will be used.

    Returns
    -------
    pvals : numpy array, shape=(n_samples)
        The p-values
    """
    if X.ndim == 1:
        X = X.reshape(1,-1)
    dsquared = mahalanobis(X, mu, cov)
    if ndof is None:
        ndof = X.shape[1]
    return chi2(ndof).cdf(dsquared)