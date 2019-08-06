import numpy as np
from scipy.stats import distributions as dist


def gen_linear(m, n_samples):
    '''Generates a linear spectrum with the slope of `m`

    Parameters
    ----------
    m : float
        Slope. Must be in [-2, 2]
    n_samples : int
        Number of samples.

    Returns
    -------
    samples : numpy.array, shape=(n_samples,)
        Samples.
    '''
    assert m > -2.0 and m < 2.0, 'Slope must be in [-2, 2]'
    y = np.random.rand(n_samples)
    if m == 0.0:
        return y
    return ((-2.0 + m + np.sqrt(4.0 - 4.0 * m + m ** 2 + 8.0 * m * y))
            / (2.0 * m))

def gen_gaussian(loc, scale, n_samples):
    '''Generates a gaussian spectrum

    Parameters
    ----------
    loc : float
        Position of peak
    scale : float
        Width of peak
    n_samples : int
        Number of samples.

    Returns
    -------
    samples : numpy.array, shape=(n_samples,)
        Samples.
    '''
    a, b = -loc / scale, (1.0 - loc) / scale
    return dist.truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)