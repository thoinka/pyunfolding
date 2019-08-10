import numpy as np
import pandas as pd
from scipy.stats import distributions as dist


def gen_gaussian_smearing(y, spread_low=0.2, spread_high=0.1, gamma=1.0,
                          truncate=False):
    """Gaussian smearing.
    
    Parameters
    ----------
    y : numpy.array, shape=(n_samples,)
        Truths, all y > 0 and y < 1
    spread_low : float
        Spread at y = 0
    spread_high : float
        Spread at y = 1
    gamma : float
        Exponent of observable function.
    truncate : bool
        Whether or not to truncate to [0,1]
        
    Returns
    -------
    X : pandas.DataFrame
        DataFrame containing dataset.
    """
    assert ((y >= 0.0) & (y <= 1.0)).all(), 'All values of y must be in [0,1]'
    y_ = y ** gamma
    spreads = spread_low * (1.0 - y_) + spread_high * y_
    if truncate:
        a, b = -y_ / spreads, (1.0 - y_) / spreads
        sample = dist.truncnorm.rvs(a=a, b=b, loc=y_, scale=spreads)
    else:
        sample = dist.norm.rvs(loc=y_, scale=spreads)
    return pd.DataFrame({'X': sample, 'y': y})


def _detector_response(lam, mu, sigma, n_pixels):
    X = np.random.normal(mu, sigma, np.random.poisson(lam))
    H = np.histogram(X, np.linspace(-3.0, 3.0, n_pixels + 1))[0].astype(float)
    H += np.random.rand(len(H))
    return H


def gen_calorimeter(y, min_n=10, max_n=10000, n_pixels=10,
                    trigger_threshold=10):
    """Calorimeter-like detector response including a number of diverse
    features.
    
    Parameters
    ----------
    y : numpy.array, shape=(n_samples,)
        Truths, all y > 0 and y < 1
    min_n : float
        Minimum number of "hits" in the calorimeter for y=0
    max_n : float
        Maximum number of "hits" in the calorimeter for y=1
    n_pixels : int
        Number of pixels in the calorimeter
    trigger_threshold : float
        Threshold number of hits to engage trigger.
        
    Returns
    -------
    X : pandas.DataFrame
        DataFrame containing dataset.
    """
    n_samples = len(y)
    mu = np.random.randn(n_samples)
    sigma = np.random.exponential(1.0, n_samples)
    N = (1.0 - y) * min_n + y * max_n 
    X = np.array([_detector_response(n, m, s, n_pixels)
                  for n, m, s in zip(N, mu, sigma)])

    qtot = np.sum(X, axis=-1)
    cog = np.sum(X * np.arange(n_pixels), axis=-1) / np.sum(X, axis=-1)
    cog2 = np.sum(X * np.arange(n_pixels) ** 2, axis=-1) / np.sum(X, axis=-1)
    std = np.sqrt(cog2 - cog ** 2)

    trigger = qtot > trigger_threshold

    dict_df = {
        'y':      y[trigger],
        'X_qtot': qtot[trigger],
        'X_cog':  cog[trigger],
        'X_std':  std[trigger]
    }
    dict_df.update({'X_p{}'.format(i): X[trigger,i]
                    for i in range(X.shape[1])})
    return pd.DataFrame(dict_df)