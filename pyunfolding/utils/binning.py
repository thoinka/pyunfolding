import numpy as np


__all__ = ["calc_bmids",
           "calc_bdiff",
           "digitize_uni",
           "bin_edges",
           "equidistant_bins",
           "equal_bins",
           "random_bins",
           "random_equal_bins"]


def calc_bmids(bin_edges, underflow=True, overflow=True):
    """Calculates the centers of bins from bin edges.

    Parameters
    ----------
    bin_edges : numpy array, shape=(n_bins + 1)
        Bin Edges.

    underflow, overflow : bool
        Whether or not to include under- and overflow bins.

    Returns
    -------
    bmids : numpy array, shape=(n_bins + underflow + overflow)
        Bin mids. For under- and overflow bins the adjacent bin sizes are
        repeated to have some half useful value.
    """
    bmids = (bin_edges[1:] + bin_edges[:-1]) * 0.5
    if underflow:
    	underflow_bmid = 2 * bin_edges[0] - bin_edges[1]
    	bmids = np.r_[underflow_bmid, bmids]
    if overflow:
    	overflow_bmid =  2 * bin_edges[-1] - bin_edges[-2]
    	bmids = np.r_[bmids, overflow]
    return bmids


def calc_bdiff(bin_edges, underflow=True, overflow=True):
    """Calculates the lengths of bins from bin edges.

    Parameters
    ----------
    bin_edges : numpy array, shape=(n_bins + 1)
        Bin Edges.

    underflow, overflow : bool
        Whether or not to include under- and overflow bins.

    Returns
    -------
    bdiff : numpy array, shape=(n_bins + underflow + overflow)
        Bin lengths. For under- and overflow bins the adjacent bin sizes are
        repeated to have some half useful value.
    """
    bdiff = np.diff(bin_edges)
    if underflow:
    	bdiff = np.r_[bdiff[0], bdiff]
    if overflow:
    	bdiff = np.r_[bdiff, bdiff[-1]]
    return bdiff


def digitize_uni(x, bins, underflow=True, overflow=True, return_all=True):
    """A wrapper around ´´numpy.digitize`` that optionalizes under- and
    overflow bins. While for ``numpy.digitize`` they are always included
    (all samples in the underflow bin are assigned 0 and all samples in the
    overflow bin are assigned `n_bins`). Here, if under- or overflow bins are
    excluded, they are assigned -1 or excluded from the return array.

    Parameters
    ----------
    x : numpy.array, shape=(n_samples,)
        Sample

    bins : numpy.array, shape=(n_bins + 1)
        Bin edges.

    underflow, overflow : bool
        Whether or not to include under- and overflow bins.

    return_all : bool
        Whether to return the full sample.

    Returns
    -------
    digitzed_array : numpy.array, shape=(n_samples,), dtype=int
        Digitzed array.
    """
    idx = np.digitize(x, bins)
    selection = np.ones(len(idx), dtype=bool)
    if not overflow:
        selection[idx == len(bins)] = False
    if not underflow:
        selection[idx == 0] = False
        idx -= 1
    idx[~selection] = -1
    if return_all:
        return idx
    else:
        return idx[selection]


def bin_edges(X):
    X_sort = np.sort(X)
    return (X_sort[1:] + X_sort[:-1]) * 0.5


def equidistant_bins(X, Xmin, Xmax, n_bins, **kwargs):
    '''Partitions `n_bins` bins with equal lengths between `Xmin` and `Xmax`

    Parameters
    ----------
    X : numpy array
        Samples

    Xmin, Xmax : float
        Minimum and maximum X-value

    n_bins : int
        Number of bins.

    Returns
    -------
    bin_edges : numpy array, shape=(n_bins + 1)
        Bin edges.
    '''
    return np.linspace(Xmin, Xmax, n_bins - 1)


def equal_bins(X, Xmin, Xmax, n_bins, **kwargs):
    '''Partitions `n_bins` bins with equal contents between `Xmin` and `Xmax`

    Parameters
    ----------
    X : numpy array
        Samples

    Xmin, Xmax : float
        Minimum and maximum X-value

    n_bins : int
        Number of bins.

    Returns
    -------
    bin_edges : numpy array, shape=(n_bins + 1)
        Bin edges.
    '''
    edges = bin_edges(X[(X > Xmin) & (X < Xmax)])
    idx = np.linspace(0, len(edges) - 1, n_bins - 1).astype(int)
    return edges[idx]


def random_bins(X, Xmin, Xmax, n_bins, rnd, **kwargs):
    '''Partitions `n_bins` random bins between `Xmin` and `Xmax`

    Parameters
    ----------
    X : numpy array
        Samples

    Xmin, Xmax : float
        Minimum and maximum X-value

    n_bins : int
        Number of bins.

    rnd : numpy.random.RandomState
        Random State

    Returns
    -------
    bin_edges : numpy array, shape=(n_bins + 1)
        Bin edges.
    '''
    edges = np.sort(rnd.uniform(Xmin, Xmax, n_bins - 3))
    return np.r_[Xmin, edges, Xmax]


def random_equal_bins(X, Xmin, Xmax, n_bins, rnd, **kwargs):
    '''Partitions `n_bins` random bins between `Xmin` and `Xmax`, in this
    case the random bins are sampled from all possible bin_edges as given
    from the sample `X`.

    Parameters
    ----------
    X : numpy array
        Samples

    Xmin, Xmax : float
        Minimum and maximum X-value

    n_bins : int
        Number of bins.

    rnd : numpy.random.RandomState
        Random State

    Returns
    -------
    bin_edges : numpy array, shape=(n_bins + 1)
        Bin edges.
    '''
    edges = bin_edges(X[(X > Xmin) & (X < Xmax)])
    idx = np.sort(rnd.choice(len(edges) - 2, n_bins - 3,
                                   replace=True) + 1)
    return np.r_[Xmin, edges[idx], Xmax]