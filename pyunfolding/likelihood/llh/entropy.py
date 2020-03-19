import numpy as np
from .base import LikelihoodTerm


class Entropy(LikelihoodTerm):
    r"""Entropy regularization. This regularization aims to maximize the
    entropy involved with the unfolding result. Generally, a minimum entropy
    result is that the result is a 'delta'-peak in the sense that all events
    are found in one single bin. A minimum entropy result is then a completely
    uniform one. Mathematically it is formulated as:

    .. math::
        S = -\tau \sum_i p_i \log p_i = -\tau \sum_i \frac{f_i}{\sum_j f_j} \log \frac{f_i}{\sum_j f_j}

    As with other regularizaations, the constant `tau` aims to either
    increase or decrease the contribution of this term. Negative `tau`s may
    actually be useful in this case in order to prefer more 'monochromatic'
    spectra.    

    Parameters
    ----------
    tau : float
        Regularization strength

    exclude_edges : bool, optional (default=True)
        Excludes the first and last bin from the curvature calculation.

    with_acceptance : bool, optional (default=False)
        Whether or not to include the acceptance correction factor from the
        model before calculating the curvature.

    Attributes
    ----------
    exclude_edges : bool
        Whether or not to include the over- and underflow bins for the
        regularization.

    initialized : bool
        Whether the object has been initialized or not.

    with_acceptance : bool
        Whether or not to include the acceptance correction factor from the
        model before calculating the curvature.

    tau : float or numpy.array
        Regularization strength.
    """
    formula = r"\mathrm{Curvature}"

    def __init__(self,
                 tau,
                 exclude_edges=True,
                 with_acceptance=False,
                 **params):
        self.exclude_edges = exclude_edges
        self.with_acceptance = with_acceptance
        self.tau = tau
        self.initialized = False

    def func(self, model, f, g):
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f

        p = f / np.sum(f)

        return -self.tau * p * np.log(p)

    def grad(self, model, f, g):
        