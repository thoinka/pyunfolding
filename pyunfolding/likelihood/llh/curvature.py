import numpy as np
from .base import LikelihoodTerm


class Curvature(LikelihoodTerm):
    r"""Curvature Regularization. The curvature of a function curve is defined
    as

    .. math::
        \kappa^2 = \frac{{y''}^2}{(1 + {y'}^2)^3}

    This regularization term approximates :math:`y''` and :math:`y''` using
    finite difference methods. This kind of regularization is a good choice
    in case the expected unfolding result has a range of different slopes
    in different regions. For small first derivatives it is roughly equal
    to what `Tikhonov`-regularization does.

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

    def init(self, n):
        """Initialize regularization matrix.

        Parameters
        ----------
        n : int
            Shape of matrix.
        """
        self._D1 = np.eye(n - 2, n, 2) - np.eye(n - 2, n)
        self._D2 = (2 * np.eye(n - 2, n, 1)
                   - np.eye(n - 2, n, 2)
                   - np.eye(n - 2, n))
        self.initialized = True

    def func(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f

        d1 = self._D1 @ f
        d2 = self._D2 @ f
        kappa_squared = d2 ** 2 / (1.0 + d1 ** 2) ** 3.0

        return -0.5 * self.tau * np.sum(kappa_squared)