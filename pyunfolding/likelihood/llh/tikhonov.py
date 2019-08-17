import numpy as np
from .base import LikelihoodTerm
from ...utils.matrices import diff0_matrix, diff1_matrix, diff2_matrix


class Tikhonov(LikelihoodTerm):
    r"""Tikhonov Regularization :math:`\mathcal{L}_\mathrm{tikhonov} = \frac{1}{2}(\Gamma \cdot f)^\top (\Gamma \cdot f)`.

    Parameters
    ----------
    tau : float or numpy.array, optional, (default=1.0)
        Regularization strength. The higher tau is chosen, the stronger the
        regularization. Stronger regularization will enforce a smoother
        solution, but also increase bin-to-bin correlations. Tau should be
        chosen as a compromise between these two aspects.
        When this is chosen to be a vector the same length as `f`, then a
        different regularization strength is used for each bin.
    C : str, optional, (default="diff2")
        Kind of regularization matrix. Possible values are:
            * ``diff0``: Identity matrix. This is equivalent to what is often
                         referred to as L2 regularization. This will prefer
                         small solutions.
            * ``diff1``: Finite difference differentiation. This will prefer
                         constant solutions.
            * ``diff2``: Second finite diffference differentiation. This will
                         prefer flat solutions.
    exclude_edges : bool, optional, (default=True)
        Whether or not to include the over- and underflow bins. It is usually
        a good idea to leave this to `True`, as in pretty much all cases
        whatever assumptions you can make about the shape of the solution
        do not apply to the under- and overflow bins.
    with_acceptance : bool, optional, (default=False)
        Whether or not to apply the Tikhonov regularization to the acceptance
        corrected spectrum. Usually you expect the solution to be smooth only
        after the acceptance correction, so this can often be a good choice,
        provided an acceptance correction is available.
    log : bool, optional, (default=False)
        If true, the logarithm of `f` is used in the regularization term.

    Attributes
    ----------
    C : numpy array, shape=(len(f), len(f))
        Regularization matrix
    c_name : str
        Denomer of the regularization matrix.
    exclude_edges : bool
        Whether or not to include the over- and underflow bins for the
        regularization.
    initialized : bool
        Whether the object has been initialized or not.
    tau : float or numpy.array
        Regularization strength.
    C, CT_C: numpy.array
        Regularization matrix and square of regularization matrix.
    log : bool, optional, (default=False)
        If true, the logarithm of `f` is used in the regularization term.

    Notes
    -----
    This class requires to be initialized -- this happens automatically when
    it is evaluated the first time. However, sometimes you may want to change
    some parameters of this term after the model has been fitted and whatnot.
    In that case you need to run ``init`` once again.
    """
    formula = r"\frac{\tau}{2}  |\mathrm{C}\mathbf{f}|^2"

    def __init__(self,
                 tau=1.0,
                 C="diff2",
                 exclude_edges=True,
                 with_acceptance=False,
                 log=False,
                 **params):
        self.c_name = C
        self.exclude_edges = exclude_edges
        self.with_acceptance = with_acceptance
        self.tau = tau
        self.log = log
        self.initialized = False

    def init(self, n):
        """Initialize regularization matrix.

        Parameters
        ----------
        n : int
            Shape of matrix.
        """
        padding = 1 if self.exclude_edges else 0
        if self.c_name == "diff1":
            self.C = diff1_matrix(n, padding)
        elif self.c_name == 'diff2':
            self.C = diff2_matrix(n, padding)
        elif self.c_name == 'diff0':
            self.C = diff0_matrix(n, padding)

        tau_vec = self.tau * np.ones(n)
        self.CT_C = np.diag(tau_vec) @ self.C.T @ self.C
        self.initialized = True

    def func(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f
        if self.log:
            if (f < 0.0).any():
                return np.finfo('float').max
            logf = np.log(f + self.epsilon)
            return logf.T @ self.CT_C @ logf * 0.5
        return 0.5 * f.T @ self.CT_C @ f

    def grad(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f
        if self.log:
            if (f < 0.0).any():
                return np.ones(len(f))
            return self.CT_C @ np.log(f) / f
        return np.dot(self.CT_C, f)
        
    def hess(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f
        if self.log:
            if (f < 0.0).any():
                return np.zeros((len(f), len(f)))
            G = self.CT_C
            return -0.5 * (np.diag((G.T + G) @ np.log(f))
                           - (G.T + G)) / (f * f.reshape(-1,1))
        return self.CT_C