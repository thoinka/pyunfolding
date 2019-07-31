import numpy as np
from .base import LikelihoodTerm
from ...utils import diff1_matrix, diff2_matrix


class Tikhonov(LikelihoodTerm):
    """Tikhonov Regularization :math:`\mathcal{L}_\mathrm{tikhonov} = \frac{1}{2}(\Gamma \cdot f)^\top (\Gamma \cdot f)`.

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
    sel : numpy array, shape=(len(f),), dtype=bool
        Boolean mask that leaves out the edges in case exclude_bins is True.
    tau : float
        Regularization strength.
    """
    formula = r"\frac{\tau}{2}  |\mathrm{C}\mathbf{f}|^2"

    def __init__(self,
                 tau=1.0,
                 C="diff2",
                 exclude_edges=True,
                 with_acceptance=False,
                 **params):
        """Initalization method

        Parameters
        ----------
        tau : float, optional, default=1.0
            Regularization strength.
        C : str, optional, default="diff2"
            Denomer of regularization strength.
        exclude_edges : bool, optional, default=True
            Whether or not to include the over- and underflow bins.
        """
        self.c_name = C
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
        padding = 1 if self.exclude_edges else 0
        if self.c_name == "diff1":
            self.C = diff1_matrix(n, padding)
        elif self.c_name == 'diff2':
            self.C = diff2_matrix(n, padding)

        tau_vec = self.tau * np.ones(n)
        self.CT_C = np.diag(tau_vec) @ self.C.T @ self.C
        self.initialized = True

    def func(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f
        return 0.5 * f.T @ self.CT_C @ f

    def grad(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f
        return np.dot(self.CT_C, f)
        
    def hess(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f
        return self.CT_C


class TikhonovLog(LikelihoodTerm):
    """Tikhonov Regularization :math:`\mathcal{L}_\mathrm{tikhonov} = \frac{1}{2}(\Gamma \cdot f)^\top (\Gamma \cdot f)`.

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
    sel : numpy array, shape=(len(f),), dtype=bool
        Boolean mask that leaves out the edges in case exclude_bins is True.
    tau : float
        Regularization strength.
    """
    formula = r"\frac{\tau}{2}  |\mathrm{C}\log\mathbf{f}|^2"

    def __init__(self,
                 tau=1.0,
                 C="diff2",
                 exclude_edges=True,
                 with_acceptance=False,
                 epsilon=1e-3,
                 **params):
        """Initalization method

        Parameters
        ----------
        tau : float, optional, default=1.0
            Regularization strength.
        C : str, optional, default="diff2"
            Kind of acceptance. Valid values:
            * "diff2": Second derivative
            * "diff1": First derivative
        exclude_edges : bool, optional, default=True
            Whether or not to include the over- and underflow bins.
        with_acceptance : bool, optional, default=False
            Whether to regulize the acceptance-corrected result vector or not.
        """
        self.c_name = C
        self.exclude_edges = exclude_edges
        self.with_acceptance = with_acceptance
        self.tau = tau
        self.epsilon = epsilon
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

        tau_vec = self.tau * np.ones(n)
        self.CT_C = np.diag(tau_vec) @ self.C.T @ self.C
        self.initialized = True

    def func(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f
        if (f < 0.0).any():
            return np.finfo('float').max
        logf = np.log(f[self.sel] + self.epsilon)
        return logf.T @ self.CT_C @ logf * 0.5

    def grad(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f
        if (f < 0.0).any():
            return np.ones(len(f))
        return self.CT_C @ np.log(f[self.sel]) / f[self.sel]

    def hess(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f
        if (f < 0.0).any():
            return np.zeros((len(f), len(f)))
        G = self.CT_C
        return -0.5 * (np.diag((G.T + G) @ np.log(f[self.sel]))
                       - (G.T + G)) / (f[self.sel] * f[self.sel].reshape(-1,1))