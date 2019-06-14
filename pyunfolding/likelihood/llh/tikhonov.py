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
        if self.c_name == "diff1":
            c_gen = diff1_matrix
        elif self.c_name == 'diff2':
            c_gen = diff2_matrix

        if self.exclude_edges:
            self.C = c_gen(n - 2)
            self.sel = slice(1, -1, None)
        else:
            self.C = c_gen(n)
            self.sel = slice(None, None, None)

        self.initialized = True

    def func(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f
        return 0.5 * self.tau * np.sum(np.dot(self.C, f[self.sel]) ** 2)

    def grad(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f
        output = np.zeros(len(f))
        output[self.sel] = self.tau * np.dot(np.dot(self.C.T, self.C),
                                             f[self.sel])
        return output

    def hess(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f
        output = np.zeros((len(f), len(f)))
        output[self.sel, self.sel] = self.tau * np.dot(self.C.T, self.C)
        return output


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
        self.initialized = False

    def init(self, n):
        """Initialize regularization matrix.

        Parameters
        ----------
        n : int
            Shape of matrix.
        """
        if self.c_name == "diff1":
            c_gen = diff1_matrix
        elif self.c_name == 'diff2':
            c_gen = diff2_matrix

        if self.exclude_edges:
            self.C = c_gen(n - 2)
            self.sel = slice(1, -1, None)
        else:
            self.C = c_gen(n)
            self.sel = slice(None, None, None)

        self.initialized = True

    def func(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f
        logf = np.log(f[self.sel])
        return logf.T @ self.C.T @ self.C @ logf * 0.5

    def grad(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f
        return self.C.T @ self.C @ np.log(f[self.sel]) / f[self.sel]

    def hess(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        if self.with_acceptance:
            f = np.diag(model.acceptance) @ f
        G = self.C @ self.C.T
        return -0.5 * (np.diag((G.T + G) @ np.log(f[self.sel]))
                       - (G.T + G)) / (f[self.sel] * f[self.sel].reshape(-1,1))