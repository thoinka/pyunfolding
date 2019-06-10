import numpy as np
from .base import LikelihoodTerm


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
        self.tau = tau
        self.initialized = False

    def init(self, n):
        """Initialize regularization matrix.

        Parameters
        ----------
        n : int
            Shape of matrix.
        """
        if self.c_name == "diff2":
            if self.exclude_edges:
                self.C = 2.0 * np.eye(n - 2) - np.roll(np.eye(n - 2), 1)\
                                             - np.roll(np.eye(n - 2), -1)
                self.sel = slice(1, -1, None)
            else:
                self.C = 2.0 * np.eye(n) - np.roll(np.eye(n), 1)\
                                         - np.roll(np.eye(n), -1)
                self.sel = slice(None, None, None)
        self.initialized = True

    def func(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        return 0.5 * self.tau * np.sum(np.dot(self.C, f[self.sel]) ** 2)

    def grad(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        output = np.zeros(len(f))
        output[self.sel] = self.tau * np.dot(np.dot(self.C.T, self.C),
                                             f[self.sel])
        return output

    def hess(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        output = np.zeros((len(f), len(f)))
        output[self.sel, self.sel] = self.tau * np.dot(self.C.T, self.C)
        return output