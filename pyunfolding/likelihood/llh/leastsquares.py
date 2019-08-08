import numpy as np
from .base import LikelihoodTerm

class LeastSquares(LikelihoodTerm):
    """Least Squares likelihood term as :math:`\frac{1}{2}(g - \lambda(f))^\top (g - \lambda(f))`
    """
    formula = r"\frac{1}{2}(\mathbf{g} - \mathrm{A}\mathbf{f})^\top (\mathbf{g} - \mathrm{A}\mathbf{f})"

    def func(self, model, f, g):
        g_est = model.predict(f)
        return 0.5 * np.dot((g - g_est).T, (g - g_est))

    def grad(self, model, f, g):
        g_est = model.predict(f)
        A = model.grad(f)
        return -(g - g_est).T @ A

    def hess(self, model, f, g):
        A = model.grad(f)
        H = model.hess(f)
        g_est = model.predict(f)
        return np.dot(A.T, A)


class WeightedLeastSquares(LikelihoodTerm):
    """Least Squares likelihood term as :math:`\frac{1}{2}(g - \lambda(f))^\top (g - \lambda(f))`
    """
    formula = r"\frac{1}{2}(\mathbf{g} - \mathrm{A}\mathbf{f})^\top (\mathbf{g} - \mathrm{A}\mathbf{f})"

    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon

    def func(self, model, f, g):
        g_est = model.predict(f)
        dg = (g - g_est)
        icov = np.diag(1.0 / (g + self.epsilon))
        return 0.5 * dg.T @ icov @ dg

    def grad(self, model, f, g):
        g_est = model.predict(f)
        A = model.grad(f)
        icov = np.diag(1.0 / (g + self.epsilon))
        return -(g - g_est).T @ icov @ A

    def hess(self, model, f, g):
        A = model.grad(f)
        H = model.hess(f)
        icov = np.diag(1.0 / (g + self.epsilon))
        g_est = model.predict(f)
        return A.T @ icov @ A