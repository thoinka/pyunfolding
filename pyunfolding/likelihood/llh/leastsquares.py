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
        return (-np.dot(g.T, A) - np.dot(A.T, g)\
               + np.dot(np.dot(A.T, A), f) + np.dot(f.T, np.dot(A.T, A))) * 0.5

    def hess(self, model, f, g):
        A = model.grad(f)
        H = model.hess(f)
        g_est = model.predict(f)
        return np.dot(A.T, A) #- np.dot((g - g_est).T, H)