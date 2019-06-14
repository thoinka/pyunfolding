import numpy as np
from .base import LikelihoodTerm


class Poisson(LikelihoodTerm):
    """Poissonian likelihood term as :math:`\sum_i (g np.log(\lmabda(f)) - \lambda(f))`.
    """
    formula = r"-\sum_i (g_i \log (\mathrm{A}\mathbf{f})_i - (\mathrm{A}\mathbf{f})_i"

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def func(self, model, f, g):
        g_est = model.predict(f)
        if (g_est < 0.0).any():
            return np.finfo(float).max
        return -np.sum(g * np.log(g_est + self.epsilon) - g_est)

    def grad(self, model, f, g):
        g_est = model.predict(f)
        return np.sum((1.0 - g / g_est) * model.grad(f).T, axis=1)

    def hess(self, model, f, g):
        g_est = model.predict(f)
        A = model.grad(f)
        return np.dot(A.T, np.dot(np.diag(g / g_est ** 2), A))