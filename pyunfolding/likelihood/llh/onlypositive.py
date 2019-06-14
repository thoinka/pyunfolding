import numpy as np
from .base import LikelihoodTerm


def safe_exp(x, clip=100.0):
    return np.exp(np.clip(x, -np.inf, clip))


class OnlyPositive(LikelihoodTerm):
    """Supresses negative terms in :math:`f` as :math:`\sum_i \frac{a}{1+\exp(f_i / s)}`, whereas
    :math:`s` is a smoothness variable, and a is the penalty for negative values.
    """
    formula = ''

    def __init__(self, a=np.finfo(float).max, s=0.0, exclude_edges=True, *args, **kwargs):
        self.s = s
        self.a = a
        self.exclude_edges = exclude_edges
        if self.s == 0.0:
            self.formula = r"\sum_i \infty * \Theta(f_i)"
        self.formula = r"\sum_i \frac{a}{1+\exp(f_i / s)}"

    def func(self, model, f, g):
        if self.exclude_edges:
            f = f[1:-1]
        if self.s == 0.0:
            return (f < 0.0).any() * self.a
        elif self.s == -0.0:
            return (f > 0.0).any() * self.a
        else:
            return np.sum(self.a / (1.0 + safe_exp(f / self.s)))

    def grad(self, model, f, g):
        if self.s == 0.0 or self.s == -0.0:
            return np.zeros_like(f)
        else:
            exp_term = safe_exp(f / self.s)
            output = self.a * exp_term / ((1.0 + exp_term) ** 2 * np.eself.s)
            if self.exclude_edges:
                output[[0,-1]] = 0.0
            return output

    def hess(self, model, f, g):
        if self.s == 0.0 or self.s == -0.0:
            return np.zeros((len(f), len(f)))
        else:
            exp_term = safe_exp(f / self.s)
            output = self.a * np.diag(exp_term * (exp_term - 1.0)
                                      / (1 + exp_term) ** 3 / self.s)
            if self.exclude_edges:
                output[[0,-1],:] = 0.0
                output[:,[0,-1]] = 0.0
            return output