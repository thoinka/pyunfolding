import numpy as np
from .base import LikelihoodTerm


class OnlyPositive(LikelihoodTerm):
    """Supresses negative terms in :math:`f` as :math:`\sum_i \exp(-f_i / s)`, whereas
    :math:`s` is a smoothness term.
    """
    formula = ''

    def __init__(self, s=0.0, exclude_edges=True, *args, **kwargs):
        self.s = s
        self.exclude_edges = exclude_edges
        if self.s == 0.0:
            self.formula = r"\sum_i \infty * \Theta(f_i)"
        self.formula = r"\sum_i \exp\left(\frac{f_i}{s}\right)"

    def func(self, model, f, g):
        if self.exclude_edges:
            f = f[1:-1]
        if self.s == 0.0:
            return (f < 0.0).any() * np.finfo('float').max
        elif self.s == -0.0:
            return (f > 0.0).any() * np.finfo('float').max
        else:
            return np.sum(np.exp(-f / self.s))

    def grad(self, model, f, g):
        if self.s == 0.0 or self.s == -0.0:
            return np.zeros_like(f)
        else:
            output =  np.exp(-f / self.s) / self.s
            if self.exclude_edges:
                output[[0,-1]] = 0.0
            return output

    def hess(self, model, f, g):
        if self.s == 0.0 or self.s == -0.0:
            return np.zeros((len(f), len(f)))
        else:
            output = np.diag(np.exp(-f / self.s) / self.s ** 2)
            if self.exclude_edges:
                output[[0,-1],:] = 0.0
                output[:,[0,-1]] = 0.0
            return output