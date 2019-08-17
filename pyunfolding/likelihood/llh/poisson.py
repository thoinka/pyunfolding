import numpy as np
from .base import LikelihoodTerm
from .onlypositive import safe_exp


class Poisson(LikelihoodTerm):
    r"""Poissonian likelihood term as

    .. math::
        \sum_i (g_i np.log(\lambda(f)_i) - \lambda(f)_i) + b \exp (a \lambda(f)_i)

    Latter term is a regularization term to smooth out the hard edge the
    unaltered likelihood term hits when negative elements emerge in
    :math:`\lambda(\mathbf{f})`.

    Parameters
    ----------
    epsilon : `float`, optional (default=1e-8)
        Offset added to :math:`\lambda(f)` in order to avoid numerical problem
        once it hits very small values.

    a, b : `float`, optional (default=1e3)
        Parameters of the additional regularization term. Set `a` to 0 to
        remove the term altogether.

    Attributes
    ----------
    epsilon : `float`, optional (default=1e-8)
        Offset added to :math:`\lambda(f)` in order to avoid numerical problem
        once it hits very small values.
        
    a, b : `float`, optional (default=1e3)
        Parameters of the additional regularization term. Set `a` to 0 to
        remove the term altogether.
    """
    formula = r"-\sum_i (g_i \log (\mathrm{A}\mathbf{f})_i - (\mathrm{A}\mathbf{f})_i"

    def __init__(self, epsilon=1e-8, a=1.0e3, b=1.0e3):
        self.epsilon = epsilon
        self.a = a
        self.b = b

    def func(self, model, f, g):
        # Some additional notes to help understand this:
        # Sometimes it happens that g_est contains zero or negative entries. In
        # that case the Likelihood isn't defined at all. To still return a
        # value, this is checked and the maximum float value is returned.
        # However, curiously, the gradient is always defined, so an additional
        # term is added to the Likelihood, i.e. something that allows the best
        # fit to move away from negative values of g_est. Then the minimum
        # that is found in this way is a proper minimum with a vanishing
        # gradient. It of course introduces a bias into the fit, but given
        # appropriate values of a and b, this is considered a negligible issue
        # and worth the agreeing best fit values for minimizers only looking
        # the gradient and other minimizers.
        g_est = model.predict(f)
        if (g_est < 0.0).any():
            return np.finfo(float).max
        bailout_grad = self.b * np.sum(safe_exp(-self.a * g_est))
        return -np.sum(g * np.log(g_est + self.epsilon) - g_est) + bailout_grad

    def grad(self, model, f, g):
        g_est = model.predict(f)
        bailout_grad = -self.a * self.b * model.grad(f).T @ np.exp(-self.a * g_est)
        return np.sum((1.0 - g / (g_est + self.epsilon)) * model.grad(f).T,
                       axis=1) + bailout_grad

    def hess(self, model, f, g):
        g_est = model.predict(f)
        A = model.grad(f)
        bailout_grad = self.a ** 2 * self.b * A @ A.T @ np.exp(-self.a * g_est)
        return np.dot(A.T, np.dot(np.diag(g / (g_est + self.epsilon) ** 2), A))