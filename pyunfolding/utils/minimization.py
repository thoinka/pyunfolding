import numpy as np
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt
from collections import namedtuple


__all__ = ["newton_minimizer",
           "num_gradient",
           "momentum_minimizer",
           "adam_minimizer",
           "adadelta_minimizer",
           "rmsprop_minimizer"]


class MinimizationResult:

    def __init__(self, x, fun, jac, n_iter, success):
        self.x = x
        self.fun = fun
        self.jac = jac
        self.n_iter = n_iter
        self.success = success

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s =  "Minimization Result\n"
        s += "-------------------\n"
        s += "x:       {}\n".format(self.x)
        s += "fun:     {}\n".format(self.fun)
        s += "jac:     {}\n".format(self.jac)
        s += "n_iter:  {}\n".format(self.n_iter)
        s += "success: {}\n".format(self.success)
        return s


def newton_minimizer(fun,
                     grad,
                     hess,
                     x0,
                     max_iter=1000,
                     tol=1e-8,
                     alpha=1e-3,
                     lr=1.0):
    """Newton Minimizer.

    Parameters
    ----------
    fun : function
        Function to be minimized.
    grad : function
        Jacobian of the fun.
    hess : function
        Hessian of the function.
    x0 : float or numpy.array
        Start values
    max_iter : int, optional
        Maximum number of iterations.
    tol : float
        Relative tolerance.
    alpha : float
        Regularization factor.
    lr : float
        Learning rate.
    """
    x = [x0]
    if grad is None:
        def grad(x): return num_gradient(fun, x)

    if hess is None:
        def hess(x): return num_gradient(grad, x)

    for i in range(max_iter):
        g = grad(x[-1])
        H = hess(x[-1])
        x += [x[-1] - lr * np.linalg.pinv(H + alpha * np.eye(len(H))) @ g]
        if np.linalg.norm(g) < tol:
            return MinimizationResult(x=x[-1], fun=fun(x[-1]), jac=grad(x[-1]),
                                      n_iter=i, success=True)
    return MinimizationResult(x=x[-1], fun=fun(x[-1]), jac=grad(x[-1]),
                              n_iter=max_iter, success=False)


def momentum_minimizer(fun,
                       grad,
                       x0,
                       max_iter=1000,
                       tol=1e-8,
                       alpha=0.03,
                       lr=0.1):
    """Momentum Minimizer.

    Parameters
    ----------
    fun : function
        Function to be minimized.
    grad : function
        Jacobian of the fun.
    x0 : float or numpy.array
        Start values
    max_iter : int, optional
        Maximum number of iterations.
    tol : float
        Relative tolerance.
    alpha : float
        Momentum coefficient.
    lr : float
        Learning rate.

    Returns
    -------
    result : MinimizationResult
             The results.
    """
    x = [x0]
    delta_x = 0.0
    if grad is None:
        def grad(x): return num_gradient(fun, x)

    for i in range(max_iter):
        g = grad(x[-1])
        x += [x[-1] - lr * g - alpha * delta_x]
        delta_x = x[-1] - x[-2]
        if np.linalg.norm(g) < tol:
            return MinimizationResult(x=x[-1], fun=fun(x[-1]), jac=grad(x[-1]),
                                      n_iter=i, success=True)
    return MinimizationResult(x=x[-1], fun=fun(x[-1]), jac=grad(x[-1]),
                              n_iter=max_iter, success=False)


def rmsprop_minimizer(fun,
                      grad,
                      x0,
                      max_iter=1000,
                      tol=1e-8,
                      gamma=0.99,
                      mu=0.0,
                      lr=0.1):
    """RMSProp Minimizer.

    Parameters
    ----------
    fun : function
        Function to be minimized.
    grad : function
        Jacobian of the fun.
    x0 : float or numpy.array
        Start values
    max_iter : int, optional
        Maximum number of iterations.
    tol : float
        Relative tolerance.
    gamma : float
        Forgetting factor.
    mu : float
        Additional momentum.
    lr : float
        Learning rate.

    Returns
    -------
    result : MinimizationResult
             The results.
    """
    x = [x0]
    v = 0.0
    m = 0.0
    if grad is None:
        def grad(x): return num_gradient(fun, x)

    for i in range(max_iter):
        g = grad(x[-1])
        v = gamma * v + (1.0 - gamma) * g ** 2
        m = mu * m + lr * g / (np.sqrt(v) + 1e-8)
        x += [x[-1] - m]
        if np.linalg.norm(g) < tol:
            return MinimizationResult(x=x[-1], fun=fun(x[-1]), jac=grad(x[-1]),
                                      n_iter=i, success=True)
    return MinimizationResult(x=x[-1], fun=fun(x[-1]), jac=grad(x[-1]),
                              n_iter=max_iter, success=False)


def adam_minimizer(fun,
                   grad,
                   x0,
                   max_iter=1000,
                   tol=1e-8,
                   beta1=0.8,
                   beta2=0.99,
                   lr=0.01):
    """ RMSProp Minimizer.

    Parameters
    ----------
    fun : function
        Function to be minimized.
    grad : function or None
        Jacobian of the fun. If None, Jacobian is evaluated numerically.
    x0 : float or numpy.array
        Start values
    tol : float
        Relative tolerance.
    beta1 : float
        First forgetting factor.
    beta2 : float
        Second forgetting factor.
    lr : float
        Learning rate.

    Returns
    -------
    result : MinimizationResult
             The results.
    """
    x = [x0]
    m = 0.0
    v = 0.0
    if grad is None:
        def grad(x): return num_gradient(fun, x)

    for i in range(max_iter):
        g = grad(x[-1])
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g ** 2
        m_ = m / (1.0 - beta1 ** (i + 1))
        v_ = v / (1.0 - beta2 ** (i + 1))
        x += [x[-1] - lr * m_ / (np.sqrt(v_) + 1e-8)]
        if np.linalg.norm(g) < tol:
            return MinimizationResult(x=x[-1], fun=fun(x[-1]), jac=grad(x[-1]),
                                      n_iter=i, success=True)
    return MinimizationResult(x=x[-1], fun=fun(x[-1]), jac=grad(x[-1]),
                              n_iter=max_iter, success=False)


def adadelta_minimizer(fun,
                       grad,
                       x0,
                       max_iter=10000,
                       tol=1e-8,
                       gamma=0.999):
    """ AdaDelta Minimizer.

    Parameters
    ----------
    fun : function
        Function to be minimized.
    grad : function or None
        Jacobian of the fun. If None, Jacobian is evaluated numerically.
    x0 : float or numpy.array
        Start values
    tol : float
        Relative tolerance.
    gamma : float
        Forgetting factor.

    Returns
    -------
    result : MinimizationResult
        The results.
    """
    x = [x0]
    w = 1.0
    v = 0.0
    if grad is None:
        def grad(x): return num_gradient(fun, x)

    for i in range(max_iter):
        g = grad(x[-1])
        v = gamma * v + (1.0 - gamma) * g ** 2
        delta_theta = -w / np.sqrt(v + 1e-8) * g
        w = gamma * w + (1.0 - gamma) * delta_theta
        x += [x[-1] + delta_theta]
        if np.linalg.norm(g) < tol:
            return MinimizationResult(x=x[-1], fun=fun(x[-1]), jac=grad(x[-1]),
                                      n_iter=i, success=True)
    return MinimizationResult(x=x[-1], fun=fun(x[-1]), jac=grad(x[-1]),
                              n_iter=max_iter, success=False)


def num_gradient(fun, x0, eps=1e-6):
    """Evaluates the gradient of fun at x0 numerically.

    Parameters
    ----------
    fun : function
        Function to evaluate the gradient for.
    x0 : numpy.array
        Point at which to evaluate the gradient for.
    eps : float
        step size.

    Returns
    -------
    gradient : numpy.array
        The gradient.
    """
    f0 = fun(x0)
    gradient = np.zeros((len(x0), *f0.shape))
    T = np.tile(x0, (len(x0), 1)) + np.eye(len(x0)) * eps
    for i in range(len(x0)):
        gradient[i] = (fun(T[i,:]) - f0) / eps
    return gradient