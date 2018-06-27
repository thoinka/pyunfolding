import numpy as np
from pyunfolding import binning
from pyunfolding import model
from pyunfolding import likelihood as llh
from pyunfolding import solver
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from pyunfolding.plot import corner_plot

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
        s = "Minimization Result:\n"
        s += "--------------------\n"
        s += "x:       {}\n".format(self.x)
        s += "fun:     {}\n".format(self.fun)
        s += "jac:     {}\n".format(self.jac)
        s += "n_iter:  {}\n".format(self.n_iter)
        s += "success: {}\n".format(self.success)
        return s


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
                                      n_iter=i, success=True), np.array(x)
    return MinimizationResult(x=x[-1], fun=fun(x[-1]), jac=grad(x[-1]),
                              n_iter=max_iter, success=False), np.array(x)


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
                                      n_iter=i, success=True), np.array(x)
    return MinimizationResult(x=x[-1], fun=fun(x[-1]), jac=grad(x[-1]),
                              n_iter=max_iter, success=False), np.array(x)


def num_gradient(fun, x0, eps=1e-4):
    """ Evaluates the gradient of fun at x0 numerically.

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
    gradient = []
    f0 = fun(x0)
    T = np.tile(x0, (len(x0), 1)) + np.eye(len(x0)) * eps
    for i in range(len(x0)):
        gradient.append((fun(T[i, :]) - f0) / eps)
    return np.array(gradient)


def paraboloid_fit(x, f):
    n_dim = x.shape[1]
    def paraboloid(x, params):
        return np.sum(x.T * np.matmul(params, x.T), axis=0)
    F = lambda p: np.sum((f - paraboloid(x, p.reshape(n_dim, n_dim))) ** 2)
    result = minimize(F, x0=np.eye(n_dim).flatten(), tol=1e-10)
    print(result)
    return result.x.reshape(n_dim, n_dim)


if __name__ == "__main__":
    fuzziness = 0.5

    np.random.seed(1337)

    test_y = (np.random.rand(50000) + 0.2) ** 1.5
    test_samples = np.random.normal(loc=test_y, scale=fuzziness)
    test_samples_2 = np.random.normal(loc=test_samples, scale=fuzziness)
    test_samples = np.vstack((test_samples, test_samples_2)).T

    np.random.seed(1338)

    train_y = (np.random.rand(50000) + 0.2) ** 1.5
    train_samples = np.random.normal(loc=train_y, scale=fuzziness)
    train_samples_2 = np.random.normal(loc=train_samples, scale=fuzziness)
    train_samples = np.vstack((train_samples, train_samples_2)).T

    b = 400

    binning_tgt = binning.GridBinning([10])
    binning_tgt.fit(test_y)
    bin_ens = [binning.TreeBinning(50, min_samples_leaf=b) for _ in range(3)]
    binning_obs = binning.EnsembleBinning(*bin_ens)
    #binning_obs = binning.TreeBinning(100, min_samples_leaf=b)
    f_class = binning_tgt.digitize(test_y)
    f_true = binning_tgt.histogram(test_y)
    binning_obs.fit(test_samples, f_class)

    m = model.Model(binning_obs, binning_tgt)
    m.fit(test_samples, test_y)

    likelihood = llh.Likelihood(m)
    likelihood.append(llh.Poisson())
    #likelihood.append(llh.TikhonovRegularization(1e-1))

    x0 = m.x0(train_samples, pos=True, tau=1e-3)
    sol = solver.NewtonMinimizer(likelihood)
    result = sol.solve(x0, train_samples)
    g = binning_obs.histogram(train_samples)

    print("x0", x0)
    print("res", result.f)

    print(num_gradient(lambda f: likelihood(f, g), result.f))
    print(result)
    
    
    sol = solver.MCMCSolver(likelihood)
    result = sol.solve(result.f, train_samples, verbose=True,
                       pass_samples=True, n_iter=5000000,
                       error_method="central", value_method="median")
    g = binning_obs.histogram(train_samples)
    
    corner_plot(result.samples_x[::100,4:7])
    plt.show()

    print(num_gradient(lambda f: likelihood(f, g), result.f))
    print(result)

    plt.plot(binning_tgt.histogram(train_y), drawstyle="steps-mid", color="k")
    plt.errorbar(np.arange(len(result.f))[1:-1], result.f[1:-1],
                 result.f_err[:,1:-1], 0.5 * np.ones(len(result.f[1:-1])),
                 linestyle="", color="k")
    plt.show()
