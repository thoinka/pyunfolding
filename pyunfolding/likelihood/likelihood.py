import numpy as np
from .. import solver


class Likelihood:
    """Base Likelihood Class.

    Attributes
    ----------
    llh : list(LikelihoodTerm)
        List of all terms summed in the likelihood.
    model : Model object
        model to evaluate.
    """

    def __init__(self, model, *args, **kwargs):
        self.llh = list(*args)
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def func(self, *args, **kwargs):
        return sum(L.func(self.model, *args, **kwargs)
                   for L in self.llh)

    def grad(self, *args, **kwargs):
        return sum(L.grad(self.model, *args, **kwargs)
                   for L in self.llh)

    def hess(self, *args, **kwargs):
        return sum(L.hess(self.model, *args, **kwargs)
                   for L in self.llh)

    def append(self, llh):
        self.llh.append(llh)

    def __add__(self, llh):
        self.append(llh)

    def __str__(self):
        return " + ".join([L.__str__() for L in self.llh])

    def solve(self, f0, X, solver_method='mcmc', **kwargs):
        '''Minimize Likelihood. Supported methods include:
        * 'mcmc': MCMC minimization
        * 'minimizer': Using scipy minimizer
        '''
        if solver_method == 'mcmc':
            self.solver = solver.MCMCSolver(self)
        elif solver_method == 'minimizer':
            self.solver = solver.Minimizer(self)
        else:
            raise NotImplementedError('Method not supported {}'.format(method))
        return self.solver.solve(f0, X, **kwargs)



class LikelihoodTerm:
    """Likelihood Term Base Class.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, model, f, g, *args, **kwargs):
        """Call method, simply calls :method:`func`

        Parameters
        ----------
        model : Model object
            model the likelihood refers to.
        f : numpy array
            truth vector
        g : numpy array
            proxy vector

        Returns
        -------
        output : float
            Likelihood value.
        """
        return self.func(model, f, g, *args, **kwargs)

    def func(self, model, f, g, *args, **kwargs):
        """Likelihood evaluation.

        Parameters
        ----------
        model : Model object
            model the likelihood refers to.
        f : numpy array
            truth vector
        g : numpy array
            proxy vector

        Returns
        -------
        output : float
            Likelihood value.
        """
        raise NotImplementedError("Function call needs to be implemented.")

    def grad(self, model, f, g, *args, **kwargs):
        """Likelihood gradient evaluation.

        Parameters
        ----------
        model : Model object
            model the likelihood refers to.
        f : numpy array
            truth vector
        g : numpy array
            proxy vector

        Returns
        -------
        output : numpy array, shape=(len(f))
            Likelihood gradient.
        """
        raise NotImplementedError("Gradient call needs to be implemented.")

    def hess(self, model, f, g, *args, **kwargs):
        """Likelihood Hessian evaluation.

        Parameters
        ----------
        model : Model object
            model the likelihood refers to.
        f : numpy array
            truth vector
        g : numpy array
            proxy vector

        Returns
        -------
        output : numpy array, shape=(len(f), len(f))
            Likelihood Hessian value.
        """
        raise NotImplementedError("Hessian call needs to be implemented.")

    def __str__(self):
        return "?"


class Poisson(LikelihoodTerm):
    """Poissonian likelihood term as :math:`\sum_i (g np.log(\lmabda(f)) - \lambda(f))`.
    """

    def func(self, model, f, g):
        g_est = model.predict(f)
        return -np.sum(g * np.log(g_est) - g_est)

    def grad(self, model, f, g):
        g_est = model.predict(f)
        return np.sum((1.0 - g / g_est) * model.grad(f).T, axis=1)

    def hess(self, model, f, g):
        g_est = model.predict(f)
        A = model.grad(f)
        return np.dot(A.T, np.dot(np.diag(g / g_est ** 2), A))


class LeastSquares(LikelihoodTerm):
    """Least Squares likelihood term as :math:`\frac{1}{2}(g - \lambda(f))^\top (g - \lambda(f))`
    """

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

    def __str__(self):
        return "(g - Af)^T (g - Af) / 2"


class OnlyPositive(LikelihoodTerm):
    """Supresses negative terms in :math:`f` as :math:`\sum_i \exp(-f_i / s)`, whereas
    :math:`s` is a smoothness term.
    """

    def __init__(self, s=0.0, exclude_edges=True, *args, **kwargs):
        self.s = s
        self.exclude_edges = exclude_edges

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

    def __str__(self):
        if self.s == 0.0:
            return "Sum_i inf * Theta(f_i)"
        return "Sum_i np.exp(s * f_i)"


class TikhonovRegularization(LikelihoodTerm):

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

    def __str__(self):
        return "tau * |C f|^2 / 2"

class TikhonovRegularization2(LikelihoodTerm):

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
        self.C = C
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
        if self.exclude_edges:
            self.sel = slice(1, -1, None)
            self.C = self.C[self.sel, self.sel]
        else:
            self.sel = slice(None, None, None)
        self.initialized = True

    def func(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        return 0.5 * self.tau * np.dot(f[self.sel].T, np.dot(self.C, f[self.sel]))

    def grad(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        output = np.zeros(len(f))
        output[self.sel] = self.tau * np.dot(self.C,
                                             f[self.sel])
        return output

    def hess(self, model, f, g):
        if not self.initialized:
            self.init(model.A.shape[1])
        output = np.zeros((len(f), len(f)))
        output[self.sel, self.sel] = self.C
        return output