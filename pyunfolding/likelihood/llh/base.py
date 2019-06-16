import numpy as np
from .. import solution
from ...utils import num_gradient


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
        s = " + ".join([L.__str__() for L in self.llh])
        return s

    def __repr__(self):
        return self.__str__()

    def solve(self, f0, X, solver_method='mcmc', **kwargs):
        '''Minimize Likelihood. Supported methods include:
        * 'mcmc': MCMC minimization
        * 'minimizer': Using scipy minimizer
        '''
        if solver_method == 'mcmc':
            self.solution = solution.MCMC(self)
        elif solver_method == 'minimizer':
            self.solution = solution.Minimizer(self)
        elif solver_method == 'newton':
            self.solution = solution.NewtonMinimizer(self)
        else:
            raise NotImplementedError('Method not supported {}'.format(method))
        return self.solution.solve(f0, X, **kwargs)


class LikelihoodTerm:
    """Likelihood Term Base Class.
    """
    formula = ''
    
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
        return num_gradient(lambda f_: self.func(model, f_, g),
                            f, **kwargs)
        # raise NotImplementedError("Gradient call needs to be implemented.")

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
        return num_gradient(lambda f_: self.grad(model, f_, g),
                            f, **kwargs)
        # raise NotImplementedError("Hessian call needs to be implemented.")

    def __str__(self):
        return self.formula

    def __repr__(self):
        return self.__str__()