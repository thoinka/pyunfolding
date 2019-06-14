import numpy as np
from .base import SolutionBase
from ...utils import UnfoldingResult


class NewtonMinimizer(SolutionBase):

    def solve(self, x0, X, max_iter=100, tol=1e-6, beta=0.5, *args, **kwargs):
        g = self.likelihood.model.binning_X.histogram(X)

        def F(p): return self.likelihood(p, g)

        def G(p): return self.likelihood.grad(p, g)

        def H(p): return self.likelihood.hess(p, g)
        x = [x0]
        f = [F(x0)]
        alpha = 1.0
        success = False
        for i in range(max_iter):
            hess = H(x[-1])
            grad = G(x[-1])
            if np.linalg.norm(grad) < tol:
                success = True
                break
            x_new = x[-1] - alpha * \
                np.dot(np.linalg.pinv(hess + 1e-8 * np.eye(len(hess))), grad)
            f_new = F(x_new)

            if f_new < f[-1]:
                x.append(x_new)
                f.append(f_new)
                alpha = 1.0
            else:
                alpha *= beta

        std = np.sqrt(np.linalg.pinv(H(x[-1])).diagonal())
        return UnfoldingResult(f=x[-1],
                               f_err=np.vstack((0.5 * std, 0.5 * std)),
                               success=success)