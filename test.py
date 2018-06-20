import numpy as np
import binning
import model
import likelihood as llh
import solver
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from plot import corner_plot

"""
General structure of this module:

Binning:
    f(x) ---> f
    g(y) ---> g

Model[Binning_x, Binning_y]:
    Model ---f_test---> g_test

Likelihood[Model]:
    Likelihood ---g_obs, f_est---> llh

Solver[Likelihood]:
    Solver ---g_obs---> f_est
"""


def condition(A):
    u, s, v = np.linalg.svd(A)
    # return np.max(s) / np.min(s)
    return np.max(s[1:-1]) / np.linalg.norm(s[1:-1])


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

    b = 40

    binning_tgt = binning.GridBinning([10])
    binning_tgt.fit(test_y)
    binning_obs = binning.TreeBinning(100, min_samples_leaf=b)
    f_class = binning_tgt.digitize(test_y)
    f_true = binning_tgt.histogram(test_y)
    binning_obs.fit(test_samples, f_class)

    m = model.Model(binning_obs, binning_tgt)
    m.fit(test_samples, test_y)
    print(b, condition(m.A))
    mx, my = np.meshgrid(np.linspace(0.0, 1.0, len(f_true)), np.linspace(0.0, 1.0, len(f_true)))
    C = np.exp(-(mx - my) ** 2 / (2.0 * 0.3 ** 2))
    #C = np.eye(len(f_true)) + np.roll(np.eye(len(f_true)),1) + np.roll(np.eye(len(f_true)),-1)
    #C[0,0] = C[-1,-1] = 1.0

    C = np.linalg.pinv(C)

    print(C.shape)
    """
    likelihood = Likelihood(LeastSquares + TikhonovRegularization + OnlyPositive, m)
    """
    likelihood = llh.Likelihood(m)
    likelihood.append(llh.LeastSquares())
    likelihood.append(llh.TikhonovRegularization2(1e-5, C=C))
    #likelihood.append(llh.OnlyPositive(0.01))

    # llh.TikhonovRegularization(model,0.0)
    x0 = m.x0(train_samples, pos=True, tau=1e-3)
    #sol = solver.Minimizer(likelihood, gradient=True)
    #min_res = sol.solve(f_true, train_samples)
    sol = solver.MCMCSolver(likelihood)
    print(f_true)
    
    #x0 = np.ones_like(x0)
    #x0 += np.random.randn(len(x0)) * 5.0

    print(x0)
    result = sol.solve(f_true, train_samples, n_iter=1000000, pass_samples=True)
    print(result)

    #corner_plot(result.samples_x[::100,5:8])
    #plt.show()

    #print(np.sqrt(1.0 / likelihood.hess(result.f, m.predict_g(train_samples)).diagonal()))
    #print(min_res.f_err)
    print(np.sum(result.f_err, axis=0))
    hist = np.vstack((
        np.tile(np.arange(len(result.f)), (len(result.samples_x),1)).flatten(),
        result.samples_x.flatten()
    ))
    plt.hist2d(*hist, bins=(np.arange(len(result.f) + 1) - 0.5, 100), norm=LogNorm())

    plt.plot(binning_tgt.histogram(train_y), drawstyle="steps-mid", color="k")
    plt.errorbar(np.arange(len(result.f)), result.f, result.f_err, 0.5 * np.ones(len(result.f)), linestyle="", color="k")
    plt.ylim([-50,None])
    plt.show()
