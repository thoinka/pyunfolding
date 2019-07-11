import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pyunfolding as pu
from sklearn.ensemble import RandomForestClassifier


def sample_linear(m, n_samples):
    y = np.random.rand(n_samples)
    if m == 0.0:
        return y
    return (-2.0 + m + np.sqrt(4.0 - 4.0 * m + m ** 2 + 8.0 * m * y)) / (2.0 * m)


def create_toydata(y, spread=0.1):
    x = y + np.random.randn(len(y)) * spread
    return x


if __name__ == '__main__':
    y_train = sample_linear(0.0, 10000)
    x_train = create_toydata(y_train)

    y_test = sample_linear(1.9, 1000)
    x_test = create_toydata(y_test)

    binning_x = pu.binning.GridBinning(100)
    binning_y = pu.binning.GridBinning(10, pmin=1, pmax=99)

    # Analytical Unfolding.
    ana_unfolding = pu.AnalyticalUnfolding(binning_x, binning_y,
                                           Sigma='poisson')
    ana_unfolding.fit(x_train, y_train)
    result_ana = ana_unfolding.predict(x_test, tau=4e-4)

    f_true = ana_unfolding.f(y_test)

    result_ana.plot(truth=f_true)
    plt.savefig('analytical_unfolding_result.pdf')

    # SVD Unfolding (with Gaussian cutoff).
    svd_unfolding = pu.SVDUnfolding(binning_x, binning_y)
    svd_unfolding.fit(x_train, y_train)
    result_svd = svd_unfolding.predict(x_test, mode='gaussian', width=4.0)

    result_svd.plot(truth=f_true)
    plt.savefig('svd_unfolding_result.pdf')

    # Bayesian Unfolding.
    bay_unfolding = pu.BayesianUnfolding(binning_x, binning_y)
    bay_unfolding.fit(x_train, y_train)
    result_bay = bay_unfolding.predict(x_test, n_iterations=5)

    result_bay.plot(truth=f_true)
    plt.savefig('bayesian_unfolding_result.pdf')

    # DSEA Unfolding.
    dsea_unfolding = pu.DSEAUnfolding(binning_y)
    dsea_unfolding.fit(x_train, y_train, RandomForestClassifier)
    result_dsea = dsea_unfolding.predict(x_test, min_samples_leaf=20,
                                         n_iterations=10)

    result_dsea.plot(truth=f_true)
    plt.savefig('dsea_unfolding_result.pdf')

    # LLH Unfolding.
    llh_unfolding = pu.LLHUnfolding(binning_x, binning_y,
                                likelihood=[pu.likelihood.llh.Poisson(),
                                            pu.likelihood.llh.Tikhonov(4e-4)])
    llh_unfolding.fit(x_train, y_train)
    result_llh = llh_unfolding.predict(x_test, solver_method='minimizer')
    result_llh = llh_unfolding.predict(x_test, result_llh.f,
                                       solver_method='newton')
    result_llh = llh_unfolding.predict(x_test, result_llh.f,
                                       solver_method='mcmc')

    result_llh.plot(truth=f_true)
    plt.savefig('llh_unfolding_result.pdf')

    result_llh.sample.plot(best_fit=result_llh.f)
    plt.savefig('llh_unfolding_corner.pdf')