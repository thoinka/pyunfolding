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
    # Takes the analytical solution of a regularized, weighted least squares
    # likelihood. The regularization is adjusted using using the keyword tau.
    ana_unfolding = pu.AnalyticalUnfolding(binning_x, binning_y,
                                           Sigma='poisson')
    ana_unfolding.fit(x_train, y_train)
    result_ana = ana_unfolding.predict(x_test, tau=4e-4)

    f_true = ana_unfolding.f(y_test)

    result_ana.plot(truth=f_true)
    plt.savefig('analytical_unfolding_result.pdf')

    # SVD Unfolding (with Gaussian cutoff).
    # Performs a singular value decomposition and smoothly surpresses small
    # singular values in order to avoid oscillations. The shape of the
    # supression function is changed by additional keywords, i.e. width.
    svd_unfolding = pu.SVDUnfolding(binning_x, binning_y)
    svd_unfolding.fit(x_train, y_train)
    result_svd = svd_unfolding.predict(x_test, mode='gaussian', width=4.0)

    result_svd.plot(truth=f_true)
    plt.savefig('svd_unfolding_result.pdf')

    # Bayesian Unfolding.
    # Iterative Bayesian Unfolding. The unfolded spectrum is estimated using
    # a bayesian guess that is fed back into the training data in every
    # iteration. The more iterations are performed using the keyword
    # n_iterations, the less regularized the unfolding becomes. There's
    # no out-of-the-box error estimation available yet.
    bay_unfolding = pu.BayesianUnfolding(binning_x, binning_y)
    bay_unfolding.fit(x_train, y_train)
    result_bay = bay_unfolding.predict(x_test, n_iterations=5)

    result_bay.plot(truth=f_true)
    plt.savefig('bayesian_unfolding_result.pdf')

    # Bayesian Unfolding in a Bootstrapping Pipeline.
    # Errors can always be estimated using bootstrapping, which is easily
    # performed using the Bootstrapper.
    bay_unfolding = pu.utils.Bootstrapper(
        pu.BayesianUnfolding(binning_x, binning_y)
        )
    bay_unfolding.fit(x_train, y_train)
    result_bay = bay_unfolding.predict(x_test, n_iterations=5)

    result_bay.plot(truth=f_true)
    plt.savefig('bayesian_bootstrap_unfolding_result.pdf')

    # DSEA Unfolding.
    # Unfolding problem is formulated as a classification problem. A classifier
    # is used to estimate the probability that one event belongs to the true
    # energy bin i. To shake off the inherent bias in the training data, its
    # weights are updated in similar fashion to iterative bayesian unfolding.
    dsea_unfolding = pu.DSEAUnfolding(binning_y)
    dsea_unfolding.fit(x_train, y_train, RandomForestClassifier)
    result_dsea = dsea_unfolding.predict(x_test, min_samples_leaf=20,
                                         n_iterations=10)

    result_dsea.plot(truth=f_true)
    plt.savefig('dsea_unfolding_result.pdf')

    # LLH Unfolding.
    # A likelihood is minimized directly using one of several methods.
    # Regularization is possible using several options. Among them is a MCMC
    # sampler that actually returns the posterior pdf.
    llh_unfolding = pu.LLHUnfolding(binning_x, binning_y,
                                likelihood=[pu.likelihood.llh.Poisson(),
                                            pu.likelihood.llh.Tikhonov(4e-4)])
    llh_unfolding.fit(x_train, y_train)
    result_llh = llh_unfolding.predict(x_test, mcmc=True)

    result_llh.plot(truth=f_true)
    plt.savefig('llh_unfolding_result.pdf')

    result_llh.sample.plot(best_fit=result_llh.f)
    plt.savefig('llh_unfolding_corner.pdf')
