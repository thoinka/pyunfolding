# pyunfolding
## Introduction
Unfolding is a problem in some fields

## Usage
pyUnfolding offers multiple algorithms to unfold a given sample on the basis of labeled data (i.e. simulations). So far those Algorithms include:

* `LLHUnfolding`: Unfolding using a likelihood approach.
* `BayesianUnfolding`: Unfolding using iterative Bayesian unfolding.
* `SVDUnfolding`: Unfolding regularized by cutting of singular values.
* `DSEAUnfolding`: Iterative unfolding that classifies each event.

## Examples
Let's take a simple example: We have two variables which are correlated.

```python
import numpy as np


def sample_linear(m, n_samples):
    y = np.random.rand(n_samples)
    return (-2.0 + m + np.sqrt(4.0 - 4.0 * m + m ** 2 + 8.0 * m * y)) / (2.0 * m)

y = sample_linear(-1.0, 100000)
X = y + np.random.randn(100000) / (2.0 + y) ** 3

y_test = sample_linear(1.0, 10000)
X_test = y_test + np.random.randn(10000) / (2.0 + y_test) ** 3
```
Let's unfold `X_test` using a Likelihood approach:

```python
import pyunfolding as pu

llhu = pu.LLHUnfolding(binning_X=pu.binning.GridBinning(100),
                       binning_y=pu.binning.GridBinning(20),
                       llh=[pu.likelihood.llh.Poisson(), pu.likelihood.llh.Tikhonov(0.01)])
llhu.fit(X, y)
result = llhu.predict(1000 * np.ones(20), X_test, solver_method='minimizer', method='SLSQP', tol=1e-12)
```
Or alternatively, we could use an iterative bayesian approach:

```python
ibu = pu.BayesianUnfolding(binning_X=pu.binning.GridBinning(100),
                              binning_y=pu.binning.GridBinning(20))
ibu.fit(X, y)
result = ibu.predict(X_test, n_iterations=10)
```

