import numpy as np


class Unfolding(object):
    """Linear Model :math:`g = A\cdot f + b`.

    Attributes
    ----------
    A : numpy.array, shape=(binning_X.n_bins, binning_y.n_bins)
        Migration matrix.
    binning_X : Binning object
        Binning of the proxy space.
    binning_y : Binning object
        Binning of the target space.
    """

    def __init__(self, binning_X, binning_y, eps=1e-8):
        self.binning_X = binning_X
        self.binning_y = binning_y
        self.fitted = False
        self.eps = eps

    def fit(self, X, y,
            weights=None,
            X_background=None,
            weights_background=None,
            acceptance=None):
        # First digitize all training event using the provided binning schemes.
        if not self.binning_y.fitted:
            self.binning_y.fit(y)
        y_class = self.binning_y.digitize(y)
        if not self.binning_X.fitted:
            self.binning_X.fit(X, y_class)
        x_class = self.binning_X.digitize(X)

        # The migration matrix is calculated by histogramming the digitized
        # observables.
        bins = [np.arange(self.binning_X.n_bins + 1) - 0.5,
                np.arange(self.binning_y.n_bins + 1) - 0.5]
        H, _, _ = np.histogram2d(x_class,
                                 y_class,
                                 bins)
        if X_background is None:
            self.b = np.zeros(self.binning_X.n_bins)
        else:
            self.b = self.binning_X.histogram(X_background,
                                              weights=weights_background)

        if acceptance is None:
            self.acceptance = np.ones(self.binning_y.n_bins)
        else:
            self.acceptance = acceptance

        self.A = (H + self.eps) / np.sum(H + self.eps, axis=0)
        self.fitted = True

    def predict(self, f):
        if not self.fitted:
            raise RuntimeError("Model not fitted! Run fit first!")
        return self.A @ f + self.b

    def predict_g(self, X):
        if not self.fitted:
            raise RuntimeError("Model not fitted! Run fit first!")
        return self.binning_X.histogram(X)

    def grad(self, f):
        if not self.fitted:
            raise RuntimeError("Model not fitted! Run fit first!")
        return self.A

    def hess(self, f):
        if not self.fitted:
            raise RuntimeError("Model not fitted! Run fit first!")
        return np.zeros_like(self.A)