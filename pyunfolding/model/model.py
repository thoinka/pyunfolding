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

    def __init__(self, binning_X, binning_y):
        self.binning_X = binning_X
        self.binning_y = binning_y
        self.fitted = False

    def fit(self, X, y):
        if not self.binning_y.fitted:
            self.binning_y.fit(y)
        y_class = self.binning_y.digitize(y)
        if not self.binning_X.fitted:
            self.binning_X.fit(X, y_class)
        x_class = self.binning_X.digitize(X)
        bins = [np.arange(self.binning_X.n_bins + 1) - 0.5,
                np.arange(self.binning_y.n_bins + 1) - 0.5, ]
        H, _, _ = np.histogram2d(x_class,
                                 np.tile(y_class, self.binning_X.rep),
                                 bins)
        #H /= float(self.binning_X.rep)
        self.A = (H + 1e-8) / np.sum(H + 1e-8, axis=0)
        self.fitted = True

    def predict(self, f):
        if not self.fitted:
            raise RuntimeError("Model not fitted! Run fit first!")
        return np.dot(self.A, f) * self.binning_X.rep

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

    def x0(self, X, pos=False, tau=0.0):
        if not self.fitted:
            raise RuntimeError("Model not fitted! Run fit first!")
        g = self.binning_X.histogram(X)
        n = self.A.shape[1]
        C = 2.0 * np.eye(n) - np.roll(np.eye(n), 1)\
                            - np.roll(np.eye(n), -1)
        A = self.A
        Aplus = np.dot(np.linalg.pinv(np.dot(A.T, A)\
                       + 0.5 * tau * np.dot(C.T, C)), A.T)
        x0 = np.dot(Aplus, g)
        if pos:
            x0[x0 < 0.0] = 1e-3
        return x0 / float(self.binning_X.rep)
