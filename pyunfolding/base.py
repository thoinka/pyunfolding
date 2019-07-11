from sklearn.utils import check_X_y, check_array


class UnfoldingBase:
    
    def __init__(self, *args, **kwargs):
        self.is_fitted = False

    def fit(self, X_train, y_train, *args, **kwargs):
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        self.n_dim = X_train.shape[1]
        return check_X_y(X_train, y_train)

    def g(self, X, weights=None):
        '''Returns an observable vector :math:`\mathbf{g}` given a sample
        of observables `X`.
        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_observables)
            Observable sample.
        weights : numpy.array, shape=(n_samples,)
            Weights corresponding to the sample.
        Returns
        -------
        g : numpy.array, shape=(n_bins_X,)
            Observable vector
        '''
        if self.is_fitted:
            return self.model.binning_X.histogram(X, weights=weights)
        raise RuntimeError('Unfolding not yet fitted! Use `fit` method first.')

    def f(self, y, weights=None):
        '''Returns a result vector :math:`\mathbf{f}` given a sample
        of target variable values `y`.
        Parameters
        ----------
        y : numpy.array, shape=(n_samples,)
            Target variable sample.
        weights : numpy.array, shape=(n_samples,)
            Weights corresponding to the sample.
        Returns
        -------
        f : numpy.array, shape=(n_bins_X,)
             Result vector.
        '''
        if self.is_fitted:
            return self.model.binning_y.histogram(y, weights=weights)
        raise RuntimeError('Unfolding not yet fitted! Use `fit` method first.')

    def predict(self, X, *args, **kwargs):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if not self.is_fitted:
            raise RuntimeError('Unfolding must be fitted first.')
        
        if self.n_dim != X.shape[1]:
            raise ValueError('Array must have same shape as training data.')

        return check_array(X)