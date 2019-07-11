from sklearn.utils import check_X_y, check_array


class UnfoldingBase:
    
    def __init__(self, *args, **kwargs):
        self.is_fitted = False

    def fit(self, X_train, y_train, *args, **kwargs):
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        self.n_dim = X_train.shape[1]
        return check_X_y(X_train, y_train)

    def predict(self, X, *args, **kwargs):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if not self.is_fitted:
            raise RuntimeError('Unfolding must be fitted first.')
        
        if self.n_dim != X.shape[1]:
            raise ValueError('Array must have same shape as training data.')

        return check_array(X)