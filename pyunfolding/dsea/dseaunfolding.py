import numpy as np


class DSEAUnfolding:
    
    def __init__(self, binning_y):
        self.binning_y = binning_y
        self.is_fitted = False
    
    def g(self, X):
        if self.is_fitted:
            return self.model.binning_X.histogram(X)
        raise SyntaxError('Unfolding not yet fitted! Use `fit` method first.')

    def f(self, y):
        if self.is_fitted:
            return self.model.binning_y.histogram(y)
        raise RuntimeError('Unfolding not yet fitted! Use `fit` method first.')
    
    def fit(self, X_train, y_train, classifier):
        self.binning_y.fit(X_train, y_train)
        if X_train.ndim == 1:
            self.X_train = X_train.reshape(-1, 1)
        else:
            self.X_train = X_train
        self.y_train_int = self.binning_y.digitize(y_train)
        self.labels = np.unique(self.y_train_int)
        self.is_fitted = True
        self.classifier = classifier
        
    def predict(self, X, n_iterations=1, alpha=1.0, plus=True, **kwargs):
        if X.ndim == 1:
            X = X.reshape(-1,1)
        if self.is_fitted:
            weights = np.ones(self.binning_y.n_bins)
            for _ in range(n_iterations):
                #cweights = {i: weights[i] for i in self.labels}
                classif = self.classifier(**kwargs)
                classif.fit(self.X_train, self.y_train_int, sample_weight=weights[self.y_train_int])
                
                prediction = np.empty((len(X), self.binning_y.n_bins))
                prediction[:,self.labels] = classif.predict_proba(X)
                prediction_training = np.empty((len(self.X_train), self.binning_y.n_bins))
                prediction_training[:,self.labels] = classif.predict_proba(self.X_train)
                
                if plus:
                    w_new = np.sum(prediction, axis=0) / np.sum(prediction_training, axis=0)
                    weights = (alpha * w_new + weights) / (1.0 + alpha)
            f = np.sum(prediction, axis=0)
            f_err = np.sqrt(np.mean(prediction ** 2, axis=0) * len(prediction))
            return UnfoldingResult(f=f,
                                   f_err=np.vstack((f_err, f_err)),
                                   success=True,
                                   samples=prediction,
                                   cov=np.cov(prediction.T) * len(prediction))
        raise RuntimeError('Unfolding not yet fitted! Use `fit` method first.')