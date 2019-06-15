import numpy as np


class DSEAUnfolding:
    """DSEA Unfolding: Instead of defining an unfolding model, the probability
    :math:`f(i|\mathbf{x})`, i.e. the probability of an event being originally
    from a target variable bin :math:`i` given it's observables
    :math:`\mathbf{x}` is estimated using a classifier. The unfolded spectrum is then estimated by accumulating all conditional probabilities. As this
    propagates the bias in the training data for the most part, an iterative
    approach very similar to the bayesian unfolding is used to 'shake off' the
    bias. To achieve that, the training data is reweighted according to the
    first, biased, estimate of :math:`\mahtbf{f}` and the procedure is rerun.

    Parameters
    ----------
    binning_y : pyunfolding.binning.Binning object
        The binning of the target space.

    Attributes
    ----------
    is_fitted : bool
        Whether or not the unfolding has been fitted.
    binning_y : pyunfolding.binning.Binning object
        The binning of the target space.
    X_train : numpy.array, shape=(n_samples, n_observables)
        Training data: Observables
    y_train : numpy.array, shape=(n_samples,)
        Training data: Target variable
    labels : numpy.array, shape=(n_labels,)
        All labels contained in training data.
    classifier : sklearn.classifier
        An sklearn classifier
    """
    def __init__(self, binning_y):
        self.binning_y = binning_y
        self.is_fitted = False

    def f(self, y):
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
            return self.binning_y.histogram(y)
        raise RuntimeError('Unfolding not yet fitted! Use `fit` method first.')
    
    def fit(self, X_train, y_train, classifier):
        '''Fit routine.

        Parameters
        ----------
        X_train : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        y_train : numpy.array, shape=(n_samples,)
            Target variable sample.
        classifier : sklearn.classifier
            An sklearn classifier 
        '''
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
        '''Calculates an estimate for the unfolding.

        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        n_iterations : int, default=1
            Number of iterations to be performed.
        alpha : float, default=1.0
            Decay rate, i.e. in order to force the algorithm to convergence
            the weights are only partially updated.
        plus : bool, default=True
            Whether or not to use partial weight updating.
        kwargs : dict
            Keywords for the classifier.
        '''
        if X.ndim == 1:
            X = X.reshape(-1,1)
        if self.is_fitted:
            weights = np.ones(self.binning_y.n_bins)
            for _ in range(n_iterations):
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