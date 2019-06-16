import numpy as np
from .uncertainties import Posterior
from .unfoldingresult import UnfoldingResult


class Bootstrapper:
    
    def __init__(self, unfolding, n_bootstraps=100):
        self.unfolding = unfolding
        self.n_bootstraps = n_bootstraps
    
    def fit(self, X_train, y_train):
        self.unfolding.fit(X_train, y_train)
    
    def predict(self, X, x0=None,
                error_method='central',
                value_method='median',
                return_sample=False,
                **kwargs):
        fs = []
        for _ in range(self.n_bootstraps):
            bs = np.random.choice(len(X), len(X), replace=True)
            result = self.unfolding.predict(X=X[bs], x0=x0, **kwargs)
            fs.append(result.f)
        fs = np.array(fs)

        ppdf = Posterior(fs)
        lower, upper = ppdf.error(error_method)
        value = ppdf.value(value_method)

        error = np.vstack((value - lower, upper - value))
        result = dict(
            f=value,
            f_err=error,
            success=True,
            cov=np.cov(fs.T),
            binning_y=self.unfolding.model.binning_y
        )
        if return_sample:
            result.update(sample=ppdf)

        return UnfoldingResult(**result)