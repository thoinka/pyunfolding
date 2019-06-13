class Bootstrapper:
    
    def __init__(self, unfolding, n_bootstraps=100):
        self.unfolding = unfolding
        self.n_bootstraps = n_bootstraps
    
    def fit(self, X_train, y_train):
        self.unfolding.fit(X_train, y_train)
    
    def predict(self, X, x0=None, **kwargs):
        fs = []
        for _ in range(self.n_bootstraps):
            bs = np.random.choice(len(X), len(X), replace=True)
            result = self.unfolding.predict(X=X[bs], x0=x0, **kwargs)
            fs.append(result.f)
        f = self.unfolding.predict(X=X, x0=x0, **kwargs).f
        return pu.utils.UnfoldingResult(f=f, f_err=np.std(fs, axis=0), success=True)