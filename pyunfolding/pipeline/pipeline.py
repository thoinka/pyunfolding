"""
binning_tgt = binning.GridBinning([20])
    binning_tgt.fit(test_y)
    binning_obs = binning.TreeBinning(100, min_samples_leaf=b)
    f_class = binning_tgt.digitize(test_y)
    f_true = binning_tgt.histogram(test_y)
    binning_obs.fit(test_samples, f_class)

    m = model.Model(binning_obs, binning_tgt)
    m.fit(test_samples, test_y)
    print(b, condition(m.A))
    likelihood = llh.Likelihood(m)
    likelihood.append(llh.Poisson())
    likelihood.append(llh.TikhonovRegularization(1e-7))
    #likelihood.append(llh.OnlyPositive(0.01))

    # llh.TikhonovRegularization(model,0.0)
    x0 = m.x0(train_samples, pos=True, tau=1e-7)
    plt.plot(x0)
    plt.show()
    sol = solver.Minimizer(likelihood, gradient=True)
    min_res = sol.solve(x0, train_samples)
    sol = solver.MCMCSolver(likelihood)
    print(f_true)
    
    #x0 = np.ones_like(x0)
    #x0 += np.random.randn(len(x0)) * 5.0
    print(x0)
    result = sol.solve(min_res.f, train_samples, n_iter=100000, pass_samples=True)
    print(result)
"""
class Pipeline:
	def __init__(self,
		         binning_obs,
                 binning_tgt,
                 model,
                 likelihood,
                 solver):
		self.binning_obs = binning_obs
		self.binning_tgt = binning_tgt
		self.model = model
		self.likelihood = likelihood
		self.solver = solver

	def fit(self, X, y):
		self.binning_obs.fit(X, y)
		self.binning_tgt.fit(y)