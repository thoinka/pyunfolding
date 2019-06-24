import numpy as np
from .matrices import diff2_matrix, diff1_matrix
from .unfoldingresult import UnfoldingResult


def analytical_solution(A, g, tau=0.0, Sigma=None, C_matrix=None):
	'''Analytical solution of the unfolding problem given a
	Tikhonov-regularized Least-Squares Likelihood approach:

	:math:`\mathcal{L}(\mathbf{f}|\mathbf{g}) = (\mathbf{g}-\mathrm{A}\mathbf{f})^\top \Sigma^{-1} (\mathbf{g}-\mathrm{A}\mathbf{f}) + \frac{\tau}{2} \mathbf{f}^\top \Gamma^\top \Gamma \mathbf{f}
	'''
	lenf = A.shape[1]

	if type(C_matrix) == str:
		if C_matrix == 'diff2':
			C_matrix = diff2_matrix(lenf)
		elif C_matrix == 'diff1':
			C_matrix = diff1_matrix(lenf)
	elif C_matrix is None:
		C_matrix = np.zeros((lenf, lenf))

	if Sigma is None:
		Sigma = np.eye(len(g))

	iSigma = np.linalg.pinv(Sigma)

	B = np.linalg.pinv(A.T @ iSigma @ A
	                   + 0.5 * tau * C_matrix.T @ C_matrix) @ A.T @ iSigma

	f_est = B @ g
	cov = B @ np.diag(g) @ B.T
	f_err = np.sqrt(np.vstack([cov.diagonal(), cov.diagonal()]))

	return UnfoldingResult(f=f_est, f_err=f_err, cov=cov, success=True)