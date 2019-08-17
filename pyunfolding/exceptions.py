__all__ = ["NotFittedError",
           "FailedMinimizationWarning",
           "InvalidCovarianceWarning",
           "FailedMCMCWarning"]


class NotFittedError(RuntimeError):
    """Raised when either `pyunfolding.binning.Binning` object or
    `pyunfolding.UnfoldingBase` object are called without fitting them with
    some training data first.
    """

class FailedMinimizationWarning(RuntimeWarning):
	"""Raised when the Minimization did not reach the demanded abort criterion
	i.e. the error tolerance has not been reached in the last iteration.
	"""

class InvalidCovarianceWarning(UserWarning):
	"""Raised when the covariance matrix returned is not symmetrical or not
	positive semi definite. In those cases additional care when using
	uncertainty estimates should be applied.
	"""

class FailedMCMCWarning(UserWarning):
	"""Raised when all attempts to burnin the Markov Chain Monte Carlo have
	failed.
	"""