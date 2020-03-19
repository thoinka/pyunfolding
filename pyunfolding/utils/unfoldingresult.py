from ..plot import plot_unfolding_result
import numpy as np
from .matrices import check_covariance


class UnfoldingResult:
    '''Class to contain the result of an unfolding.

    Parameters
    ----------
    f : numpy.array, shape=(n_bins_y)
        Unfolding result

    f_err : numpy.array, shape=(2, n_bins_y)
        Uncertainty estimate, potentially asymmetrical

    success : bool
        Whether the unfolding was deemed successful or not.

    kwargs : dict
        Additional keys for the result object.

    Attributes
    ----------
    f : numpy.array, shape=(n_bins_y)
        Unfolding result

    f_err : numpy.array, shape=(2, n_bins_y)
        Uncertainty estimate, potentially asymmetrical

    success : bool
        Whether the unfolding was deemed successful or not.

    Raises
    ------
    InvalidCovarianceWarning
        Raised when the covariance matrix is improper.
    '''

    def __init__(self, f, f_err, success, *args, **kwargs):
        self.f = f
        self.f_err = f_err
        self.success = success
        for key, val in kwargs.items():
            self.__dict__[key] = val

        if 'cov' in self.__dict__.keys():
            check_covariance(self.cov)

    def __str__(self):
        if self.success:
            s = "Successful unfolding:\n"
        else:
            s = "Unsuccessful unfolding:\n"
        sym_err = np.allclose(self.f_err[0], self.f_err[1])
        for i, (f, ferr1, ferr2) in enumerate(zip(self.f, self.f_err[0], self.f_err[1])):
            if sym_err:
                s += u"Var {}:\t{:.2f} ±{:.2f}\n".format(i + 1,
                                                        f, ferr1)
            else:
                s += "Var {}:\t{:.2f} +{:.2f} -{:.2f}\n".format(i + 1, f,
                                                                ferr1, ferr2)
        try:
            s += "Error-Message: {}".format(self.error)
        except:
            s += "No error message was left."
        return s

    def plot(self,
             ax=None,
             truth=None,
             exclude_edges=True,
             correlations=True,
             *args,
             **kwargs):
        """Plots this unfolding result.
        
        Parameters
        ----------
        ax : matplotlib.pyplot.axis, optional (default=None)
            Matplotlib axis. If None provided, one is created.

        truth : numpy.array or None, optional (default=None)
            Baseline truth if provided, otherwise None.

        exclude_edges : bool, optional (default=True)
            Whether to leave out the edges.

        correlations : bool, optional (default=True)
            Whether to encode the bin-to-bin correlations.
        
        Returns
        -------
        matplotlib.pyplot.axis
            Axis containing plot.
        """
        return plot_unfolding_result(self, ax, truth, exclude_edges,
                                     correlations, *args, **kwargs)

    def __repr__(self):
        return self.__str__()

    def update(self, **kwargs):
        self.__dict__.update(kwargs)