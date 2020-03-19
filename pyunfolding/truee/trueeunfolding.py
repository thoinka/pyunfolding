import numpy as np
import pandas as pd
import os
import warnings

try:
    import root_pandas as rp
    import ROOT
    check = os.system('truee')
    if check != 0:
        raise RuntimeError()
    TRUEE_AVAILABLE = True
except:
    TRUEE_AVAILABLE = False
    warnings.warn('TRUEE Unfolding is unavailable on this system.')

from ..utils import UnfoldingResult
from ..base import UnfoldingBase


class TRUEEUnfolding(UnfoldingBase):
    '''Unfolding using TRUEE [10.1016/j.nima.2012.08.105]. It requires a lot
    of stuff to be installed, namely ROOT with root_pandas and obviously truee.
    The binning TRUEE uses is completely externalized, so the only thing TRUEE
    really does is solving the unfolding equation.

    Parameters
    ----------
    binning_X : pyunfolding.binning.Binning object
        The binning of the observable space.
    binning_y : pyunfolding.binning.Binning object
        The binning of the target variable.

    Attributes
    ----------
    is_fitted : bool
        Whether or not the unfolding has been fitted.
    
    TRUEE_CALL : str
        location of truee binaray (or alias)
    
    TRUEE_RESULT : str
        Name of result file. I guess it's always TrueeResultFile.root, but just
        in case.
    
    binning_X : pyunfolding.binning.Binning object
        The binning of the observable space.
    
    binning_y : pyunfolding.binning.Binning object
        The binning of the target variable.
    '''
    # System-specific 
    TRUEE_CALL = 'truee'
    TRUEE_RESULT = 'TrueeResultFile.root'
    
    def __init__(self, binning_X, binning_y):
        if not TRUEE_AVAILABLE:
            raise RuntimeError('TRUEE is not available on this system.')
        super(TRUEEUnfolding, self).__init__()
        self.binning_X = binning_X
        self.binning_y = binning_y
        
        self._config = {}
        # Mode of unfolding is always 'unfolding'. We never use the built-in
        # 'test' or 'pull' modes.
        self._config.update(mode='unfolding')
        
        # Make truee shut up as much as we can
        self._config.update(printflag='0')
        
        # The number of variables is always 1 in order to use pyunfolding's
        # binning. Instead of giving truee the actual observables, we just give
        # truee the digitized observables, meaning the bin numbers.
        self._config.update(number_all_variables='1')
        
    def _write_config_file(self, loc):
        output = ('\n'.join('{}: {}'.format(key, val)
                            for key, val in self._config.items()))
        with open(loc, 'w+') as f:
            f.write(output)
        
    def fit(self, X_train, y_train):
        '''Fit routine.

        Parameters
        ----------
        X_train : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
        
        y_train : numpy.array, shape=(n_samples,)
            Target variable sample.
        '''
        X_train, y_train = super(TRUEEUnfolding, self).fit(X_train, y_train)
        self.binning_X.fit(X_train, y_train)
        self.binning_y.fit(y_train)
        X_digit = self.binning_X.digitize(X_train)
        y_digit = self.binning_y.digitize(y_train)
        df_train = pd.DataFrame(np.column_stack((X_digit, y_digit)),
                                columns=['x', 'y'])
        
        file_mc = 'temp_truee_train.root'
        self.tempdir = '_truee_temp_dir'
        self.path_mc = os.path.join(self.tempdir, file_mc)
        if not os.path.exists('_truee_temp_dir'):
            os.mkdir('_truee_temp_dir')
        rp.to_root(df_train, self.path_mc, 'data')
        
        self._config.update(source_file_moca=self.path_mc)
        self._config.update(roottree_moca='data')
        
        self._config.update(branch_y='x')
        self._config.update(number_y_bins=self.binning_X.n_bins)
        self._config.update(limits_y='{} {}'.format(-0.5, self.binning_X.n_bins - 0.5))
        
        self._config.update(branch_x='y')
        self._config.update(number_bins=self.binning_y.n_bins)
        self._config.update(max_number_bins=self.binning_y.n_bins)
        self._config.update(limits_x='{} {}'.format(-0.5, self.binning_y.n_bins - 0.5))
        self.is_fitted = True
    
    def predict(self, X, n_knots, n_dof,
                data_luminosity=1.0,
                moca_luminosity=1.0,
                moca_weight=1.0,
                fx_positive=False,
                smooth_x=False,
                zero_left=False,
                zero_right=False,
                constraints='',
                weight_first=0,
                cleanup=True, **kwargs):
        '''Calculates an estimate for the unfolding by calling TRUEE.

        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_obervables)
            Observable sample.
       
        n_knots : int
            Number of knots for the spline representation used in TRUEE.
            Rule of thumb: Should be about twice the number of bins in the
            target variable space.
       
        n_dof : int
            Number of degrees of freedom, the more, the less regularized the
            unfolding.
       
        data_luminosity : float
            I guess weights for X?
        
        moca_luminosity : float
            I guess weights for y?
        
        fx_positive : bool
            Whether to enforce positive results for the unfolded spectrum.
        
        smooth_x : bool
            Whether to smooth ... the observable vector? I don't know.
        
        zero_left, zero_right : bool
            I think supposedly, this is supposed to set the left/right-most bin
            to zero. However, I don't think it does anything at all
        
        constraints : str
            A string containing a C-style formula (without spaces!). No idea.
        
        weight_first : int
            Who knows
        
        cleanup : bool
            Whether or not to delete all temporary files after TRUEE was called.

        Returns
        -------
        result : ``pyunfolding.utils.UnfoldingResult`` object
            The result of the unfolding, see documentation for 
            `UnfoldingResult`.
        '''
        if not self.is_fitted:
            raise RuntimeError('Unfolding not yet fitted. Use `fit` routine first.')
    
        X = super(TRUEEUnfolding, self).predict(X)

        # Storing parameters to config dictionary
        self._config.update(number_deg_free=n_dof)
        self._config.update(max_number_deg_free=n_dof)
        self._config.update(number_knots=n_knots)
        self._config.update(max_number_knots=n_knots)
        self._config.update(data_luminosity=data_luminosity)
        self._config.update(moca_luminosity=moca_luminosity)
        self._config.update(moca_weight=moca_weight)
        self._config.update(fx_positive=int(fx_positive))
        self._config.update(smooth_x=int(smooth_x))
        self._config.update(zero_left=int(zero_left))
        self._config.update(zero_right=int(zero_right))
        self._config.update(constraints=constraints)
        self._config.update(weight_first=weight_first)
        
        X_digit = self.binning_X.digitize(X)
        file_dt = 'temp_truee_test.root'
        self.path_dt = os.path.join(self.tempdir, file_dt)
        df_test = pd.DataFrame(np.column_stack((X_digit, np.zeros(len(X_digit)))), columns=['x', 'y'])
        rp.to_root(df_test, self.path_dt, 'data')
        
        self._config.update(roottree_data='data')
        self._config.update(source_file_data=self.path_dt)
        
        # Write config file and run TRUEE
        file_conf = 'parameters.config'
        self.path_conf = os.path.join(self.tempdir, file_conf)
        self._write_config_file(self.path_conf)
        os.system('{} {}'.format(self.TRUEE_CALL, self.path_conf))
        
        f = ROOT.TFile.Open(self.TRUEE_RESULT)
        g = f.GetDirectory('RealDataResults')
        string = 'bins_{}_knots_{}_degFree_{}'.format(self.binning_y.n_bins, n_knots, n_dof)
        cov = np.array([[g.Get('Tcovar_matrix_{};1'.format(string))(i, j)
                         for i in range(self.binning_y.n_bins)]
                        for j in range(self.binning_y.n_bins)]) 
        h = g.Get('events_result_{};1'.format(string))
        f_vals = np.array([h.GetBinContent(i) for i in range(self.binning_y.n_bins + 1)])
        f_err = np.sqrt(cov.diagonal())
        
        # Cleanup temp files
        if cleanup:
            os.remove(self.path_mc)
            os.remove(self.path_dt)
            os.remove(self.path_conf)
            os.remove(self.TRUEE_RESULT)
            os.rmdir(self.tempdir)
            
        # I'm not sure why this is necessary, but it is. And it's not an elegant solution either.
        scaling = np.sum(f_vals) / len(X)
        
        return UnfoldingResult(f=f_vals[1:] / scaling,
                               f_err = np.vstack((f_err, f_err)) / scaling,
                               cov=cov,
                               binning_y=self.binning_y,
                               success=True)