"""Sklearn-style API for PSN (Partitioning Signal and Noise).

This module provides a scikit-learn compatible interface for PSN denoising,
with fit(), transform(), and fit_transform() methods.
"""

import numpy as np
from .psn import psn


class PSN:
    """Sklearn-style interface for PSN denoising.

    Mimics sklearn's PCA API with fit(), transform(), and fit_transform() methods.

    Parameters
    ----------
    mode : str, optional
        Denoising mode preset. One of:
        - 'conservative': prioritizes retaining signal
        - 'standard': prioritizes out-of-sample generalization (default)
        - 'aggressive': flexible adaptation per unit
        If None, uses custom options only.
    basis : str or ndarray, optional
        Basis type: 'signal', 'difference', 'noise', 'pca', 'random',
        or custom matrix [nunits x D] with orthonormal columns.
        Default: 'signal'.
    criterion : str, optional
        Threshold selection criterion: 'prediction', 'variance',
        or 'variance_eigenvalues'. Default: 'prediction'.
    threshold_method : str, optional
        How to select thresholds: 'global', 'hybrid', or 'unit'.
        Default: 'hybrid'.
    basis_ordering : str, optional
        How to rank dimensions: 'eigenvalues' or 'signalvariance'.
        Default: 'eigenvalues'.
    variance_threshold : float, optional
        Fraction for variance criteria, in [0, 1]. Default: 0.99.
    allowable_thresholds : array-like, optional
        Constrain thresholds to these values. Default: None.
    unit_groups : array-like, optional
        [nunits] array specifying which units share thresholds.
        Default: each unit is its own group.
    gsn_args : dict, optional
        Options passed to GSN routine. Default: None.
    wantverbose : bool, optional
        Whether to print progress messages. Default: True.

    Attributes (available after fit)
    ---------------------------------
    denoiseddata_ : ndarray [nunits x nconds]
        Denoised training data.
    residuals_ : ndarray [nunits x nconds x ntrials]
        Training residuals (original - denoised).
    denoiser_ : ndarray [nunits x nunits]
        The learned denoising matrix.
    unit_means_ : ndarray [nunits]
        Mean per unit (used for centering).
    svnv_before_ : ndarray [nunits x 2]
        Signal/noise variance before denoising.
    svnv_after_ : ndarray [nunits x 2]
        Signal/noise variance after denoising.
    best_threshold_ : scalar or ndarray
        Selected threshold(s).
    fullbasis_ : ndarray [nunits x dims]
        Basis vectors after global ordering.
    basis_eigenvalues_ : ndarray or None
        Eigenvalues matching fullbasis_ columns.
    unitreorderings_ : ndarray [nunits x dims]
        Per-unit dimension orderings.
    gsn_result_ : dict
        GSN results containing cSb, cNb covariances.
    signalvar_ : ndarray
        Signal variance per dimension.
    noisevar_ : ndarray
        Noise variance per dimension.
    objective_ : ndarray
        Cumulative objective curve.
    basis_viz_ : ndarray
        Basis in original order (for visualization).
    signal_proj_viz_ : ndarray
        Signal projections in original order.
    noise_proj_viz_ : ndarray
        Noise projections in original order.
    basis_eigenvalues_viz_ : ndarray or None
        Eigenvalues in original order.
    input_data_ : ndarray
        Copy of training data.
    signalsubspace_ : ndarray or None
        Selected basis vectors (global mode only).
    dimreduce_ : ndarray or None
        Low-dimensional representation (global mode only).
    unit_signal_vars_ : list or None
        Per-unit signal variances (hybrid/unit modes).
    unit_noise_vars_ : list or None
        Per-unit noise variances (hybrid/unit modes).
    unit_objectives_ : list or None
        Per-unit objective curves (hybrid/unit modes).
    opt_used_ : dict
        Options that were actually used.

    Examples
    --------
    >>> from psn import PSN
    >>> # Fit and transform in one call
    >>> model = PSN(mode='standard')
    >>> denoised = model.fit_transform(data)
    >>>
    >>> # Or fit first, then transform new data
    >>> model = PSN(mode='standard')
    >>> model.fit(train_data)
    >>> denoised_test = model.transform(test_data)
    >>>
    >>> # Access learned parameters
    >>> print(f"Retained {model.best_threshold_} dimensions")
    >>> print(f"Denoiser shape: {model.denoiser_.shape}")
    >>>
    >>> # Generate diagnostic figure
    >>> fig = model.plot_diagnostics()
    """

    def __init__(self, mode='standard', **options):
        """Initialize PSN with mode and options.

        Parameters
        ----------
        mode : str, optional
            One of 'conservative', 'standard', 'aggressive', or None.
        **options
            Additional options passed to psn().
        """
        self.mode = mode
        self.options = options
        self._is_fitted = False

    def fit(self, data, visualize=False):
        """Learn the denoiser from training data.

        Parameters
        ----------
        data : ndarray [nunits x nconds x ntrials]
            Training data. Must have ntrials >= 2.
        visualize : bool, optional
            Whether to generate diagnostic figure during fit.
            Default: False.

        Returns
        -------
        self
            Fitted estimator.
        """
        # Build options dict
        opt = self.options.copy()
        opt['wantfig'] = visualize

        # Call functional API
        if self.mode in ('conservative', 'standard', 'aggressive'):
            results = psn(data, self.mode, opt)
        elif self.mode is None:
            results = psn(data, opt)
        else:
            raise ValueError(
                f"mode must be 'conservative', 'standard', 'aggressive', or None. "
                f"Got: {self.mode}"
            )

        # Store all results as attributes with trailing underscore
        # Core outputs
        self.denoiseddata_ = results['denoiseddata']
        self.residuals_ = results['residuals']
        self.unit_means_ = results['unit_means']
        self.denoiser_ = results['denoiser']

        # Diagnostic outputs
        self.svnv_before_ = results['svnv_before']
        self.svnv_after_ = results['svnv_after']
        self.best_threshold_ = results['best_threshold']
        self.fullbasis_ = results['fullbasis']
        self.unitreorderings_ = results['unitreorderings']

        # GSN outputs
        self.gsn_result_ = results['gsn_result']

        # Variance outputs
        self.signalvar_ = results['signalvar']
        self.noisevar_ = results['noisevar']
        self.objective_ = results['objective']

        # Visualization basis (original order)
        self.basis_viz_ = results['basis_viz']
        self.signal_proj_viz_ = results['signal_proj_viz']
        self.noise_proj_viz_ = results['noise_proj_viz']
        self.basis_eigenvalues_viz_ = results['basis_eigenvalues_viz']

        # Sorted eigenvalues
        self.basis_eigenvalues_ = results['basis_eigenvalues']

        # Input data (for diagnostics)
        self.input_data_ = results['input_data']

        # Global mode special outputs
        self.signalsubspace_ = results.get('signalsubspace')
        self.dimreduce_ = results.get('dimreduce')

        # Unit-specific mode outputs
        self.unit_signal_vars_ = results.get('unit_signal_vars')
        self.unit_noise_vars_ = results.get('unit_noise_vars')
        self.unit_objectives_ = results.get('unit_objectives')

        # Options used
        self.opt_used_ = results['opt_used']

        self._is_fitted = True
        return self

    def transform(self, data):
        """Apply learned denoiser to new data.

        Parameters
        ----------
        data : ndarray
            Data to denoise. Shape [nunits x nconds x ntrials] or
            [nunits x nconds]. If 3D, will be trial-averaged first.
            Must have same nunits as training data.

        Returns
        -------
        denoised : ndarray [nunits x nconds]
            Denoised data.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "PSN must be fitted before calling transform(). "
                "Call fit() first."
            )

        # Check dimensions first
        if data.ndim not in (2, 3):
            raise ValueError(
                f"Data must be 2D or 3D, got {data.ndim}D array."
            )

        # Validate nunits matches
        nunits_expected = self.denoiser_.shape[0]
        if data.shape[0] != nunits_expected:
            raise ValueError(
                f"Data has {data.shape[0]} units, but model was fitted with "
                f"{nunits_expected} units."
            )

        # Trial average if 3D
        if data.ndim == 3:
            if np.any(np.isnan(data)):
                trial_avg = np.nanmean(data, axis=2)
            else:
                trial_avg = np.mean(data, axis=2)
        else:
            trial_avg = data

        # Apply denoiser using learned unit_means_
        # denoiser.T @ (data - means) + means
        denoised = (
            self.denoiser_.T @ (trial_avg - self.unit_means_[:, np.newaxis])
            + self.unit_means_[:, np.newaxis]
        )

        return denoised

    def fit_transform(self, data, visualize=False):
        """Fit and return denoised training data.

        Parameters
        ----------
        data : ndarray [nunits x nconds x ntrials]
            Training data.
        visualize : bool, optional
            Whether to generate diagnostic figure. Default: False.

        Returns
        -------
        denoised : ndarray [nunits x nconds]
            Denoised training data.
        """
        self.fit(data, visualize=visualize)
        return self.denoiseddata_

    def plot_diagnostics(self, **kwargs):
        """Generate diagnostic figure using training data.

        Must be called after fit().

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to plot_diagnostic_figures.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The diagnostic figure.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "PSN must be fitted before calling plot_diagnostics(). "
                "Call fit() first."
            )

        # Import visualization
        from .utilities.plotting.visualization import plot_diagnostic_figures

        # Reconstruct results dict from stored attributes
        # This avoids storing the dict directly (pickle safety)
        results = {
            'denoiseddata': self.denoiseddata_,
            'residuals': self.residuals_,
            'unit_means': self.unit_means_,
            'denoiser': self.denoiser_,
            'svnv_before': self.svnv_before_,
            'svnv_after': self.svnv_after_,
            'best_threshold': self.best_threshold_,
            'fullbasis': self.fullbasis_,
            'unitreorderings': self.unitreorderings_,
            'gsn_result': self.gsn_result_,
            'signalvar': self.signalvar_,
            'noisevar': self.noisevar_,
            'objective': self.objective_,
            'basis_viz': self.basis_viz_,
            'signal_proj_viz': self.signal_proj_viz_,
            'noise_proj_viz': self.noise_proj_viz_,
            'basis_eigenvalues_viz': self.basis_eigenvalues_viz_,
            'basis_eigenvalues': self.basis_eigenvalues_,
            'input_data': self.input_data_,
            'opt_used': self.opt_used_,
        }

        # Add optional outputs if they exist
        if self.signalsubspace_ is not None:
            results['signalsubspace'] = self.signalsubspace_
        if self.dimreduce_ is not None:
            results['dimreduce'] = self.dimreduce_
        if self.unit_signal_vars_ is not None:
            results['unit_signal_vars'] = self.unit_signal_vars_
        if self.unit_noise_vars_ is not None:
            results['unit_noise_vars'] = self.unit_noise_vars_
        if self.unit_objectives_ is not None:
            results['unit_objectives'] = self.unit_objectives_

        fig = plot_diagnostic_figures(self.input_data_, results, **kwargs)
        return fig

    def __repr__(self):
        """String representation."""
        if self._is_fitted:
            threshold_info = (
                f"{self.best_threshold_}"
                if np.isscalar(self.best_threshold_)
                else f"mean={np.mean(self.best_threshold_):.1f}"
            )
            return (
                f"PSN(mode={self.mode!r}, fitted=True, "
                f"threshold={threshold_info})"
            )
        else:
            return f"PSN(mode={self.mode!r}, fitted=False)"

    def __getstate__(self):
        """Prepare state for pickling, removing non-picklable items."""
        state = self.__dict__.copy()

        # Sanitize opt_used_ to remove lambda functions
        if 'opt_used_' in state and state['opt_used_'] is not None:
            state['opt_used_'] = self._sanitize_dict_for_pickle(state['opt_used_'])

        return state

    def __setstate__(self, state):
        """Restore state from pickle."""
        self.__dict__.update(state)

    @staticmethod
    def _sanitize_dict_for_pickle(d):
        """Remove non-picklable items (callables) from a dict recursively."""
        result = {}
        for k, v in d.items():
            if callable(v):
                # Skip callable items (lambdas, functions)
                continue
            elif isinstance(v, dict):
                # Recursively sanitize nested dicts
                result[k] = PSN._sanitize_dict_for_pickle(v)
            else:
                result[k] = v
        return result
