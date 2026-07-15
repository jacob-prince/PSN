"""Scikit-learn-style estimator for PSN (Partitioning Signal and Noise).

The :class:`PSN` estimator wraps the functional :func:`psn` API behind the
standard scikit-learn ``fit`` / ``transform`` / ``fit_transform`` interface.
Subclassing :class:`sklearn.base.BaseEstimator` and
:class:`sklearn.base.TransformerMixin` gives it ``get_params`` / ``set_params`` /
``clone`` and lets it act as a single step in a :class:`~sklearn.pipeline.Pipeline`.

It is NOT a drop-in for scikit-learn's model-selection machinery
(:class:`~sklearn.model_selection.GridSearchCV`, ``cross_val_score``,
``check_estimator``). PSN data is shaped ``[nunits, nconds, ntrials]`` with UNITS
on axis 0, whereas scikit-learn treats axis 0 as samples, so the default CV
splitters would partition units rather than conditions. PSN is also unsupervised
(no ``y``, no ``score``), so ``GridSearchCV`` has nothing to optimise without a
custom scorer. To cross-validate, split the conditions axis yourself; see the
Notes in :class:`PSN` and SKLEARN_API.md.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .psn import psn


class PSN(BaseEstimator, TransformerMixin):
    """Scikit-learn estimator for PSN denoising.

    Learns a denoising operator from ``[nunits x nconds x ntrials]`` data in
    ``fit`` and applies it to trial-averaged data in ``transform``.

    Parameters
    ----------
    mode : {None, 'conservative', 'standard', 'aggressive', 'compare', 'wiener'}, optional
        Named preset, or ``None`` (the default) to use the library defaults
        (equivalent to 'standard': signal basis, max-tradeoff threshold, global).
        Any non-None parameter below OVERRIDES the preset's corresponding default,
        except for 'wiener', which is basis-free and rejects conflicting options
        (see :func:`psn`). The presets are:

        - 'conservative' : signal basis, variance criterion, 99% retained (global)
        - 'standard'     : signal basis at the max-tradeoff threshold (global)
        - 'aggressive'   : difference basis at the prediction peak (global)
        - 'compare'      : pick signal vs difference by split-half r at each max-tradeoff K (global)
        - 'wiener'       : full-rank matrix Wiener filter (basis-free, untruncated)

    basis : str or ndarray, optional
        'signal', 'difference', 'noise', 'pca', 'random', 'compare', or a custom
        ``[nunits x D]`` orthonormal matrix. Default (None): library default.
    criterion : str, optional
        'max-tradeoff', 'prediction', 'variance', 'variance_eigenvalues', or 'wiener'.
    threshold_method : {'global', 'hybrid'}, optional
        Single population threshold ('global') or per-unit thresholds ('hybrid').
    basis_ordering : {'eigenvalues', 'signalvariance'}, optional
        How to rank basis dimensions.
    variance_threshold : float, optional
        Fraction retained for the 'variance' criteria, in [0, 1].
    allowable_thresholds : array-like, optional
        Restrict the threshold to these values (PSN picks the best one among them).
    unit_groups : array-like, optional
        ``[nunits]`` integer labels; units sharing a label share a threshold (hybrid).
    alpha : float, optional
        Interpolation in [0, 1] between the prediction peak (0) and the
        trial-average / do-nothing point (1). Does not use variance_threshold.
    gsn_result : dict, optional
        Precomputed ``{'cSb', 'cNb'}`` to skip GSN estimation (e.g. for sweeps).
    gsn_args : dict, optional
        Extra options forwarded to the GSN routine.
    device : str, optional
        'cpu' (default), 'cuda', 'mps', or 'auto'.
    split_half_metric : {'correlation', 'mse'}, optional
        Metric used for the split-half reliability diagnostic.
    cmap : matplotlib colormap, optional
        Colormap for the data/denoised/residual panels in diagnostic figures.
    wantverbose : bool, optional
        Print progress messages. Default: True.
    wantfig : bool, optional
        Generate the diagnostic figure during ``fit``. Default: False.
        Can be overridden per-call via ``fit(..., visualize=True)``.

    Attributes
    ----------
    denoiseddata_ : ndarray [nunits x nconds]
        Denoised trial-averaged training data.
    residuals_ : ndarray [nunits x nconds x ntrials]
        Training residuals (original minus denoised).
    denoiser_ : ndarray [nunits x nunits]
        Learned denoising matrix (applied as ``denoiser_.T @ x``).
    unit_means_ : ndarray [nunits]
        Per-unit means used for centering.
    best_threshold_ : int or ndarray
        Retained dimensions (scalar for global, per-unit array for hybrid).
    svnv_before_, svnv_after_ : ndarray [nunits x 2]
        Per-unit signal/noise variance before and after denoising.
    fullbasis_, basis_eigenvalues_ : ndarray
        Basis vectors (global order) and their eigenvalues.
    signalvar_, noisevar_, objective_ : ndarray
        Per-dimension signal/noise variance and the threshold-selection curve.
    gsn_result_ : dict
        GSN covariances (``cSb``, ``cNb``) and related outputs.
    opt_used_ : dict
        The fully-resolved options PSN actually used.
    threshold_selection_, recovery_tradeoff_, diagnostics_ : dict or None
        Present for 'compare'/'wiener' (and recovery diagnostics on every run).
    signalsubspace_, dimreduce_ : ndarray or None
        Selected subspace and low-dim representation (global mode only).
    unit_signal_vars_, unit_noise_vars_, unit_objectives_ : list or None
        Per-unit diagnostics (hybrid mode only).
    input_data_ : ndarray
        Copy of the training data (kept for plot_diagnostics).

    Notes
    -----
    ``PSN`` follows the scikit-learn estimator/transformer protocol (``fit``,
    ``transform``, ``fit_transform``, ``get_params``, ``set_params``, ``clone``)
    and works as a single :class:`~sklearn.pipeline.Pipeline` step. It is
    deliberately NOT wired for scikit-learn model selection: ``X`` is
    ``[nunits, nconds, ntrials]`` (units on axis 0, not samples) and PSN is
    unsupervised, so ``GridSearchCV`` / ``cross_val_score`` / ``check_estimator``
    do not apply out of the box - their splitters would partition units. To
    cross-validate, split the CONDITIONS axis yourself: fit on a subset of
    conditions and call ``transform`` on the held-out conditions.

    Examples
    --------
    >>> from psn import PSN
    >>> model = PSN(mode='standard', wantverbose=False)
    >>> denoised = model.fit_transform(train)        # [nunits x nconds]
    >>> denoised_test = model.fit(train).transform(test)
    >>> # Hyperparameters are introspectable / settable (sklearn protocol):
    >>> model.get_params()['threshold_method']
    >>> model.set_params(criterion='prediction')
    """

    # Option keys forwarded to psn() when the corresponding parameter is not None.
    _PASSTHROUGH = (
        'basis', 'criterion', 'threshold_method', 'basis_ordering',
        'variance_threshold', 'allowable_thresholds', 'unit_groups', 'alpha',
        'gsn_result', 'gsn_args', 'device', 'split_half_metric', 'cmap',
    )
    _VALID_MODES = ('conservative', 'standard', 'aggressive', 'compare', 'wiener')

    def __init__(self, mode=None, *, basis=None, criterion=None,
                 threshold_method=None, basis_ordering=None, variance_threshold=None,
                 allowable_thresholds=None, unit_groups=None, alpha=None,
                 gsn_result=None, gsn_args=None, device=None, split_half_metric=None,
                 cmap=None, wantverbose=True, wantfig=False):
        # sklearn contract: __init__ only stores params verbatim (no logic).
        self.mode = mode
        self.basis = basis
        self.criterion = criterion
        self.threshold_method = threshold_method
        self.basis_ordering = basis_ordering
        self.variance_threshold = variance_threshold
        self.allowable_thresholds = allowable_thresholds
        self.unit_groups = unit_groups
        self.alpha = alpha
        self.gsn_result = gsn_result
        self.gsn_args = gsn_args
        self.device = device
        self.split_half_metric = split_half_metric
        self.cmap = cmap
        self.wantverbose = wantverbose
        self.wantfig = wantfig

    # sklearn fitted-state protocol ----------------------------------------
    @property
    def _is_fitted(self):
        """True once fit() has populated the learned attributes."""
        return hasattr(self, 'denoiser_')

    def __sklearn_is_fitted__(self):
        return self._is_fitted

    def _build_opt(self, visualize):
        """Assemble the psn() options dict from the non-None parameters."""
        opt = {}
        for name in self._PASSTHROUGH:
            val = getattr(self, name)
            if val is not None:
                opt[name] = val
        opt['wantverbose'] = self.wantverbose
        opt['wantfig'] = self.wantfig if visualize is None else bool(visualize)
        return opt

    def fit(self, X, y=None, visualize=None):
        """Learn the denoiser from training data.

        Parameters
        ----------
        X : ndarray [nunits x nconds x ntrials]
            Training data (ntrials >= 2).
        y : ignored
            Present for scikit-learn API compatibility.
        visualize : bool, optional
            Override <wantfig> for this call (generate the diagnostic figure).

        Returns
        -------
        self
        """
        opt = self._build_opt(visualize)

        if self.mode is None:
            results = psn(X, opt)
        elif self.mode in self._VALID_MODES:
            results = psn(X, self.mode, opt)
        else:
            raise ValueError(
                f"mode must be None or one of {self._VALID_MODES}; got {self.mode!r}")

        # Core outputs
        self.denoiseddata_ = results['denoiseddata']
        self.residuals_ = results['residuals']
        self.unit_means_ = results['unit_means']
        self.denoiser_ = results['denoiser']

        # Diagnostics
        self.svnv_before_ = results['svnv_before']
        self.svnv_after_ = results['svnv_after']
        self.best_threshold_ = results['best_threshold']
        self.fullbasis_ = results['fullbasis']
        self.basis_eigenvalues_ = results['basis_eigenvalues']
        self.signalvar_ = results['signalvar']
        self.noisevar_ = results['noisevar']
        self.objective_ = results['objective']

        # GSN
        self.gsn_result_ = results['gsn_result']

        # Visualization (original order)
        self.basis_viz_ = results['basis_viz']
        self.signal_proj_viz_ = results['signal_proj_viz']
        self.noise_proj_viz_ = results['noise_proj_viz']
        self.basis_eigenvalues_viz_ = results['basis_eigenvalues_viz']
        self.input_data_ = results['input_data']
        self.opt_used_ = results['opt_used']

        # Mode-dependent / optional outputs
        self.signalsubspace_ = results.get('signalsubspace')
        self.dimreduce_ = results.get('dimreduce')
        self.unit_signal_vars_ = results.get('unit_signal_vars')
        self.unit_noise_vars_ = results.get('unit_noise_vars')
        self.unit_objectives_ = results.get('unit_objectives')
        self.threshold_selection_ = results.get('threshold_selection')
        self.recovery_tradeoff_ = results.get('recovery_tradeoff')
        self.diagnostics_ = results.get('diagnostics')

        return self

    def transform(self, X):
        """Apply the learned denoiser to new data.

        Parameters
        ----------
        X : ndarray
            ``[nunits x nconds x ntrials]`` (trial-averaged first) or
            ``[nunits x nconds]``. Must match the fitted ``nunits``.

        Returns
        -------
        denoised : ndarray [nunits x nconds]
        """
        if not self._is_fitted:
            raise RuntimeError("PSN must be fitted before calling transform(). "
                               "Call fit() first.")
        X = np.asarray(X)
        if X.ndim not in (2, 3):
            raise ValueError(f"Data must be 2D or 3D, got {X.ndim}D array.")

        nunits_expected = self.denoiser_.shape[0]
        if X.shape[0] != nunits_expected:
            raise ValueError(f"Data has {X.shape[0]} units, but model was fitted "
                             f"with {nunits_expected} units.")

        if X.ndim == 3:
            trial_avg = np.nanmean(X, axis=2) if np.any(np.isnan(X)) else np.mean(X, axis=2)
        else:
            trial_avg = X

        means = self.unit_means_[:, np.newaxis]
        return self.denoiser_.T @ (trial_avg - means) + means

    def fit_transform(self, X, y=None, visualize=None):
        """Fit, then return the denoised trial-averaged training data."""
        self.fit(X, visualize=visualize)
        return self.denoiseddata_

    def plot_diagnostics(self, **kwargs):
        """Generate the diagnostic figure for the fitted (training) data.

        Parameters
        ----------
        figurepath : str, optional
            If given, save the figure there.
        cmap : matplotlib colormap, optional
            Override the data/denoised/residual colormap.
        **kwargs
            Forwarded to ``plot_diagnostic_figures``.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if not self._is_fitted:
            raise RuntimeError("PSN must be fitted before calling plot_diagnostics(). "
                               "Call fit() first.")

        from .utilities.plotting.visualization import plot_diagnostic_figures

        # Reconstruct the results dict from stored attributes (avoids holding the
        # full dict; keeps pickling lean).
        results = {
            'denoiseddata': self.denoiseddata_,
            'residuals': self.residuals_,
            'unit_means': self.unit_means_,
            'denoiser': self.denoiser_,
            'svnv_before': self.svnv_before_,
            'svnv_after': self.svnv_after_,
            'best_threshold': self.best_threshold_,
            'fullbasis': self.fullbasis_,
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
        for key in ('signalsubspace', 'dimreduce', 'unit_signal_vars',
                    'unit_noise_vars', 'unit_objectives', 'threshold_selection',
                    'recovery_tradeoff', 'diagnostics'):
            val = getattr(self, key + '_', None)
            if val is not None:
                results[key] = val

        if 'cmap' not in kwargs and self.opt_used_.get('cmap') is not None:
            kwargs['cmap'] = self.opt_used_['cmap']

        return plot_diagnostic_figures(self.input_data_, results, **kwargs)

    # pickling: strip callables (e.g. cmap) from opt_used_ ------------------
    def __getstate__(self):
        state = super().__getstate__()
        opt = state.get('opt_used_')
        if isinstance(opt, dict):
            state = {**state, 'opt_used_': self._sanitize_dict_for_pickle(opt)}
        return state

    @staticmethod
    def _sanitize_dict_for_pickle(d):
        """Recursively drop callables (lambdas, colormaps) from a dict."""
        result = {}
        for k, v in d.items():
            if callable(v):
                continue
            result[k] = PSN._sanitize_dict_for_pickle(v) if isinstance(v, dict) else v
        return result
