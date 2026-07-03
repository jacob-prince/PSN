"""PSN (Partitioning Signal and Noise) - Main denoising function.

This module provides the main psn() function for denoising neural data using
Partitioning Signal and Noise (PSN) methodology.
"""

import time
import numpy as np

# Import utilities
from ._device import report_device_status, select_pipeline_device
from .utils import perform_gsn
from .utilities.input.parse_inputs import parse_inputs
from .utilities.input.validate_data import validate_data
from .utilities.input.set_default_options import set_default_options
from .utilities.input.load_gsn_result import load_gsn_result
from .utilities.basis.construct_basis import construct_basis
from .utilities.basis.project_covs import project_covs
from .utilities.basis.use_cached_eigvecs import use_cached_eigvecs
from .utilities.denoise.denoise_global import denoise_global
from .utilities.denoise.denoise_unitwise import denoise_unitwise
from .utilities.denoise.denoise_wiener import denoise_fullrank_wiener
from .utilities.threshold.select_threshold import select_threshold, format_threshold
from .utilities.diagnostics.compute_signal_noise_diagnostics import compute_signal_noise_diagnostics
from .utilities.diagnostics.recovery_tradeoff import attach_recovery_tradeoff
from .utilities.plotting.safe_visualize_results import safe_visualize_results


def psn(*args, **kwargs):
    """PSN  Denoise neural data using PSN (Partitioning Signal and Noise).

    -------------------------------------------------------------------------
    Algorithm:
    -------------------------------------------------------------------------

    PSN works by:
      1) Estimating signal and noise covariances via GSN
      2) Constructing an orthonormal basis (eigenvectors of cSb or cSb - cNb/ntrials_avg)
      3) Projecting the signal and noise covariances into that basis
      4) Ranking dimensions by signal variance (or eigenvalues for the difference basis)
      5) Selecting a threshold on the analytic recovery curve (or a variance target)
      6) Building a denoiser that projects data onto the selected dimensions
      7) Reconstructing the denoised data
      8) Using this signal estimate to compute the noise estimate (residuals)

    For unit-specific methods (threshold_method='hybrid'), each unit gets weighted
    projections based on the squared basis coefficients for that unit, and
    potentially different thresholds.

    -------------------------------------------------------------------------
    Usage:
    -------------------------------------------------------------------------

    results = psn(data) is the default version of PSN and is identical to psn(data,'standard').
              It uses the signal basis at the "max-tradeoff" threshold: a deterministic
              operating point that captures the bulk of the achievable recovery while retaining
              more signal variance than the prediction peak, with a single population (global)
              threshold. It is shorthand for
              psn(data,{'basis':'signal','criterion':'max-tradeoff','threshold_method':'global'})

    results = psn(data,'conservative', opt) prioritizes retaining signal, and is shorthand for
              psn(data,{'basis':'signal','criterion':'variance','threshold_method':'global','variance_threshold':0.99})

    results = psn(data,'standard', opt) is identical to the default psn(data): the signal basis at the
              max-tradeoff threshold with a single population (global) threshold. It is shorthand for
              psn(data,{'basis':'signal','criterion':'max-tradeoff','threshold_method':'global'})

    results = psn(data,'aggressive', opt) uses a more aggressive denoising approach (difference basis at
              the prediction peak). This may yield higher analytic recovery compared to
              'standard' but may yield unstable results in cases of limited data. It is shorthand for
              psn(data,{'basis':'difference','criterion':'prediction','threshold_method':'global'})

    results = psn(data,'wiener', opt) applies the full-rank matrix Wiener filter (the optimal
              linear estimator that minimizes expected mean-squared error given the signal and
              noise covariances). It is shorthand for psn(data,{'criterion':'wiener'})

    results = psn(data,'compare', opt) builds both the signal and difference bases at their
              max-tradeoff thresholds and keeps whichever has the higher empirical split-half
              reliability (split-half r) at that threshold. It is shorthand for
              psn(data,{'basis':'compare','criterion':'max-tradeoff','threshold_method':'global'})

    results = psn(data, opt) is a version of PSN where the user customizes the settings.

    In all cases, <opt> can be omitted and default parameters are used.

    When a named mode is combined with an <opt> dict, the dict OVERRIDES the mode's
    defaults (the user's options take priority). For example,
    psn(data,'aggressive',{'threshold_method':'hybrid'}) keeps the 'aggressive' basis
    and criterion but switches to hybrid thresholds. The 'wiener' mode is the
    exception: it is basis-free and untruncated, so combining it with a conflicting
    <basis>, <criterion>, <threshold_method>, <basis_ordering>, <allowable_thresholds>,
    <variance_threshold>, <alpha>, or <unit_groups> raises a ValueError instead.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <data> - shape [nunits x nconds x ntrials]. Measured responses for each
      unit, condition, and trial. Requires ntrials >= 2.

      NaN HANDLING (uneven trials): conditions need not have the full set of
      trials: set data[:,i,j] to NaN for missing trials. Each condition must
      still have at least one trial valid across ALL units (no NaNs). PSN uses
      the average number of valid trials per condition in noise/ntrials formulas
      (following GSN). The denoised output contains no NaNs (PSN fills them in);
      residuals preserve the input NaN positions. (See perform_gsn for details.)

    <opt> (optional) - dict with the following fields. The options most users
    touch are in the first group; the rest are advanced and can be ignored.

    This dict can be passed positionally (psn(data, {...})) or as the opt keyword
    (psn(data, opt={...})). Every option below can also be given individually as a
    keyword argument, e.g. psn(data, device='cuda') or psn(data, 'standard',
    basis='signal'); keyword options take precedence over a positional options dict.

    ----- BASIS & THRESHOLD SELECTION -----

      <basis> (optional) - basis specifier. Either string or matrix:
        'signal'     -> use signal basis (eigenvectors of signal covariance, cSb)
        'difference' -> use "difference" basis (eigenvectors of cSb - cNb / ntrials_avg)
                        where ntrials_avg is the average number of trials (handles NaNs)
        B            -> user-supplied basis vectors B, size [nunits x D], D >= 1, with
                        orthonormal columns (B.T @ B = I).
        'noise'      -> [NOT RECOMMENDED] use noise basis (eigenvectors of noise covariance, cNb)
        'pca'        -> [NOT RECOMMENDED] use eigenvectors of covariance of trial-averaged data
        'random'     -> [NOT RECOMMENDED] generate a random orthonormal basis
        'wiener'     -> [DEPRECATED] alias for criterion='wiener' (the full-rank Wiener
                        filter). Wiener is not a basis; see <criterion>.
        'compare'    -> build both the 'signal' and 'difference' bases, each at its
                        own max-tradeoff threshold, then keep whichever has the higher
                        empirical split-half reliability (split-half r) evaluated AT
                        those two thresholds. The chosen basis and the per-candidate
                        results for both bases (each basis's max-tradeoff K, analytic
                        recovery, and split-half r) are returned in
                        results['threshold_selection'] and results['diagnostics'].
        Default: 'signal'.

      <criterion> (optional) - string. How to determine the threshold. Several
        criteria operate on the ANALYTIC RECOVERY curve: the analytic
        (GSN-covariance-based) estimate of how well the denoised output recovers
        the true underlying signal (the noise-free responses you would get with
        infinite data), out-of-sample. It is cumsum(signal - noise/ntrials_avg)
        over the retained dimensions: each kept dimension adds its signal
        variance but also injects noise variance into the trial average, so the
        cumulative net is the signal variance recovered net of injected noise.
        Its empirical counterpart in the diagnostic figure is the split-half
        reliability.
        'prediction'            -> retain the dimensions that maximize the analytic
                                   recovery (its cumulative peak)
        'max-tradeoff'          -> a deterministic operating point between 'prediction' and
                                   'variance' (in practice closer to 'variance'): it keeps the
                                   analytic recovery high while retaining more signal variance
                                   (dropping fewer dimensions) than 'prediction'. Geometrically,
                                   on the descending (prediction-peak -> trial-average / do-nothing)
                                   limb of the recovery curve it is the point farthest from the
                                   straight chord joining those two anchors. Equivalently, after
                                   rescaling each axis to [0,1] (x = fraction of signal variance
                                   retained, y = analytic recovery), it is the point that maximizes
                                   x + y. Fully analytic. Defined for any basis; cleanest with
                                   the signal basis (the default).
        'variance'              -> retain dimensions until a target fraction of signal variance is reached
        'variance_eigenvalues'  -> retain dimensions until a target fraction of the total sum of
                                   positive eigenvalues associated with the basis is reached
        'wiener'                -> full-rank matrix Wiener filter: D = Σ_S @ (Σ_S + Σ_N/t)^{-1},
                                   the linear estimator that minimizes expected mean-squared
                                   error given the signal and noise covariances. Unlike the
                                   other criteria
                                   this applies NO truncation (all dimensions are kept and
                                   continuously weighted) and it is basis-free. Because it
                                   bypasses the basis/criterion/threshold pipeline, supplying a
                                   conflicting <basis>, <threshold_method>, <basis_ordering>,
                                   <allowable_thresholds>, <variance_threshold>, <alpha>, or
                                   <unit_groups> raises a ValueError rather than being ignored.
        Default: 'max-tradeoff'.
        (Note that <criterion> set to 'variance_eigenvalues' is not compatible with
         the B and 'random' cases of <basis>, and is also not compatible with
         <threshold_method> as 'hybrid'.)
        (Also, note that when <basis> is 'signal' and <threshold_method> is 'global', then
         identical results are produced by 'variance' vs. 'variance_eigenvalues'.)

      <threshold_method> (optional) - string. How to select thresholds (i.e. the
        number of dimensions to retain):
        'global' -> single threshold for all units (symmetric denoiser)
        'hybrid' -> global ordering of basis vectors, unit-specific thresholds
        Default: 'global'.

      <alpha> (optional) - scalar in [0,1] or None. Interpolation parameter
        between the prediction peak and the trial-average (do-nothing) point,
        in signal-variance space:
          alpha=0   -> prediction peak (same as criterion='prediction')
          alpha=1   -> retain all signal variance (the trial average / do
                       nothing, keep every dimension)
          alpha=0.3 -> retain an additional 30% of the signal variance gap
                       between the prediction peak and the full signal variance
        When set, overrides the criterion setting. alpha does NOT use
        variance_threshold (its right endpoint is fixed at the full signal
        variance). Does not apply to criterion='wiener'.
        Default: None (disabled; existing criterion logic used).

      <variance_threshold> (optional) - scalar in [0,1]. Target fraction used
        when <criterion> is 'variance' or 'variance_eigenvalues'. (It does not
        affect <alpha>.)
        Default: 0.99.

    ----- ADVANCED -----

      <basis_ordering> (optional) - string. How to set the initial global order of basis vectors:
        'eigenvalues'    -> use descending order of eigenvalues (if available)
        'signalvariance' -> measure signal variance and use descending order of signal variance
        Default: 'eigenvalues'.
        (Note that when <basis> is B or 'random', eigenvalues are not available, so we
         necessarily fall back to 'signalvariance'.)

      <allowable_thresholds> (optional) is an array of thresholds that are acceptable.
        For example, a threshold of 7 means to retain the first 7 dimensions.
        PSN selects the BEST threshold among the allowable values for the chosen
        criterion, never evaluating a threshold outside the set: the allowable
        dimensionality that maximizes the criterion ('prediction'/'max-tradeoff')
        or first reaches the variance target ('variance'/'variance_eigenvalues').
        Setting <allowable_thresholds> to a single value forces PSN to use exactly
        that many dimensions.
        Default: None (no constraint; any threshold between 0 and D is allowed).

      <unit_groups> (optional) - [nunits] array of non-negative integers specifying
          which units must share the same threshold (applies when <threshold_method>
          is 'hybrid'). Units with the same integer label are treated
          as a group and receive the same threshold (determined by averaging
          the criterion across units of that group). If None or omitted, the
          default behavior is equivalent to np.arange(nunits), which indicates that
          each unit forms its own group (i.e., distinct threshold for each unit).
          For <threshold_method> = 'global', <unit_groups> is ignored (defaults to
          all zeros internally).

    ----- PERFORMANCE -----

      <gsn_result> (optional) - dict with keys 'cSb' and 'cNb' from a previous
        PSN call (i.e. results['gsn_result']). When provided, PSN skips the
        expensive GSN estimation and uses these covariances directly. This is
        useful for sweeping over hyperparameters (alpha, basis, criterion, etc.)
        on the same data without re-running GSN each time.
        Default: None (run GSN normally).

      <gsn_args> (optional) - dict of options passed directly to the GSN routine
          (perform_gsn). Typical fields might include:
              wantshrinkage  - whether to use covariance shrinkage (default: True)
              wantverbose    - whether to print diagnostic output (default: False)
              random_seed    - RNG seed used inside GSN
          If None or omitted, defaults are used.

      <device> (optional) - string or torch.device. Hardware for the heavy linear
        algebra (GSN covariance estimation, basis eigendecomposition, covariance
        projections, denoiser construction). Pass your data as an ordinary numpy
        array exactly as usual - PSN moves the internal arrays onto the device for
        you; you do NOT pre-move the data. A GPU is used ONLY when explicitly asked:
            'cpu'          - (default) run on CPU: torch-CPU when torch is
                             importable, else numpy. No GPU is touched.
            'cuda'         - run on the CUDA GPU (needs a torch build with CUDA;
                             check torch.cuda.is_available()).
            'mps'          - run on the Apple-Silicon GPU (Metal).
            'auto' / None  - treated as 'cpu'. This deliberately does NOT auto-pick
                             a GPU, so a cluster job never lands on one by accident.
            a torch.device - torch.device('cuda')/('mps') is honored; a CPU
                             torch.device maps to 'cpu'.
        The value is normalized once and propagated to every stage, so the whole
        pipeline runs on the chosen device. GPU acceleration pays off mainly at
        large nunits (~10k+), where the matmul/solve work dominates; for small
        problems CPU is usually as fast (no host<->device transfer overhead).
        Requires PyTorch (with CUDA support for 'cuda'). Default: 'cpu'.

        Example (CUDA GPU on a cluster):
            results = psn(data, device='cuda')                # keyword form
            results = psn(data, 'standard', device='cuda')    # a mode + keyword
            results = psn(data, {'basis': 'signal', 'device': 'cuda'})  # options dict

    ----- FIGURE -----

      <split_half_metric> (optional) - 'correlation' or 'mse'. Metric for the
        split-half reliability panel of the diagnostic figure ONLY ('correlation'
        = Pearson r per unit, 'mse' = mean squared error per unit). Does not
        affect the denoising in any way.
        Default: 'correlation'.

      <wantfig> - boolean. Whether to generate diagnostic figures.
        Default: True.

      <figurepath> (optional) - string. Path to save the diagnostic figure.
        Can be a full path (e.g., '/path/to/figure.png') or just a filename
        (e.g., 'psn_diagnostics.png') to save in the current directory.
        If specified, the figure is saved to this path and closed after saving.
        If None or omitted, figures are displayed but not automatically saved.
        Default: None.

      <wantverbose> - boolean. Whether to show messages during execution.
        Default: True.

      <skip_split_half> - boolean. When True, keep the analytic recovery curve
        but skip the empirical split-half computation in the recovery-tradeoff
        diagnostic (the per-K, per-unit reliability loop that dominates runtime
        at large nunits and on NaN data). results['recovery_tradeoff'] is still
        populated with the analytic curve; its split-half fields are None.
        Default: False.

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    The function returns a single dict <results> with the following fields:

    Returned in all cases:

      results['denoiseddata'] - [nunits x nconds]. Trial-averaged data after applying
                         the denoiser. This is PSN's estimate of the signal.

      results['residuals']    - [nunits x nconds x ntrials]. The original data minus
                         the <denoiseddata>. This is PSN's estimate of the noise.

      results['unit_means']  - [nunits]. Mean response per unit. Note that PSN
                        subtracts the mean response per unit before the denoising
                        projection and then re-adds the means afterwards.

      results['denoiser']    - [nunits x nunits]. The denoising matrix. For global mode,
                        this is symmetric. For hybrid mode, this is generally
                        non-symmetric. The denoising is: denoiser.T @ (data - means) + means.

      results['svnv_before'] - [nunits x 2]. The total signal variance (column 0) and
                        total noise variance (column 1) for each unit, before PSN.

      results['svnv_after']  - [nunits x 2]. The total signal variance (column 0) and
                        total noise variance (column 1) for each unit, after PSN.

      results['best_threshold'] - optimal threshold(s) chosen by the algorithm.
          • If thresholding is global, this is a scalar,
            i.e. the number of dimensions retained.
          • If thresholding is unit-specific, this is [nunits], giving the
            threshold for each unit. (Note that units in the same <unit_groups>
            will share the same value).

      results['fullbasis']   - [nunits x dims]. The full set of basis vectors after global
                        ordering. All units share this ordering.

      results['basis_eigenvalues'] - [dims]. Eigenvalues from basis construction (e.g.,
                        from signal covariance, difference matrix, etc.), sorted to
                        match the column order of fullbasis. None for custom
                        or random bases. For visualization of original order, see
                        basis_eigenvalues_viz.

      results['gsn_result']  - dict. Full results from the GSN algorithm containing:
                                .cSb  - signal covariance
                                .cNb  - noise covariance
                            plus any additional outputs from the GSN code.
                            This can be passed back via opt['gsn_result'] to skip
                            recomputing GSN on subsequent calls with the same data.

      results['input_data']  - copy of the original input data

      results['signalvar']   - [dims] or list. Signal variance per dimension.

      results['noisevar']    - [dims] or list. Noise variance per dimension.

      results['objective']   - [dims+1]. Cumulative objective curve that was actually
                        used for threshold selection. For 'prediction' and 'max-tradeoff'
                        criteria (and 'alpha'), this is the analytic recovery curve
                        cumsum(signal - noise/ntrials). For 'variance', this is cumsum(signal).
                        For 'variance_eigenvalues', this is cumsum(positive eigenvalues).

    Special outputs returned only when <threshold_method> is 'global':

      results['signalsubspace'] - [nunits x K]. The final set of basis vectors selected
        for denoising (i.e. the subspace into which data are projected).

      results['dimreduce'] - [K x nconds]. This is the low-dimensional representation
        of the denoised data in the selected subspace.

    Special outputs for unit-specific thresholds (threshold_method 'hybrid'):

      results['unit_signal_vars'] - list of signal variances per unit
      results['unit_noise_vars']  - list of noise variances per unit
      results['unit_objectives']  - list of objective curves per unit

    Visualization outputs (original order before global ranking):

      results['basis_viz']       - [nunits x dims]. Basis vectors in original order
                                (before global ranking/sorting).
      results['signal_proj_viz'] - [dims]. Signal variance projections in original order.
      results['noise_proj_viz']  - [dims]. Noise variance projections in original order.
      results['basis_eigenvalues_viz'] - [dims]. Eigenvalues in original order,
                                matching column order of basis_viz.

    Metadata:

      results['opt_used']   - dict of the resolved options actually used.
      results['runtime']    - scalar. Wall-clock seconds for the psn() call.
      results['recovery_tradeoff'] - dict (always present). Diagnostic data backing
                            the recovery-tradeoff figure: per-basis analytic-recovery
                            and split-half curves vs. fraction of signal variance
                            retained, the trial-average and (when applicable) Wiener
                            reference points, and the chosen operating point.
      results['threshold_selection'] - dict, set for 'compare' and 'wiener'. Records
                            the chosen mode/basis/criterion/best_threshold (+ recovery
                            and sv_frac for 'compare').
      results['diagnostics'] - dict, set for 'compare'. Per-candidate selection
                            metrics (best_threshold/recovery/sv_frac for signal and
                            difference) plus the cached candidate eigenbases.
      results['alpha_info']  - dict, set when <alpha> is active: the prediction-peak
                            (k_pred) and do-nothing (k_var) dimension counts the
                            alpha threshold interpolates between, and alpha.
      results['wiener_matrix'] - [nunits x nunits], set only for criterion='wiener':
                            the full-rank Wiener filter D that is applied to the data.

    """

    # =========================================================================
    # STEP 1: Parse and validate inputs
    # =========================================================================

    _t_start = time.time()

    data, opt = parse_inputs(*args, **kwargs)
    nunits, nconds, ntrials, ntrials_avg, has_nans = validate_data(data)
    opt = set_default_options(opt, nunits)

    # Normalize the requested device ONCE, then propagate this single value to
    # every stage (GSN, basis eigh, projections, denoising, figure). Policy: a
    # GPU is used only when explicitly requested ('cuda'/'mps'); 'cpu'/'auto'/
    # unset stay on CPU (best CPU backend per component). This is the single
    # source of truth so device handling is seamless across the whole pipeline.
    opt['device'] = select_pipeline_device(opt.get('device', 'cpu'))

    # Remember the basis the user asked for, before use_cached_eigvecs may swap
    # it to a matrix - the recovery-tradeoff policy keys off the original basis.
    _orig_basis = opt.get('basis')

    report_device_status(opt)

    # Verbose decision-trace helpers. Each line is tagged with the pipeline stage
    # it belongs to ([PSN] <stage> | ...); _vsub indents a detail under its stage.
    def _vlog(stage, msg):
        print(f"[PSN] {stage:<8}| {msg}")

    def _vsub(msg):
        print(f"[PSN] {'':<8}|   {msg}")

    # Per-stage wall-clock: GSN estimation, figure generation, and (by remainder)
    # PSN's own work. _done prints the breakdown + total on completion.
    _gsn_time = 0.0
    _fig_time = 0.0

    def _done(msg):
        total = time.time() - _t_start
        psn_t = max(0.0, total - _gsn_time - _fig_time)
        _vlog('done', f"{msg}; runtime: GSN {_gsn_time:.3f}s + PSN {psn_t:.3f}s + "
              f"figure {_fig_time:.3f}s = total {total:.3f}s")

    if opt['wantverbose']:
        if has_nans:
            _vlog('input', f"{nunits} units x {nconds} conditions x {ntrials} trials; "
                  f"NaNs present -> uneven-trials path, avg {ntrials_avg:.2f} valid trials/condition")
        else:
            _vlog('input', f"{nunits} units x {nconds} conditions x {ntrials} trials; "
                  f"no NaNs -> using all {ntrials} trials")
        _bdesc = _orig_basis if isinstance(_orig_basis, str) else 'custom matrix'
        _cfg = f"basis={_bdesc}, criterion={opt['criterion']}, threshold_method={opt['threshold_method']}"
        if opt.get('alpha') is not None:
            _cfg += f", alpha={opt['alpha']} (overrides criterion)"
        _vlog('config', _cfg)

    # =========================================================================
    # STEP 2: Compute unit means
    # =========================================================================
    # PSN removes the mean response of each unit before denoising, then adds
    # it back afterward. This ensures we're only denoising the fluctuations
    # around the mean, not the mean itself.

    if has_nans:
        trial_avg = np.nanmean(data, axis=2)  # [nunits x nconds] - ignore NaNs
    else:
        trial_avg = np.mean(data, axis=2)  # [nunits x nconds]
    unit_means = np.mean(trial_avg, axis=1)  # [nunits]

    # =========================================================================
    # STEP 3: Estimate signal and noise covariances using GSN
    # =========================================================================
    # GSN (Generative Modeling of Signal and Noise) estimates the covariance matrices
    # that describe the signal and noise in the data.

    _t_gsn = time.time()
    if opt.get('gsn_result') is not None:
        gsn_result = load_gsn_result(opt['gsn_result'])
        if 'cSb' not in gsn_result or 'cNb' not in gsn_result:
            raise ValueError("opt['gsn_result'] must contain 'cSb' and 'cNb' keys")
        if gsn_result['cSb'].shape[0] != nunits:
            raise ValueError(
                f"gsn_result covariance size ({gsn_result['cSb'].shape[0]}) "
                f"does not match data ({nunits} units)")
        # Auto-upgrade: when the user asked for basis='signal' or
        # 'difference' AND the loaded gsn_result has matching eigvecs +
        # eigvals cached, swap basis to the matrix + propagate the
        # eigvalues. PSN's downstream then skips its own eigh (which is
        # the dominant cost at large nunits) while producing bit-
        # equivalent results to the string-basis path.
        opt = use_cached_eigvecs(opt, gsn_result)
        if opt['wantverbose']:
            _vlog('GSN', f"reusing provided gsn_result (skipping GSN); covariances {nunits}x{nunits}")
    else:
        if opt['wantverbose']:
            _vlog('GSN', 'estimating covariances via GSN (no gsn_result provided)')
        gsn_opt = opt['gsn_args'] if opt['gsn_args'] is not None else {}
        if 'wantverbose' not in gsn_opt:
            gsn_opt['wantverbose'] = False
        if 'wantshrinkage' not in gsn_opt:
            gsn_opt['wantshrinkage'] = True
        # Propagate the pipeline device so GSN's shrinkage estimation (and, on
        # the cluster, its covariance math) runs on the same backend PSN uses.
        # An explicit gsn_args['device'] still wins.
        if 'device' not in gsn_opt:
            gsn_opt['device'] = opt['device']
        gsn_result = perform_gsn(data, gsn_opt)
    _gsn_time = time.time() - _t_gsn

    cSb = gsn_result['cSb']  # signal covariance (symmetric)
    cNb = gsn_result['cNb']  # noise covariance (symmetric)

    # =========================================================================
    # HIGH-LEVEL SELECTION (basis='compare', or the full-rank Wiener
    # criterion). select_threshold resolves these to a concrete run; explicit
    # 'signal'/'difference', custom matrices, and 'noise'/'pca'/'random' fall
    # through to STEP 4 unchanged.
    # =========================================================================
    _crit = opt.get('criterion')
    _basis_str = opt['basis'] if isinstance(opt['basis'], str) else None
    if _basis_str in ('compare', 'wiener') or _crit == 'wiener':
        tsel = select_threshold(cSb, cNb, ntrials_avg, opt, device=opt['device'],
                                data=data, unit_means=unit_means, has_nans=has_nans)

        # ---- full-rank matrix Wiener (criterion='wiener' or legacy basis='wiener') ----
        # D = Σ_S @ (Σ_S + Σ_N/t)^{-1}, the optimal linear estimator. Basis-free
        # and untruncated, so steps 4-7 are skipped.
        if tsel['family'] == 'fullrank':
            if opt['wantverbose']:
                _vlog('select', 'criterion=wiener -> full-rank Wiener filter; skipping basis & truncation')
            results = denoise_fullrank_wiener(
                cSb, cNb, data, trial_avg, unit_means, ntrials_avg, nunits, gsn_result, opt)
            results['threshold_selection'] = {
                'mode': 'wiener', 'basis': None, 'criterion': 'wiener',
                'best_threshold': results.get('best_threshold')}
            attach_recovery_tradeoff(
                results, cSb, cNb, ntrials_avg, data, unit_means, has_nans,
                'wiener', nunits, device=opt['device'],
                skip_split_half=opt.get('skip_split_half', False))
            if opt['wantfig']:
                if opt['wantverbose']:
                    _vlog('figure', 'generating diagnostic figure')
                _t_fig = time.time()
                safe_visualize_results(results, opt)
                _fig_time += time.time() - _t_fig
            results['runtime'] = time.time() - _t_start
            if opt['wantverbose']:
                _done(f"full-rank Wiener filter ({results['best_threshold']:.1f} effective dims)")
            return results

        # ---- low-rank: run the chosen concrete basis once, reusing the GSN fit ----
        sub_opt = dict(opt)
        # Reuse the eigenbasis select_threshold already built for the winner, so the
        # recursive run skips construct_basis's eigh (the dominant O(N^3) cost). Passing
        # the eigvecs as a matrix + their eigenvalues is end-to-end equivalent to the
        # string basis (same use_cached_eigvecs path; locked by test_gsn_eigenbasis_handoff).
        if tsel.get('eigvecs') is not None and tsel.get('eigvals') is not None:
            sub_opt['basis'] = tsel['eigvecs']
            sub_opt['basis_eigenvalues'] = tsel['eigvals']
        else:
            sub_opt['basis'] = tsel['basis']
        sub_opt['criterion'] = tsel['criterion']
        sub_opt['wantfig'] = False
        sub_opt['wantverbose'] = False
        sub_opt['gsn_result'] = gsn_result          # reuse one GSN estimate (apples-to-apples)
        sub_opt['_skip_recovery_tradeoff'] = True    # attached once below
        results = psn(data, sub_opt)

        results['threshold_selection'] = {
            'mode': _orig_basis, 'basis': tsel['basis'], 'criterion': tsel['criterion'],
            'best_threshold': results.get('best_threshold'),
            'recovery': tsel.get('recovery'), 'sv_frac': tsel.get('sv_frac'),
            'split_half_r': tsel.get('split_half_r')}
        # For 'compare', diagnostics carries the per-candidate analytic recovery.
        # The empirical split-half curves for both bases are added by the figure
        # layer (results['recovery_tradeoff']) as out-of-sample validation only.
        results['diagnostics'] = tsel.get('diagnostics', {})

        # Recovery-tradeoff curves (analytic + split-half). Reuse any candidate
        # eigenbases select_threshold already built so no extra eigh is run.
        attach_recovery_tradeoff(
            results, cSb, cNb, ntrials_avg, data, unit_means, has_nans,
            _orig_basis, nunits, device=opt['device'],
            extra_bases=tsel.get('diagnostics', {}).get('eigvecs_by_basis'),
            skip_split_half=opt.get('skip_split_half', False))

        if opt['wantverbose']:
            ts = results['threshold_selection']
            cands = (results.get('diagnostics') or {}).get('candidates', {})
            if cands:
                cstr = '; '.join(
                    f"{b}: K={format_threshold(cands[b]['best_threshold'])}, "
                    f"split-half r={cands[b]['split_half_r']:.3f}"
                    for b in ('signal', 'difference') if b in cands)
                _vlog('basis', f"compare (split-half r at each max-tradeoff K): {cstr}")
            shr_ch = ts.get('split_half_r')
            shr_str = f"{shr_ch:.3f}" if shr_ch is not None else "n/a"
            _vsub(f"chose '{ts['basis']}' (higher split-half r={shr_str}; "
                  f"K={format_threshold(ts['best_threshold'])})")

        # Figure honors the user's settings but reflects the chosen basis.
        fig_opt = dict(results.get('opt_used', opt))
        for key in ('wantfig', 'figurepath', 'wantverbose', 'cmap', 'split_half_metric'):
            if key in opt:
                fig_opt[key] = opt[key]
        results['opt_used'] = fig_opt
        if opt['wantfig']:
            if opt['wantverbose']:
                _vlog('figure', 'generating diagnostic figure')
            _t_fig = time.time()
            safe_visualize_results(results, fig_opt)
            _fig_time += time.time() - _t_fig

        results['runtime'] = time.time() - _t_start
        if opt['wantverbose']:
            _done(f"chose '{ts['basis']}', retained "
                  f"{format_threshold(results.get('best_threshold'))} dimensions")
        return results

    # =========================================================================
    # STEP 4: Construct the denoising basis
    # =========================================================================
    # The basis is a set of orthonormal vectors that span the space in which
    # we'll perform denoising. Different basis choices emphasize different
    # aspects of the data structure.

    if opt['wantverbose']:
        _why = {'signal': 'eigendecomposition of cSb',
                'difference': 'eigendecomposition of cSb - cNb/ntrials',
                'pca': 'eigendecomposition of trial-averaged data covariance',
                'noise': 'eigendecomposition of cNb',
                'random': 'random orthonormal basis'}
        if isinstance(opt['basis'], str):
            _vlog('basis', f"{opt['basis']} -> {_why.get(opt['basis'], opt['basis'])}")
        elif isinstance(_orig_basis, str) and _orig_basis in ('signal', 'difference'):
            _vlog('basis', f"{_orig_basis} -> reusing cached eigvecs from gsn_result (no eigh)")
        else:
            _vlog('basis', f"custom matrix [{opt['basis'].shape[0]}x{opt['basis'].shape[1]}]")

    basis, basis_eigenvalues = construct_basis(
        cSb, cNb, opt['basis'], data, trial_avg, unit_means, ntrials_avg, has_nans,
        custom_basis_eigenvalues=opt.get('basis_eigenvalues'), device=opt['device'])

    # Validate allowable_thresholds against actual basis dimensions
    ndims = basis.shape[1]
    if opt['allowable_thresholds'] is not None:
        if np.any(np.array(opt['allowable_thresholds']) > ndims):
            raise ValueError(f'allowable_thresholds contains values exceeding number of basis dimensions ({ndims})')

    # =========================================================================
    # STEP 5: Project covariances into basis space
    # =========================================================================
    # We compute how much signal and noise variance each basis dimension contains
    # by projecting the GSN-derived signal and noise covariance matrices (cSb, cNb)
    # into the coordinate system of the basis that will be used for PSN denoising.

    if opt['wantverbose']:
        _vlog('basis', 'projecting covariances into the basis (per-dim signal/noise variance)')

    # Always use GSN-based projection for all basis types
    signal_proj, noise_proj = project_covs(cSb, cNb, basis,
                                            device=opt.get('device', 'cpu'))

    # Save original basis and projections for visualization (before reordering)
    basis_viz = basis.copy()
    signal_proj_viz = signal_proj.copy()
    noise_proj_viz = noise_proj.copy()
    basis_eigenvalues_viz = basis_eigenvalues.copy() if basis_eigenvalues is not None else None

    # =========================================================================
    # STEP 6: Rank basis dimensions (global ordering)
    # =========================================================================
    # We order the basis dimensions according to their importance based on basis_ordering:
    # - 'eigenvalues': rank by eigenvalues (if available)
    # - 'signalvariance': rank by signal variance
    # - 'prediction': rank by signal variance - noise variance / ntrials

    if opt['basis_ordering'] == 'eigenvalues' and basis_eigenvalues is not None:
        # Use eigenvalue-based ranking
        sort_idx_global = np.argsort(basis_eigenvalues)[::-1]  # Descending
        if opt['wantverbose']:
            _vlog('basis', 'ordering dimensions by eigenvalues (available for this basis)')
    elif opt['basis_ordering'] == 'prediction':
        # Use prediction objective (signal - noise/ntrials) for ranking
        prediction_obj = signal_proj - noise_proj / ntrials_avg
        sort_idx_global = np.argsort(prediction_obj)[::-1]  # Descending
        if opt['wantverbose']:
            _vlog('basis', 'ordering dimensions by prediction (signalvar - noisevar/ntrials)')
    else:
        # Use signal variance-based ranking (or fallback when eigenvalues unavailable)
        sort_idx_global = np.argsort(signal_proj)[::-1]  # Descending
        if opt['wantverbose']:
            if opt['basis_ordering'] == 'eigenvalues':
                _vlog('basis', 'eigenvalues unavailable (custom/random basis) -> ordering by signal variance')
            else:
                _vlog('basis', 'ordering dimensions by signal variance')

    # Reorder basis and projections according to global ranking
    basis = basis[:, sort_idx_global]
    signal_proj = signal_proj[sort_idx_global]
    noise_proj = noise_proj[sort_idx_global]
    if basis_eigenvalues is not None:
        basis_eigenvalues = basis_eigenvalues[sort_idx_global]

    # =========================================================================
    # STEP 7: Select thresholds and build denoising matrix
    # =========================================================================
    # Determine how many dimensions to retain. This depends on threshold_method:
    #   - 'global': single threshold for all units (symmetric denoiser)
    #   - 'hybrid': unit-specific thresholds with global basis ordering

    if opt['wantverbose']:
        if opt.get('alpha') is not None:
            _vlog('select', f"method={opt['threshold_method']}, alpha={opt['alpha']} overrides criterion")
        else:
            _vlog('select', f"method={opt['threshold_method']}, criterion={opt['criterion']}")

    if opt['threshold_method'] == 'global':
        # GLOBAL (POPULATION) MODE with hard truncation
        denoiser, best_threshold, objective, signalvar, noisevar, unit_cumsum_curves, \
        unit_signal_vars, unit_noise_vars = \
            denoise_global(basis, signal_proj, noise_proj, basis_eigenvalues,
                          ntrials_avg, opt)

    else:
        # HYBRID MODE: unit-specific thresholds with a shared global basis ordering
        denoiser, best_threshold, objective, signalvar, noisevar, unit_cumsum_curves, \
        unit_signal_vars, unit_noise_vars = \
            denoise_unitwise(basis, signal_proj, noise_proj, basis_eigenvalues,
                           ntrials_avg, opt)

    # Compute alpha_info for visualization (if alpha is active)
    alpha_info = None
    if opt.get('alpha') is not None:
        diff = signalvar - noisevar / ntrials_avg
        pred_obj = np.concatenate([[0], np.cumsum(diff)])
        k_pred = int(np.argmax(pred_obj))
        sig_cs = np.concatenate([[0], np.cumsum(signalvar)])
        total_signal = sig_cs[-1]
        # alpha interpolates toward the trial-average (do-nothing) endpoint:
        # full signal variance. It does NOT use variance_threshold, so k_var is
        # the do-nothing dimensionality (all dims).
        S_var = total_signal
        if total_signal <= 0:
            k_var = 0
        else:
            idx = np.where(sig_cs >= S_var)[0]
            k_var = int(idx[0]) if len(idx) > 0 else int(basis.shape[1])
            k_var = min(k_var, int(basis.shape[1]))
        alpha_info = {'k_pred': k_pred, 'k_var': k_var, 'alpha': opt['alpha']}

    # Decision trace: report the chosen threshold and *why* (wantverbose).
    if opt['wantverbose']:
        crit = opt['criterion']
        if opt['threshold_method'] != 'global':
            bt = np.atleast_1d(np.asarray(best_threshold))
            _vsub(f"hybrid per-unit thresholds -> mean {bt.mean():.1f} "
                  f"(range {int(bt.min())}-{int(bt.max())})")
        else:
            k = int(best_threshold)
            sig_cs = np.concatenate([[0.0], np.cumsum(np.asarray(signalvar, dtype=float))])
            sv_pct = 100.0 * sig_cs[k] / sig_cs[-1] if sig_cs[-1] > 0 else 0.0
            nd = int(np.asarray(signalvar).size)
            if opt.get('alpha') is not None:
                _vsub(f"alpha={opt['alpha']} -> K={k} "
                      f"(interpolated {int(round(100 * opt['alpha']))}% from prediction peak "
                      f"K={alpha_info['k_pred']} toward do-nothing K={alpha_info['k_var']}; "
                      f"retains {sv_pct:.1f}% signal var)")
            elif crit == 'max-tradeoff':
                diff = np.asarray(signalvar, dtype=float) - np.asarray(noisevar, dtype=float) / ntrials_avg
                rec = np.concatenate([[0.0], np.cumsum(diff)])
                kpk = int(np.argmax(rec))
                _vsub(f"max-tradeoff -> K={k} (knee of the recovery curve, between "
                      f"prediction peak K={kpk} and do-nothing K={nd}; retains {sv_pct:.1f}% signal var)")
            elif crit == 'prediction':
                _vsub(f"prediction -> K={k} (analytic recovery peak; retains {sv_pct:.1f}% signal var)")
            elif crit in ('variance', 'variance_eigenvalues'):
                vt = float(opt.get('variance_threshold', 0.99))
                _vsub(f"{crit} -> K={k} (smallest K reaching {vt:.0%} target; "
                      f"retains {sv_pct:.1f}% signal var)")
            if opt.get('allowable_thresholds') is not None:
                allow = np.asarray(opt['allowable_thresholds']).ravel().tolist()
                _vsub(f"constrained to allowable {allow} -> K={k} (best among allowed)")

    # =========================================================================
    # STEP 8: Apply denoising to data
    # =========================================================================
    # Project the trial-averaged data through the denoiser to get the
    # final denoised estimates.

    if opt['wantverbose']:
        _vlog('denoise', 'applying denoiser to data')

    # Apply denoiser (transpose works for both symmetric and non-symmetric cases)
    denoiseddata = denoiser.T @ (trial_avg - unit_means[:, np.newaxis]) + unit_means[:, np.newaxis]

    # =========================================================================
    # STEP 9: Compute residuals (noise estimates)
    # =========================================================================
    # The residuals are what's left after subtracting the denoised data from
    # the original data. These represent PSN's estimate of the noise.

    residuals = data - denoiseddata[:, :, np.newaxis]

    # =========================================================================
    # STEP 10: Compute signal and noise variances before/after denoising
    # =========================================================================

    svnv_before, svnv_after = compute_signal_noise_diagnostics(
        opt['threshold_method'], unit_signal_vars, unit_noise_vars, best_threshold, nunits, ntrials_avg)

    # =========================================================================
    # STEP 11: Package results
    # =========================================================================

    if opt['wantverbose']:
        _vlog('denoise', 'packaging results')

    results = {}

    # Core outputs
    results['denoiseddata'] = denoiseddata
    results['residuals'] = residuals
    results['unit_means'] = unit_means
    results['denoiser'] = denoiser

    # Diagnostic outputs
    results['svnv_before'] = svnv_before
    results['svnv_after'] = svnv_after
    results['best_threshold'] = best_threshold
    results['fullbasis'] = basis

    # GSN outputs
    results['gsn_result'] = gsn_result

    # Variance outputs
    results['signalvar'] = signalvar
    results['noisevar'] = noisevar
    results['objective'] = objective

    # Visualization basis (original order before global ranking)
    results['basis_viz'] = basis_viz
    results['signal_proj_viz'] = signal_proj_viz
    results['noise_proj_viz'] = noise_proj_viz
    results['basis_eigenvalues_viz'] = basis_eigenvalues_viz

    # Sorted eigenvalues (match fullbasis column order)
    results['basis_eigenvalues'] = basis_eigenvalues

    # Input data
    results['input_data'] = data

    # Special outputs for global thresholding
    if opt['threshold_method'] == 'global':
        if best_threshold > 0:
            results['signalsubspace'] = basis[:, :best_threshold]
            # Project data onto signal subspace (dimensionality reduction)
            results['dimreduce'] = results['signalsubspace'].T @ (trial_avg - unit_means[:, np.newaxis])
        else:
            results['signalsubspace'] = None
            results['dimreduce'] = None
    else:
        results['signalsubspace'] = None
        results['dimreduce'] = None

    # Unit-specific outputs
    if opt['threshold_method'] != 'global':
        results['unit_signal_vars'] = unit_signal_vars
        results['unit_noise_vars'] = unit_noise_vars
        results['unit_objectives'] = unit_cumsum_curves

    # Alpha info for visualization
    if alpha_info is not None:
        results['alpha_info'] = alpha_info

    # Store options for visualization
    results['opt_used'] = opt

    # =========================================================================
    # STEP 12: Recovery-tradeoff data (always - independent of wantfig).
    # Skipped for the internal sub-run of 'compare' selection (computed once for
    # 'compare' itself, reusing both candidate bases).
    # =========================================================================
    if not opt.get('_skip_recovery_tradeoff'):
        attach_recovery_tradeoff(
            results, cSb, cNb, ntrials_avg, data, unit_means, has_nans,
            _orig_basis, nunits, device=opt['device'],
            skip_split_half=opt.get('skip_split_half', False))

    # =========================================================================
    # STEP 13: Visualization
    # =========================================================================

    if opt['wantfig']:
        if opt['wantverbose']:
            _vlog('figure', 'generating diagnostic figure')
        _t_fig = time.time()
        safe_visualize_results(results, opt)
        _fig_time += time.time() - _t_fig

    results['runtime'] = time.time() - _t_start

    if opt['wantverbose']:
        _done(f"retained {format_threshold(best_threshold)} dimensions")

    return results
