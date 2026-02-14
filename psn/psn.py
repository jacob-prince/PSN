"""PSN (Partitioning Signal and Noise) - Main denoising function.

This module provides the main psn() function for denoising neural data using
Partitioning Signal and Noise (PSN) methodology.
"""

import numpy as np

# Import utilities
from .utilities.input.parse_inputs import parse_inputs
from .utilities.input.validate_data import validate_data
from .utilities.input.set_default_options import set_default_options
from .utilities.basis.construct_basis import construct_basis
from .utilities.basis.project_covs import project_covs
from .utilities.denoise.denoise_global import denoise_global
from .utilities.denoise.denoise_unitwise import denoise_unitwise
from .utilities.denoise.denoise_wiener import (
    denoise_wiener_global, denoise_fullrank_wiener
)
from .utilities.diagnostics.compute_signal_noise_diagnostics import compute_signal_noise_diagnostics
from .utilities.plotting.visualize_results import visualize_results
from .utils import perform_gsn


def psn(*args):
    """PSN  Denoise neural data using PSN (Partitioning Signal and Noise).

    results = psn(data) is the default version of PSN and is shorthand for psn(data,'standard')

    results = psn(data,'conservative', opt) prioritizes retaining signal, and is shorthand for
              psn(data,{'basis':'signal','criterion':'variance','threshold_method':'global'})

    results = psn(data,'standard', opt) prioritizes out-of-sample generalization at the possible expense
              of removing some signal in dimensions dominated by noise. It is shorthand for
              psn(data,{'basis':'signal','criterion':'prediction','threshold_method':'hybrid'})

    results = psn(data,'aggressive', opt) uses an aggressive denoising approach that flexibly adapts
              to every unit. This approach may yield improved out-of-sample generalization compared to
              'standard' but may yield unstable results in cases of limited data. It is shorthand for
              psn(data,{'basis':'difference','criterion':'prediction','threshold_method':'hybrid'})

    results = psn(data, opt) is a version of PSN where the user customizes the settings.

    In all cases, <opt> can be omitted and default parameters are used.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <data> - shape [nunits x nconds x ntrials]. Measured responses for each
      unit, condition, and trial. Requires ntrials >= 2.

      NaN HANDLING (uneven trials across conditions):
      - Not all conditions need to have the full set of trials. To indicate
        the lack of data for certain trials, you can include NaNs ---
        specifically, it is okay if data[:,i,j] consists of NaNs for some
        combination(s) of i and j.
      - IMPORTANT: Each condition must have at least one trial with valid
        data across ALL units (i.e., data[:,i,j] must contain at least one
        trial j where no units have NaNs).
      - PSN computes the average number of trials across conditions (for
        conditions with >=2 valid trials) and uses this average in formulas
        involving noise/ntrials. This follows GSN's approach.
      - The denoised output will NOT contain NaNs (PSN fills them in based
        on the available data). Residuals will preserve NaNs in the same
        positions as the input data.
      (See perform_gsn for more details about uneven trials.)

    <opt> (optional) - dict with the following fields:

      <basis> (optional) - basis specifier. Either string or matrix:
        'signal'     -> use signal basis (eigenvectors of signal covariance, cSb)
        'difference' -> use "difference" basis (eigenvectors of cSb - cNb / ntrials_avg)
                        where ntrials_avg is the average number of trials (handles NaNs)
        'wiener'     -> full-rank matrix Wiener filter: D = Σ_S @ (Σ_S + Σ_N/t)^{-1}.
                        Bypasses basis construction, ordering, criterion, and thresholding.
                        All other options (criterion, threshold_method, basis_ordering,
                        denoiser_type) are ignored. Only ntrials_eval and gsn_args apply.
        B            -> user-supplied basis vectors B, size [nunits x D], D >= 1, with
                        orthonormal columns (B.T @ B = I).
        'noise'      -> [NOT RECOMMENDED] use noise basis (eigenvectors of noise covariance, cNb)
        'pca'        -> [NOT RECOMMENDED] use eigenvectors of covariance of trial-averaged data
        'random'     -> [NOT RECOMMENDED] generate a random orthonormal basis
        Default: 'signal'.

      <criterion> (optional) - string. How to determine the threshold:
        'prediction'            -> maximize out-of-sample generalization by analytically
                                   maximizing cumulative signal - noise/ntrials_avg
        'variance'              -> retain dimensions until a target fraction of signal variance is reached
        'variance_eigenvalues'  -> retain dimensions until a target fraction of the total sum of
                                   positive eigenvalues associated with the basis is reached
        Default: 'prediction'.
        (Note that <criterion> set to 'variance_eigenvalues' is not compatible with
         the B and 'random' cases of <basis>, and is also not compatible with
         <threshold_method> as 'hybrid' or 'unit'.)
        (Also, note that when <basis> is 'signal' and <threshold_method> is 'global', then
         identical results are produced by 'variance' vs. 'variance_eigenvalues'.)

      <threshold_method> (optional) - string. How to select thresholds (i.e. the
        number of dimensions to retain):
        'global' -> single threshold for all units (symmetric denoiser)
        'hybrid' -> global ordering of basis vectors, unit-specific thresholds
        'unit'   -> unit-specific ordering of basis vectors, unit-specific thresholds
        Default: 'hybrid'.

      <basis_ordering> (optional) - string. How to set the initial global order of basis vectors:
        'eigenvalues'    -> use descending order of eigenvalues (if available)
        'signalvariance' -> measure signal variance and use descending order of signal variance
        Default: 'eigenvalues'.
        (Note that when <basis> is B or 'random', eigenvalues are not available, so we
         necessarily fall back to 'signalvariance'.)

      <variance_threshold> (optional) - scalar in [0,1]. Fraction used
        when <criterion> is 'variance' or 'variance_eigenvalues'.
        Default: 0.99.

      <allowable_thresholds> (optional) is an array of thresholds that are acceptable.
        For example, a threshold of 7 means to retain the first 7 dimensions.
        If an optimal threshold is found that is not listed in <allowable_thresholds>,
        we force to the nearest acceptable threshold (rounding up in cases of a tie).
        Note that setting <allowable_thresholds> to a single threshold will force
        PSN to use exactly that many dimensions.
        Default: None (no constraint; any threshold between 0 and D is allowed).

      <unit_groups> (optional) - [nunits] array of non-negative integers specifying
          which units must share the same threshold (applies when <threshold_method>
          is 'hybrid' or 'unit'). Units with the same integer label are treated
          as a group and receive the same threshold (determined by averaging
          the criterion across units of that group). If None or omitted, the
          default behavior is equivalent to np.arange(nunits), which indicates that
          each unit forms its own group (i.e., distinct threshold for each unit).
          For <threshold_method> = 'global', <unit_groups> is ignored (defaults to
          all zeros internally).

      <denoiser_type> (optional) - string. Type of denoising operator:
        'truncation'    -> hard truncation (keep first K dims, drop rest) [default]
        'wiener'        -> Wiener shrinkage (apply continuous weights w_k to each dim)
                           w_k = s_k / (s_k + n_k / t_eval)
        Note: Wiener is only supported with threshold_method='global'.

      <ntrials_eval> (optional) - scalar. Number of trials used to form the
        trial average being denoised. This affects Wiener weights via noise/t_eval.
        Default: None (uses ntrials_avg from the data).
        Set this if denoising held-out data with different trial counts.

      <gsn_args> (optional) - dict of options passed directly to the GSN routine
          (perform_gsn). Typical fields might include:
              wantshrinkage  - whether to use covariance shrinkage (default: True)
              wantverbose    - whether to print diagnostic output (default: False)
              random_seed    - RNG seed used inside GSN
          If None or omitted, defaults are used.

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
                        this is symmetric. For hybrid/unit modes, this is generally
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
                        ordering. For unit-specific methods, individual units may
                        use different orderings (see unitreorderings).

      results['basis_eigenvalues'] - [dims]. Eigenvalues from basis construction (e.g.,
                        from signal covariance, difference matrix, etc.), sorted to
                        match the column order of fullbasis. None for custom
                        or random bases. For visualization of original order, see
                        basis_eigenvalues_viz.

      results['unitreorderings'] - [nunits x dimindices]. Each row indicates the
        chosen unit-specific ordering for the dimensions. If <threshold_method> is
        'global' or 'hybrid', each row is simply np.arange(D) in that order.

      results['gsn_result']  - dict. Full results from the GSN algorithm containing:
                                .cSb  - signal covariance
                                .cNb  - noise covariance
                            plus any additional outputs from the GSN code.

      results['input_data']  - copy of the original input data

      results['signalvar']   - [dims] or list. Signal variance per dimension.

      results['noisevar']    - [dims] or list. Noise variance per dimension.

      results['objective']   - [dims+1]. Cumulative objective curve that was actually
                        used for threshold selection. For 'prediction' criterion, this is
                        cumsum(signal - noise/ntrials). For 'variance', this is cumsum(signal).
                        For 'variance_eigenvalues', this is cumsum(positive eigenvalues).

    Special outputs returned only when <threshold_method> is 'global':

      results['signalsubspace'] - [nunits x K]. The final set of basis vectors selected
        for denoising (i.e. the subspace into which data are projected).

      results['dimreduce'] - [K x nconds]. This is the low-dimensional representation
        of the denoised data in the selected subspace.

    Special outputs for unit-specific methods ('hybrid' or 'unit'):

      results['unit_signal_vars'] - list of signal variances per unit
      results['unit_noise_vars']  - list of noise variances per unit
      results['unit_objectives']  - list of objective curves per unit

    Special outputs for Wiener denoiser (denoiser_type='wiener'):

      results['wiener_weights'] - [dims]. Shrinkage weight applied to each dimension.
                            w_k = s_k / (s_k + n_k / t_eval) in [0, 1].

    Visualization outputs (original order before global ranking):

      results['basis_viz']       - [nunits x dims]. Basis vectors in original order
                                (before global ranking/sorting).
      results['signal_proj_viz'] - [dims]. Signal variance projections in original order.
      results['noise_proj_viz']  - [dims]. Noise variance projections in original order.
      results['basis_eigenvalues_viz'] - [dims]. Eigenvalues in original order,
                                matching column order of basis_viz.

    -------------------------------------------------------------------------
    Algorithm Details:
    -------------------------------------------------------------------------

    PSN works by:
      1) Estimating signal and noise covariances via GSN
      2) Constructing an orthonormal basis (eigenvectors of cSb or cSb - cNb/ntrials_avg)
      3) Projecting signal and noise covariances into this basis
      4) Ranking dimensions by signal variance (or eigenvalues for difference basis)
      5) Selecting threshold to maximize signal - noise/ntrials_avg (or retain variance fraction)
      6) Building a denoiser that projects data onto selected dimensions
      7) Reconstructing denoised data
      8) Using this updated signal estimate to compute a noise estimate (residuals)

    For unit-specific methods, each unit gets weighted projections based on
    the squared basis coefficients for that unit, and potentially different
    dimension orderings and thresholds.
    """

    # =========================================================================
    # STEP 1: Parse and validate inputs
    # =========================================================================

    data, opt = parse_inputs(*args)
    nunits, nconds, ntrials, ntrials_avg, has_nans = validate_data(data)
    opt = set_default_options(opt, nunits)

    if opt['wantverbose']:
        if has_nans:
            print(f"PSN: Starting denoising for {nunits} units, {nconds} conditions, {ntrials} max trials (avg {ntrials_avg:.2f} trials)")
        else:
            print(f"PSN: Starting denoising for {nunits} units, {nconds} conditions, {ntrials} trials")

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

    if opt['wantverbose']:
        print('PSN: Running GSN to estimate signal and noise covariances...')

    gsn_opt = opt['gsn_args'] if opt['gsn_args'] is not None else {}
    if 'wantverbose' not in gsn_opt:
        gsn_opt['wantverbose'] = False
    if 'wantshrinkage' not in gsn_opt:
        gsn_opt['wantshrinkage'] = True

    gsn_result = perform_gsn(data, gsn_opt)
    cSb = gsn_result['cSb']  # signal covariance (symmetric)
    cNb = gsn_result['cNb']  # noise covariance (symmetric)

    # =========================================================================
    # FULL-RANK WIENER SHORT-CIRCUIT (basis='wiener')
    # =========================================================================
    # If basis='wiener', skip steps 4-7 and apply the full-rank matrix Wiener
    # filter directly: D = Σ_S @ (Σ_S + Σ_N/t)^{-1}. This is the Bayes-optimal
    # linear estimator when the signal and noise covariances are known.

    if isinstance(opt['basis'], str) and opt['basis'] == 'wiener':
        if opt['wantverbose']:
            print('PSN: Applying full-rank matrix Wiener filter...')

        results = denoise_fullrank_wiener(
            cSb, cNb, data, trial_avg, unit_means, ntrials_avg, nunits, gsn_result, opt
        )

        if opt['wantfig']:
            if opt['wantverbose']:
                print('PSN: Generating diagnostic figures...')
            visualize_results(results, opt)

        if opt['wantverbose']:
            print(f"PSN: Complete! Full-rank Wiener filter ({results['best_threshold']:.1f} effective dimensions).")

        return results

    # =========================================================================
    # STEP 4: Construct the denoising basis
    # =========================================================================
    # The basis is a set of orthonormal vectors that span the space in which
    # we'll perform denoising. Different basis choices emphasize different
    # aspects of the data structure.

    if opt['wantverbose']:
        if isinstance(opt['basis'], str):
            print(f"PSN: Constructing denoising basis (type: {opt['basis']})...")
        else:
            print(f"PSN: Constructing denoising basis (type: custom matrix [{opt['basis'].shape[0]}x{opt['basis'].shape[1]}])...")

    basis, basis_eigenvalues = construct_basis(cSb, cNb, opt['basis'], data, trial_avg, unit_means, ntrials_avg, has_nans)

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
        print('PSN: Computing signal and noise variance per dimension...')

    # Always use GSN-based projection for all basis types
    signal_proj, noise_proj = project_covs(cSb, cNb, basis)

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

    if opt['wantverbose']:
        print('PSN: Ranking basis dimensions globally...')

    if opt['basis_ordering'] == 'eigenvalues' and basis_eigenvalues is not None:
        # Use eigenvalue-based ranking
        sort_idx_global = np.argsort(basis_eigenvalues)[::-1]  # Descending
        if opt['wantverbose']:
            print('PSN: Using eigenvalue-based ordering')
    elif opt['basis_ordering'] == 'prediction':
        # Use prediction objective (signal - noise/ntrials) for ranking
        prediction_obj = signal_proj - noise_proj / ntrials_avg
        sort_idx_global = np.argsort(prediction_obj)[::-1]  # Descending
        if opt['wantverbose']:
            print('PSN: Using prediction-based ordering (signalvar - noisevar/ntrials)')
    else:
        # Use signal variance-based ranking (or fallback when eigenvalues unavailable)
        sort_idx_global = np.argsort(signal_proj)[::-1]  # Descending
        if opt['wantverbose']:
            if opt['basis_ordering'] == 'eigenvalues':
                print('PSN: Eigenvalues unavailable, falling back to signal variance ordering')
            else:
                print('PSN: Using signal variance ordering')

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
    #   - 'unit': unit-specific thresholds AND unit-specific orderings

    if opt['wantverbose']:
        print(f"PSN: Selecting thresholds (method: {opt['threshold_method']}, criterion: {opt['criterion']})...")

    # Initialize wiener_weights (only populated for Wiener denoiser)
    wiener_weights = None

    if opt['denoiser_type'] == 'wiener':
        # WIENER SHRINKAGE MODE (threshold_method forced to 'global' in set_default_options)
        if opt['wantverbose']:
            print('PSN: Using Wiener shrinkage denoiser...')

        denoiser, best_threshold, objective, signalvar, noisevar, unit_cumsum_curves, \
        unit_signal_vars, unit_noise_vars, wiener_weights = \
            denoise_wiener_global(basis, signal_proj, noise_proj, basis_eigenvalues,
                                  ntrials_avg, opt)

        unit_orderings = np.tile(np.arange(basis.shape[1]), (nunits, 1))

    elif opt['threshold_method'] == 'global':
        # GLOBAL (POPULATION) MODE with hard truncation
        denoiser, best_threshold, objective, signalvar, noisevar, unit_cumsum_curves, \
        unit_signal_vars, unit_noise_vars = \
            denoise_global(basis, signal_proj, noise_proj, basis_eigenvalues,
                          ntrials_avg, opt)

        unit_orderings = np.tile(np.arange(basis.shape[1]), (nunits, 1))

    else:
        # UNIT-SPECIFIC MODES (hybrid or unit)
        unitwise_threshold_only = (opt['threshold_method'] == 'hybrid')

        denoiser, best_threshold, objective, signalvar, noisevar, unit_cumsum_curves, \
        unit_signal_vars, unit_noise_vars, unit_orderings = \
            denoise_unitwise(basis, signal_proj, noise_proj, basis_eigenvalues,
                           ntrials_avg, opt, unitwise_threshold_only)

    # =========================================================================
    # STEP 8: Apply denoising to data
    # =========================================================================
    # Project the trial-averaged data through the denoiser to get the
    # final denoised estimates.

    if opt['wantverbose']:
        print('PSN: Applying denoiser to data...')

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

    if opt['denoiser_type'] == 'wiener':
        # For Wiener-family denoisers, compute weighted signal/noise variances
        # using the Wiener weights instead of hard thresholding
        svnv_before = np.zeros((nunits, 2))
        svnv_after = np.zeros((nunits, 2))
        for u in range(nunits):
            sig_u = unit_signal_vars[u]
            noi_u = unit_noise_vars[u]
            # Before: sum of all variances
            svnv_before[u, :] = [np.sum(sig_u), np.sum(noi_u) / ntrials_avg]
            # After: weighted sum using Wiener weights (w_k^2 for variance)
            w_sq = wiener_weights ** 2
            svnv_after[u, :] = [np.sum(w_sq * sig_u), np.sum(w_sq * noi_u) / ntrials_avg]
    else:
        svnv_before, svnv_after = compute_signal_noise_diagnostics(
            opt['threshold_method'], unit_signal_vars, unit_noise_vars, best_threshold, nunits, ntrials_avg)

    # =========================================================================
    # STEP 11: Package results
    # =========================================================================

    if opt['wantverbose']:
        print('PSN: Packaging results...')

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
    results['unitreorderings'] = unit_orderings

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

    # Special outputs for global thresholding (not applicable for Wiener-family denoisers)
    if opt['threshold_method'] == 'global' and opt['denoiser_type'] != 'wiener':
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

    # Wiener denoiser outputs
    if wiener_weights is not None:
        results['wiener_weights'] = wiener_weights

    # Store options for visualization
    results['opt_used'] = opt

    # =========================================================================
    # STEP 12: Visualization
    # =========================================================================

    if opt['wantfig']:
        if opt['wantverbose']:
            print('PSN: Generating diagnostic figures...')
        visualize_results(results, opt)

    if opt['wantverbose']:
        if opt['denoiser_type'] == 'wiener':
            print(f"PSN: Complete! Wiener denoiser with {best_threshold:.1f} effective dimensions.")
        elif opt['threshold_method'] == 'global':
            print(f"PSN: Complete! Retained {best_threshold} dimensions.")
        else:
            print(f"PSN: Complete! Retained {np.mean(best_threshold):.1f} dimensions on average (range: {np.min(best_threshold)}-{np.max(best_threshold)}).")

    return results
