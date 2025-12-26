"""Global (population-level) denoising utility for PSN."""

import numpy as np
from ..threshold.select_threshold_analytic import select_threshold_analytic
from ..threshold.constrain_to_allowable import constrain_to_allowable
from .compute_unit_weighted_projections import compute_unit_weighted_projections


def denoise_global(basis, signal_proj, noise_proj, basis_eigenvalues, ntrials, opt):
    """DENOISE_GLOBAL  Population-level denoising (symmetric denoiser)

    [denoiser, best_threshold, objective, ...] = denoise_global(basis, signal_proj,
    noise_proj, basis_eigenvalues, ntrials, opt) builds a symmetric denoising
    matrix using a single threshold applied to all units.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <basis> - [nunits x ndims] orthonormal basis matrix

    <signal_proj> - [ndims] signal variance per dimension

    <noise_proj> - [ndims] noise variance per dimension

    <basis_eigenvalues> - [ndims] eigenvalues from basis construction, or None

    <ntrials> - scalar, number of trials (or average if NaNs present)

    <opt> - dict with PSN options (criterion, allowable_thresholds, etc.)

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <denoiser> - [nunits x nunits] symmetric denoising matrix. If k dimensions
      are retained, denoiser = basis[:,:k] @ basis[:,:k].T

    <best_threshold> - scalar, number of dimensions retained (0 to ndims)

    <objective> - [ndims+1] cumulative objective curve used for threshold
      selection. Depends on criterion: cumsum(signal - noise/ntrials) for
      'prediction', cumsum(signal) for 'variance', or cumsum(eigenvalues)
      for 'variance_eigenvalues'

    <signalvar> - [ndims] signal variance per dimension (copy of signal_proj)

    <noisevar> - [ndims] noise variance per dimension (copy of noise_proj)

    <unit_cumsum_curves> - list of length nunits of unit-specific objective curves

    <unit_signal_vars> - list of length nunits of unit-specific signal variances

    <unit_noise_vars> - list of length nunits of unit-specific noise variances

    -------------------------------------------------------------------------
    Implementation notes:
    -------------------------------------------------------------------------

    Fast path: If using difference basis + prediction criterion, eigenvalues
    already encode signal - noise/ntrials, so we directly maximize cumsum(eigenvalues)
    """

    nunits = basis.shape[0]
    ndims = basis.shape[1]
    use_diff_basis = isinstance(opt['basis'], str) and opt['basis'] == 'difference'
    use_prediction = opt['criterion'] == 'prediction'

    # Check if allowable_thresholds is a scalar (forced threshold)
    if opt['allowable_thresholds'] is not None:
        allowable_arr = np.asarray(opt['allowable_thresholds'])
        if allowable_arr.ndim == 1 and len(allowable_arr) == 1:
            # FORCED THRESHOLD: Skip optimization, use the scalar value directly
            k = int(allowable_arr[0])
            # Still compute objective curve for visualization
            if use_diff_basis and use_prediction and basis_eigenvalues is not None:
                objective = np.concatenate([[0], np.cumsum(basis_eigenvalues)])
            else:
                _, objective = select_threshold_analytic(signal_proj, noise_proj, basis_eigenvalues, ntrials, opt)
        else:
            # Normal optimization with constraint
            # Threshold selection
            if use_diff_basis and use_prediction and basis_eigenvalues is not None:
                # FAST PATH: difference basis eigenvalues ARE the net benefit
                objective = np.concatenate([[0], np.cumsum(basis_eigenvalues)])
                k = np.argmax(objective)
                # k is already the number of dims (0-indexed argmax)
            else:
                # Standard path (including variance_eigenvalues criterion)
                k, objective = select_threshold_analytic(signal_proj, noise_proj, basis_eigenvalues, ntrials, opt)

            # Apply allowable_thresholds constraint
            k = constrain_to_allowable(k, opt['allowable_thresholds'])
    else:
        # No constraint: normal optimization
        if use_diff_basis and use_prediction and basis_eigenvalues is not None:
            # FAST PATH: difference basis eigenvalues ARE the net benefit
            objective = np.concatenate([[0], np.cumsum(basis_eigenvalues)])
            k = np.argmax(objective)
            # k is already the number of dims (0-indexed argmax)
        else:
            # Standard path (including variance_eigenvalues criterion)
            k, objective = select_threshold_analytic(signal_proj, noise_proj, basis_eigenvalues, ntrials, opt)

    best_threshold = k

    # Build symmetric denoiser
    if k > 0:
        denoiser = basis[:, :k] @ basis[:, :k].T
    else:
        denoiser = np.zeros((nunits, nunits))

    # Outputs
    signalvar = signal_proj
    noisevar = noise_proj

    # Compute unit-specific weighted variances using same logic as unit-specific method
    # Even though we use a global threshold, we can still compute how much each
    # dimension contributes to each unit's signal and noise
    unit_cumsum_curves, unit_signal_vars, unit_noise_vars, _ = \
        compute_unit_weighted_projections(basis, signal_proj, noise_proj, ntrials, False)

    return denoiser, best_threshold, objective, signalvar, noisevar, unit_cumsum_curves, \
           unit_signal_vars, unit_noise_vars
