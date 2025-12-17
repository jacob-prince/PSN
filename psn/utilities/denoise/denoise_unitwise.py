"""Unit-specific denoising utility for PSN."""

import numpy as np
from ..threshold.constrain_to_allowable import constrain_to_allowable
from .compute_unit_weighted_projections import compute_unit_weighted_projections


def denoise_unitwise(basis, signal_proj, noise_proj, basis_eigenvalues, ntrials, opt, threshold_only):
    """DENOISE_UNITWISE  Unit-specific denoising (non-symmetric denoiser)

    [denoiser, best_threshold, objective, ...] = denoise_unitwise(basis, signal_proj,
    noise_proj, basis_eigenvalues, ntrials, opt, threshold_only) builds a generally
    non-symmetric denoising matrix with unit-specific thresholds and optionally
    unit-specific dimension orderings.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <basis> - [nunits x ndims] orthonormal basis matrix

    <signal_proj> - [ndims] signal variance per dimension

    <noise_proj> - [ndims] noise variance per dimension

    <basis_eigenvalues> - [ndims] eigenvalues from basis construction, or None

    <ntrials> - scalar, number of trials (or average if NaNs present)

    <opt> - dict with PSN options (criterion, allowable_thresholds, unit_groups, etc.)

    <threshold_only> - boolean. If True, use global dimension ordering with
      unit-specific thresholds (hybrid mode). If False, use unit-specific
      dimension ordering and thresholds (full unit-specific mode)

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <denoiser> - [nunits x nunits] generally non-symmetric denoising matrix.
      Column u is Bu @ Bu[u,:].T where Bu are the selected dimensions for unit u

    <best_threshold> - [nunits] number of dimensions retained per unit.
      Units in the same unit_groups share the same threshold

    <objective> - [ndims+1] population-averaged cumulative objective curve

    <signalvar> - [ndims] population-averaged signal variance per dimension

    <noisevar> - [ndims] population-averaged noise variance per dimension

    <unit_cumsum_curves> - list of length nunits of unit-specific objective curves

    <unit_signal_vars> - list of length nunits of unit-specific weighted signal variances

    <unit_noise_vars> - list of length nunits of unit-specific weighted noise variances

    <unit_orderings> - [nunits x ndims] dimension ordering for each unit

    -------------------------------------------------------------------------
    Algorithm:
    -------------------------------------------------------------------------

    Each unit receives:
      - Weighted signal/noise projections: w = basis[u,:]^2, sig_u = w * signal_proj
      - Optional unit-specific ranking (if threshold_only=False)
      - Unit-specific threshold selection with optional unit_groups averaging
      - Denoiser column: Bu @ Bu[u,:].T where Bu = basis[:, dims_for_unit_u]

    The denoiser is generally non-symmetric. Apply as: denoiser.T @ data
    """

    nunits, ndims = basis.shape

    denoiser = np.zeros((nunits, nunits))

    # First pass: compute weighted projections and objectives for each unit
    # If threshold_only=True (hybrid mode), use global ordering
    # If threshold_only=False (full unit-specific), rank by each unit's signal variance
    do_unit_ranking = not threshold_only
    unit_cumsum_curves, unit_signal_vars, unit_noise_vars, unit_orderings = \
        compute_unit_weighted_projections(basis, signal_proj, noise_proj, ntrials, do_unit_ranking)

    # Second pass: select thresholds considering unit_groups
    unique_groups = np.unique(opt['unit_groups'])
    best_threshold = np.zeros(nunits, dtype=int)

    for g in unique_groups:
        group_mask = (opt['unit_groups'] == g)
        group_indices = np.where(group_mask)[0]

        if opt['criterion'] == 'prediction':
            # Average objective curves across units in this group
            # All curves should have the same length (ndims+1)
            avg_curve = np.mean(np.column_stack([unit_cumsum_curves[i] for i in group_indices]), axis=1)
            k_group = np.argmax(avg_curve)
            # k_group is already the number of dims (0-indexed argmax)
        elif opt['criterion'] == 'variance':
            # Average signal variances across units in this group
            avg_signal = np.mean(np.column_stack([unit_signal_vars[i] for i in group_indices]), axis=1)
            vt = np.clip(opt['variance_threshold'], 0, 1)
            if vt == 0:
                k_group = 0
            else:
                # Prepend 0 for consistency with global mode (index 0 = 0 dims)
                cs = np.concatenate([[0], np.cumsum(avg_signal)])
                total = cs[-1]
                if total <= 0:
                    k_group = 0
                else:
                    idx = np.where(cs >= vt * total)[0]
                    if len(idx) == 0:
                        k_group = 0
                    else:
                        k_group = idx[0]  # First index where threshold is reached
                    k_group = min(k_group, ndims)
        else:
            raise ValueError("criterion 'variance_eigenvalues' not supported for unit-specific modes")

        # Apply allowable_thresholds constraint
        if opt['allowable_thresholds'] is not None:
            k_group = constrain_to_allowable(k_group, opt['allowable_thresholds'])

        # Assign this threshold to all units in the group
        best_threshold[group_mask] = k_group

    # Third pass: build denoiser columns
    # Optimize by grouping units with same threshold and ordering
    unique_thresholds = np.unique(best_threshold[best_threshold > 0])

    for k in unique_thresholds:
        units_with_k = np.where(best_threshold == k)[0]

        if threshold_only:
            # Hybrid mode: all units share same ordering, vectorize fully
            Bu = basis[:, :k]
            # Vectorized: denoiser[:, units] = Bu @ Bu[units, :].T
            denoiser[:, units_with_k] = Bu @ Bu[units_with_k, :].T
        else:
            # Full unit-specific: group by ordering within same threshold
            for u in units_with_k:
                sort_idx_u = unit_orderings[u, :]
                Bu = basis[:, sort_idx_u[:k]]
                denoiser[:, u] = Bu @ Bu[u, :]

    # Population-level averages for visualization
    if len(unit_signal_vars) > 0:
        signalvar = np.mean(np.column_stack(unit_signal_vars), axis=1)
        noisevar = np.mean(np.column_stack(unit_noise_vars), axis=1)
        objective = np.concatenate([[0], np.cumsum(signalvar - noisevar / ntrials)])
    else:
        signalvar = np.array([])
        noisevar = np.array([])
        objective = np.zeros(1)

    return denoiser, best_threshold, objective, signalvar, noisevar, unit_cumsum_curves, \
           unit_signal_vars, unit_noise_vars, unit_orderings
