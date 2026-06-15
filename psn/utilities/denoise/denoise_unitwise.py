"""Unit-specific denoising utility for PSN."""

import numpy as np
from ..threshold.constrain_to_allowable import constrain_to_allowable
from ..threshold.max_tradeoff import max_tradeoff_threshold
from .compute_unit_weighted_projections import compute_unit_weighted_projections
from psn._device import resolve_device, to_device, from_device, is_cpu


def denoise_unitwise(basis, signal_proj, noise_proj, basis_eigenvalues, ntrials, opt):
    """DENOISE_UNITWISE  Unit-specific denoising (non-symmetric denoiser)

    [denoiser, best_threshold, objective, ...] = denoise_unitwise(basis, signal_proj,
    noise_proj, basis_eigenvalues, ntrials, opt) builds a generally non-symmetric
    denoising matrix with unit-specific thresholds applied on a shared global
    dimension ordering (threshold_method='hybrid').

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <basis> - [nunits x ndims] orthonormal basis matrix

    <signal_proj> - [ndims] signal variance per dimension

    <noise_proj> - [ndims] noise variance per dimension

    <basis_eigenvalues> - [ndims] eigenvalues from basis construction, or None

    <ntrials> - scalar, number of trials (or average if NaNs present)

    <opt> - dict with PSN options (criterion, allowable_thresholds, unit_groups, etc.)

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

    -------------------------------------------------------------------------
    Algorithm:
    -------------------------------------------------------------------------

    Each unit receives:
      - Weighted signal/noise projections: w = basis[u,:]^2, sig_u = w * signal_proj
      - Unit-specific threshold selection with optional unit_groups averaging
      - Denoiser column: Bu @ Bu[u,:].T where Bu = basis[:, :k_u]

    The denoiser is generally non-symmetric. Apply as: denoiser.T @ data
    """

    nunits, ndims = basis.shape
    device = resolve_device(opt.get('device', 'cpu'))

    denoiser = np.zeros((nunits, nunits))

    # First pass: compute weighted projections and objectives for each unit.
    # Hybrid mode uses the shared global ordering (do_unit_ranking=False).
    unit_cumsum_curves, unit_signal_vars, unit_noise_vars, _ = \
        compute_unit_weighted_projections(basis, signal_proj, noise_proj, ntrials,
                                            False, device=device)

    # Second pass: select thresholds considering unit_groups
    unique_groups = np.unique(opt['unit_groups'])
    best_threshold = np.zeros(nunits, dtype=int)

    for g in unique_groups:
        group_mask = (opt['unit_groups'] == g)
        group_indices = np.where(group_mask)[0]

        if opt.get('alpha') is not None:
            # Alpha interpolation: blend prediction peak and variance target
            alpha_val = opt['alpha']
            vt = np.clip(opt['variance_threshold'], 0, 1)
            # Average signal vars and prediction curves across group
            avg_signal = np.mean(np.column_stack([unit_signal_vars[i] for i in group_indices]), axis=1)
            avg_curve = np.mean(np.column_stack([unit_cumsum_curves[i] for i in group_indices]), axis=1)
            k_pred = np.argmax(avg_curve)
            # Signal cumsum
            sig_cs = np.concatenate([[0], np.cumsum(avg_signal)])
            S_pred = sig_cs[k_pred]
            total = sig_cs[-1]
            S_var = vt * total
            target = S_pred + alpha_val * max(0, S_var - S_pred)
            if total <= 0:
                k_group = 0
            else:
                idx = np.where(sig_cs >= target)[0]
                k_group = idx[0] if len(idx) > 0 else ndims
                k_group = max(k_group, k_pred)
                k_group = min(k_group, ndims)
        elif opt['criterion'] == 'prediction':
            # Average objective curves across units in this group
            # All curves should have the same length (ndims+1)
            avg_curve = np.mean(np.column_stack([unit_cumsum_curves[i] for i in group_indices]), axis=1)
            k_group = np.argmax(avg_curve)
            # k_group is already the number of dims (0-indexed argmax)
        elif opt['criterion'] == 'max-tradeoff':
            # Max-tradeoff on this group's averaged recovery curve (per-unit thresholds)
            avg_signal = np.mean(np.column_stack([unit_signal_vars[i] for i in group_indices]), axis=1)
            avg_noise = np.mean(np.column_stack([unit_noise_vars[i] for i in group_indices]), axis=1)
            k_group = max_tradeoff_threshold(avg_signal, avg_noise, ntrials)
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

    # Third pass: build denoiser columns. All units share the global ordering,
    # so group units by their threshold and build each group with one matmul.
    unique_thresholds = np.unique(best_threshold[best_threshold > 0])

    if not is_cpu(device) and len(unique_thresholds) > 0:
        # GPU path: do all threshold-group matmuls on device in one shot,
        # then bring the (n, n) denoiser back to host. The matmul is the
        # dominant cost at large nunits — GPU cuts it by 1-2 orders of
        # magnitude over multi-threaded CPU.
        import torch
        tdtype = (torch.float64 if basis.dtype == np.float64
                  else torch.float32)
        basis_t = to_device(basis, device, dtype=tdtype)
        denoiser_t = torch.zeros((nunits, nunits), dtype=tdtype, device=device)
        for k in unique_thresholds:
            units_k = np.where(best_threshold == k)[0]
            units_k_t = torch.as_tensor(units_k, device=device, dtype=torch.long)
            Bu = basis_t[:, :k]
            # denoiser[:, units] = Bu @ Bu[units, :].T
            denoiser_t[:, units_k_t] = Bu @ Bu.index_select(0, units_k_t).T
        denoiser = from_device(denoiser_t).astype(np.float64, copy=False)
        del denoiser_t, basis_t
        if str(device).startswith('cuda'):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    else:
        for k in unique_thresholds:
            units_with_k = np.where(best_threshold == k)[0]
            # All units share the same ordering, so vectorize fully:
            # denoiser[:, units] = Bu @ Bu[units, :].T
            Bu = basis[:, :k]
            denoiser[:, units_with_k] = Bu @ Bu[units_with_k, :].T

    # Population-level totals for visualization
    # Sum across units to get total variance (since unit weights sum to 1,
    # mean * nunits = sum). This makes signalvar/noisevar consistent with
    # eigenvalues and global mode.
    if len(unit_signal_vars) > 0:
        signalvar = np.mean(np.column_stack(unit_signal_vars), axis=1) * nunits
        noisevar = np.mean(np.column_stack(unit_noise_vars), axis=1) * nunits
        objective = np.concatenate([[0], np.cumsum(signalvar - noisevar / ntrials)])

        # Note: unit_cumsum_curves remain unscaled (per-unit contributions)
        # Their sum equals the objective (green line)
    else:
        signalvar = np.array([])
        noisevar = np.array([])
        objective = np.zeros(1)

    return denoiser, best_threshold, objective, signalvar, noisevar, unit_cumsum_curves, \
           unit_signal_vars, unit_noise_vars
