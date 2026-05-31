"""Unit-specific weighted projections utility for PSN."""

import numpy as np

from psn._device import resolve_device, to_device, from_device, is_cpu


def compute_unit_weighted_projections(basis, signal_proj, noise_proj, ntrials,
                                       do_unit_ranking, device='cpu'):
    """COMPUTE_UNIT_WEIGHTED_PROJECTIONS  Compute unit-specific weighted variances and objective curves

    [unit_cumsum_curves, unit_signal_vars, unit_noise_vars, unit_orderings] =
    compute_unit_weighted_projections(basis, signal_proj, noise_proj, ntrials, do_unit_ranking)
    computes how much signal and noise variance each basis dimension contributes
    to each individual unit, and builds objective curves for unit-specific thresholding.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <basis> - [nunits x ndims] orthonormal basis matrix

    <signal_proj> - [ndims] signal variance per dimension (from project_covs)

    <noise_proj> - [ndims] noise variance per dimension (from project_covs)

    <ntrials> - scalar, number of trials (or average number of trials if NaNs present)

    <do_unit_ranking> - boolean. If True, rank dimensions by each unit's weighted
      signal variance (full unit-specific mode). If False, use global ordering
      (hybrid mode with unit-specific thresholds only)

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <unit_cumsum_curves> - list of length nunits. Each element contains
      [ndims+1] cumulative objective curve for that unit, computed as
      cumsum(weighted_signal - weighted_noise/ntrials)

    <unit_signal_vars> - list of length nunits. Each element contains [ndims]
      weighted signal variance for that unit

    <unit_noise_vars> - list of length nunits. Each element contains [ndims]
      weighted noise variance for that unit

    <unit_orderings> - [nunits x ndims] dimension ordering indices. Row u gives
      the dimension ordering for unit u. If do_unit_ranking=False, all rows
      are np.arange(ndims)

    -------------------------------------------------------------------------
    Algorithm:
    -------------------------------------------------------------------------

    For each unit u, computes weights w(d) = basis(u,d)^2, which measure how
    much dimension d affects unit u. Then computes weighted variances:
      sig_u(d) = w(d) * signal_proj(d)
      noi_u(d) = w(d) * noise_proj(d)
    """

    nunits, ndims = basis.shape
    device = resolve_device(device)

    if is_cpu(device):
        # Vectorized numpy path. The original implementation looped
        # over nunits and built per-unit lists — at nunits=24640
        # that's 24640 Python iterations with overhead dominating
        # the cheap inner ops. The vectorized form below is 10-50×
        # faster on CPU and trivially portable to GPU.
        W = basis * basis                                              # (n, d)
        sig_all = W * signal_proj[None, :]                             # (n, d)
        noi_all = W * noise_proj[None, :]                              # (n, d)
        if do_unit_ranking:
            unit_orderings = np.argsort(sig_all, axis=1)[:, ::-1].copy()
            row_idx = np.arange(nunits)[:, None]
            sig_sorted = sig_all[row_idx, unit_orderings]
            noi_sorted = noi_all[row_idx, unit_orderings]
        else:
            unit_orderings = np.broadcast_to(
                np.arange(ndims, dtype=int), (nunits, ndims)).copy()
            sig_sorted = sig_all
            noi_sorted = noi_all
        diff = sig_sorted - noi_sorted / ntrials                       # (n, d)
        curves = np.concatenate(
            [np.zeros((nunits, 1), dtype=diff.dtype),
             np.cumsum(diff, axis=1)], axis=1)                         # (n, d+1)
    else:
        import torch
        tdtype = (torch.float64 if (basis.dtype == np.float64
                                     and signal_proj.dtype == np.float64
                                     and noise_proj.dtype == np.float64)
                  else torch.float32)
        B_t = to_device(basis, device, dtype=tdtype)
        sp_t = to_device(signal_proj, device, dtype=tdtype)
        np_t = to_device(noise_proj, device, dtype=tdtype)
        W = B_t * B_t                                                  # (n, d)
        sig_all = W * sp_t[None, :]
        noi_all = W * np_t[None, :]
        if do_unit_ranking:
            order_t = torch.argsort(sig_all, dim=1, descending=True)
            sig_sorted_t = torch.gather(sig_all, 1, order_t)
            noi_sorted_t = torch.gather(noi_all, 1, order_t)
            unit_orderings = from_device(order_t).astype(int)
        else:
            sig_sorted_t = sig_all
            noi_sorted_t = noi_all
            unit_orderings = np.broadcast_to(
                np.arange(ndims, dtype=int), (nunits, ndims)).copy()
        diff_t = sig_sorted_t - noi_sorted_t / ntrials
        zeros = torch.zeros((nunits, 1), dtype=diff_t.dtype, device=diff_t.device)
        curves_t = torch.cat(
            [zeros, torch.cumsum(diff_t, dim=1)], dim=1)
        sig_sorted = from_device(sig_sorted_t)
        noi_sorted = from_device(noi_sorted_t)
        curves = from_device(curves_t)

    # Repackage to the legacy per-unit-list contract so callers stay
    # unchanged. The lists hold views into the underlying ndarrays
    # so we don't make full copies.
    unit_cumsum_curves = list(curves)
    unit_signal_vars = list(sig_sorted)
    unit_noise_vars = list(noi_sorted)

    return unit_cumsum_curves, unit_signal_vars, unit_noise_vars, unit_orderings
