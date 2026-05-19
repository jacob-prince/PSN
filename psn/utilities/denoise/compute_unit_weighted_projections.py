"""Unit-specific weighted projections utility for PSN."""

import numpy as np


def compute_unit_weighted_projections(basis, signal_proj, noise_proj, ntrials, do_unit_ranking):
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

    unit_cumsum_curves = []
    unit_signal_vars = []
    unit_noise_vars = []
    unit_orderings = np.zeros((nunits, ndims), dtype=int)

    for u in range(nunits):
        # Compute weighted projections for this unit
        # w = squared basis coefficients (how much each dimension affects this unit)
        w = basis[u, :] ** 2
        sig_u = w * signal_proj
        noi_u = w * noise_proj

        if do_unit_ranking:
            # Rank by this unit's signal variance
            sort_idx_u = np.argsort(sig_u)[::-1]  # Descending order
            sig_sorted = sig_u[sort_idx_u]
            noi_sorted = noi_u[sort_idx_u]
        else:
            # Use global ordering
            sig_sorted = sig_u
            noi_sorted = noi_u
            sort_idx_u = np.arange(ndims)

        unit_orderings[u, :] = sort_idx_u

        # Compute objective curve for this unit
        # Always use prediction-style objective (signal - noise/ntrials)
        # even for variance criterion (threshold selection handles the difference)
        scaled_noise = noi_sorted / ntrials
        diff = sig_sorted - scaled_noise
        curve_u = np.concatenate([[0], np.cumsum(diff)])

        unit_cumsum_curves.append(curve_u)
        unit_signal_vars.append(sig_sorted)
        unit_noise_vars.append(noi_sorted)

    return unit_cumsum_curves, unit_signal_vars, unit_noise_vars, unit_orderings
