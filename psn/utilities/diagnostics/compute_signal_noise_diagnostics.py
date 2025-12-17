"""Signal/noise diagnostics utility for PSN."""

import numpy as np


def compute_signal_noise_diagnostics(threshold_method, unit_signal_vars, unit_noise_vars, best_threshold, nunits):
    """COMPUTE_SIGNAL_NOISE_DIAGNOSTICS  Sum signal/noise variance before/after thresholding

    [svnv_before, svnv_after] = compute_signal_noise_diagnostics(threshold_method,
    unit_signal_vars, unit_noise_vars, best_threshold, nunits) computes the total
    signal and noise variance for each unit before and after applying PSN thresholding.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <threshold_method> - string, 'global', 'hybrid', or 'unit'

    <unit_signal_vars> - list of length nunits, each element contains [ndims]
      weighted signal variances for that unit

    <unit_noise_vars> - list of length nunits, each element contains [ndims]
      weighted noise variances for that unit

    <best_threshold> - scalar (for global) or [nunits] (for hybrid/unit),
      number of dimensions retained per unit

    <nunits> - scalar, number of units

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <svnv_before> - [nunits x 2] matrix. Column 0 is total signal variance,
      column 1 is total noise variance, summed across all dimensions (before
      thresholding)

    <svnv_after> - [nunits x 2] matrix. Column 0 is total signal variance,
      column 1 is total noise variance, summed only across retained dimensions
      (after thresholding)
    """

    svnv_before = np.zeros((nunits, 2))
    svnv_after = np.zeros((nunits, 2))

    if threshold_method == 'global':
        # Global: all units share same threshold, but each unit gets different
        # amounts of signal/noise variance based on weighted projections
        for u in range(nunits):
            sig_u = unit_signal_vars[u]
            noi_u = unit_noise_vars[u]
            k = best_threshold  # Same threshold for all units
            svnv_before[u, :] = [np.sum(sig_u), np.sum(noi_u)]
            svnv_after[u, :] = [(k > 0) * np.sum(sig_u[:k]), (k > 0) * np.sum(noi_u[:k])]
    else:
        # Unit-specific: each unit has weighted projections and individual threshold
        for u in range(nunits):
            sig_u = unit_signal_vars[u]
            noi_u = unit_noise_vars[u]
            k_u = best_threshold[u]
            svnv_before[u, :] = [np.sum(sig_u), np.sum(noi_u)]
            svnv_after[u, :] = [(k_u > 0) * np.sum(sig_u[:k_u]), (k_u > 0) * np.sum(noi_u[:k_u])]

    return svnv_before, svnv_after
