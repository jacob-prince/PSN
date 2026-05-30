"""Split-half reliability metric used for PSN's auto basis selection."""

import numpy as np


def split_half_cross_median(data, denoiser, unit_means, has_nans):
    """Median per-unit 'TAvg vs Denoised' split-half correlation.

    This is the median of the *yellow dots* in the diagnostic split-half panel.
    Trials are split odd/even; for each unit it correlates one half's
    trial-average (an unbiased, noisy estimate of the true signal) against the
    OTHER half's denoised estimate, averaged both ways. Because the
    trial-average side is unbiased, this is a data-driven estimate of the
    correlation between the denoised output and the true (unbiased) signal,
    without the denoiser's own bias appearing on both sides.

    Parameters
    ----------
    data : ndarray [nunits x nconds x ntrials]   the input data
    denoiser : ndarray [nunits x nunits]          the denoising matrix (applied
        as ``denoiser.T @ (x - means) + means``)
    unit_means : ndarray [nunits]                 per-unit means
    has_nans : bool                               whether ``data`` contains NaNs

    Returns
    -------
    float   median across units (-inf if no unit is usable).
    """
    nunits, nconds, ntrials = data.shape
    odd = np.arange(0, ntrials, 2)
    even = np.arange(1, ntrials, 2)
    A, B = data[:, :, odd], data[:, :, even]
    tavg_A = np.nanmean(A, axis=2) if has_nans else np.mean(A, axis=2)
    tavg_B = np.nanmean(B, axis=2) if has_nans else np.mean(B, axis=2)
    dn_A = denoiser.T @ (tavg_A - unit_means[:, np.newaxis]) + unit_means[:, np.newaxis]
    dn_B = denoiser.T @ (tavg_B - unit_means[:, np.newaxis]) + unit_means[:, np.newaxis]

    vals = []
    for u in range(nunits):
        if not (np.nanstd(tavg_A[u]) > 0 and np.nanstd(tavg_B[u]) > 0 and
                np.nanstd(dn_A[u]) > 0 and np.nanstd(dn_B[u]) > 0):
            continue
        mAB = ~(np.isnan(tavg_A[u]) | np.isnan(dn_B[u]))
        mBA = ~(np.isnan(dn_A[u]) | np.isnan(tavg_B[u]))
        if mAB.sum() > 1 and mBA.sum() > 1:
            cAB = np.corrcoef(tavg_A[u, mAB], dn_B[u, mAB])[0, 1]
            cBA = np.corrcoef(dn_A[u, mBA], tavg_B[u, mBA])[0, 1]
            vals.append(0.5 * (cAB + cBA))
    return float(np.nanmedian(vals)) if vals else -np.inf
