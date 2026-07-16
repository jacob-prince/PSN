"""Split-half reliability of a denoiser.

Used by basis='compare' to choose between the signal and difference bases by
empirical split-half r at each basis's max-tradeoff threshold. The metric matches
the diagnostic figure's split-half curve (recovery_tradeoff.py, numpy path) and
the MATLAB matlab/utilities/diagnostics/split_half_r.m.
"""

import numpy as np


def _row_corr(X, Y, has_nans):
    """Per-row (per-unit) Pearson correlation across conditions; NaN-aware."""
    if has_nans:
        out = np.full(X.shape[0], np.nan)
        for u in range(X.shape[0]):
            m = ~(np.isnan(X[u]) | np.isnan(Y[u]))
            if m.sum() > 1 and np.std(X[u, m]) > 0 and np.std(Y[u, m]) > 0:
                out[u] = np.corrcoef(X[u, m], Y[u, m])[0, 1]
        return out
    Xc = X - X.mean(1, keepdims=True)
    Yc = Y - Y.mean(1, keepdims=True)
    den = np.sqrt((Xc ** 2).sum(1) * (Yc ** 2).sum(1))
    out = np.full(X.shape[0], np.nan)
    nz = den > 0
    out[nz] = (Xc * Yc).sum(1)[nz] / den[nz]
    return out


def split_half_r(data, D, unit_means, has_nans=False):
    """Median per-unit split-half reliability of denoiser <D>.

    Splits trials into even/odd halves; for each half takes the trial-average,
    denoises the OTHER half's average with D, and correlates them per unit across
    conditions (symmetrized). Returns the median across units. NaN entries are
    handled pairwise. Reduces to the diagnostic figure's split-half value for the
    same denoiser.

    <data>       - [nunits x nconds x ntrials]
    <D>          - [nunits x nunits] denoiser (applied as D @ (avg - means) + means)
    <unit_means> - [nunits] per-unit means used for centering
    <has_nans>   - whether <data> contains NaNs
    """
    ntrials = data.shape[2]
    A = data[:, :, np.arange(0, ntrials, 2)]
    B = data[:, :, np.arange(1, ntrials, 2)]
    tavg_A = np.nanmean(A, axis=2) if has_nans else A.mean(axis=2)
    tavg_B = np.nanmean(B, axis=2) if has_nans else B.mean(axis=2)
    um = np.asarray(unit_means).reshape(-1, 1)
    dn_A = D @ (tavg_A - um) + um
    dn_B = D @ (tavg_B - um) + um
    both = 0.5 * (_row_corr(tavg_A, dn_B, has_nans) + _row_corr(dn_A, tavg_B, has_nans))
    return float(np.nanmedian(both)) if np.any(~np.isnan(both)) else np.nan
