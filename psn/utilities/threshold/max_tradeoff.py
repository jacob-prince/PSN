"""Max-tradeoff threshold selection for PSN."""

import numpy as np

from .constrain_to_allowable import constrain_to_allowable
from .select_allowable import allowable_candidates


def max_tradeoff_threshold(signal, noise, ntrials, allowable=None):
    """Select the "max-tradeoff" threshold on the recovery curve.

    Returns an operating point that captures the bulk of
    the achievable recovery: the point of maximum recovery-gain per unit of
    signal variance sacrificed, relative to the trial-average baseline. It is
    the point on the descending (peak -> trial-average) limb of the recovery
    curve that lies farthest from the chord joining those two anchors, measured
    in (fraction-of-signal-variance-retained, normalized-recovery) coordinates.

    Equivalently (in coordinates where x is normalized so peak->trial-average
    spans [0, 1]) it is argmax(x + y), the point maximizing the equal-weight
    bias/recovery tradeoff. The selection is invariant to affine rescaling of
    either axis (it depends only on the two anchors and the curve).

    Concretely, with per-dimension signal/noise variances in basis order:
      recovery(k)  = cumsum(signal - noise/ntrials)      (rises, peaks, falls)
      sv_frac(k)   = cumsum(signal) / total_signal       (fraction retained)

    Well-defined for any basis. With the signal basis (dimensions ordered by
    signal variance) the descending limb spans the high signal-variance-retained
    region; with other bases the recovery curve may include noise-dominated
    dimensions. Fully analytic; no cross-validation.

    Parameters
    ----------
    signal : array [ndims]   per-dimension signal variance (basis order)
    noise  : array [ndims]   per-dimension noise variance (basis order)
    ntrials : scalar         number of trials (or average if NaNs present)
    allowable : array-like or None   if given, restrict the choice to these
        threshold values (best-among-allowable): the allowable threshold on the
        descending limb farthest from the chord, snapping degenerate fall-backs
        to the nearest allowable value. None (default) searches all thresholds.

    Returns
    -------
    k : int   selected number of dimensions to retain, in [0, ndims]
    """
    signal = np.asarray(signal, dtype=float)
    noise = np.asarray(noise, dtype=float)
    ndims = len(signal)
    if ndims == 0:
        return 0

    def _finalize(k):
        # When restricted to an allowable set, a degenerate fall-back value may
        # not itself be allowable; snap it to the nearest allowable threshold.
        if allowable is None:
            return int(k)
        C = allowable_candidates(allowable, ndims)
        if C.size == 0 or int(k) in C:
            return int(k)
        return int(constrain_to_allowable(int(k), C))

    diff = signal - noise / ntrials
    rec = np.concatenate([[0.0], np.cumsum(diff)])          # recovery curve, len ndims+1
    sig_cum = np.concatenate([[0.0], np.cumsum(signal)])    # cumulative signal variance
    total_S = sig_cum[-1]

    k_peak = int(np.argmax(rec))                            # recovery peak (= max recovery)

    # Degenerate cases: no signal, or no usable descending limb -> the peak is
    # already the most we can do, so fall back to it.
    if total_S <= 0 or (ndims - k_peak) < 2:
        return _finalize(k_peak)
    rec_peak = rec[k_peak]
    rec_ta = rec[ndims]                                     # trial-average (retain all dims)
    if (rec_peak - rec_ta) <= 1e-12:
        return _finalize(k_peak)

    seg = np.arange(k_peak, ndims + 1)
    x = sig_cum[seg] / total_S                              # fraction signal var retained
    y = (rec[seg] - rec_ta) / (rec_peak - rec_ta)          # recovery rescaled to [0, 1]
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    denom = np.hypot(x1 - x0, y1 - y0)
    if denom <= 1e-12:
        return _finalize(k_peak)
    # perpendicular distance from the peak -> trial-average chord
    dist = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom

    if allowable is not None:
        # Best-among-allowable: pick the allowable threshold on the descending
        # limb that is farthest from the chord. If none lie on the limb, snap the
        # unconstrained pick to the nearest allowable value.
        C = allowable_candidates(allowable, ndims)
        mask = np.isin(seg, C)
        if np.any(mask):
            seg_c = seg[mask]
            dist_c = dist[mask]
            return int(seg_c[int(np.argmax(dist_c))])
        return _finalize(int(seg[int(np.argmax(dist))]))

    return int(seg[int(np.argmax(dist))])
