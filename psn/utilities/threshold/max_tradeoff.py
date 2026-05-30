"""Max-tradeoff threshold selection for PSN."""

import numpy as np


def max_tradeoff_threshold(signal, noise, ntrials):
    """Select the "max-tradeoff" threshold on the recovery curve.

    Returns the most-unbiased operating point that still captures the bulk of
    the achievable recovery: the point of maximum recovery-gain per unit of
    signal variance sacrificed, relative to the trial-average baseline. It is
    the point on the descending (peak -> trial-average) limb of the recovery
    curve that lies farthest from the chord joining those two anchors, measured
    in (fraction-of-signal-variance-retained, normalized-recovery) coordinates.

    Equivalently (in coordinates where x is normalized so peak->trial-average
    spans [0, 1]) it is argmax(x + y) -- the point maximizing the equal-weight
    bias/recovery tradeoff. The selection is invariant to affine rescaling of
    either axis (it depends only on the two anchors and the curve).

    Concretely, with per-dimension signal/noise variances in basis order:
      recovery(k)  = cumsum(signal - noise/ntrials)      (rises, peaks, falls)
      sv_frac(k)   = cumsum(signal) / total_signal       (fraction retained)

    Most meaningful when dimensions are ordered by signal variance (the signal
    basis), where the descending limb spans the near-unbiased, high-sv_frac
    region. Fully analytic; no cross-validation.

    Parameters
    ----------
    signal : array [ndims]   per-dimension signal variance (basis order)
    noise  : array [ndims]   per-dimension noise variance (basis order)
    ntrials : scalar         number of trials (or average if NaNs present)

    Returns
    -------
    k : int   selected number of dimensions to retain, in [0, ndims]
    """
    signal = np.asarray(signal, dtype=float)
    noise = np.asarray(noise, dtype=float)
    ndims = len(signal)
    if ndims == 0:
        return 0

    diff = signal - noise / ntrials
    rec = np.concatenate([[0.0], np.cumsum(diff)])          # recovery curve, len ndims+1
    sig_cum = np.concatenate([[0.0], np.cumsum(signal)])    # cumulative signal variance
    total_S = sig_cum[-1]

    k_peak = int(np.argmax(rec))                            # recovery peak (= max recovery)

    # Degenerate cases: no signal, or no usable descending limb -> the peak is
    # already the most we can do, so fall back to it.
    if total_S <= 0 or (ndims - k_peak) < 2:
        return k_peak
    rec_peak = rec[k_peak]
    rec_ta = rec[ndims]                                     # trial-average (retain all dims)
    if (rec_peak - rec_ta) <= 1e-12:
        return k_peak

    seg = np.arange(k_peak, ndims + 1)
    x = sig_cum[seg] / total_S                              # fraction signal var retained
    y = (rec[seg] - rec_ta) / (rec_peak - rec_ta)          # recovery rescaled to [0, 1]
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    denom = np.hypot(x1 - x0, y1 - y0)
    if denom <= 1e-12:
        return k_peak
    # perpendicular distance from the peak -> trial-average chord
    dist = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
    return int(seg[int(np.argmax(dist))])
