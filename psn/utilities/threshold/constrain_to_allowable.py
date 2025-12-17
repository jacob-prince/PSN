"""Threshold constraint utility for PSN."""

import numpy as np


def constrain_to_allowable(k, allowable):
    """CONSTRAIN_TO_ALLOWABLE  Force threshold to nearest allowable value

    k_constrained = constrain_to_allowable(k, allowable) snaps the threshold(s)
    to the nearest value in the allowable set, rounding up in case of ties.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <k> - scalar or [nunits] array of threshold values to constrain

    <allowable> - array of allowed threshold values (e.g., [0, 5, 10, 15])

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <k_constrained> - same size as k, with each value replaced by the nearest
      value from <allowable>. In case of a tie (equal distance to two allowable
      values), rounds up to the larger value
    """

    allowable = np.asarray(allowable)

    if np.isscalar(k):
        if k not in allowable:
            diffs = np.abs(allowable - k)
            min_diff = np.min(diffs)
            tied_values = allowable[diffs == min_diff]
            k_constrained = np.max(tied_values)  # Round up on tie
        else:
            k_constrained = k
    else:
        # Vector case (unit-specific)
        k = np.asarray(k)
        k_constrained = k.copy()
        for i in range(len(k)):
            if k[i] not in allowable:
                diffs = np.abs(allowable - k[i])
                min_diff = np.min(diffs)
                tied_values = allowable[diffs == min_diff]
                k_constrained[i] = np.max(tied_values)

    return k_constrained
