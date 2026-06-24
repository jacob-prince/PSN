"""Best-among-allowable threshold-selection helpers for PSN.

When opt['allowable_thresholds'] restricts which thresholds PSN may choose,
selection picks the BEST threshold *among* the allowable values: it does not
search outside the set and snap. These primitives implement that rule for the
two selection shapes used across criteria:

  - maximize a per-dimension score          -> argmax_allowable    (prediction)
  - first to reach a cumulative target       -> first_reach_allowable (variance,
                                                variance_eigenvalues, alpha)

The max-tradeoff (chord-distance) criterion restricts inside
max_tradeoff_threshold itself. A single allowable value forces exactly that many
dimensions, which falls out of these helpers naturally (a one-element candidate
set). Mirrors the MATLAB utilities/threshold/{allowable_candidates,
argmax_allowable,first_reach_allowable}.m.
"""

import numpy as np


def allowable_candidates(allowable, ndims):
    """Sorted unique integer candidates from <allowable>, clipped to [0, ndims]."""
    C = np.unique(np.asarray(allowable).ravel().astype(int))
    return C[(C >= 0) & (C <= ndims)]


def argmax_allowable(objective, allowable):
    """Number of dims in <allowable> maximizing objective[k]; fewest dims on ties.

    <objective> is the length ndims+1 cumulative curve (index = number of dims).
    Matches the unconstrained np.argmax convention (first maximizer => fewest
    dims). Falls back to the unconstrained argmax if the candidate set is empty.
    """
    objective = np.asarray(objective)
    ndims = len(objective) - 1
    C = allowable_candidates(allowable, ndims)
    if C.size == 0:
        return int(np.argmax(objective))
    return int(C[int(np.argmax(objective[C]))])


def first_reach_allowable(cum_curve, target, allowable, floor_k=0):
    """Smallest k in <allowable> with cum_curve[k] >= target and k >= floor_k.

    If none qualify, returns the largest allowable candidate (the closest we can
    get from below). <cum_curve> is the length ndims+1 cumulative curve (index =
    number of dims). Returns 0 if the candidate set is empty.
    """
    cum_curve = np.asarray(cum_curve)
    ndims = len(cum_curve) - 1
    C = allowable_candidates(allowable, ndims)
    if C.size == 0:
        return 0
    elig = C[(C >= floor_k) & (cum_curve[C] >= target)]
    if elig.size > 0:
        return int(elig.min())
    return int(C.max())
