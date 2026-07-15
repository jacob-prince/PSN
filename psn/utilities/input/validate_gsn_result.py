"""Structurally validate a cached gsn_result before PSN trusts it.

The cache's covariances feed the projection/threshold math and its cached
eigenvectors can replace PSN's eigh, so a malformed cache (wrong shape, NaN,
non-symmetric, 1-D eigvecs) would fail cryptically or denoise silently wrong.

These checks are structural only (shape, finiteness, symmetry, a cheap PSD-ish
diagonal check). They cannot confirm the cache describes the same population of
units as the data being denoised - matching unit count is necessary, not
sufficient - so that pairing stays the caller's responsibility.
"""

import numpy as np

# Scale-relative tolerances (GSN covariances are symmetric and PSD).
_SYM_RTOL = 1e-6      # max|M - M.T| <= _SYM_RTOL * max|M|
_PSD_RTOL = 1e-8      # min(diag) >= -_PSD_RTOL * max|diag|


def _check_covariance(M, name, nunits):
    """Validate one covariance matrix; return it coerced to a float ndarray."""
    M = np.asarray(M)
    if M.ndim != 2 or M.shape != (nunits, nunits):
        raise ValueError(
            f"gsn_result['{name}'] must be a ({nunits}, {nunits}) matrix "
            f"matching the data's unit count; got shape {M.shape}.")
    if not np.isfinite(M).all():
        raise ValueError(
            f"gsn_result['{name}'] contains non-finite values (NaN/Inf); "
            f"the cache is corrupt.")
    scale = float(np.max(np.abs(M))) if M.size else 0.0
    if scale > 0.0:
        asym = float(np.max(np.abs(M - M.T)))
        if asym > _SYM_RTOL * scale:
            raise ValueError(
                f"gsn_result['{name}'] is not symmetric "
                f"(max asymmetry {asym:.3e} vs scale {scale:.3e}).")
    # Cheap PSD sanity: diagonal variances can't be negative (full check is O(N^3)).
    diag = np.diag(M)
    dscale = float(np.max(np.abs(diag))) if diag.size else 0.0
    if diag.size and diag.min() < -_PSD_RTOL * max(1.0, dscale):
        raise ValueError(
            f"gsn_result['{name}'] has a negative diagonal value "
            f"({diag.min():.3e}); a covariance has non-negative variances.")
    return M


def _check_cached_eigvecs(gsn_result, nunits):
    """Validate cached eigvecs/eigvals before the swap; construct_basis would
    silently re-orthonormalize a NaN/1-D array into a valid-shaped wrong basis."""
    for kind in ('signal', 'difference'):
        vkey, wkey = f'eigvecs_{kind}', f'eigvals_{kind}'
        has_v, has_w = vkey in gsn_result, wkey in gsn_result
        if not has_v and not has_w:
            continue
        if has_v != has_w:
            raise ValueError(
                f"gsn_result has '{vkey if has_v else wkey}' but not "
                f"'{wkey if has_v else vkey}'; cached eigvecs and eigvals must "
                f"be provided together.")
        V = np.asarray(gsn_result[vkey])
        w = np.asarray(gsn_result[wkey]).reshape(-1)
        if V.ndim != 2 or V.shape[0] != nunits:
            raise ValueError(
                f"gsn_result['{vkey}'] must be a 2-D ({nunits}, k) matrix; "
                f"got shape {V.shape}.")
        k = V.shape[1]
        if k < 1 or k > nunits:
            raise ValueError(
                f"gsn_result['{vkey}'] must have between 1 and {nunits} "
                f"columns; got {k}.")
        if w.shape[0] != k:
            raise ValueError(
                f"gsn_result['{wkey}'] length ({w.shape[0]}) must match the "
                f"number of cached eigvec columns ({k}).")
        if not np.isfinite(V).all() or not np.isfinite(w).all():
            raise ValueError(
                f"gsn_result['{vkey}']/'{wkey}' contain non-finite values "
                f"(NaN/Inf); the cache is corrupt.")


def validate_gsn_result(gsn_result, nunits):
    """Structurally validate a cached gsn_result against a data unit count.

    Raises ValueError on the first problem found. Returns the same dict with
    'cSb'/'cNb' coerced to ndarrays. <gsn_result> must contain 'cSb' and 'cNb'
    and may carry cached eigvecs_{signal,difference}/eigvals_{signal,difference};
    <nunits> is the unit count of the data being denoised. See the module
    docstring for what these checks cannot cover.
    """
    if 'cSb' not in gsn_result or 'cNb' not in gsn_result:
        raise ValueError("gsn_result must contain 'cSb' and 'cNb' keys")
    gsn_result['cSb'] = _check_covariance(gsn_result['cSb'], 'cSb', nunits)
    gsn_result['cNb'] = _check_covariance(gsn_result['cNb'], 'cNb', nunits)
    _check_cached_eigvecs(gsn_result, nunits)
    return gsn_result
