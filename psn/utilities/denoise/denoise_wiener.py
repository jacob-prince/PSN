"""Wiener shrinkage denoiser for PSN.

Applies dimension-wise shrinkage weights instead of hard truncation.
Uses PSN's estimated signal/noise variances per dimension to compute
optimal Wiener weights. Also provides the full-rank matrix Wiener filter
(basis='wiener') which bypasses basis construction entirely.
"""

import numpy as np
from scipy.linalg import solve

from psn._device import resolve_device, to_device, from_device, is_cpu


def compute_wiener_weights(signal_proj, noise_proj, ntrials_eval):
    """Compute Wiener shrinkage weights for each dimension.

    The Wiener filter weight for dimension k is:
        w_k = s_k / (s_k + n_k / t_eval)

    where:
        s_k = signal variance in dimension k
        n_k = noise variance in dimension k
        t_eval = number of trials used in evaluation (affects noise in trial average)

    Parameters
    ----------
    signal_proj : ndarray, shape (ndims,)
        Signal variance per basis dimension.
    noise_proj : ndarray, shape (ndims,)
        Noise variance per basis dimension.
    ntrials_eval : float
        Number of trials used to form the trial average being denoised.
        This is typically the same as ntrials_avg from training, but can
        differ if denoising held-out data with different trial counts.

    Returns
    -------
    weights : ndarray, shape (ndims,)
        Wiener shrinkage weights in [0, 1] for each dimension.
    """
    # Compute denominator: signal + noise/t_eval
    denom = signal_proj + noise_proj / ntrials_eval

    # Avoid division by zero: where denom is 0 or negative, set weight to 0
    weights = np.zeros_like(signal_proj)
    valid = denom > 0
    weights[valid] = signal_proj[valid] / denom[valid]

    # Clip to [0, 1] for safety (should already be in this range)
    weights = np.clip(weights, 0.0, 1.0)

    return weights


def denoise_wiener(basis, signal_proj, noise_proj, ntrials_eval, device='cpu'):
    """Build Wiener shrinkage denoiser matrix.

    Instead of hard truncation (keep K dims, drop rest), applies continuous
    shrinkage weights to each dimension:

        X_denoised = B @ diag(w) @ B.T @ X

    where w_k = s_k / (s_k + n_k / t_eval)

    Parameters
    ----------
    basis : ndarray, shape (nunits, ndims)
        Orthonormal basis matrix (columns are basis vectors).
    signal_proj : ndarray, shape (ndims,)
        Signal variance per basis dimension.
    noise_proj : ndarray, shape (ndims,)
        Noise variance per basis dimension.
    ntrials_eval : float
        Number of trials used to form the trial average being denoised.

    Returns
    -------
    denoiser : ndarray, shape (nunits, nunits)
        Symmetric Wiener denoising matrix.
    weights : ndarray, shape (ndims,)
        Wiener shrinkage weights used.
    effective_dims : float
        Effective number of dimensions retained (sum of weights).
    """
    nunits = basis.shape[0]
    ndims = basis.shape[1]

    # Compute Wiener weights
    weights = compute_wiener_weights(signal_proj, noise_proj, ntrials_eval)

    # Build Wiener denoiser: B @ diag(w) @ B.T = (B * w) @ B.T.
    # Same shape as the hybrid denoise's matmul — GPU when requested.
    dev = resolve_device(device)
    if is_cpu(dev):
        denoiser = (basis * weights) @ basis.T
    else:
        import torch
        tdtype = (torch.float64 if basis.dtype == np.float64
                  else torch.float32)
        basis_t = to_device(basis, dev, dtype=tdtype)
        weights_t = to_device(weights, dev, dtype=tdtype)
        denoiser_t = (basis_t * weights_t) @ basis_t.T
        denoiser = from_device(denoiser_t)
        del basis_t, weights_t, denoiser_t
        if str(dev).startswith('cuda'):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # Effective number of dimensions (sum of weights)
    effective_dims = np.sum(weights)

    return denoiser, weights, effective_dims


def denoise_wiener_global(basis, signal_proj, noise_proj, basis_eigenvalues, ntrials, opt):
    """Global Wiener denoising (drop-in replacement for denoise_global).

    Applies Wiener shrinkage using PSN's signal/noise estimates.
    Returns outputs compatible with denoise_global interface.

    Parameters
    ----------
    basis : ndarray, shape (nunits, ndims)
        Orthonormal basis matrix.
    signal_proj : ndarray, shape (ndims,)
        Signal variance per dimension.
    noise_proj : ndarray, shape (ndims,)
        Noise variance per dimension.
    basis_eigenvalues : ndarray or None
        Eigenvalues from basis construction.
    ntrials : float
        Average number of trials.
    opt : dict
        PSN options dict. Uses opt['ntrials_eval'] if present, else ntrials.

    Returns
    -------
    denoiser : ndarray, shape (nunits, nunits)
        Symmetric Wiener denoising matrix.
    best_threshold : float
        Effective number of dimensions (sum of weights).
    objective : ndarray, shape (ndims+1,)
        Cumulative prediction objective for visualization.
    signalvar : ndarray
        Signal variance per dimension.
    noisevar : ndarray
        Noise variance per dimension.
    unit_cumsum_curves : list
        Unit-specific objective curves (for compatibility).
    unit_signal_vars : list
        Unit-specific signal variances.
    unit_noise_vars : list
        Unit-specific noise variances.
    wiener_weights : ndarray, shape (ndims,)
        The Wiener weights applied to each dimension.
    """
    from .compute_unit_weighted_projections import compute_unit_weighted_projections

    # Get ntrials_eval (defaults to ntrials if not specified or None)
    ntrials_eval = opt.get('ntrials_eval')
    if ntrials_eval is None:
        ntrials_eval = ntrials

    # Build Wiener denoiser
    denoiser, wiener_weights, effective_dims = denoise_wiener(
        basis, signal_proj, noise_proj, ntrials_eval,
        device=opt.get('device', 'cpu')
    )

    # Compute objective curve for visualization (prediction objective)
    objective = np.concatenate([[0], np.cumsum(signal_proj - noise_proj / ntrials)])

    # For compatibility: "threshold" is the effective number of dimensions
    best_threshold = effective_dims

    # Outputs
    signalvar = signal_proj
    noisevar = noise_proj

    # Compute unit-specific weighted variances for diagnostics
    unit_cumsum_curves, unit_signal_vars, unit_noise_vars, _ = \
        compute_unit_weighted_projections(basis, signal_proj, noise_proj, ntrials, False)

    return (denoiser, best_threshold, objective, signalvar, noisevar,
            unit_cumsum_curves, unit_signal_vars, unit_noise_vars, wiener_weights)


def denoise_fullrank_wiener(cSb, cNb, data, trial_avg, unit_means, ntrials_avg, nunits, gsn_result, opt):
    """Full-rank matrix Wiener filter: D = Σ_S @ (Σ_S + Σ_N/t)^{-1}.

    Bypasses basis construction, ordering, criterion, and thresholding.
    This is the Bayes-optimal linear estimator when signal and noise
    covariances are known.

    Parameters
    ----------
    cSb : ndarray, shape (nunits, nunits)
        Signal covariance matrix from GSN.
    cNb : ndarray, shape (nunits, nunits)
        Noise covariance matrix from GSN.
    data : ndarray, shape (nunits, nconds, ntrials)
        Original input data.
    trial_avg : ndarray, shape (nunits, nconds)
        Trial-averaged data.
    unit_means : ndarray, shape (nunits,)
        Mean response per unit.
    ntrials_avg : float
        Average number of trials.
    nunits : int
        Number of units.
    gsn_result : dict
        Full GSN results dict.
    opt : dict
        PSN options dict. Uses opt['ntrials_eval'] if present.

    Returns
    -------
    results : dict
        Complete PSN results dict ready for visualization and return.
    """
    ntrials_eval = opt.get('ntrials_eval')
    if ntrials_eval is None:
        ntrials_eval = ntrials_avg

    # D = Σ_S @ (Σ_S + Σ_N/t)^{-1}
    # Store denoiser such that denoiser.T @ x = D @ x
    # So denoiser = inv(A) @ Σ_S, and denoiser.T = Σ_S @ inv(A) = D
    device = resolve_device(opt.get('device', 'cpu'))
    A = cSb + cNb / ntrials_eval
    jitter = 1e-10 * np.trace(A) / nunits
    A += jitter * np.eye(nunits)
    if is_cpu(device):
        denoiser = solve(A, cSb, assume_a='sym')
    else:
        # GPU solve. Symmetric-positive-definite (after the jitter
        # term we just added), so use cholesky_solve which is the
        # GPU-native equivalent of scipy.linalg.solve(assume_a='sym').
        import torch
        tdtype = (torch.float64 if cSb.dtype == np.float64
                  else torch.float32)
        A_t   = to_device(A,   device, dtype=tdtype)
        cSb_t = to_device(cSb, device, dtype=tdtype)
        L = torch.linalg.cholesky(A_t)
        denoiser_t = torch.cholesky_solve(cSb_t, L)
        denoiser = from_device(denoiser_t)
        del A_t, cSb_t, L, denoiser_t
        if str(device).startswith('cuda'):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # Apply denoiser
    denoiseddata = denoiser.T @ (trial_avg - unit_means[:, np.newaxis]) + unit_means[:, np.newaxis]

    # Residuals
    residuals = data - denoiseddata[:, :, np.newaxis]

    # Diagnostics: signal/noise variance per unit before and after
    D = denoiser.T  # the actual filter matrix
    svnv_before = np.column_stack([np.diag(cSb), np.diag(cNb) / ntrials_avg])
    D_cSb_Dt = D @ cSb @ D.T
    D_cNb_Dt = D @ cNb @ D.T
    svnv_after = np.column_stack([np.diag(D_cSb_Dt), np.diag(D_cNb_Dt) / ntrials_avg])

    effective_dims = np.trace(D)

    # Compute cSb eigenvectors for visualization (same as signal basis)
    eigvals_cSb, eigvecs_cSb = np.linalg.eigh(cSb)
    sort_idx = np.argsort(eigvals_cSb)[::-1]
    eigvals_cSb = eigvals_cSb[sort_idx]
    eigvecs_cSb = eigvecs_cSb[:, sort_idx]
    signal_proj = np.sum(eigvecs_cSb * (cSb @ eigvecs_cSb), axis=0)
    noise_proj = np.sum(eigvecs_cSb * (cNb @ eigvecs_cSb), axis=0)

    return {
        'denoiseddata': denoiseddata,
        'residuals': residuals,
        'unit_means': unit_means,
        'denoiser': denoiser,
        'svnv_before': svnv_before,
        'svnv_after': svnv_after,
        'best_threshold': effective_dims,
        'fullbasis': eigvecs_cSb,
        'basis_eigenvalues': eigvals_cSb,
        'unitreorderings': np.tile(np.arange(nunits), (nunits, 1)),
        'gsn_result': gsn_result,
        'signalvar': signal_proj,
        'noisevar': noise_proj,
        'objective': None,
        'basis_viz': eigvecs_cSb,
        'signal_proj_viz': signal_proj,
        'noise_proj_viz': noise_proj,
        'basis_eigenvalues_viz': eigvals_cSb,
        'input_data': data,
        'signalsubspace': None,
        'dimreduce': None,
        'wiener_matrix': D,
        'opt_used': opt,
    }
