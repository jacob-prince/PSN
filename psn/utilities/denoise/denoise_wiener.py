"""Full-rank matrix Wiener filter for PSN.

Provides the full-rank matrix Wiener filter D = Σ_S @ (Σ_S + Σ_N/t)^{-1}
(criterion='wiener'), the optimal linear estimator. It applies no
truncation and bypasses basis construction entirely.
"""

import numpy as np

from psn._device import resolve_device, to_device, from_device, is_cpu
from ..basis.eigh_descending_sym import eigh_descending_sym


def denoise_fullrank_wiener(cSb, cNb, data, trial_avg, unit_means, ntrials_avg, nunits, gsn_result, opt):
    """Full-rank matrix Wiener filter: D = Σ_S @ (Σ_S + Σ_N/t)^{-1}.

    Bypasses basis construction, ordering, criterion, and thresholding.
    This is the optimal linear estimator when signal and noise
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
        PSN options dict.

    Returns
    -------
    results : dict
        Complete PSN results dict ready for visualization and return.
    """
    # D = Σ_S @ (Σ_S + Σ_N/t)^{-1}
    # Store denoiser such that denoiser.T @ x = D @ x
    # So denoiser = inv(A) @ Σ_S, and denoiser.T = Σ_S @ inv(A) = D
    device = resolve_device(opt.get('device', 'cpu'))
    A = cSb + cNb / ntrials_avg
    jitter = 1e-10 * np.trace(A) / nunits
    A += jitter * np.eye(nunits)
    if is_cpu(device):
        # A is symmetric positive-definite after the jitter term, so
        # np.linalg.solve (inv(A) @ cSb == D.T) is exact here. We avoid
        # scipy.linalg.solve(assume_a='sym') on purpose: its LAPACK sysv
        # path segfaults when torch is co-imported (MKL/libomp duplicate
        # OpenMP runtime), and `import psn` always loads torch via _device.
        denoiser = np.linalg.solve(A, cSb)
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

    # Compute cSb eigenvectors for visualization (same as the signal basis,
    # using the standardized sign convention so it matches construct_basis and
    # the MATLAB implementation deterministically).
    eigvals_cSb, eigvecs_cSb = eigh_descending_sym(cSb)
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
