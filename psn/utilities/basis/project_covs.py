"""Covariance projection utility for PSN."""

import numpy as np

from psn._device import from_device, is_cpu, resolve_device, to_device


def project_covs(cS, cN, B, device='cpu'):
    """PROJECT_COVS  Project covariances into basis

    [sig, noi] = project_covs(cS, cN, B) computes the signal and noise
    variance along each dimension of the basis B by projecting the
    covariance matrices into the basis coordinate system.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <cS> - [nunits x nunits] symmetric signal covariance matrix from GSN

    <cN> - [nunits x nunits] symmetric noise covariance matrix from GSN

    <B> - [nunits x ndims] orthonormal basis matrix with basis vectors as columns

    <device> - 'cpu' (default), 'cuda', 'mps', or 'auto'. Set to a
      non-cpu device to run the two matmuls on a torch device - at
      nunits ≥ 10000 the GPU is 20-50× faster than CPU numpy.

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <sig> - [ndims] signal variance along each basis dimension.
      Mathematically: sig[i] = B[:,i].T @ cS @ B[:,i], which is the i-th
      diagonal element of B.T @ cS @ B

    <noi> - [ndims] noise variance along each basis dimension.
      Mathematically: noi[i] = B[:,i].T @ cN @ B[:,i]

    -------------------------------------------------------------------------
    Implementation notes:
    -------------------------------------------------------------------------

    Uses efficient element-wise computation: diag(B.T @ C @ B) = sum((C @ B) * B, axis=0)
    This is O(N^2 * K) instead of O(N^3) for full matrix multiplication.
    Small negative values from numerical error are clamped to zero.
    """
    device = resolve_device(device)

    if is_cpu(device):
        # CPU/numpy path. diag(B.T @ C @ B)[i] = sum(B[:,i] * (C @ B[:,i]))
        sig = np.sum((cS @ B) * B, axis=0)
        noi = np.sum((cN @ B) * B, axis=0)
        return np.maximum(sig, 0), np.maximum(noi, 0)

    # GPU/torch path. Pick a working precision: f32 unless the input
    # was already f64, in which case stay f64.
    import torch
    out_dtype = np.float64 if (cS.dtype == np.float64 and cN.dtype == np.float64
                                and B.dtype == np.float64) else np.float32
    tdtype = torch.float64 if out_dtype == np.float64 else torch.float32
    cS_t = to_device(cS, device, dtype=tdtype)
    cN_t = to_device(cN, device, dtype=tdtype)
    B_t  = to_device(B,  device, dtype=tdtype)
    # Use einsum so it's a single fused kernel.
    sig_t = torch.einsum('ij,ik,jk->k', cS_t, B_t, B_t)
    noi_t = torch.einsum('ij,ik,jk->k', cN_t, B_t, B_t)
    sig_t = torch.clamp(sig_t, min=0)
    noi_t = torch.clamp(noi_t, min=0)
    return from_device(sig_t).astype(out_dtype), from_device(noi_t).astype(out_dtype)
