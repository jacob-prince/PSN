"""Covariance projection utility for PSN."""

import numpy as np


def project_covs(cS, cN, B):
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

    # Efficient diagonal extraction (avoids full matrix multiplication)
    # diag(B.T @ C @ B)[i] = B[:,i].T @ C @ B[:,i] = sum(B[:,i] * (C @ B[:,i]))
    sig = np.sum((cS @ B) * B, axis=0)
    noi = np.sum((cN @ B) * B, axis=0)

    # Clamp tiny negatives from numerical error
    sig = np.maximum(sig, 0)
    noi = np.maximum(noi, 0)

    return sig, noi
