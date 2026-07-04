"""Basis construction utilities for PSN."""

import numpy as np

from .eigh_descending_sym import eigh_descending_sym as _eigh_descending_sym
from .normalize_orthonormalize_basis import (
    normalize_orthonormalize_basis as _normalize_orthonormalize_basis,
)


def construct_basis(cSb, cNb, basis_spec, data, trial_avg, unit_means, ntrials_avg, has_nans,
                    custom_basis_eigenvalues=None, device='cpu'):
    """CONSTRUCT_BASIS  Create the orthonormal basis for denoising

    [basis, basis_eigenvalues] = construct_basis(cSb, cNb, basis_spec, ...)
    constructs an orthonormal basis for PSN denoising according to the
    specified basis type.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <cSb> - [nunits x nunits] symmetric signal covariance matrix from GSN

    <cNb> - [nunits x nunits] symmetric noise covariance matrix from GSN

    <basis_spec> - basis specifier. Either string or matrix:
      'signal'     - Eigenvectors of signal covariance (cSb)
      'difference' - Eigenvectors of cSb - cNb/ntrials_avg (emphasize signal-dominated directions)
      'noise'      - Eigenvectors of noise covariance (cNb)
      'pca'        - Standard PCA on trial-averaged data
      'random'     - Random orthonormal basis (uses fixed seed for reproducibility)
      B            - User-provided basis matrix [nunits x D] with orthonormal columns

    <data> - [nunits x nconds x ntrials] original data array

    <trial_avg> - [nunits x nconds] pre-computed trial-averaged data

    <unit_means> - [nunits] mean response per unit

    <ntrials_avg> - scalar, average number of valid trials (handles NaN case correctly)

    <has_nans> - boolean, whether data contains NaNs

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <basis> - [nunits x ndims] orthonormal basis vectors. Columns are sorted
      by descending eigenvalue magnitude (for eigenvalue-based methods) or
      left as provided (for custom/random bases)

    <basis_eigenvalues> - [ndims] eigenvalues associated with basis, sorted
      to match basis columns. None for custom or random bases. For 'pca'
      basis, contains PCA eigenvalues (used for ordering, but not appropriate
      for criterion='variance_eigenvalues')
    """

    nunits = data.shape[0]

    if isinstance(basis_spec, str):
        if basis_spec == 'signal':
            # Eigenvectors of signal covariance (GSN returns symmetric)
            basis_eigenvalues, basis = _eigh_descending_sym(cSb, device=device)

        elif basis_spec == 'difference':
            # Eigenvectors of signal - scaled noise
            # Eigenvalues encode the net benefit per dimension
            # Use ntrials_avg to properly handle NaN case
            A = cSb - cNb / ntrials_avg
            # Symmetrize derived matrix to handle numerical errors
            A = (A + A.T) / 2
            basis_eigenvalues, basis = _eigh_descending_sym(A, device=device)  # already symmetrized above

        elif basis_spec == 'noise':
            # Eigenvectors of noise covariance (GSN returns symmetric)
            basis_eigenvalues, basis = _eigh_descending_sym(cNb, device=device)

        elif basis_spec == 'pca':
            # Standard PCA on trial-averaged data
            # Eigenvectors from empirical covariance, but treated exactly like signal basis
            # in all subsequent ranking/thresholding (uses GSN signal_proj, not PCA eigenvalues).
            # PCA eigenvalues are kept for visualization purposes only.
            # Use pre-computed trial_avg to avoid redundant computation
            trial_avg_demeaned = trial_avg - unit_means[:, np.newaxis]
            cov_matrix = np.cov(trial_avg_demeaned, ddof=1)  # numpy cov returns symmetric matrix
            basis_eigenvalues, basis = _eigh_descending_sym(cov_matrix, device=device)  # no symmetrization needed
            # Note: PCA eigenvalues ARE used for ordering (default behavior), but should
            # NOT be used with criterion='variance_eigenvalues' as they don't represent
            # GSN-estimated signal variance.

        elif basis_spec == 'random':
            # Random orthonormal basis (no meaningful eigenvalues)
            # NOTE: This sets the RNG state for reproducibility
            np.random.seed(42)
            basis, _ = np.linalg.qr(np.random.randn(nunits, nunits))
            basis_eigenvalues = None

        else:
            raise ValueError(f'Unknown basis type: {basis_spec}')

    else:
        # User-provided custom basis. Eigenvalues optional - pass them
        # via `custom_basis_eigenvalues` when the caller already has
        # them (e.g. when reusing eigvecs cached from a previous GSN
        # run). When provided, downstream uses 'eigenvalues' ordering
        # instead of the signalvariance fallback, which makes
        # opt['basis']=<eigvecs of cSb> equivalent to
        # opt['basis']='signal' end-to-end.
        basis = basis_spec

        if basis.shape[0] != nunits:
            raise ValueError(f'Custom basis must have {nunits} rows (matching nunits)')
        if basis.shape[1] < 1 or basis.shape[1] > nunits:
            raise ValueError(f'Custom basis must have between 1 and {nunits} columns')

        # Ensure basis is real (in case user provides complex-valued basis from eigendecomposition)
        if np.iscomplexobj(basis):
            basis = np.real(basis)

        basis = _normalize_orthonormalize_basis(basis)
        if custom_basis_eigenvalues is not None:
            ev = np.asarray(custom_basis_eigenvalues).reshape(-1)
            if ev.shape[0] != basis.shape[1]:
                raise ValueError(
                    f'custom_basis_eigenvalues length ({ev.shape[0]}) must match '
                    f'basis column count ({basis.shape[1]}).')
            basis_eigenvalues = ev
        else:
            basis_eigenvalues = None

    return basis, basis_eigenvalues
