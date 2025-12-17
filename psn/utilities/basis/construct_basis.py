"""Basis construction utilities for PSN."""

import numpy as np


def construct_basis(cSb, cNb, basis_spec, data, trial_avg, unit_means, ntrials_avg, has_nans):
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
      basis, contains PCA eigenvalues for visualization only (not used for ranking)
    """

    nunits = data.shape[0]

    if isinstance(basis_spec, str):
        if basis_spec == 'signal':
            # Eigenvectors of signal covariance (GSN returns symmetric)
            basis_eigenvalues, basis = _eigh_descending_sym(cSb)

        elif basis_spec == 'difference':
            # Eigenvectors of signal - scaled noise
            # Eigenvalues encode the net benefit per dimension
            # Use ntrials_avg to properly handle NaN case
            A = cSb - cNb / ntrials_avg
            # Symmetrize derived matrix to handle numerical errors
            A = (A + A.T) / 2
            basis_eigenvalues, basis = _eigh_descending_sym(A)  # already symmetrized above

        elif basis_spec == 'noise':
            # Eigenvectors of noise covariance (GSN returns symmetric)
            basis_eigenvalues, basis = _eigh_descending_sym(cNb)

        elif basis_spec == 'pca':
            # Standard PCA on trial-averaged data
            # Eigenvectors from empirical covariance, but treated exactly like signal basis
            # in all subsequent ranking/thresholding (uses GSN signal_proj, not PCA eigenvalues).
            # PCA eigenvalues are kept for visualization purposes only.
            # Use pre-computed trial_avg to avoid redundant computation
            trial_avg_demeaned = trial_avg - unit_means[:, np.newaxis]
            cov_matrix = np.cov(trial_avg_demeaned, ddof=1)  # numpy cov returns symmetric matrix
            basis_eigenvalues, basis = _eigh_descending_sym(cov_matrix)  # no symmetrization needed
            # Note: PCA eigenvalues stored but NOT used for ranking/thresholding

        elif basis_spec == 'random':
            # Random orthonormal basis (no meaningful eigenvalues)
            # NOTE: This sets the RNG state for reproducibility
            np.random.seed(42)
            basis, _ = np.linalg.qr(np.random.randn(nunits, nunits))
            basis_eigenvalues = None

        else:
            raise ValueError(f'Unknown basis type: {basis_spec}')

    else:
        # User-provided custom basis (no eigenvalues available)
        basis = basis_spec

        if basis.shape[0] != nunits:
            raise ValueError(f'Custom basis must have {nunits} rows (matching nunits)')
        if basis.shape[1] < 1 or basis.shape[1] > nunits:
            raise ValueError(f'Custom basis must have between 1 and {nunits} columns')

        basis = _normalize_orthonormalize_basis(basis)
        basis_eigenvalues = None

    return basis, basis_eigenvalues


def _eigh_descending_sym(matrix, do_symmetrize=False):
    """EIGH_DESCENDING_SYM  Compute eigendecomposition with consistent sorting

    [evals_sorted, evecs_sorted] = _eigh_descending_sym(matrix) computes
    the eigendecomposition of a symmetric matrix and returns eigenvalues
    and eigenvectors sorted by descending eigenvalue magnitude, with
    standardized eigenvector signs for reproducibility.

    [evals_sorted, evecs_sorted] = _eigh_descending_sym(matrix, do_symmetrize)
    optionally forces the matrix to be symmetric before eigendecomposition.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <matrix> - [n x n] numeric matrix (should be symmetric or nearly symmetric)

    <do_symmetrize> (optional) - boolean. If True, enforces symmetry via
      (matrix + matrix.T)/2 before eigendecomposition. Default: False.
      Note: GSN returns symmetric cSb/cNb, so symmetrization is typically
      only needed for derived matrices like cSb - cNb/ntrials_avg

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <evals_sorted> - [n] eigenvalues sorted in descending order

    <evecs_sorted> - [n x n] eigenvectors with columns sorted to match
      <evals_sorted>. Signs standardized so that the largest-magnitude
      element in each column is positive
    """

    if do_symmetrize:
        matrix = (matrix + matrix.T) / 2

    # Compute eigendecomposition
    evals, evecs = np.linalg.eigh(matrix)

    # Sort by eigenvalue magnitude (descending)
    order = np.argsort(evals)[::-1]
    evals_sorted = evals[order]
    evecs_sorted = evecs[:, order]

    # Deterministic sign: make largest-magnitude element positive
    piv = np.argmax(np.abs(evecs_sorted), axis=0)
    idx = (piv, np.arange(evecs_sorted.shape[1]))
    sgn = np.sign(evecs_sorted[idx])
    sgn[sgn == 0] = 1
    evecs_sorted = evecs_sorted * sgn

    return evals_sorted, evecs_sorted


def _normalize_orthonormalize_basis(basis):
    """NORMALIZE_ORTHONORMALIZE_BASIS  Ensure basis has orthonormal columns

    basis = _normalize_orthonormalize_basis(basis) takes a matrix and
    ensures its columns are orthonormal (unit length and mutually orthogonal).

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <basis> - [n x k] numeric matrix with k basis vectors as columns

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <basis> - [n x k] matrix with orthonormal columns. First normalizes each
      column to unit length, then checks orthogonality. If not orthogonal
      (Gram matrix not identity within tolerance 1e-10), applies QR
      decomposition to enforce orthonormality
    """

    norms = np.sqrt(np.sum(basis**2, axis=0))
    norms[norms == 0] = 1
    basis = basis / norms

    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(gram.shape[0]), atol=1e-10, rtol=0):
        basis, _ = np.linalg.qr(basis)

    return basis
