"""Eigendecomposition utility with consistent sorting."""

import numpy as np


def eigh_descending_sym(matrix, do_symmetrize=False):
    """Compute eigendecomposition with consistent sorting.

    Computes the eigendecomposition of a symmetric matrix and returns
    eigenvalues and eigenvectors sorted by descending eigenvalue magnitude,
    with standardized eigenvector signs for reproducibility.

    Parameters
    ----------
    matrix : ndarray, shape (n, n)
        Numeric matrix (should be symmetric or nearly symmetric).
    do_symmetrize : bool, optional
        If True, enforces symmetry via (matrix + matrix.T)/2 before
        eigendecomposition. Default: False.
        Note: GSN returns symmetric cSb/cNb, so symmetrization is typically
        only needed for derived matrices like cSb - cNb/ntrials_avg.

    Returns
    -------
    evals_sorted : ndarray, shape (n,)
        Eigenvalues sorted in descending order.
    evecs_sorted : ndarray, shape (n, n)
        Eigenvectors with columns sorted to match evals_sorted.
        Signs standardized so that the largest-magnitude element
        in each column is positive.
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
