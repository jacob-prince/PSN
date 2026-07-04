"""Eigendecomposition utility with consistent sorting."""

import numpy as np

from psn._device import from_device, is_cpu, resolve_device, to_device


def eigh_descending_sym(matrix, do_symmetrize=False, device='cpu'):
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
    device : str or torch.device, optional
        Where to run the O(n^3) eigh. 'cpu' (default) uses numpy. A GPU device
        ('cuda'/'mps') runs it via torch on-device. Note: a GPU eigh picks a
        different orthonormal basis on degenerate eigenspaces than numpy, so
        downstream thresholds can differ by a few percent from a CPU run - an
        inherent property of the backend, not a bug.

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

    # Compute eigendecomposition on the requested device.
    dev = resolve_device(device)
    if is_cpu(dev):
        evals, evecs = np.linalg.eigh(matrix)
    else:
        import torch
        tdtype = (torch.float64 if np.asarray(matrix).dtype == np.float64
                  else torch.float32)
        M_t = to_device(matrix, dev, dtype=tdtype)
        evals_t, evecs_t = torch.linalg.eigh(M_t)
        evals = from_device(evals_t)
        evecs = from_device(evecs_t)
        del M_t, evals_t, evecs_t

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
