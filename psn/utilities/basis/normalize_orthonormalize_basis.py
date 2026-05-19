"""Basis orthonormalization utility."""

import numpy as np


def normalize_orthonormalize_basis(basis):
    """Ensure basis has orthonormal columns.

    Takes a matrix and ensures its columns are orthonormal
    (unit length and mutually orthogonal).

    Parameters
    ----------
    basis : ndarray, shape (n, k)
        Numeric matrix with k basis vectors as columns.

    Returns
    -------
    basis : ndarray, shape (n, k)
        Matrix with orthonormal columns. First normalizes each column
        to unit length, then checks orthogonality. If not orthogonal
        (Gram matrix not identity within tolerance 1e-10), applies QR
        decomposition to enforce orthonormality.
    """
    norms = np.sqrt(np.sum(basis**2, axis=0))
    norms[norms == 0] = 1
    basis = basis / norms

    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(gram.shape[0]), atol=1e-10, rtol=0):
        basis, _ = np.linalg.qr(basis)

    return basis
