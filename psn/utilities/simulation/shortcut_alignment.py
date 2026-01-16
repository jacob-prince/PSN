"""Direct alignment utility for basis matrices."""

import numpy as np


def shortcut_alignment(U_signal, U_noise, k):
    """Directly align the first k noise PCs to the signal PCs.

    This method provides exact alignment (alpha=1.0) and maintains
    orthonormality through Gram-Schmidt orthogonalization.

    Parameters
    ----------
    U_signal : ndarray, shape (nvox, nvox)
        Orthonormal signal basis.
    U_noise : ndarray, shape (nvox, nvox)
        Orthonormal noise basis.
    k : int
        Number of top PCs to align.

    Returns
    -------
    U_noise_adj : ndarray, shape (nvox, nvox)
        Orthonormal basis with first k columns aligned to U_signal.
    """
    U_noise_adj = U_noise.copy()
    nvox = U_signal.shape[0]

    # Set top k noise PCs equal to signal PCs
    U_noise_adj[:, :k] = U_signal[:, :k]

    # Orthonormalize remaining PCs via Gram-Schmidt
    for i in range(k, nvox):
        v = U_noise_adj[:, i].copy()

        # Orthogonalize against all previous columns (including the aligned ones)
        for j in range(i):
            v -= np.dot(v, U_noise_adj[:, j]) * U_noise_adj[:, j]

        norm = np.linalg.norm(v)
        if norm < 1e-12:
            # Choose a random orthogonal vector if degenerate
            attempts = 0
            while norm < 1e-12 and attempts < nvox:
                v = np.random.randn(nvox)
                # Orthogonalize against all previous columns
                for j in range(i):
                    v -= np.dot(v, U_noise_adj[:, j]) * U_noise_adj[:, j]
                norm = np.linalg.norm(v)
                attempts += 1

            if norm < 1e-12:
                # Last resort: use standard basis vector
                for basis_idx in range(nvox):
                    v = np.zeros(nvox)
                    v[basis_idx] = 1.0
                    # Orthogonalize against all previous columns
                    for j in range(i):
                        v -= np.dot(v, U_noise_adj[:, j]) * U_noise_adj[:, j]
                    norm = np.linalg.norm(v)
                    if norm > 1e-12:
                        break

        U_noise_adj[:, i] = v / norm

    return U_noise_adj
