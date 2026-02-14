"""Gradient descent alignment utility for basis matrices."""

import numpy as np

from .shortcut_alignment import shortcut_alignment


def adjust_alignment_gradient_descent(U_signal, U_noise_init, alpha, k,
                                       lr=5e-1, lambda_orth=1.0,
                                       num_steps=10000,
                                       tol_align=1e-6, tol_orth=1e-6,
                                       verbose=True):
    """Gradient descent method to align noise basis to signal basis.

    Aligns the top-k columns of U_noise to U_signal such that
    dot(U_noise[:, i], U_signal[:, i]) ≈ alpha.

    Parameters
    ----------
    U_signal : ndarray, shape (nvox, nvox)
        Orthonormal signal basis.
    U_noise_init : ndarray, shape (nvox, nvox)
        Initial orthonormal noise basis.
    alpha : float
        Target alignment strength in [0, 1].
        0 = orthogonal, 1 = perfectly aligned.
    k : int
        Number of top PCs to align.
    lr : float, optional
        Learning rate. Default: 0.5.
    lambda_orth : float, optional
        Orthogonality constraint weight. Default: 1.0.
    num_steps : int, optional
        Maximum optimization steps. Default: 10000.
    tol_align : float, optional
        Alignment error tolerance. Default: 1e-6.
    tol_orth : float, optional
        Orthogonality error tolerance. Default: 1e-6.
    verbose : bool, optional
        Whether to print convergence info. Default: True.

    Returns
    -------
    U_noise_aligned : ndarray, shape (nvox, nvox)
        New orthonormal basis with first k columns aligned to U_signal
        with strength alpha.
    """
    # Handle edge cases
    if k == 0:
        return U_noise_init.copy()

    # Use shortcut alignment for perfect alignment (alpha = 1.0)
    # This avoids convergence issues and ensures exact orthonormality
    if np.isclose(alpha, 1.0, atol=1e-10):
        return shortcut_alignment(U_signal, U_noise_init, k)

    U = U_noise_init.copy()
    nvox = U.shape[0]
    I = np.eye(nvox)

    for step in range(1, num_steps + 1):
        grad = np.zeros_like(U)
        align_vals = []
        for i in range(k):
            dot = np.dot(U[:, i], U_signal[:, i])
            align_vals.append(dot)
            grad[:, i] = (dot - alpha) * U_signal[:, i]
        M = U.T @ U - I
        grad += lambda_orth * ((U @ M))
        U -= lr * grad

        max_align_err = max(abs(a - alpha) for a in align_vals) if align_vals else 0
        orth_err = np.linalg.norm(U.T @ U - I)

        if max_align_err < tol_align and orth_err < tol_orth:
            if verbose:
                print(f"\t\tOptimization complete. Step {step}/{num_steps}: align_err={max_align_err:.2e}, orth_err={orth_err:.2e}")
            # Ensure perfect orthonormality before returning
            U = _qr_preserve_alignment(U, U_signal, k)
            return U

    if verbose:
        print(f"Optimization did not converge. Step {step}/{num_steps}: align_err={max_align_err:.2e}, orth_err={orth_err:.2e}")

    # Ensure orthonormality even if didn't fully converge
    U = _qr_preserve_alignment(U, U_signal, k)
    return U


def _qr_preserve_alignment(U, U_signal, k):
    """QR orthonormalization that preserves alignment signs for the first k columns."""
    Q, _ = np.linalg.qr(U)
    # QR can flip column signs — fix by ensuring dot(Q[:,i], U_signal[:,i])
    # has the same sign as dot(U[:,i], U_signal[:,i]) for the aligned columns
    for i in range(k):
        if np.dot(Q[:, i], U_signal[:, i]) < 0:
            Q[:, i] = -Q[:, i]
    return Q
