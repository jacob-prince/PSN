"""
Functions for generating simulated neural data with controlled signal and noise properties.

This module provides tools to generate synthetic neural data with specific covariance
structures for both signal and noise components. The data generation process allows for:
- Control over signal and noise decay rates
- Alignment between signal and noise principal components
- Separate train and test datasets with matched properties
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .adjust_alignment_gradient_descent import adjust_alignment_gradient_descent as _adjust_alignment_gradient_descent
from ..plotting.plot_data_diagnostic import plot_data_diagnostic


def _random_orthonormal_columns(rng: np.random.RandomState, nvox: int, r: int) -> np.ndarray:
    """Return an (nvox, r) matrix with orthonormal columns."""
    if r <= 0:
        raise ValueError("r must be positive")
    A = rng.randn(nvox, r)
    Q, _ = np.linalg.qr(A, mode="reduced")
    return Q


def _align_noise_basis_lowrank(
    U_signal: np.ndarray,
    U_noise_init: np.ndarray,
    alpha: float,
    k: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Low-rank alignment of top-k noise PCs to signal PCs.

    Produces an orthonormal (nvox, rN) matrix whose first k columns satisfy
    dot(U_noise[:,i], U_signal[:,i]) == alpha (approximately, but very close).

    This is a scalable alternative to _adjust_alignment_gradient_descent, which
    requires full (nvox, nvox) matrices.
    """
    if k <= 0:
        return U_noise_init

    if not (0.0 <= alpha <= 1.0):
        warnings.warn("align_alpha must be in [0,1]; will be clamped.")
        alpha = float(max(0.0, min(1.0, alpha)))

    nvox = U_signal.shape[0]
    rS = U_signal.shape[1]
    rN = U_noise_init.shape[1]
    k_eff = int(min(k, rS, rN, nvox // 2))
    if k_eff <= 0:
        return U_noise_init

    Usk = U_signal[:, :k_eff]

    # Construct V: k_eff orthonormal vectors orthogonal to span(Usk)
    A = rng.randn(nvox, k_eff)
    A = A - Usk @ (Usk.T @ A)
    V, _ = np.linalg.qr(A, mode="reduced")

    # Aligned block; columns remain orthonormal because Usk ⟂ V
    aligned = alpha * Usk + np.sqrt(max(0.0, 1.0 - alpha**2)) * V

    # Fill remaining noise directions from the initial noise basis, projected to the complement
    B = np.concatenate([Usk, V], axis=1)  # (nvox, 2k)
    rest = U_noise_init[:, k_eff:]
    if rest.size == 0:
        return aligned

    rest = rest - B @ (B.T @ rest)
    rest, _ = np.linalg.qr(rest, mode="reduced")

    U_noise = np.concatenate([aligned, rest], axis=1)
    return U_noise


def generate_data(nvox=50, ncond=200, ntrial=5, signal_decay=2.0, noise_decay=1.25,
                 noise_multiplier=3.0, align_alpha=0.5, align_k=10, random_seed=42,
                 want_fig=False, signal_cov=None, true_signal=None, noise_cov=None, cluster_units=False, verbose=True,
                 *, fast: bool = False, rank_signal: int = 50, rank_noise: int = 200,
                 isotropic_noise: float = 0.0, return_cov: bool | None = None,
                 max_nvox_for_cov: int = 2000):
    """
    Generate synthetic neural data with controlled signal and noise properties.
    
    Args:
        nvox (int, optional):    Number of voxels/units. If true_signal is provided, this will be inferred from true_signal.shape[1]
        ncond (int, optional):   Number of conditions. If true_signal is provided, this will be inferred from true_signal.shape[0]
        ntrial (int):  Number of trials per condition
        signal_decay (float): Rate of eigenvalue decay for signal covariance
        noise_decay (float):  Rate of eigenvalue decay for noise covariance
        noise_multiplier (float): Scaling factor for noise variance
        align_alpha (float): Alignment between signal & noise PCs (1=aligned, 0=orthogonal)
        align_k (int): Number of top PCs to align
        random_seed (int, optional): Random seed for reproducibility
        want_fig (bool): Whether to display a diagnostic figure of the generated data
        signal_cov (ndarray, optional): User-provided signal covariance matrix (nvox, nvox)
                                        If provided, overrides signal_decay parameter
        true_signal (ndarray, optional): User-provided ground truth signal (ncond, nvox) or (height, width, channels)
                                        If provided, overrides signal_cov and signal_decay.
                                        For 3D arrays (like images), spatial dimensions are flattened: nvox = width * channels
                                        Note: When provided, signal_cov will be calculated
                                        as the sample covariance of this signal.
        noise_cov (ndarray, optional): User-provided noise covariance matrix (nvox, nvox)
                                       If provided, overrides noise_decay parameter
        cluster_units (bool): Whether to reorder units based on hierarchical clustering
                            of the signal covariance matrix. This is purely cosmetic
                            for visualization and does not affect data properties.
    
    Returns:
        (train_data, test_data, ground_truth)
         - train_data: (nvox, ncond, ntrial)
         - test_data:  (nvox, ncond, ntrial)
         - ground_truth: dict w/ keys:
             'signal'     -> (ncond, nvox)
             'signal_cov' -> (nvox, nvox)
             'noise_cov'  -> (nvox, nvox)
             'U_signal'   -> Original eigenvectors for signal
             'U_noise'    -> Original eigenvectors for noise
             'signal_eigs'
             'noise_eigs'
             'unit_order' -> Original indices of units after clustering (if cluster_units=True)

                Fast mode notes
                ---------------
                If fast=True, this function avoids constructing full (nvox, nvox) covariance matrices
                and avoids full (nvox, nvox) orthonormal bases. It instead uses low-rank factors with
                power-law spectral decay and samples via those factors.

                In fast mode:
                - ground_truth['U_signal'] and ['U_noise'] are (nvox, rank_*) not (nvox, nvox)
                - ground_truth['signal_cov'] and ['noise_cov'] may be None unless return_cov=True
                    and nvox <= max_nvox_for_cov
    """
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random.RandomState()

    # Fast path: scalable low-rank spectral-decay simulation
    if fast:
        if want_fig:
            raise ValueError("want_fig=True is not supported in fast mode (requires full covariances)")
        if signal_cov is not None or noise_cov is not None:
            raise ValueError("signal_cov/noise_cov are not supported in fast mode (would be huge)")
        if cluster_units:
            raise ValueError("cluster_units is not supported in fast mode (requires full covariances)")

        if nvox is None or ncond is None or ntrial is None:
            raise ValueError("nvox, ncond, and ntrial must be provided")
        if ntrial < 1:
            raise ValueError("ntrial must be >= 1")

        rS = int(min(rank_signal, nvox))
        rN = int(min(rank_noise, nvox))

        # Signal basis and spectrum
        if true_signal is None:
            U_signal = _random_orthonormal_columns(rng, nvox=nvox, r=rS)
            signal_eigs = 1.0 / (np.arange(1, rS + 1) ** signal_decay)
            zS = rng.randn(rS, ncond)
            true_signal_full = (U_signal * np.sqrt(signal_eigs)) @ zS  # (nvox, ncond)
            true_signal_full = true_signal_full.T  # (ncond, nvox)
        else:
            if len(true_signal.shape) != 2:
                raise ValueError(f"true_signal must be 2D (ncond, nvox), got {true_signal.shape}")
            if true_signal.shape != (ncond, nvox):
                raise ValueError(f"Provided true_signal has shape {true_signal.shape}, expected ({ncond}, {nvox})")
            true_signal_full = np.asarray(true_signal, dtype=float)

            # Approximate covariance eigen-structure via randomized SVD of the (ncond, nvox) signal matrix.
            try:
                from sklearn.utils.extmath import randomized_svd

                Uc, Sc, Vt = randomized_svd(true_signal_full, n_components=rS, random_state=random_seed)
                # Covariance eigenvalues: (Sc^2)/(ncond-1); eigenvectors are columns of V
                U_signal = Vt.T
                signal_eigs = (Sc**2) / max(1, (ncond - 1))
            except Exception:
                # Fallback: exact SVD on (ncond, nvox) with full_matrices=False
                Uc, Sc, Vt = np.linalg.svd(true_signal_full, full_matrices=False)
                U_signal = Vt[:rS, :].T
                signal_eigs = (Sc[:rS] ** 2) / max(1, (ncond - 1))

        # Noise basis and spectrum (+ optional alignment)
        U_noise_init = _random_orthonormal_columns(rng, nvox=nvox, r=rN)
        if align_k and align_k > 0:
            U_noise = _align_noise_basis_lowrank(U_signal, U_noise_init, alpha=align_alpha, k=int(align_k), rng=rng)
        else:
            U_noise = U_noise_init
        noise_eigs = noise_multiplier / (np.arange(1, rN + 1) ** noise_decay)

        # Generate train/test data: same signal, independent noise
        train_data = np.zeros((ntrial, nvox, ncond), dtype=float)
        test_data = np.zeros((ntrial, nvox, ncond), dtype=float)

        for t in range(ntrial):
            zN_train = rng.randn(rN, ncond)
            zN_test = rng.randn(rN, ncond)

            train_noise = (U_noise * np.sqrt(noise_eigs)) @ zN_train  # (nvox, ncond)
            test_noise = (U_noise * np.sqrt(noise_eigs)) @ zN_test

            if isotropic_noise and isotropic_noise > 0:
                train_noise = train_noise + isotropic_noise * rng.randn(nvox, ncond)
                test_noise = test_noise + isotropic_noise * rng.randn(nvox, ncond)

            train_data[t, :, :] = (true_signal_full.T + train_noise)
            test_data[t, :, :] = (true_signal_full.T + test_noise)

        train_data = train_data.transpose(1, 2, 0)
        test_data = test_data.transpose(1, 2, 0)

        # Optionally materialize full covariances for small problems
        if return_cov is None:
            return_cov_eff = nvox <= max_nvox_for_cov
        else:
            return_cov_eff = bool(return_cov)

        if return_cov_eff and nvox > max_nvox_for_cov:
            raise ValueError(
                f"return_cov=True would build ({nvox}x{nvox}) matrices; set return_cov=False "
                f"or increase max_nvox_for_cov (currently {max_nvox_for_cov})."
            )

        signal_cov_full = None
        noise_cov_full = None
        if return_cov_eff:
            signal_cov_full = U_signal @ np.diag(signal_eigs) @ U_signal.T
            noise_cov_full = U_noise @ np.diag(noise_eigs) @ U_noise.T
            if isotropic_noise and isotropic_noise > 0:
                noise_cov_full = noise_cov_full + (isotropic_noise**2) * np.eye(nvox)

        ground_truth = {
            'signal':      true_signal_full,
            'signal_cov':  signal_cov_full,
            'noise_cov':   noise_cov_full,
            'U_signal':    U_signal,
            'U_noise':     U_noise,
            'signal_eigs': signal_eigs,
            'noise_eigs':  noise_eigs,
            'cov_factors': {
                'signal': {'U': U_signal, 'eigs': signal_eigs},
                'noise':  {'U': U_noise, 'eigs': noise_eigs, 'isotropic_noise': float(isotropic_noise)},
            },
            'user_provided': {
                'signal_cov': False,
                'true_signal': true_signal is not None,
                'noise_cov': False,
            },
            'fast': True,
        }

        return train_data, test_data, ground_truth

    # Infer nvox and ncond from true_signal if provided and not explicitly set
    if true_signal is not None:
        if len(true_signal.shape) != 2:
            raise ValueError(f"true_signal must be a 2D array with shape (ncond, nvox), got shape {true_signal.shape}")
            
        true_signal_ncond, true_signal_nvox = true_signal.shape
        
        if nvox is None:
            nvox = true_signal_nvox
        elif nvox != true_signal_nvox:
            raise ValueError(f"Provided nvox={nvox} doesn't match true_signal shape[1]={true_signal_nvox}")
            
        if ncond is None:
            ncond = true_signal_ncond
        elif ncond != true_signal_ncond:
            raise ValueError(f"Provided ncond={ncond} doesn't match true_signal shape[0]={true_signal_ncond}")
    
    # Check that required parameters are now available
    if nvox is None:
        raise ValueError("nvox must be provided either explicitly or via true_signal")
    if ncond is None:
        raise ValueError("ncond must be provided either explicitly or via true_signal")
    if ntrial is None:
        raise ValueError("ntrial must be provided")

    # Track what was user-provided before we potentially modify these variables
    user_provided_signal_cov = signal_cov is not None
    user_provided_true_signal = true_signal is not None
    user_provided_noise_cov = noise_cov is not None

    # Check input dimensions if provided
    if signal_cov is not None:
        if signal_cov.shape != (nvox, nvox):
            raise ValueError(f"Provided signal_cov has shape {signal_cov.shape}, expected ({nvox}, {nvox})")

    if true_signal is not None:
        if true_signal.shape != (ncond, nvox):
            raise ValueError(f"Provided true_signal has shape {true_signal.shape}, expected ({ncond}, {nvox})")

    if noise_cov is not None:
        if noise_cov.shape != (nvox, nvox):
            raise ValueError(f"Provided noise_cov has shape {noise_cov.shape}, expected ({nvox}, {nvox})")

    # Generate random orthonormal matrices for signal & noise
    U_noise, _, _ = np.linalg.svd(rng.randn(nvox, nvox), full_matrices=True)

    # For signal, either use SVD of provided covariance or generate random
    if signal_cov is not None:
        # Use provided signal covariance
        U_signal, signal_eigs, _ = np.linalg.svd(signal_cov)
        signal_cov = np.copy(signal_cov)  # Ensure we have a copy to avoid modifying the input
    else:
        # Generate random orthonormal matrix for signal
        U_signal, _, _ = np.linalg.svd(rng.randn(nvox, nvox), full_matrices=True)
        # Create diagonal eigenvalues
        signal_eigs = 1.0 / (np.arange(1, nvox+1) ** signal_decay)
        # Build signal covariance matrix
        signal_cov = U_signal @ np.diag(signal_eigs) @ U_signal.T

    # For noise, either use SVD of provided covariance or generate random
    if user_provided_noise_cov:
        # Use provided noise covariance
        U_noise, noise_eigs, _ = np.linalg.svd(noise_cov)
        noise_cov = np.copy(noise_cov)  # Ensure we have a copy to avoid modifying the input

        # Warn if alignment was requested but noise_cov is provided
        if align_k > 0 and verbose:
            print("Warning: align_k > 0 but noise_cov was provided. Using provided noise covariance without alignment.")
    else:
        # Generate noise covariance after potential alignment
        # Align noise PCs to signal PCs if requested
        if align_k > 0:
            # Cap align_k to not exceed available dimensions
            effective_k = min(align_k, nvox)
            U_noise = _adjust_alignment_gradient_descent(
                U_signal, U_noise, align_alpha, effective_k, verbose=verbose
            )

        # Create diagonal eigenvalues for noise
        noise_eigs = noise_multiplier / (np.arange(1, nvox+1) ** noise_decay)
        # Build noise covariance matrix
        noise_cov = U_noise @ np.diag(noise_eigs) @ U_noise.T

    # Generate the ground truth signal
    if true_signal is not None:
        # Use provided ground truth signal
        true_signal = np.copy(true_signal)  # Ensure we have a copy

        # Recalculate signal covariance based on the provided true signal
        # This ensures signal_cov matches the actual covariance of true_signal
        signal_cov = np.cov(true_signal, rowvar=False)

        # Recompute signal eigendecomposition for consistency
        U_signal, signal_eigs, _ = np.linalg.svd(signal_cov)

        # Re-align noise after recalculating U_signal (only if noise_cov was not user-provided)
        if align_k > 0 and not user_provided_noise_cov:
            U_noise = _adjust_alignment_gradient_descent(
                U_signal, U_noise, align_alpha, align_k, verbose=verbose
            )
            # Rebuild noise covariance matrix with the realigned eigenvectors
            noise_cov = U_noise @ np.diag(noise_eigs) @ U_noise.T
        elif align_k > 0 and user_provided_noise_cov:
            if verbose:
                print("Warning: align_k > 0 but noise_cov was provided. Skipping noise alignment.")
    else:
        # Generate from covariance
        true_signal = rng.multivariate_normal(
            mean=np.zeros(nvox),
            cov=signal_cov,
            size=ncond
        )  # shape (ncond, nvox)

    # Preallocate train/test data in shape (ntrial, nvox, ncond)
    train_data = np.zeros((ntrial, nvox, ncond))
    test_data = np.zeros((ntrial, nvox, ncond))

    # Generate data
    for t in range(ntrial):
        # Independent noise for each trial
        train_noise = rng.multivariate_normal(
            mean=np.zeros(nvox),
            cov=noise_cov,
            size=ncond
        )  # shape (ncond, nvox)
        test_noise = rng.multivariate_normal(
            mean=np.zeros(nvox),
            cov=noise_cov,
            size=ncond
        )   # shape (ncond, nvox)

        # Add noise to signal
        train_data[t, :, :] = (true_signal + train_noise).T
        test_data[t, :, :]  = (true_signal + test_noise).T

    # Reshape to (nvox, ncond, ntrial)
    train_data = train_data.transpose(1, 2, 0)
    test_data  = test_data.transpose(1, 2, 0)

    # Optionally reorder units based on hierarchical clustering
    unit_order = None
    if cluster_units:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist
        from scipy.stats import zscore

        # Cluster based on ground truth signal patterns
        # true_signal shape is (ncond, nvox), so transpose to get (nvox, ncond)
        # Standardize each unit's activity pattern for better clustering
        signal_for_clustering = zscore(true_signal, axis=0).T  # zscore across conditions for each unit

        # Use correlation distance and average linkage for more balanced clusters
        dist = pdist(signal_for_clustering, metric='correlation')
        Z = linkage(dist, method='average')  # Average linkage often gives more balanced clusters
        unit_order = leaves_list(Z)  # Get the reordered indices

        # Reorder all relevant matrices and arrays
        train_data = train_data[unit_order]
        test_data = test_data[unit_order]
        true_signal = true_signal[:, unit_order]
        signal_cov = signal_cov[unit_order][:, unit_order]
        noise_cov = noise_cov[unit_order][:, unit_order]
        U_signal = U_signal[unit_order]
        U_noise = U_noise[unit_order]

    ground_truth = {
        'signal':      true_signal,
        'signal_cov':  signal_cov,
        'noise_cov':   noise_cov,
        'U_signal':    U_signal,
        'U_noise':     U_noise,
        'signal_eigs': signal_eigs,
        'noise_eigs':  noise_eigs,
        'user_provided': {
            'signal_cov': user_provided_signal_cov,
            'true_signal': user_provided_true_signal,
            'noise_cov': user_provided_noise_cov
        }
    }

    if unit_order is not None:
        ground_truth['unit_order'] = unit_order

    if want_fig:
        fig = plot_data_diagnostic(train_data, ground_truth, {
            'nvox': nvox,
            'ncond': ncond,
            'ntrial': ntrial,
            'signal_decay': signal_decay,
            'noise_decay': noise_decay,
            'noise_multiplier': noise_multiplier,
            'align_alpha': align_alpha,
            'align_k': align_k,
            'random_seed': random_seed,
            'user_provided': ground_truth['user_provided'],
            'clustered': cluster_units
        })
        ground_truth['diagnostic_fig'] = fig

    return train_data, test_data, ground_truth


def generate_heterogeneous_populations(
    n_populations=3,
    units_per_pop=20,
    ncond=100,
    ntrial=3,
    signal_decay=2.0,
    noise_decay=1.25,
    noise_multiplier=3.0,
    population_orthogonality=0.9,
    random_seed=42,
    want_fig=False,
    verbose=True
):
    """
    Generate data with heterogeneous subpopulations that have conflicting preferences.
    
    This creates a challenging scenario where different groups of units have different
    optimal basis orderings, making global/population-based approaches suboptimal.
    
    Args:
        n_populations (int): Number of distinct subpopulations
        units_per_pop (int): Number of units per subpopulation
        ncond (int): Number of conditions
        ntrial (int): Number of trials per condition
        signal_decay (float): Rate of eigenvalue decay for signal covariance
        noise_decay (float): Rate of eigenvalue decay for noise covariance  
        noise_multiplier (float): Scaling factor for noise variance
        population_orthogonality (float): How different the populations are (0=identical, 1=orthogonal)
                                         Controls the angle between population-specific signal subspaces
        random_seed (int): Random seed for reproducibility
        want_fig (bool): Whether to display diagnostic figures
        verbose (bool): Whether to print diagnostic information
    
    Returns:
        (train_data, test_data, ground_truth)
         - train_data: (nvox, ncond, ntrial)
         - test_data: (nvox, ncond, ntrial)
         - ground_truth: dict with keys:
             'signal' -> (ncond, nvox) - ground truth signal
             'signal_cov' -> (nvox, nvox) - overall signal covariance
             'noise_cov' -> (nvox, nvox) - overall noise covariance
             'population_labels' -> (nvox,) - which population each unit belongs to
             'population_bases' -> list of (nvox, nvox) - optimal basis for each population
             'population_signals' -> list of (ncond, units_per_pop) - signal for each population
    
    Example:
        >>> # Create 3 populations with 15 units each, where each population has different preferences
        >>> train, test, gt = generate_heterogeneous_populations(
        ...     n_populations=3, 
        ...     units_per_pop=15,
        ...     population_orthogonality=0.8,
        ...     want_fig=True
        ... )
    """
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random.RandomState()
    
    nvox = n_populations * units_per_pop
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"GENERATING HETEROGENEOUS POPULATION DATA")
        print(f"{'='*80}")
        print(f"  Populations: {n_populations}")
        print(f"  Units per population: {units_per_pop}")
        print(f"  Total units: {nvox}")
        print(f"  Conditions: {ncond}")
        print(f"  Trials: {ntrial}")
        print(f"  Population orthogonality: {population_orthogonality:.2f}")
        print(f"{'='*80}\n")
    
    # Create population-specific signal subspaces
    # Each population will have a different optimal basis ordering
    population_bases = []
    population_signals = []
    population_labels = np.zeros(nvox, dtype=int)
    
    # Generate a base random orthonormal matrix
    U_base, _, _ = np.linalg.svd(rng.randn(units_per_pop, units_per_pop), full_matrices=True)
    
    for pop_idx in range(n_populations):
        # Create population-specific basis by rotating from base
        if pop_idx == 0:
            # First population uses the base
            U_pop = U_base.copy()
        else:
            # Subsequent populations are rotated versions
            # Use Givens rotations to create controlled orthogonality
            U_pop = U_base.copy()
            
            # Apply rotation based on population_orthogonality
            # Higher orthogonality = more rotation = more different preferences
            n_rotations = max(1, int(units_per_pop * population_orthogonality))
            
            for _ in range(n_rotations):
                # Random Givens rotation
                i, j = rng.choice(units_per_pop, size=2, replace=False)
                theta = rng.uniform(0, np.pi * population_orthogonality)
                
                # Apply rotation in plane (i, j)
                c, s = np.cos(theta), np.sin(theta)
                G = np.eye(units_per_pop)
                G[i, i] = c
                G[i, j] = -s
                G[j, i] = s
                G[j, j] = c
                U_pop = G @ U_pop
        
        population_bases.append(U_pop)
        
        # Generate population-specific signal with decaying eigenvalues
        signal_eigs_pop = 1.0 / (np.arange(1, units_per_pop + 1) ** signal_decay)
        signal_cov_pop = U_pop @ np.diag(signal_eigs_pop) @ U_pop.T
        
        # Generate signal for this population
        signal_pop = rng.multivariate_normal(
            mean=np.zeros(units_per_pop),
            cov=signal_cov_pop,
            size=ncond
        )  # (ncond, units_per_pop)
        
        population_signals.append(signal_pop)
        
        # Track which units belong to which population
        start_idx = pop_idx * units_per_pop
        end_idx = start_idx + units_per_pop
        population_labels[start_idx:end_idx] = pop_idx
    
    # Assemble full signal matrix
    true_signal = np.hstack(population_signals)  # (ncond, nvox)
    
    # Compute overall signal covariance (will be block-diagonal-ish)
    signal_cov = np.cov(true_signal, rowvar=False)
    
    # Generate noise covariance (could be global or population-specific)
    # For simplicity, use a global noise structure
    U_noise, _, _ = np.linalg.svd(rng.randn(nvox, nvox), full_matrices=True)
    noise_eigs = noise_multiplier / (np.arange(1, nvox + 1) ** noise_decay)
    noise_cov = U_noise @ np.diag(noise_eigs) @ U_noise.T
    
    # Preallocate train/test data
    train_data = np.zeros((ntrial, nvox, ncond))
    test_data = np.zeros((ntrial, nvox, ncond))
    
    # Generate noisy data
    for t in range(ntrial):
        train_noise = rng.multivariate_normal(
            mean=np.zeros(nvox),
            cov=noise_cov,
            size=ncond
        )  # (ncond, nvox)
        
        test_noise = rng.multivariate_normal(
            mean=np.zeros(nvox),
            cov=noise_cov,
            size=ncond
        )  # (ncond, nvox)
        
        train_data[t, :, :] = (true_signal + train_noise).T
        test_data[t, :, :] = (true_signal + test_noise).T
    
    # Reshape to (nvox, ncond, ntrial)
    train_data = train_data.transpose(1, 2, 0)
    test_data = test_data.transpose(1, 2, 0)
    
    # Create ground truth dictionary
    ground_truth = {
        'signal': true_signal,
        'signal_cov': signal_cov,
        'noise_cov': noise_cov,
        'population_labels': population_labels,
        'population_bases': population_bases,
        'population_signals': population_signals,
        'n_populations': n_populations,
        'units_per_pop': units_per_pop,
        'population_orthogonality': population_orthogonality
    }
    
    if want_fig:
        _visualize_heterogeneous_populations(
            train_data, true_signal, ground_truth, 
            signal_cov, noise_cov, ntrial
        )
    
    return train_data, test_data, ground_truth


def _visualize_heterogeneous_populations(train_data, true_signal, ground_truth, 
                                         signal_cov, noise_cov, ntrial):
    """Visualize heterogeneous population data structure."""
    
    population_labels = ground_truth['population_labels']
    n_populations = ground_truth['n_populations']
    units_per_pop = ground_truth['units_per_pop']
    nvox = len(population_labels)
    ncond = true_signal.shape[0]
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Population colors
    colors = plt.cm.tab10(np.linspace(0, 1, n_populations))
    
    # Plot 1: Ground truth signal with population boundaries
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(true_signal.T, aspect='auto', cmap='RdBu_r', interpolation='none')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('Ground Truth Signal\n(with population structure)')
    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Unit')
    
    # Add population boundaries
    for pop_idx in range(1, n_populations):
        ax1.axhline(pop_idx * units_per_pop - 0.5, color='yellow', linewidth=3, linestyle='--')
    
    # Plot 2: Signal covariance (should show block structure)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(signal_cov, aspect='equal', cmap='RdBu_r', interpolation='none')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('Signal Covariance\n(block structure)')
    ax2.set_xlabel('Unit')
    ax2.set_ylabel('Unit')
    
    # Add population boundaries
    for pop_idx in range(1, n_populations):
        ax2.axhline(pop_idx * units_per_pop - 0.5, color='yellow', linewidth=2)
        ax2.axvline(pop_idx * units_per_pop - 0.5, color='yellow', linewidth=2)
    
    # Plot 3: Noise covariance
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(noise_cov, aspect='equal', cmap='RdBu_r', interpolation='none')
    plt.colorbar(im3, ax=ax3)
    ax3.set_title('Noise Covariance\n(global structure)')
    ax3.set_xlabel('Unit')
    ax3.set_ylabel('Unit')
    
    # Plot 4: Population labels
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.barh(np.arange(nvox), np.ones(nvox), color=[colors[pop] for pop in population_labels])
    ax4.set_yticks(np.arange(0, nvox, max(1, nvox // 10)))
    ax4.set_xlabel('Population')
    ax4.set_ylabel('Unit')
    ax4.set_title(f'Population Labels\n({n_populations} populations)')
    ax4.set_xlim([0, 1.5])
    
    # Plot 5-7: Per-population signal patterns
    for pop_idx in range(min(3, n_populations)):
        ax = fig.add_subplot(gs[1, pop_idx])
        pop_start = pop_idx * units_per_pop
        pop_end = pop_start + units_per_pop
        pop_signal = true_signal[:, pop_start:pop_end].T
        
        im = ax.imshow(pop_signal, aspect='auto', cmap='RdBu_r', interpolation='none')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Population {pop_idx + 1} Signal\n({units_per_pop} units)', color=colors[pop_idx])
        ax.set_xlabel('Condition')
        ax.set_ylabel('Unit (within pop)')
    
    # Plot 8: Trial-averaged data
    ax8 = fig.add_subplot(gs[1, 3])
    trial_avg = np.mean(train_data, axis=2)
    im8 = ax8.imshow(trial_avg, aspect='auto', cmap='RdBu_r', interpolation='none')
    plt.colorbar(im8, ax=ax8)
    ax8.set_title(f'Trial-Averaged Data\n({ntrial} trials)')
    ax8.set_xlabel('Condition')
    ax8.set_ylabel('Unit')
    
    # Add population boundaries
    for pop_idx in range(1, n_populations):
        ax8.axhline(pop_idx * units_per_pop - 0.5, color='yellow', linewidth=2, linestyle='--')
    
    # Plot 9: Cross-population basis alignment
    ax9 = fig.add_subplot(gs[2, 0:2])
    if n_populations >= 2:
        # Show alignment between first two populations
        U1 = ground_truth['population_bases'][0]
        U2 = ground_truth['population_bases'][1]
        alignment = np.abs(U1.T @ U2)
        
        im9 = ax9.imshow(alignment, aspect='equal', cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im9, ax=ax9)
        ax9.set_title('Cross-Population Basis Alignment\n(Pop 1 vs Pop 2)')
        ax9.set_xlabel('Population 2 basis dimension')
        ax9.set_ylabel('Population 1 basis dimension')
    
    # Plot 10: Signal variance per population
    ax10 = fig.add_subplot(gs[2, 2])
    pop_variances = []
    for pop_idx in range(n_populations):
        pop_start = pop_idx * units_per_pop
        pop_end = pop_start + units_per_pop
        pop_var = np.var(true_signal[:, pop_start:pop_end])
        pop_variances.append(pop_var)
    
    ax10.bar(range(n_populations), pop_variances, color=colors[:n_populations])
    ax10.set_xlabel('Population')
    ax10.set_ylabel('Signal Variance')
    ax10.set_title('Signal Variance per Population')
    ax10.set_xticks(range(n_populations))
    
    # Plot 11: Explanation text
    ax11 = fig.add_subplot(gs[2, 3])
    ax11.axis('off')
    explanation = (
        f"HETEROGENEOUS POPULATIONS\n\n"
        f"• {n_populations} distinct subpopulations\n"
        f"• {units_per_pop} units per population\n"
        f"• Orthogonality: {ground_truth['population_orthogonality']:.2f}\n\n"
        f"Each population has different\n"
        f"optimal basis orderings.\n\n"
        f"Global approaches will be\n"
        f"suboptimal because they\n"
        f"average across conflicting\n"
        f"preferences."
    )
    ax11.text(0.1, 0.5, explanation, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Heterogeneous Population Data Structure', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig


