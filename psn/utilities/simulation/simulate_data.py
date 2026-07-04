"""
Functions for generating simulated neural data with controlled signal and noise properties.

This module provides tools to generate synthetic neural data with specific covariance
structures for both signal and noise components. The data generation process allows for:
- Control over signal and noise decay rates
- Alignment between signal and noise principal components
- Separate train and test datasets with matched properties
"""

import numpy as np

from ..plotting.plot_data_diagnostic import plot_data_diagnostic
from .adjust_alignment_gradient_descent import (
    adjust_alignment_gradient_descent as _adjust_alignment_gradient_descent,
)
from .helpers import (
    _align_noise_basis_lowrank,
    _coerce_true_signal,
    _random_orthonormal_columns,
    _visualize_heterogeneous_populations,
)


def generate_data(nvox=50, ncond=200, ntrial=5, signal_decay=2.0, noise_decay=1.25,
                 noise_multiplier=3.0, align_alpha=0.5, align_k=10, random_seed=42,
                 want_fig=False, signal_cov=None, true_signal=None, noise_cov=None, cluster_units=False, verbose=True,
                 *, fast: bool = False, rank_signal: int = 50, rank_noise: int = 200,
                 isotropic_noise: float = 0.0, return_cov=None,
                 max_nvox_for_cov: int = 2000):
    """Generate synthetic neural data with controlled signal and noise structure.

    Builds a ground-truth signal (from a power-law spectrum, a supplied
    covariance, or an image) and adds trial noise with its own spectrum and a
    tunable alignment to the signal, returning matched train and test datasets.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <nvox> - int. Number of units/voxels. Inferred from true_signal when given.
        Default: 50.

    <ncond> - int. Number of conditions. Inferred from true_signal when given.
        Default: 200.

    <ntrial> - int. Number of trials per condition. Default: 5.

    <signal_decay> - float. Power-law decay of the signal eigenvalues
        (eig_i ~ 1/i^signal_decay). Ignored if signal_cov/true_signal given.
        Default: 2.0.

    <noise_decay> - float. Power-law decay of the noise eigenvalues. Ignored if
        noise_cov given. Default: 1.25.

    <noise_multiplier> - float. Overall noise scaling (higher = noisier).
        Default: 3.0.

    <align_alpha> - float in [0,1]. Alignment of the top noise PCs to the signal
        PCs (1 = aligned, 0 = orthogonal). Ignored if noise_cov given.
        Default: 0.5.

    <align_k> - int. Number of top PCs to align; 0 disables alignment.
        Default: 10.

    <random_seed> - int or None. RNG seed for reproducibility (None = fresh
        state). Default: 42.

    <want_fig> - bool. Render a diagnostic figure of the generated data. Not
        supported in fast mode. Default: False.

    <signal_cov> - [nvox x nvox] or None. Supplied signal covariance; overrides
        signal_decay. Default: None.

    <true_signal> - array, str, Path, or None. Supplied ground-truth signal;
        overrides signal_cov and signal_decay. May be a 2D (ncond, nvox) array,
        an image-shaped (H,W)/(H,W,C) array, or an image filepath (loaded and
        normalized to [0,1]; a bare name like 'pliny' resolves in the bundled
        images/ dir). For image-derived inputs the spatial dims are flattened to
        2D and the SMALLER dim is used as nvox, to keep the O(nvox^3) work cheap.
        Default: None.

    <noise_cov> - [nvox x nvox] or None. Supplied noise covariance; overrides
        noise_decay and disables alignment. Default: None.

    <cluster_units> - bool. Reorder units by hierarchical clustering of the
        signal covariance (cosmetic; original order kept in
        ground_truth['unit_order']). Default: False.

    <verbose> - bool. Print progress during generation. Default: True.

    Keyword-only (fast / large-N path):

    <fast> - bool. Use the scalable low-rank simulation: no full (nvox, nvox)
        covariances or bases; samples from low-rank power-law factors instead.
        Incompatible with want_fig and signal_cov/noise_cov. Default: False.

    <rank_signal>, <rank_noise> - int. Ranks of the low-rank signal/noise factors
        in fast mode. Defaults: 50, 200.

    <isotropic_noise> - float. Fast mode only: std of an added isotropic (white)
        noise floor (adds isotropic_noise^2 * I to the noise covariance).
        Default: 0.0.

    <return_cov> - bool or None. Fast mode: materialize ground_truth signal_cov/
        noise_cov when True and nvox <= max_nvox_for_cov. Default: None.

    <max_nvox_for_cov> - int. Upper nvox bound for materializing covariances in
        fast mode. Default: 2000.

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <train_data> - [nvox x ncond x ntrial]. Training set (independent noise).

    <test_data> - [nvox x ncond x ntrial]. Test set (matched signal, independent
        noise).

    <ground_truth> - dict with keys:
        'signal'     -> [ncond x nvox] ground-truth signal
        'signal_cov' -> [nvox x nvox] signal covariance (None in fast mode unless
                        return_cov and nvox <= max_nvox_for_cov)
        'noise_cov'  -> [nvox x nvox] noise covariance (same caveat)
        'U_signal'   -> signal eigenvectors ([nvox x nvox], or [nvox x rank_signal]
                        in fast mode)
        'U_noise'    -> noise eigenvectors (likewise [nvox x rank_noise] in fast)
        'signal_eigs', 'noise_eigs' -> eigenvalue spectra
        'unit_order' -> original unit indices after clustering (if cluster_units)
    """
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random.RandomState()

    # Normalize true_signal up front so every downstream path (fast and
    # non-fast) sees a 2D (ncond, nvox) array. This is where a filepath to
    # an image gets loaded and image-shaped arrays get flattened. The signal
    # is authoritative for the data dimensions, so derive ncond/nvox from it
    # (overriding the defaults) - this is what makes `generate_data(
    # true_signal='picture.jpg')` work without restating its shape.
    if true_signal is not None:
        true_signal = _coerce_true_signal(true_signal)
        ncond, nvox = true_signal.shape

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

                _, Sc, Vt = randomized_svd(true_signal_full, n_components=rS, random_state=random_seed)
                # Covariance eigenvalues: (Sc^2)/(ncond-1); eigenvectors are columns of V
                U_signal = Vt.T
                signal_eigs = (Sc**2) / max(1, (ncond - 1))
            except Exception:
                # Fallback: exact SVD on (ncond, nvox) with full_matrices=False
                _, Sc, Vt = np.linalg.svd(true_signal_full, full_matrices=False)
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
        print("GENERATING HETEROGENEOUS POPULATION DATA")
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
