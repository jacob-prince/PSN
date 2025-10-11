"""Refactored PSN implementation with improved efficiency and readability.

This module provides a cleaner, more efficient implementation of the PSN denoising algorithm
while maintaining exact numerical equivalence with the original implementation.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import FastICA

from .utils import (
    compute_noise_ceiling,
    make_orthonormal,
    negative_mse_columns,
    perform_gsn,
    r2_score_columns,
    split_half_reliability_3d,
)
from .visualization import plot_diagnostic_figures


# ========== HELPER FUNCTIONS ==========

def _validate_data(data):
    """Validate input data shape and values."""
    assert np.isfinite(data).all(), "Data contains infinite or NaN values."

    nunits, nconds, ntrials = data.shape

    if ntrials < 2:
        raise ValueError("Data must have at least 2 trials.")

    if nconds < 2:
        raise ValueError("Data must have at least 2 conditions to estimate covariance.")

    return nunits, nconds, ntrials


def _standardize_eigenvector_signs(evecs):
    """Standardize eigenvector signs by making the mean of each eigenvector positive."""
    standardized_evecs = evecs.copy()

    for i in range(evecs.shape[1]):
        if np.mean(evecs[:, i]) < 0:
            standardized_evecs[:, i] = -evecs[:, i]

    return standardized_evecs


def _compute_symmetric_eigen(matrix):
    """Compute eigendecomposition of a matrix, enforcing symmetry and sorting by magnitude."""
    # Force symmetry for numerical stability
    matrix_sym = (matrix + matrix.T) / 2

    # Eigendecomposition
    evals, evecs = np.linalg.eigh(matrix_sym)

    # Sort by absolute value of eigenvalues (descending)
    idx = np.argsort(np.abs(evals))[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Standardize eigenvector signs
    evecs = _standardize_eigenvector_signs(evecs)

    magnitudes = np.abs(evals)

    return evecs, magnitudes, matrix_sym


def _validate_and_normalize_basis(V, nunits):
    """Validate and normalize user-supplied basis vectors."""
    if V.shape[0] != nunits:
        raise ValueError(f"Basis must have {nunits} rows, got {V.shape[0]}")
    if V.shape[1] < 1:
        raise ValueError("Basis must have at least 1 column")

    # Check and fix unit length
    vector_norms = np.sqrt(np.sum(V**2, axis=0))
    if not np.allclose(vector_norms, 1, rtol=0, atol=1e-10):
        print('Normalizing basis vectors to unit length...')
        V = V / vector_norms

    # Check and fix orthogonality
    gram = V.T @ V
    if not np.allclose(gram, np.eye(gram.shape[0]), rtol=0, atol=1e-10):
        print('Adjusting basis vectors to ensure orthogonality...')
        V = make_orthonormal(V)

    return V


def _compute_basis_from_gsn(V, gsn_results, trial_avg_demeaned):
    """Compute basis vectors from GSN results based on V mode.
    
    Args:
        V: Basis mode selector
        gsn_results: Results from GSN algorithm
        trial_avg_demeaned: Trial-averaged, demeaned data (nunits, nconds)
    
    Returns:
        basis: Orthonormal basis matrix
        magnitudes: Eigenvalues/variances for each basis dimension
        basis_source: Source matrix used to compute basis (for visualization)
    """
    
    if V in [0, 1, 2]:
        cSb = gsn_results['cSb']
        cNb = gsn_results['cNb']

    if V == 0:
        # Signal covariance eigenvectors
        basis, magnitudes, basis_source = _compute_symmetric_eigen(cSb)

    elif V == 1:
        # Whitened signal covariance eigenvectors
        cNb_inv = np.linalg.pinv(cNb)
        transformed_cov = cNb_inv @ cSb
        basis, magnitudes, basis_source = _compute_symmetric_eigen(transformed_cov)

    elif V == 2:
        # Noise covariance eigenvectors
        basis, magnitudes, basis_source = _compute_symmetric_eigen(cNb)

    elif V == 3:
        # PCA eigenvectors
        cov_mat = np.cov(trial_avg_demeaned, ddof=1)
        basis, magnitudes, basis_source = _compute_symmetric_eigen(cov_mat)

    elif V == 4:
        # Random orthonormal basis
        np.random.seed(42)
        rand_mat = np.random.randn(trial_avg_demeaned.shape[0], trial_avg_demeaned.shape[0])
        basis, _ = np.linalg.qr(rand_mat)
        basis = basis[:, :trial_avg_demeaned.shape[0]]
        magnitudes = np.ones(trial_avg_demeaned.shape[0])
        basis_source = None

    elif V == 5:
        # ICA (Independent Component Analysis) basis
        # Store the unnormalized mixing matrix before QR decomposition
        # This will be retrieved and reranked later in the main psn() function
        nunits = trial_avg_demeaned.shape[0]
        nconds = trial_avg_demeaned.shape[1]
        
        n_components = min(nunits, nconds)
        # Use deterministic random initialization matching MATLAB
        # Generate initialization matrix with seed 42
        np.random.seed(42)
        w_init = np.random.randn(n_components, n_components)
        ica = FastICA(n_components=n_components, random_state=42, max_iter=1000, 
                      tol=1e-4, whiten='unit-variance', w_init=w_init)

        try:
            # Fit ICA: FastICA expects (n_samples, n_features)
            # data_for_ica is (nunits, nconds), so we transpose to (nconds, nunits)
            ica.fit(trial_avg_demeaned.T)

            # Get the mixing directions (columns of A in MATLAB notation)
            # In sklearn: mixing_ matrix has shape (n_features, n_components) = (nunits, n_components)
            # Each column is a mixing direction
            mixing_directions = ica.mixing_  # (nunits, n_components)
            
            # Make them unit length (normalize each column)
            mixing_directions_normalized = mixing_directions / np.linalg.norm(
                mixing_directions, axis=0, keepdims=True)
            
            # Apply QR decomposition to make orthonormal
            # NOTE: The original mixing_directions will be stored in basis_source for later use
            # Signal variance ranking will be done later in main psn() function
            # using compute_noise_ceiling on the raw trial data
            basis, _ = np.linalg.qr(mixing_directions_normalized)
            
            # Store the original (normalized) mixing matrix in basis_source
            # This will be used for reranking and visualization
            basis_source = mixing_directions_normalized.copy()
            
            # Add random basis vectors if necessary to complete the basis
            if basis.shape[1] < nunits:
                remaining_dims = nunits - basis.shape[1]
                np.random.seed(43)
                rand_mat = np.random.randn(nunits, remaining_dims)

                # Orthogonalize against existing basis using modified Gram-Schmidt
                for i in range(remaining_dims):
                    v = rand_mat[:, i]
                    # Project out existing basis vectors
                    for j in range(basis.shape[1]):
                        v = v - np.dot(v, basis[:, j]) * basis[:, j]
                    # Normalize
                    norm_v = np.linalg.norm(v)
                    if norm_v > 1e-10:  # Avoid division by zero
                        v = v / norm_v
                        rand_mat[:, i] = v
                    else:
                        # Generate a new random vector if current one is too small
                        v = np.random.randn(nunits)
                        for j in range(basis.shape[1]):
                            v = v - np.dot(v, basis[:, j]) * basis[:, j]
                        v = v / np.linalg.norm(v)
                        rand_mat[:, i] = v

                basis = np.hstack([basis, rand_mat])
            
            # Placeholder magnitudes (will be recomputed with noise ceiling in main function)
            projections_avg = trial_avg_demeaned.T @ basis  # (nconds, nunits)
            magnitudes = np.var(projections_avg, axis=0, ddof=1)

        except Exception as e:
            print(f"Warning: ICA failed ({e}), falling back to PCA")
            # Fallback to PCA if ICA fails
            cov_mat = np.cov(trial_avg_demeaned, ddof=1)
            basis, magnitudes, basis_source = _compute_symmetric_eigen(cov_mat)

    else:
        raise ValueError(f"Invalid V value: {V}")

    return basis, magnitudes, basis_source


def _rank_basis_dimensions(basis, basis_source, data, magnitudes, ranking='eigs'):
    """Rank basis dimensions by various criteria.
    
    This function ranks basis dimensions using different methods:
    - 'eigs': By eigenvalue magnitude (decreasing)
    - 'eig-inv': By eigenvalue magnitude (increasing)
    - 'signal': By signal variance (decreasing)
    - 'ncsnr': By noise-ceiling SNR (decreasing)
    - 'sig-noise': By difference between signal % and noise % (decreasing)
    
    Args:
        basis: Orthonormal basis (nunits, ndims)
        basis_source: Source matrix before orthonormalization (nunits, ncomponents), can be None
        data: Raw trial data (nunits, nconds, ntrials)
        magnitudes: Initial eigenvalues/magnitudes for each dimension
        ranking: Ranking method - 'eigs', 'eig-inv', 'signal', 'ncsnr', or 'sig-noise'
    
    Returns:
        basis_ranked: Reranked orthonormal basis
        magnitudes_ranked: Ranking values in appropriate order
        basis_source_ranked: Reordered source matrix (or None)
    """
    nunits, nconds, ntrials = data.shape
    ndims = basis.shape[1]
    
    if ranking == 'eigs':
        # Rank by eigenvalue magnitude (decreasing) - magnitudes already computed
        sort_idx = np.argsort(magnitudes)[::-1]
        magnitudes_ranked = magnitudes[sort_idx]
        
    elif ranking == 'eig-inv':
        # Rank by eigenvalue magnitude (increasing)
        sort_idx = np.argsort(magnitudes)
        magnitudes_ranked = magnitudes[sort_idx]
        
    elif ranking in ['signal', 'ncsnr', 'sig-noise']:
        # Need to compute signal and noise components for each dimension
        data_reshaped = data.transpose(1, 2, 0)  # (nconds, ntrials, nunits)
        signal_variances = np.zeros(ndims)
        noise_variances = np.zeros(ndims)
        ncsnrs = np.zeros(ndims)
        
        for i in range(ndims):
            # Get this basis vector
            this_basis_vec = basis[:, i]
            
            # Project data onto this direction: (nconds, ntrials, nunits) @ (nunits,) = (nconds, ntrials)
            proj_data = np.dot(data_reshaped, this_basis_vec)
            
            # Use compute_noise_ceiling to get signal variance, noise variance, and ncsnr
            # proj_data needs shape (1, nconds, ntrials) for compute_noise_ceiling
            _, ncsnr, sigvar, noisevar = compute_noise_ceiling(proj_data[np.newaxis, ...])
            
            signal_variances[i] = float(sigvar[0])
            noise_variances[i] = float(noisevar[0])
            ncsnrs[i] = float(ncsnr[0])
        
        if ranking == 'signal':
            # Rank by signal variance (decreasing)
            sort_idx = np.argsort(signal_variances)[::-1]
            magnitudes_ranked = signal_variances[sort_idx]
            
        elif ranking == 'ncsnr':
            # Rank by noise-ceiling SNR (decreasing)
            sort_idx = np.argsort(ncsnrs)[::-1]
            magnitudes_ranked = ncsnrs[sort_idx]
            
        elif ranking == 'sig-noise':
            # Rank by difference between signal % and noise %
            total_signal = np.sum(signal_variances)
            total_noise = np.sum(noise_variances)
            
            # Avoid division by zero
            if total_signal > 0:
                signal_pcts = 100 * signal_variances / total_signal
            else:
                signal_pcts = np.zeros(ndims)
                
            if total_noise > 0:
                noise_pcts = 100 * noise_variances / total_noise
            else:
                noise_pcts = np.zeros(ndims)
            
            sig_noise_diff = signal_pcts - noise_pcts
            sort_idx = np.argsort(sig_noise_diff)[::-1]
            magnitudes_ranked = sig_noise_diff[sort_idx]
    
    else:
        raise ValueError(f"Invalid ranking method: {ranking}. Must be one of: "
                        "'eigs', 'eig-inv', 'signal', 'ncsnr', 'sig-noise'")
    
    # Reorder basis and basis_source
    basis_ranked = basis[:, sort_idx]
    
    if basis_source is not None and basis_source.shape[1] == len(sort_idx):
        basis_source_ranked = basis_source[:, sort_idx]
    else:
        basis_source_ranked = None
    
    return basis_ranked, magnitudes_ranked, basis_source_ranked


def _compute_user_basis_magnitudes(basis, data):
    """Compute magnitudes for user-supplied basis based on variance."""
    trial_avg = np.mean(data, axis=2)  # (nunits, nconds)
    proj_data = trial_avg.T @ basis  # (nconds, basis_dim)
    magnitudes = np.var(proj_data, axis=0, ddof=1)
    return magnitudes


def _create_denoiser_matrix(basis, dims_to_keep, truncate=0):
    """Create denoising matrix from basis and dimension indices.

    Args:
        basis: Basis matrix (nunits, total_dims)
        dims_to_keep: Number of dimensions to keep or array of indices
        truncate: Number of early dimensions to skip

    Returns:
        Denoising matrix (nunits, nunits)
    """
    denoising_fn = np.zeros(basis.shape[1])

    if isinstance(dims_to_keep, (int, np.integer)):
        # Keep dims from truncate to truncate+dims_to_keep
        start_idx = truncate
        end_idx = min(start_idx + dims_to_keep, basis.shape[1])
        if end_idx > start_idx:
            denoising_fn[start_idx:end_idx] = 1
    else:
        # dims_to_keep is an array of indices
        denoising_fn[dims_to_keep] = 1

    return basis @ np.diag(denoising_fn) @ basis.T


def _apply_denoiser(data, denoiser, unit_means, denoisingtype):
    """Apply denoiser to data with proper demeaning and mean restoration.

    Args:
        data: Input data (nunits, nconds) or (nunits, nconds, ntrials)
        denoiser: Denoising matrix (nunits, nunits)
        unit_means: Mean values to subtract and add back (nunits,)
        denoisingtype: 0 for trial-averaged, 1 for single-trial

    Returns:
        Denoised data with same shape as input
    """
    if denoisingtype == 0:
        # Trial-averaged denoising
        trial_avg = np.mean(data, axis=2) if data.ndim == 3 else data
        trial_avg_demeaned = trial_avg - unit_means[:, np.newaxis]
        denoiseddata = (trial_avg_demeaned.T @ denoiser).T
        denoiseddata = denoiseddata + unit_means[:, np.newaxis]
    else:
        # Single-trial denoising
        denoiseddata = np.zeros_like(data)
        ntrials = data.shape[2]
        for t in range(ntrials):
            data_demeaned = data[:, :, t] - unit_means[:, np.newaxis]
            denoiseddata[:, :, t] = (data_demeaned.T @ denoiser).T
        denoiseddata = denoiseddata + unit_means[:, np.newaxis, np.newaxis]

    return denoiseddata


# ========== CROSS-VALIDATION ==========

def _perform_cross_validation(data, basis, opt, unit_means):
    """Perform cross-validation to determine optimal denoising dimensions."""
    np.random.seed(42)

    nunits, nconds, ntrials = data.shape
    cv_mode = opt['cv_mode']
    thresholds = opt['cv_thresholds']
    threshold_per = opt['cv_threshold_per']
    scoring_fn = opt['cv_scoring_fn']
    truncate = opt.get('truncate', 0)

    cv_scores = np.zeros((len(thresholds), ntrials, nunits))

    # Cross-validation loop
    for tr in range(ntrials):
        if cv_mode == 0:
            # Leave-one-out: train on n-1 trials, test on 1
            train_trials = np.setdiff1d(np.arange(ntrials), tr)
            train_avg = np.mean(data[:, :, train_trials], axis=2)
            test_data = data[:, :, tr]

            for tt, threshold in enumerate(thresholds):
                denoiser = _create_denoiser_matrix(basis, threshold, truncate)
                train_avg_demeaned = train_avg - unit_means[:, np.newaxis]
                train_denoised = (train_avg_demeaned.T @ denoiser).T
                cv_scores[tt, tr] = scoring_fn(test_data.T, train_denoised.T)

        elif cv_mode == 1:
            # Keep-one-in: train on 1 trial, test on n-1
            train_data = data[:, :, tr].T
            test_avg = np.mean(data[:, :, np.setdiff1d(np.arange(ntrials), tr)], axis=2).T

            for tt, threshold in enumerate(thresholds):
                denoiser = _create_denoiser_matrix(basis, threshold, truncate)
                train_demeaned = train_data - unit_means
                train_denoised = train_demeaned @ denoiser
                cv_scores[tt, tr] = scoring_fn(test_avg, train_denoised)

    # Determine best threshold(s)
    if threshold_per == 'population':
        # Single threshold for all units
        avg_scores = np.mean(cv_scores, axis=(1, 2))
        best_ix = np.argmax(avg_scores)
        best_threshold = int(thresholds[best_ix])

        # Create denoiser
        denoiser = _create_denoiser_matrix(basis, best_threshold, truncate)

        # Compute signal subspace and dimreduce
        safe_thr = min(best_threshold, basis.shape[1])
        start_idx = truncate
        end_idx = min(start_idx + safe_thr, basis.shape[1])
        signalsubspace = basis[:, start_idx:end_idx] if end_idx > start_idx else basis[:, :0]
        dimreduce = None  # Computed later if needed

    else:
        # Unit-wise thresholding with unit groups
        avg_scores = np.mean(cv_scores, axis=1)  # (len(thresholds), nunits)
        unit_groups = opt['unit_groups']
        unique_groups = np.unique(unit_groups)

        best_threshold = np.zeros(nunits, dtype=int)

        for group_id in unique_groups:
            group_mask = unit_groups == group_id
            group_avg_scores = np.mean(avg_scores[:, group_mask], axis=1)
            best_idx = np.argmax(group_avg_scores)
            best_threshold[group_mask] = thresholds[best_idx]

        # Create unit-wise denoiser
        denoiser = np.zeros((nunits, nunits))
        for unit_i in range(nunits):
            unit_denoiser = _create_denoiser_matrix(basis, int(best_threshold[unit_i]), truncate)
            denoiser[:, unit_i] = unit_denoiser[:, unit_i]

        signalsubspace = None
        dimreduce = None

    return denoiser, cv_scores, best_threshold, signalsubspace, dimreduce


# ========== MAGNITUDE THRESHOLDING ==========

def _perform_magnitude_thresholding(data, basis, opt, magnitudes, unit_means):
    """Select dimensions using magnitude thresholding.
    
    Note: The magnitudes are assumed to already be sorted according to the ranking method.
    For most ranking methods (eigs, signal, ncsnr, sig-noise), this is decreasing order.
    For eig-inv, this is increasing order. We use them as-is without re-sorting.
    """
    mag_frac = opt['mag_frac']
    truncate = opt.get('truncate', 0)

    # Use magnitudes as provided (already sorted by _rank_basis_dimensions)
    sorted_indices = np.arange(len(magnitudes))
    sorted_magnitudes = magnitudes

    # Select dimensions by cumulative variance
    total_variance = np.sum(sorted_magnitudes)
    cumulative_fraction = np.cumsum(sorted_magnitudes) / total_variance

    dims_needed = np.sum(cumulative_fraction < mag_frac) + 1
    dims_needed = min(dims_needed, len(sorted_magnitudes))

    # Filter out truncated dimensions
    selected_indices = sorted_indices[:dims_needed]
    filtered_indices = selected_indices[selected_indices >= truncate]

    # Add more dimensions if needed after truncation
    if len(filtered_indices) < dims_needed:
        remaining_indices = sorted_indices[dims_needed:]
        remaining_valid = remaining_indices[remaining_indices >= truncate]
        needed_additional = dims_needed - len(filtered_indices)
        additional_indices = remaining_valid[:needed_additional]
        filtered_indices = np.concatenate([filtered_indices, additional_indices])

    best_threshold_indices = filtered_indices
    dimsretained = len(best_threshold_indices)

    if dimsretained == 0:
        # No dimensions retained
        nunits = basis.shape[0]
        denoiser = np.zeros((nunits, nunits))
        signalsubspace = basis[:, :0]
        dimreduce = None
    else:
        # Create denoiser from selected dimensions
        denoiser = _create_denoiser_matrix(basis, best_threshold_indices, 0)
        signalsubspace = basis[:, best_threshold_indices]
        dimreduce = None  # Computed later if needed

    cv_scores = np.array([])

    return denoiser, cv_scores, best_threshold_indices, signalsubspace, dimreduce, sorted_magnitudes, dimsretained


# ========== MAIN PSN FUNCTION ==========

def psn(data, V=None, opt=None, wantfig=True):
    """
    Denoise neural data using Partitioning Signal and Noise (PSN).

    Algorithm Details:
    -----------------
    The PSN denoising algorithm works by identifying dimensions in the neural data that contain
    primarily signal rather than noise. It does this in several steps:

    1. Signal and Noise Estimation:
        - For each condition, computes mean response across trials (signal estimate)
        - For each condition, computes variance across trials (noise estimate)
        - Builds signal (cSb) and noise (cNb) covariance matrices across conditions

    2. Basis Selection (<V> parameter):
        - V=0: Uses eigenvectors of signal covariance (cSb)
        - V=1: Uses eigenvectors of signal covariance transformed by inverse noise covariance
        - V=2: Uses eigenvectors of noise covariance (cNb)
        - V=3: Uses PCA on trial-averaged data
        - V=4: Uses random orthonormal basis
        - V=matrix: Uses user-supplied orthonormal basis

    3. Dimension Selection:
        The algorithm must decide how many dimensions to keep. This can be done in two ways:

        a) Cross-validation (<cv_mode> >= 0):
            - Splits trials into training and testing sets
            - For training set:
                * Projects data onto different numbers of basis dimensions
                * Creates denoising matrix for each dimensionality
            - For test set:
                * Measures how well denoised training data predicts test data
                * Uses mean squared error (MSE) as prediction metric
            - Selects number of dimensions that gives best prediction
            - Can be done per-unit or for whole population

        b) Magnitude Thresholding (<cv_mode> = -1):
            - Computes "magnitude" for each dimension:
                * Either eigenvalues (signal strength)
                * Or variance explained in the data
            - Sets threshold as fraction of maximum magnitude
            - Keeps dimensions above threshold either:
                * Contiguously from strongest dimension
                * Or any dimension above threshold

    4. Denoising:
        - Creates denoising matrix using selected dimensions
        - For trial-averaged denoising:
            * Averages data across trials
            * Projects through denoising matrix
        - For single-trial denoising:
            * Projects each trial through denoising matrix
        - Returns denoised data and diagnostic information

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <data> - shape (nunits, nconds, ntrials). This indicates the measured
        responses to different conditions on distinct trials.
        The number of trials (ntrials) must be at least 2.
    <V> - shape (nunits, nunits) or scalar. Indicates the set of basis functions to use.
        0 means perform PSN and use the eigenvectors of the
          signal covariance estimate (cSb)
        1 means perform PSN and use the eigenvectors of the
          signal covariance estimate, transformed by the inverse of
          the noise covariance estimate (inv(cNb)*cSb)
        2 means perform PSN and use the eigenvectors of the
          noise covariance estimate (cNb)
        3 means naive PCA (i.e. eigenvectors of the covariance
          of the trial-averaged data)
        4 means use a randomly generated orthonormal basis (nunits, nunits)
        B means use user-supplied basis B. The dimensionality of B
          should be (nunits, D) where D >= 1. The columns of B should
          unit-length and pairwise orthogonal.
        Default: 0.
    <opt> - dict with the following optional fields:
        <cv_mode> - scalar. Indicates how to determine the optimal threshold:
          0 means cross-validation using n-1 (train) / 1 (test) splits of trials.
          1 means cross-validation using 1 (train) / n-1 (test) splits of trials.
         -1 means do not perform cross-validation and instead set the threshold
            based on when the magnitudes of components drop below
            a certain fraction (see <mag_frac>).
          Default: 0.
        <cv_threshold_per> - string. 'population' or 'unit', specifying
          whether to use unit-wise thresholding (possibly different thresholds
          for different units) or population thresholding (one threshold for
          all units). Matters only when <cv_mode> is 0 or 1. Default: 'unit'.
        <unit_groups> - shape (nunits,). Integer array specifying which units should
          receive the same cv threshold. This is only applicable when <cv_threshold_per>
          is 'unit'. Units with the same integer value get the same cv threshold
          (computed by averaging scores for those groups of units). If <cv_threshold_per>
          is 'population', all units should have unit_group = 0. Default: np.arange(nunits)
          (each unit gets its own threshold).
        <cv_thresholds> - shape (1, n_thresholds). Vector of thresholds to evaluate in
          cross-validation. Matters only when <cv_mode> is 0 or 1.
          Each threshold is a positive integer indicating a potential
          number of dimensions to retain. Should be in sorted order and
          elements should be unique. Default: 1:D where D is the
          maximum number of dimensions.
        <cv_scoring_fn> - function handle. For <cv_mode> 0 or 1 only.
          It is a function handle to compute denoiser performance.
          Default: negative_mse_columns.
        <mag_type> - scalar. Indicates how to obtain component magnitudes.
          Matters only when <cv_mode> is -1.
          0 means use signal variance computed from the data
          1 means use eigenvalues (<V> must be 0, 1, 2, or 3)
          Default: 0.
        <mag_frac> - scalar. Indicates the fraction of total variance to retain.
          Matters only when <cv_mode> is -1. The algorithm will sort dimensions
          by magnitude and select the top dimensions that cumulatively account
          for this fraction of the total variance.
          Default: 0.95.
        <denoisingtype> - scalar. Indicates denoising type:
          0 means denoising in the trial-averaged sense
          1 means single-trial-oriented denoising
          Note that if <cv_mode> is 0, you probably want <denoisingtype> to be 0,
          and if <cv_mode> is 1, you probably want <denoisingtype> to be 1, but
          the code is deliberately flexible for users to specify what they want.
          Default: 0.
        <truncate> - scalar. Number of early PCs to remove from the retained dimensions.
          If set to 1, the first PC will be excluded from denoising in addition to
          whatever later dimensions are deemed optimal to remove via cross validation.
          Default: 0.
    <wantfig> - bool. Whether to generate diagnostic figures showing the denoising results.
        Can also be specified in the opt dictionary (see above). If specified in both
        locations, the value in opt takes precedence.
        Default: True.

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    Returns a dictionary with the following fields:

    Return in all cases:
        <denoiser> - shape (nunits, nunits). This is the denoising matrix.
        <fullbasis> - shape (nunits, dims). This is the full set of basis functions.
        <gsn_result> - dict. Full results from the GSN algorithm (when V is 0, 1, or 2).
          Contains 'cSb' (signal covariance) and 'cNb' (noise covariance) matrices,
          plus any additional outputs from the GSN package. Will be None if V is 3, 4,
          or a custom basis matrix.

    In the case that <denoisingtype> is 0, we return:
        <denoiseddata> - shape (nunits, nconds). This is the trial-averaged data
          after applying the denoiser.

    In the case that <denoisingtype> is 1, we return:
        <denoiseddata> - shape (nunits, nconds, ntrials). This is the
          single-trial data after applying the denoiser.

    In the case that <cv_mode> is 0 or 1 (cross-validation):
        If <cv_threshold_per> is 'population', we return:
          <best_threshold> - shape (1, 1). The optimal threshold (a single integer),
            indicating how many dimensions are retained.
          <signalsubspace> - shape (nunits, best_threshold). This is the final set of basis
            functions selected for denoising (i.e. the subspace into which
            we project). The number of basis functions is equal to <best_threshold>.
          <dimreduce> - shape (best_threshold, nconds) or (best_threshold, nconds, ntrials). This
            is the trial-averaged data (or single-trial data) after denoising.
            Importantly, we do not reconstruct the original units but leave
            the data projected into the set of reduced dimensions.
        If <cv_threshold_per> is 'unit', we return:
          <best_threshold> - shape (1, nunits). The optimal threshold for each unit.
        In both cases ('population' or 'unit'), we return:
          <denoised_cv_scores> - shape (n_thresholds, ntrials, nunits).
            Cross-validation performance scores for each threshold.

    In the case that <cv_mode> is -1 (magnitude-based):
        <mags> - shape (1, dims). Component magnitudes used for thresholding.
        <dimsretained> - shape (1, n_retained). The indices of the dimensions retained.
        <signalsubspace> - shape (nunits, n_retained). This is the final set of basis
          functions selected for denoising (i.e. the subspace into which
          we project).
        <dimreduce> - shape (n_retained, nconds) or (n_retained, nconds, ntrials). This
          is the trial-averaged data (or single-trial data) after denoising.
          Importantly, we do not reconstruct the original units but leave
          the data projected into the set of reduced dimensions.

    -------------------------------------------------------------------------
    Examples:
    -------------------------------------------------------------------------

        # Basic usage with default options
        data = np.random.randn(100, 200, 3)  # 100 units, 200 conditions, 3 trials
        opt = {
            'cv_mode': 0,  # n-1 train / 1 test split
            'cv_threshold_per': 'unit',  # Unit-wise thresholding
            'cv_thresholds': np.arange(1,101),  # Test all possible dimensions
            'cv_scoring_fn': negative_mse_columns,  # Use negative MSE as scoring function (default)
            'denoisingtype': 1  # Single-trial denoising
        }
        results = psn(data, None, opt)

        # Using magnitude thresholding
        opt = {
            'cv_mode': -1,  # Use magnitude thresholding
            'mag_frac': 0.95,  # Keep components that account for 95% of variance
            'mag_type': 0  # Use signal variance
        }
        results = psn(data, 0, opt)

        # Unit-wise cross-validation
        opt = {
            'cv_mode': 0,  # Leave-one-out CV
            'cv_threshold_per': 'unit',  # Unit-specific thresholds
            'cv_thresholds': [1, 2, 3]  # Test these dimensions
        }
        results = psn(data, 0, opt)

        # Unit-wise cross-validation with custom unit groupings
        opt = {
            'cv_mode': 0,  # Leave-one-out CV
            'cv_threshold_per': 'unit',  # Unit-specific thresholds
            'unit_groups': np.array([0, 0, 1, 1, 2, 2] + [3]*94),  # First 4 units in 2 groups, next 2 in another, rest in one group
            'cv_thresholds': [1, 2, 3]  # Test these dimensions
        }
        results = psn(data, 0, opt)

        # Single-trial denoising with population threshold
        opt = {
            'denoisingtype': 1,  # Single-trial mode
            'cv_threshold_per': 'population'  # Same dims for all units
        }
        results = psn(data, 0, opt)
        denoised_trials = results['denoiseddata']  # [nunits x nconds x ntrials]

        # Custom basis
        nunits = data.shape[0]
        custom_basis = np.linalg.qr(np.random.randn(nunits, nunits))[0]
        results = psn(data, custom_basis)

    -------------------------------------------------------------------------
    History:
    -------------------------------------------------------------------------

        - 2025/01/06 - Initial version.
        - 2025/10/07 - Refactored for improved efficiency and readability.
    """

    # Validate input data
    nunits, nconds, ntrials = _validate_data(data)

    # Handle defaults
    if V is None:
        V = 0

    if opt is None:
        opt = {}

    # Validate cv_threshold_per early
    if 'cv_threshold_per' in opt:
        if opt['cv_threshold_per'] not in ['unit', 'population']:
            raise KeyError("cv_threshold_per must be 'unit' or 'population'")

    # Set defaults
    opt.setdefault('cv_scoring_fn', negative_mse_columns)
    opt.setdefault('cv_mode', 0)
    opt.setdefault('cv_threshold_per', 'unit')
    opt.setdefault('mag_type', 0)
    opt.setdefault('mag_frac', 0.95)
    opt.setdefault('denoisingtype', 0)
    opt.setdefault('truncate', 0)
    # ranking default will be set based on V value after basis is determined

    # Check if wantfig is in opt and override
    if 'wantfig' in opt:
        wantfig = opt['wantfig']

    # Set default unit_groups
    if 'unit_groups' not in opt:
        if opt['cv_threshold_per'] == 'population':
            opt['unit_groups'] = np.zeros(nunits, dtype=int)
        else:
            opt['unit_groups'] = np.arange(nunits, dtype=int)

    # Validate unit_groups
    unit_groups = np.array(opt['unit_groups'], dtype=int)
    if len(unit_groups) != nunits:
        raise ValueError(f"unit_groups must have length {nunits}, got {len(unit_groups)}")
    if not np.all(unit_groups >= 0):
        raise ValueError("unit_groups must contain only non-negative integers")
    if opt['cv_threshold_per'] == 'population' and not np.all(unit_groups == 0):
        raise ValueError("When cv_threshold_per='population', all unit_groups must be 0")

    opt['unit_groups'] = unit_groups

    # Compute unit means
    trial_avg = np.mean(data, axis=2)
    unit_means = np.mean(trial_avg, axis=1)

    # Determine basis
    gsn_result = None

    if isinstance(V, int):
        if V not in [0, 1, 2, 3, 4, 5]:
            raise ValueError("V must be in [0..5] (int) or a 2D numpy array.")

        # Compute GSN for modes 0-2
        if V in [0, 1, 2]:
            gsn_result = perform_gsn(data, {'wantverbose': False, 'random_seed': 42})

        # Demean trial average for PCA
        trial_avg_demeaned = (trial_avg.T - unit_means).T

        # Get basis
        basis, magnitudes, basis_source = _compute_basis_from_gsn(
            V, gsn_result or {}, trial_avg_demeaned)
        
        # Set default ranking based on V value and threshold type if not explicitly provided
        if 'ranking' not in opt:
            # Special case: V=2 (noise) or V=4 (random) with unit thresholding -> use sig-noise
            if V in [2, 4] and opt.get('cv_threshold_per', 'unit') == 'unit':
                opt['ranking'] = 'sig-noise'
            else:
                # Default for all other cases: use ncsnr (noise-ceiling SNR)
                # This has been empirically validated as the most robust across scenarios
                opt['ranking'] = 'ncsnr'
        
        # Rank basis dimensions according to the specified ranking method
        # basis_source contains the original matrix before orthonormalization (e.g., ICA mixing)
        ica_mixing = None
        basis, magnitudes, ranked_source = _rank_basis_dimensions(
            basis, basis_source, data, magnitudes, ranking=opt['ranking'])
        
        # For ICA, store the ranked mixing matrix separately
        if V == 5:
            ica_mixing = ranked_source
            # For ICA, we don't want basis_source in visualization, replace with None
            basis_source = None
        else:
            # For other basis types, update basis_source with ranked version
            basis_source = ranked_source

    elif isinstance(V, np.ndarray):
        # User-supplied basis
        original_V = V.copy()
        V = _validate_and_normalize_basis(V, nunits)
        basis = V.copy()
        magnitudes = _compute_user_basis_magnitudes(basis, data)
        basis_source = None
        ica_mixing = None
        
        # Set default ranking for user-supplied basis
        if 'ranking' not in opt:
            # Use ncsnr as default (most robust across scenarios)
            opt['ranking'] = 'ncsnr'
        
        # Rank user-supplied basis if ranking is specified
        if opt['ranking'] != 'eigs' or not np.allclose(magnitudes, np.sort(magnitudes)[::-1]):
            basis, magnitudes, _ = _rank_basis_dimensions(
                basis, None, data, magnitudes, ranking=opt['ranking'])

    else:
        raise ValueError("If V is not int, it must be a numpy array.")

    # Set default cv_thresholds
    if 'cv_thresholds' not in opt:
        opt['cv_thresholds'] = np.arange(0, basis.shape[1] + 1)
    else:
        thresholds = np.array(opt['cv_thresholds'])
        if not np.all(thresholds == thresholds.astype(int)):
            raise ValueError("cv_thresholds must be integers")
        if not np.all(np.diff(thresholds) > 0):
            raise ValueError("cv_thresholds must be in sorted order with unique values")

    # Perform dimension selection and denoising
    if opt['cv_mode'] >= 0:
        # Cross-validation
        denoiser, cv_scores, best_threshold, signalsubspace, dimreduce = _perform_cross_validation(
            data, basis, opt, unit_means
        )
        denoiseddata = _apply_denoiser(data, denoiser, unit_means, opt['denoisingtype'])

        # Compute dimreduce if population mode
        if opt['cv_threshold_per'] == 'population' and signalsubspace is not None:
            if opt['denoisingtype'] == 0:
                trial_avg_demeaned = (trial_avg.T - unit_means).T
                dimreduce = signalsubspace.T @ trial_avg_demeaned
            else:
                effective_dims = signalsubspace.shape[1]
                dimreduce = np.zeros((effective_dims, nconds, ntrials))
                for t in range(ntrials):
                    data_demeaned = (data[:, :, t].T - unit_means).T
                    dimreduce[:, :, t] = signalsubspace.T @ data_demeaned

        # dimsretained is not populated for CV mode in MATLAB - match this behavior
        dimsretained = np.array([])  # Empty array to match MATLAB

        results = {
            'denoiser': denoiser,
            'cv_scores': cv_scores,
            'best_threshold': best_threshold,
            'denoiseddata': denoiseddata,
            'fullbasis': basis,
            'signalsubspace': signalsubspace,
            'dimreduce': dimreduce,
            'dimsretained': dimsretained,
            'opt': opt,
            'V': V if isinstance(V, int) else 'custom',
            'gsn_result': gsn_result,
            'unit_means': unit_means,
            'mags': magnitudes,
            'basis_source': basis_source if isinstance(V, int) else None,
            'ica_mixing': ica_mixing,
        }

    else:
        # Magnitude thresholding
        denoiser, cv_scores, best_threshold, signalsubspace, dimreduce, mags, dimsretained = _perform_magnitude_thresholding(
            data, basis, opt, magnitudes, unit_means
        )
        denoiseddata = _apply_denoiser(data, denoiser, unit_means, opt['denoisingtype'])

        # Compute dimreduce
        if signalsubspace is not None and signalsubspace.shape[1] > 0:
            if opt['denoisingtype'] == 0:
                trial_avg_demeaned = trial_avg - unit_means[:, np.newaxis]
                dimreduce = signalsubspace.T @ trial_avg_demeaned
            else:
                dimreduce = np.zeros((len(best_threshold), nconds, ntrials))
                for t in range(ntrials):
                    data_demeaned = data[:, :, t] - unit_means[:, np.newaxis]
                    dimreduce[:, :, t] = signalsubspace.T @ data_demeaned

        results = {
            'denoiser': denoiser,
            'cv_scores': cv_scores,
            'best_threshold': best_threshold,
            'denoiseddata': denoiseddata,
            'fullbasis': basis,
            'mags': mags,
            'dimsretained': dimsretained,
            'signalsubspace': signalsubspace,
            'dimreduce': dimreduce,
            'opt': opt,
            'V': V if isinstance(V, int) else 'custom',
            'gsn_result': gsn_result,
            'unit_means': unit_means,
            'basis_source': basis_source if isinstance(V, int) else None,
            'ica_mixing': ica_mixing,
        }

    # Store input data for visualization
    results['input_data'] = data.copy()

    # Add visualization function handle
    def regenerate_visualization(test_data=None):
        """Regenerate the diagnostic visualization."""
        plot_diagnostic_figures(results['input_data'], results, test_data)

    results['plot'] = regenerate_visualization

    # Generate figures if requested
    if wantfig:
        plot_diagnostic_figures(data, results)

    return results


# ========== SKLEARN API ==========

class PSN(BaseEstimator, TransformerMixin):
    """
    Partitioning Signal and Noise (PSN) denoiser with sklearn-compatible interface.

    PSN denoises neural data by identifying dimensions that contain primarily signal
    rather than noise. This implementation provides both sklearn-compatible fit/transform
    methods and maintains backward compatibility with functional usage.

    Parameters
    ----------
    basis : str or ndarray, default='signal'
        The set of basis functions to use for denoising:
        - 'signal': GSN cSb (V=0) - eigenvectors of signal covariance
        - 'whitened-signal': GSN cNb * GSN cSb (V=1) - signal covariance transformed by inverse noise
        - 'noise': GSN cNb (V=2) - eigenvectors of noise covariance
        - 'pca': naive PCA (V=3) - eigenvectors of trial-averaged data covariance
        - 'random': random orthonormal basis (V=4) - not recommended
        - 'ica': ICA basis (V=5) - orthonormalized independent components from FastICA
        - ndarray: user-supplied orthonormal basis matrix, shape (nunits, dims)

    cv : str or None, default='unit'
        Cross-validation strategy for threshold selection:
        - 'unit': unit-wise thresholding, separate threshold chosen per unit
        - 'population': population thresholding, one threshold for all units
        - None: magnitude thresholding, retains dimensions for specified variance fraction

    scoring : str or callable, default='mse'
        Scoring function for cross-validation (when cv is 'unit' or 'population'):
        - 'mse': mean squared error (default)
        - 'split_half': split-half reliability correlation
        - 'r2': coefficient of determination (R²)
        - callable: custom scoring function with signature score(y_true, y_pred)

    mag_threshold : float, default=0.95
        Proportion of variance to retain when cv=None (magnitude thresholding mode)

    unit_groups : array-like or None, default=None
        Integer array of shape (nunits,) specifying which units should receive the same
        CV threshold. Only applicable when cv='unit'. Units with the same integer value
        get the same threshold. If None, each unit gets its own threshold (for 'unit' mode)
        or all units get the same threshold (for 'population' mode).

    truncate : int, default=0
        Number of early PCs to remove from the retained dimensions. If set to 1, the first
        PC will be excluded from denoising in addition to whatever later dimensions are
        deemed optimal to remove via cross validation or magnitude thresholding.

    verbose : bool, default=False
        Whether to print progress messages during fitting

    wantfig : bool, default=True
        Whether to generate diagnostic figures from fit() and transform() calls

    gsn_kwargs : dict or None, default=None
        Additional keyword arguments to pass to the GSN algorithm

    Attributes
    ----------
    denoiser_ : ndarray
        The fitted denoising matrix, shape (nunits, nunits)

    best_threshold_ : int or ndarray
        The optimal threshold(s) selected during fitting

    fullbasis_ : ndarray
        The complete basis used for denoising, shape (nunits, dims)

    signalsubspace_ : ndarray
        The final set of basis functions used for denoising

    unit_means_ : ndarray
        The mean response for each unit, shape (nunits,)

    cv_scores_ : ndarray
        Cross-validation scores for each threshold (if applicable)

    fitted_results_ : dict
        Complete results from the PSN algorithm

    Examples
    --------
    >>> import numpy as np
    >>> from psn import PSN
    >>>
    >>> # Generate sample data: 50 units, 100 conditions, 5 trials
    >>> data = np.random.randn(50, 100, 5)
    >>>
    >>> # Basic usage with default parameters
    >>> denoiser = PSN()
    >>> denoiser.fit(data)
    >>> denoised_data = denoiser.transform(data)
    >>>
    >>> # Population thresholding with PCA basis
    >>> denoiser = PSN(basis='pca', cv='population')
    >>> denoiser.fit(data)
    >>> denoised_data = denoiser.transform(data)
    >>>
    >>> # Magnitude thresholding retaining 90% variance
    >>> denoiser = PSN(basis='signal', cv=None, mag_threshold=0.90)
    >>> denoiser.fit(data)
    >>> denoised_data = denoiser.transform(data)
    >>>
    >>> # Custom basis matrix
    >>> custom_basis = np.linalg.qr(np.random.randn(50, 50))[0]
    >>> denoiser = PSN(basis=custom_basis, cv='unit')
    >>> denoiser.fit(data)
    >>> denoised_data = denoiser.transform(data)
    """

    def __init__(self, basis='signal', cv='unit', scoring='mse', mag_threshold=None,
                 unit_groups=None, truncate=0, verbose=False, wantfig=True, gsn_kwargs=None):
        self.basis = basis
        self.cv = cv
        self.scoring = scoring
        self.mag_threshold = mag_threshold
        self.unit_groups = unit_groups
        self.truncate = truncate
        self.verbose = verbose
        self.wantfig = wantfig
        self.gsn_kwargs = gsn_kwargs

    def _validate_params(self):
        """Validate input parameters."""
        valid_basis_strings = ['signal', 'whitened-signal', 'noise', 'pca', 'random', 'ica']
        if isinstance(self.basis, str):
            if self.basis not in valid_basis_strings:
                raise ValueError(f"basis must be one of {valid_basis_strings} or an ndarray")
        elif not isinstance(self.basis, np.ndarray):
            raise ValueError(f"basis must be one of {valid_basis_strings} or an ndarray")

        if self.cv not in ['unit', 'population', None]:
            raise ValueError("cv must be 'unit', 'population', or None")

        valid_scoring_strings = ['split_half', 'mse', 'r2']
        if isinstance(self.scoring, str):
            if self.scoring not in valid_scoring_strings:
                raise ValueError(f"scoring must be one of {valid_scoring_strings} or a callable")
        elif not callable(self.scoring):
            raise ValueError(f"scoring must be one of {valid_scoring_strings} or a callable")

        if self.mag_threshold is not None:
            if not isinstance(self.mag_threshold, (int, float)) or not 0 < self.mag_threshold <= 1:
                raise ValueError("mag_threshold must be a number between 0 and 1")

        if not isinstance(self.truncate, int) or self.truncate < 0:
            raise ValueError("truncate must be a non-negative integer")

    def _convert_params_to_functional(self, data):
        """Convert sklearn-style parameters to functional PSN parameters."""
        self._validate_params()

        # Convert basis parameter
        if isinstance(self.basis, str):
            basis_map = {
                'signal': 0,
                'whitened-signal': 1,
                'noise': 2,
                'pca': 3,
                'random': 4,
                'ica': 5
            }
            V = basis_map[self.basis]
        else:
            # numpy array
            V = self.basis

        # Convert cv and scoring parameters
        if self.cv is None:
            # cv=None means magnitude thresholding
            cv_mode = -1
            cv_threshold_per = 'population'
        else:
            # cv='unit' or cv='population' means cross-validation
            cv_mode = 0
            cv_threshold_per = self.cv

        # Convert scoring function
        scoring_map = {
            'mse': negative_mse_columns,
            'r2': r2_score_columns,
        }
        cv_scoring_fn = scoring_map.get(self.scoring, self.scoring)

        # Build options dictionary
        opt = {
            'cv_mode': cv_mode,
            'cv_threshold_per': cv_threshold_per,
            'cv_scoring_fn': cv_scoring_fn,
            'mag_frac': self.mag_threshold if self.mag_threshold is not None else 0.95,
            'denoisingtype': 0,
            'truncate': self.truncate,
        }

        if self.unit_groups is not None:
            opt['unit_groups'] = np.asarray(self.unit_groups, dtype=int)

        if self.gsn_kwargs is not None:
            opt.update(self.gsn_kwargs)

        return V, opt

    def fit(self, X, y=None):
        """
        Fit the PSN denoiser to the data.

        Parameters
        ----------
        X : ndarray, shape (nunits, nconds, ntrials)
            Neural response data where:
            - nunits: number of recording units/neurons
            - nconds: number of experimental conditions
            - ntrials: number of trials per condition

        y : Ignored
            Not used, present for sklearn compatibility

        Returns
        -------
        self : object
            Returns the instance itself
        """
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("Input data must be 3-dimensional (nunits, nconds, ntrials)")
        if X.shape[2] < 2:
            raise ValueError("Data must have at least 2 trials")
        if X.shape[1] < 2:
            raise ValueError("Data must have at least 2 conditions")

        V, opt = self._convert_params_to_functional(X)

        if self.verbose:
            print("Fitting PSN denoiser...")

        results = psn(X, V=V, opt=opt, wantfig=self.wantfig)

        self.denoiser_ = results['denoiser']
        self.best_threshold_ = results['best_threshold']
        self.fullbasis_ = results['fullbasis']
        self.signalsubspace_ = results.get('signalsubspace')
        self.unit_means_ = results['unit_means']
        self.cv_scores_ = results.get('cv_scores')
        self.fitted_results_ = results

        if self.verbose:
            print("PSN fitting completed")
            if hasattr(self, 'best_threshold_'):
                if np.isscalar(self.best_threshold_):
                    print(f"Selected {self.best_threshold_} dimensions")
                else:
                    print(f"Selected dimensions per unit: min={np.min(self.best_threshold_)}, "
                          f"max={np.max(self.best_threshold_)}, mean={np.mean(self.best_threshold_):.1f}")

        return self

    def transform(self, X):
        """
        Apply the fitted PSN denoiser to data.

        Parameters
        ----------
        X : ndarray, shape (nunits, nconds, ntrials) or (nunits, nconds)
            Neural response data to denoise. Can be:
            - 3D array: (nunits, nconds, ntrials) for single-trial denoising
            - 2D array: (nunits, nconds) for trial-averaged denoising

        Returns
        -------
        X_denoised : ndarray, same shape as X
            Denoised neural response data
        """
        check_is_fitted(self, 'denoiser_')

        X = np.asarray(X)

        if X.ndim == 2:
            nunits, nconds = X.shape
            if nunits != self.denoiser_.shape[0]:
                raise ValueError(f"Number of units ({nunits}) doesn't match fitted denoiser "
                               f"({self.denoiser_.shape[0]})")

            X_demeaned = X - self.unit_means_[:, np.newaxis]
            X_denoised = (X_demeaned.T @ self.denoiser_).T
            X_denoised = X_denoised + self.unit_means_[:, np.newaxis]

        elif X.ndim == 3:
            nunits, nconds, ntrials = X.shape
            if nunits != self.denoiser_.shape[0]:
                raise ValueError(f"Number of units ({nunits}) doesn't match fitted denoiser "
                               f"({self.denoiser_.shape[0]})")

            X_denoised = np.zeros_like(X)
            for t in range(ntrials):
                X_trial_demeaned = X[:, :, t] - self.unit_means_[:, np.newaxis]
                X_denoised[:, :, t] = (X_trial_demeaned.T @ self.denoiser_).T
            X_denoised = X_denoised + self.unit_means_[:, np.newaxis, np.newaxis]

        else:
            raise ValueError("Input data must be 2D (nunits, nconds) or 3D (nunits, nconds, ntrials)")

        return X_denoised

    def fit_transform(self, X, y=None):
        """
        Fit the denoiser and transform the data in one step.

        Parameters
        ----------
        X : ndarray, shape (nunits, nconds, ntrials)
            Neural response data to fit and transform

        y : Ignored
            Not used, present for sklearn compatibility

        Returns
        -------
        X_denoised : ndarray, shape (nunits, nconds)
            Denoised trial-averaged neural response data
        """
        self.fit(X, y)
        trial_avg = np.mean(X, axis=2)
        return self.transform(trial_avg)

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names. If None, generic names are used.

        Returns
        -------
        feature_names_out : ndarray of shape (n_features_out,), dtype=str
            Output feature names
        """
        check_is_fitted(self, 'denoiser_')

        n_features_out = self.denoiser_.shape[0]

        if input_features is None:
            return np.array([f"unit_{i}" for i in range(n_features_out)])
        else:
            input_features = np.asarray(input_features, dtype=str)
            if len(input_features) != n_features_out:
                raise ValueError(f"input_features has {len(input_features)} elements, "
                               f"expected {n_features_out}")
            return input_features.copy()

    def score(self, X, y=None):
        """
        Return the mean split-half reliability score on the given test data.

        This computes split-half reliability for the denoised data as a measure
        of denoising quality. Higher values indicate better reliability preservation.

        Parameters
        ----------
        X : ndarray, shape (nunits, nconds, ntrials)
            Test data

        y : Ignored
            Not used, present for sklearn compatibility

        Returns
        -------
        score : float
            Mean split-half reliability across all units
        """
        check_is_fitted(self, 'denoiser_')

        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("Input data must be 3-dimensional for scoring")

        X_denoised = self.transform(X)
        denoised_reliability = split_half_reliability_3d(X_denoised)

        return np.mean(denoised_reliability)

    def plot_diagnostics(self, test_data=None):
        """
        Generate diagnostic plots for the fitted denoiser.

        Parameters
        ----------
        test_data : ndarray, optional
            Test data for diagnostic plots, shape (nunits, nconds, ntrials).
            If None, uses leave-one-out cross-validation on training data.
        """
        check_is_fitted(self, 'fitted_results_')

        if hasattr(self.fitted_results_, 'plot'):
            self.fitted_results_['plot'](test_data)
        else:
            plot_diagnostic_figures(self.fitted_results_['input_data'],
                                  self.fitted_results_, test_data)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values
        """
        return {
            'basis': self.basis,
            'cv': self.cv,
            'scoring': self.scoring,
            'mag_threshold': self.mag_threshold,
            'unit_groups': self.unit_groups,
            'truncate': self.truncate,
            'verbose': self.verbose,
            'wantfig': self.wantfig,
            'gsn_kwargs': self.gsn_kwargs
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters

        Returns
        -------
        self : estimator instance
            Estimator instance
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self
