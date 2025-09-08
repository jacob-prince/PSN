import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from .utils import compute_noise_ceiling, make_orthonormal, negative_mse_columns, perform_gsn, r2_score_columns
from .visualization import plot_diagnostic_figures

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
    <wantfig> - bool. Whether to generate diagnostic figures showing the denoising results.
        Default: True.

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    Returns a dictionary with the following fields:

    Return in all cases:
        <denoiser> - shape (nunits, nunits). This is the denoising matrix.
        <fullbasis> - shape (nunits, dims). This is the full set of basis functions.

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
            'cv_threshold_per': 'unit',  # Same threshold for all units
            'cv_thresholds': np.arange(100),  # Test all possible dimensions
            'cv_scoring_fn': negative_mse_columns,  # Use negative MSE as scoring function
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
    """

    # 1) Check for infinite or NaN data => some tests want an AssertionError.
    assert np.isfinite(data).all(), "Data contains infinite or NaN values."

    nunits, nconds, ntrials = data.shape

    # 2) If we have fewer than 2 trials, raise an error
    if ntrials < 2:
        raise ValueError("Data must have at least 2 trials.")

    # 2b) Check for minimum number of conditions
    assert nconds >= 2, "Data must have at least 2 conditions to estimate covariance."

    # 3) If V is None => treat it as 0
    if V is None:
        V = 0

    # 4) Prepare default opts
    if opt is None:
        opt = {}
        
    # Validate cv_threshold_per before setting defaults
    if 'cv_threshold_per' in opt:
        if opt['cv_threshold_per'] not in ['unit', 'population']:
            raise KeyError("cv_threshold_per must be 'unit' or 'population'")

    # Initialize return dictionary with None values
    results = {
        'denoiser': None,
        'cv_scores': None,
        'best_threshold': None,
        'denoiseddata': None,
        'fullbasis': None,
        'signalsubspace': None,
        'dimreduce': None,
        'dimsretained': None,
        'opt': opt, 
        'V': V 
    }

    # Check if basis vectors are unit length and normalize if not
    if isinstance(V, np.ndarray):
        # First check and fix unit length
        vector_norms = np.sqrt(np.sum(V**2, axis=0))
        if not np.allclose(vector_norms, 1, rtol=0, atol=1e-10):
            print('Normalizing basis vectors to unit length...')
            V = V / vector_norms

        # Then check orthogonality
        gram = V.T @ V
        if not np.allclose(gram, np.eye(gram.shape[0]), rtol=0, atol=1e-10):
            print('Adjusting basis vectors to ensure orthogonality...')
            V = make_orthonormal(V)

    # Now set defaults
    opt.setdefault('cv_scoring_fn', negative_mse_columns)
    opt.setdefault('cv_mode', 0)
    opt.setdefault('cv_threshold_per', 'unit')
    
    opt.setdefault('mag_type', 0)
    opt.setdefault('mag_frac', 0.95)
    opt.setdefault('denoisingtype', 0)  # Default to trial-averaged denoising
    
    # Set default unit_groups based on cv_threshold_per
    if 'unit_groups' not in opt:
        if opt['cv_threshold_per'] == 'population':
            opt['unit_groups'] = np.zeros(nunits, dtype=int)  # All units in group 0
        else:  # 'unit'
            opt['unit_groups'] = np.arange(nunits, dtype=int)  # Each unit gets its own group
    
    # Validate unit_groups
    unit_groups = np.array(opt['unit_groups'], dtype=int)
    if len(unit_groups) != nunits:
        raise ValueError(f"unit_groups must have length {nunits}, got {len(unit_groups)}")
    if not np.all(unit_groups >= 0):
        raise ValueError("unit_groups must contain only non-negative integers")
    if opt['cv_threshold_per'] == 'population' and not np.all(unit_groups == 0):
        raise ValueError("When cv_threshold_per='population', all unit_groups must be 0")
    
    # Store validated unit_groups back in opt
    opt['unit_groups'] = unit_groups
    
    # compute the unit means since they are removed during denoising and will be added back
    trial_avg = np.mean(data, axis=2)
    results['unit_means'] = np.mean(trial_avg, axis=1)

    # 5) If V is an integer => glean basis from PSN results
    if isinstance(V, int):
        if V not in [0, 1, 2, 3, 4]:
            raise ValueError("V must be in [0..4] (int) or a 2D numpy array.")
            
        gsn_results = perform_gsn(data, {'wantverbose': False, 'random_seed': 42})

        cSb = gsn_results['cSb']
        cNb = gsn_results['cNb']

        # Helper for pseudo-inversion (in case cNb is singular)
        def inv_or_pinv(mat):
            return np.linalg.pinv(mat)
        
        def standardize_eigenvector_signs(evecs):
            """Standardize eigenvector signs by making the mean of each eigenvector positive."""
            standardized_evecs = evecs.copy()
            
            # For each eigenvector, flip sign if mean is negative
            for i in range(evecs.shape[1]):
                if np.mean(evecs[:, i]) < 0:
                    standardized_evecs[:, i] = -evecs[:, i]
            
            return standardized_evecs

        if V == 0:
            # Just eigen-decompose cSb
            # Force symmetry for consistency with MATLAB
            cSb_sym = (cSb + cSb.T) / 2
            evals, evecs = np.linalg.eigh(cSb_sym)
            # Sort by absolute value of eigenvalues
            idx = np.argsort(np.abs(evals))[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            # Standardize eigenvector signs
            evecs = standardize_eigenvector_signs(evecs)
            basis = evecs
            magnitudes = np.abs(evals)  # No need to flip, already sorted
            results['basis_source'] = cSb_sym
        elif V == 1:
            cNb_inv = inv_or_pinv(cNb)
            transformed_cov = cNb_inv @ cSb
            # enforce symmetry of transformed_cov
            transformed_cov = (transformed_cov + transformed_cov.T) / 2
            evals, evecs = np.linalg.eigh(transformed_cov)
            # Sort by absolute value of eigenvalues
            idx = np.argsort(np.abs(evals))[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            # Standardize eigenvector signs
            evecs = standardize_eigenvector_signs(evecs)
            basis = evecs
            magnitudes = np.abs(evals)  # No need to flip, already sorted
            results['basis_source'] = transformed_cov
        elif V == 2:
            # Force symmetry for consistency with MATLAB
            cNb_sym = (cNb + cNb.T) / 2
            evals, evecs = np.linalg.eigh(cNb_sym)
            # Sort by absolute value of eigenvalues
            idx = np.argsort(np.abs(evals))[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            # Standardize eigenvector signs
            evecs = standardize_eigenvector_signs(evecs)
            basis = evecs
            magnitudes = np.abs(evals)  # No need to flip, already sorted
            results['basis_source'] = cNb_sym
        elif V == 3:
            # de-mean each row of trial_avg
            trial_avg = (trial_avg.T - results['unit_means']).T
            cov_mat = np.cov(trial_avg, ddof=1)
            # Force symmetry for consistency with MATLAB
            cov_mat_sym = (cov_mat + cov_mat.T) / 2
            evals, evecs = np.linalg.eigh(cov_mat_sym)
            idx = np.argsort(np.abs(evals))[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            # Standardize eigenvector signs
            evecs = standardize_eigenvector_signs(evecs)
            basis = evecs
            magnitudes = np.abs(evals)
            results['basis_source'] = cov_mat_sym
        else:  # V == 4
            # Generate a random basis with same dimensions as eigenvector basis
            # Set random seed for reproducibility with MATLAB
            np.random.seed(42)
            rand_mat = np.random.randn(nunits, nunits)  # Start with square matrix
            basis, _ = np.linalg.qr(rand_mat)
            # Only keep first nunits columns to match eigenvector basis dimensions
            basis = basis[:, :nunits]
            magnitudes = np.ones(nunits)  # No meaningful magnitudes for random basis
            results['basis_source'] = None  # No meaningful source matrix for random basis        
    else:
        # If V not int => must be a numpy array
        if not isinstance(V, np.ndarray):
            raise ValueError("If V is not int, it must be a numpy array.")
        
        # Check orthonormality of user-supplied basis
        if V.shape[0] != nunits:
            raise ValueError(f"Basis must have {nunits} rows, got {V.shape[0]}")
        if V.shape[1] < 1:
            raise ValueError("Basis must have at least 1 column")
            
        # Check unit-length columns
        norms = np.linalg.norm(V, axis=0)
        if not np.allclose(norms, 1):
            raise ValueError("Basis columns must be unit length")
            
        # Check orthogonality
        gram = V.T @ V
        if not np.allclose(gram, np.eye(V.shape[1])):
            raise ValueError("Basis columns must be orthogonal")
            
        basis = V.copy()
        # For user-supplied basis, compute magnitudes based on variance in basis
        trial_avg = np.mean(data, axis=2)  # shape (nunits, nconds)
        trial_avg_reshaped = trial_avg.T  # shape (ncond, nvox)
        proj_data = trial_avg_reshaped @ basis  # shape (ncond, basis_dim)
        magnitudes = np.var(proj_data, axis=0, ddof=1)  # variance along conditions for each basis dimension
        results['basis_source'] = None
        
    # Store the full basis and magnitudes for return
    fullbasis = basis.copy()

    # Update results with computed values
    results['fullbasis'] = fullbasis
    # Only set mags for magnitude thresholding mode (will be set later if needed)

    # 6) Default cross-validation thresholds if not provided
    if 'cv_thresholds' not in opt:
        opt['cv_thresholds'] = np.arange(1, basis.shape[1] + 1)
    else:
        # Validate cv_thresholds
        thresholds = np.array(opt['cv_thresholds'])
        if not np.all(thresholds > 0):
            raise ValueError("cv_thresholds must be positive integers")
        if not np.all(thresholds == thresholds.astype(int)):
            raise ValueError("cv_thresholds must be integers")
        if not np.all(np.diff(thresholds) > 0):
            raise ValueError("cv_thresholds must be in sorted order with unique values")

    # 7) Decide cross-validation or magnitude-threshold
    # We'll treat negative cv_mode as "do magnitude thresholding."
    if opt['cv_mode'] >= 0:
        denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce = perform_cross_validation(data, basis, opt, results=results)
        
        # Update results dictionary
        results.update({
            'denoiser': denoiser,
            'cv_scores': cv_scores,
            'best_threshold': best_threshold,
            'denoiseddata': denoiseddata,
            'fullbasis': fullbasis,
            'mags': magnitudes
        })
        
        # Add population-specific returns if applicable
        if opt['cv_threshold_per'] == 'population':
            results.update({
                'signalsubspace': signalsubspace,
                'dimreduce': dimreduce
            })
    else:
        denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce, mags, dimsretained = perform_magnitude_thresholding(data, basis, opt, results)
        
        # Update results dictionary with all magnitude thresholding returns
        results.update({
            'denoiser': denoiser,
            'cv_scores': cv_scores,
            'best_threshold': best_threshold,
            'denoiseddata': denoiseddata,
            'fullbasis': fullbasis,
            'mags': mags,
            'dimsretained': dimsretained,
            'signalsubspace': signalsubspace,
            'dimreduce': dimreduce
        })

    # Store the input data and parameters in results for later visualization
    results['input_data'] = data.copy()
    results['V'] = V
    
    # Add a function handle to regenerate the visualization
    def regenerate_visualization(test_data=None):
        """
        Regenerate the diagnostic visualization.
        
        Parameters:
        -----------
        test_data : ndarray, optional
            Data to use for testing in the bottom row plots, shape (nunits, nconds, ntrials).
            If None, will use leave-one-out cross-validation on the training data.
        """
        plot_diagnostic_figures(results['input_data'], results, test_data)
    
    results['plot'] = regenerate_visualization

    if wantfig:
        plot_diagnostic_figures(data, results)

    return results

def perform_cross_validation(data, basis, opt, results=None):
    """
    Perform cross-validation to determine optimal denoising dimensions.

    Uses cross-validation to determine how many dimensions to retain for denoising:
    1. Split trials into training and testing sets
    2. Project training data into basis
    3. Create denoising matrix for each dimensionality
    4. Measure prediction quality on test set
    5. Select threshold that gives best predictions

    The splitting can be done in two ways:
    - Leave-one-out: Use n-1 trials for training, 1 for testing
    - Keep-one-in: Use 1 trial for training, n-1 for testing

    Inputs:
    -----------
    <data> - shape (nunits, nconds, ntrials). Neural response data to denoise.
    <basis> - shape (nunits, dims). Orthonormal basis for denoising.
    <opt> - dict with fields:
        <cv_mode> - scalar. 
            0: n-1 train / 1 test split
            1: 1 train / n-1 test split
        <cv_threshold_per> - string.
            'unit': different thresholds per unit or unit group
            'population': same threshold for all units
        <unit_groups> - shape (nunits,). Integer array specifying which units 
            should receive the same cv threshold. Only applicable when cv_threshold_per='unit'.
            Units with the same integer value get the same cv threshold.
        <cv_thresholds> - shape (1, n_thresholds).
            Dimensions to test
        <cv_scoring_fn> - function handle.
            Function to compute prediction error
        <denoisingtype> - scalar.
            0: trial-averaged denoising
            1: single-trial denoising

    Returns:
    --------
    <denoiser> - shape (nunits, nunits). Matrix that projects data onto denoised space.
    <cv_scores> - shape (n_thresholds, ntrials, nunits). Cross-validation scores for each threshold.
    <best_threshold> - shape (1, nunits) or scalar. Selected threshold(s).
    <denoiseddata> - shape (nunits, nconds) or (nunits, nconds, ntrials). Denoised neural responses.
    <fullbasis> - shape (nunits, dims). Complete basis used for denoising.
    <signalsubspace> - shape (nunits, best_threshold) or []. Final basis functions used for denoising.
    <dimreduce> - shape (best_threshold, nconds) or (best_threshold, nconds, ntrials). 
        Data projected onto signal subspace.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    nunits, nconds, ntrials = data.shape
    cv_mode = opt['cv_mode']
    thresholds = opt['cv_thresholds']
    opt.setdefault('cv_scoring_fn', negative_mse_columns)
    threshold_per = opt['cv_threshold_per']
    scoring_fn = opt['cv_scoring_fn']
    denoisingtype = opt['denoisingtype']
    
    # Initialize cv_scores
    cv_scores = np.zeros((len(thresholds), ntrials, nunits))

    for tr in range(ntrials):
        # Define cross-validation splits based on cv_mode
        if cv_mode == 0:
            # Denoise average of n-1 trials, test against held out trial
            train_trials = np.setdiff1d(np.arange(ntrials), tr)
            train_avg = np.mean(data[:, :, train_trials], axis=2)  # Average n-1 trials
            test_data = data[:, :, tr]  # Single held-out trial
            
            for tt, threshold in enumerate(thresholds):
                safe_thr = min(threshold, basis.shape[1])
                denoising_fn = np.concatenate([np.ones(safe_thr), np.zeros(basis.shape[1] - safe_thr)])
                denoiser = basis @ np.diag(denoising_fn) @ basis.T
                
                # Demean training average before denoising
                train_avg_demeaned = train_avg - results['unit_means'][:, np.newaxis]
                train_denoised = (train_avg_demeaned.T @ denoiser).T
                cv_scores[tt, tr] = scoring_fn(test_data.T, train_denoised.T)
                  
        elif cv_mode == 1:
            # Denoise single trial, test against average of n-1 trials
            dataA = data[:, :, tr].T  # Single trial (nconds x nunits)
            dataB = np.mean(data[:, :, np.setdiff1d(np.arange(ntrials), tr)], axis=2).T  # Mean of other trials
            
            for tt, threshold in enumerate(thresholds):
                safe_thr = min(threshold, basis.shape[1])
                denoising_fn = np.concatenate([np.ones(safe_thr), np.zeros(basis.shape[1] - safe_thr)])
                denoiser = basis @ np.diag(denoising_fn) @ basis.T
                
                # Demean single trial before denoising
                dataA_demeaned = dataA - results['unit_means']
                dataA_denoised = dataA_demeaned @ denoiser
                cv_scores[tt, tr] = scoring_fn(dataB, dataA_denoised)
                
    # Decide best threshold
    if threshold_per == 'population':
        # Average over trials and units for population threshold
        avg_scores = np.mean(cv_scores, axis=(1, 2))  # (len(thresholds),)
        best_ix = np.argmax(avg_scores)
        best_threshold = int(thresholds[best_ix])  # Return scalar for population mode
        safe_thr = min(best_threshold, basis.shape[1])
        denoiser = basis[:, :safe_thr] @ basis[:, :safe_thr].T
    else:
        # unit-wise: average over trials only, then group by unit_groups
        avg_scores = np.mean(cv_scores, axis=1)  # (len(thresholds), nunits)
        unit_groups = opt['unit_groups']
        unique_groups = np.unique(unit_groups)
        
        best_thresh_unitwise = np.zeros(nunits, dtype=int)
        
        # For each group, find the best threshold by averaging CV scores within the group
        for group_id in unique_groups:
            group_mask = unit_groups == group_id
            group_units = np.where(group_mask)[0]
            
            # Average CV scores across units in this group
            group_avg_scores = np.mean(avg_scores[:, group_mask], axis=1)  # (len(thresholds),)
            best_idx = np.argmax(group_avg_scores)
            best_thresh_for_group = thresholds[best_idx]
            
            # Assign this threshold to all units in the group
            best_thresh_unitwise[group_mask] = best_thresh_for_group
            
        best_threshold = best_thresh_unitwise  # Return 1D array for unit-wise mode
                
        # Construct unit-wise denoiser
        denoiser = np.zeros((nunits, nunits))
        for unit_i in range(nunits):
            # For each unit, create its own denoising vector using its threshold
            safe_thr = min(int(best_threshold[unit_i]), basis.shape[1])
            unit_denoiser = basis[:, :safe_thr] @ basis[:, :safe_thr].T
            # Use the column corresponding to this unit
            denoiser[:, unit_i] = unit_denoiser[:, unit_i]

    # Calculate denoiseddata based on denoisingtype
    if denoisingtype == 0:
        # Trial-averaged denoising
        trial_avg = np.mean(data, axis=2)
        # Demean trial average before denoising
        trial_avg_demeaned = trial_avg - results['unit_means'][:, np.newaxis]
        denoiseddata = (trial_avg_demeaned.T @ denoiser).T
    else:
        # Single-trial denoising
        denoiseddata = np.zeros_like(data)
        for t in range(ntrials):
            # Demean each trial before denoising
            data_demeaned = data[:, :, t] - results['unit_means'][:, np.newaxis]
            denoiseddata[:, :, t] = (data_demeaned.T @ denoiser).T
            
    if results is not None and 'unit_means' in results:
        if denoiseddata.ndim == 3:  # Single-trial case
            denoiseddata = denoiseddata + results['unit_means'][:, np.newaxis, np.newaxis]
        else:  # Trial-averaged case
            denoiseddata = denoiseddata + results['unit_means'][:, np.newaxis]

    # Calculate additional return values
    fullbasis = basis.copy()
    if threshold_per == 'population':
        signalsubspace = basis[:, :safe_thr]
        
        # Project data onto signal subspace
        if denoisingtype == 0:
            trial_avg = np.mean(data, axis=2)
            # Demean before projecting to signal subspace for consistency
            trial_avg_demeaned = (trial_avg.T - results['unit_means']).T
            dimreduce = signalsubspace.T @ trial_avg_demeaned  # (safe_thr, nconds)
        else:
            dimreduce = np.zeros((safe_thr, nconds, ntrials))
            for t in range(ntrials):
                # Demean before projecting to signal subspace for consistency
                data_demeaned = (data[:, :, t].T - results['unit_means']).T
                dimreduce[:, :, t] = signalsubspace.T @ data_demeaned
    else:
        signalsubspace = None
        dimreduce = None

    return denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce

def perform_magnitude_thresholding(data, basis, opt, results=None):
    """
    Select dimensions using magnitude thresholding.

    Implements the magnitude thresholding procedure for PSN denoising.
    Selects dimensions based on cumulative variance explained rather than 
    using cross-validation.

    Algorithm Details:
    1. Get magnitudes either:
       - From signal variance of the data projected into the basis (mag_type=0)
       - Or precomputed basis eigenvalues (mag_type=1)
    2. Sort dimensions by magnitude in descending order
    3. Select the top dimensions that cumulatively account for mag_frac 
       of the total variance
    4. Create denoising matrix using selected dimensions (in original order)

    Parameters:
    -----------
    <data> - shape (nunits, nconds, ntrials). Neural response data to denoise.
    <basis> - shape (nunits, dims). Orthonormal basis for denoising.
    <opt> - dict with fields:
        <mag_type> - scalar. How to obtain component magnitudes:
            0: use signal variance computed from data
            1: use pre-computed eigenvalues from results
        <mag_frac> - scalar. Fraction of total variance to retain (e.g., 0.95).
        <denoisingtype> - scalar. Type of denoising:
            0: trial-averaged
            1: single-trial
    <results> - dict containing pre-computed magnitudes in results['mags'] if mag_type=1

    Returns:
    --------
    <denoiser> - shape (nunits, nunits). Matrix that projects data onto denoised space.
    <cv_scores> - shape (0, 0). Empty array (not used in magnitude thresholding).
    <best_threshold> - shape (1, n_retained). Selected dimension indices.
    <denoiseddata> - shape (nunits, nconds) or (nunits, nconds, ntrials). Denoised neural responses.
    <basis> - shape (nunits, dims). Complete basis used for denoising.
    <signalsubspace> - shape (nunits, n_retained). Final basis functions used for denoising.
    <dimreduce> - shape (n_retained, nconds) or (n_retained, nconds, ntrials). 
        Data projected onto signal subspace.
    <magnitudes> - shape (1, dims). Component magnitudes used for thresholding.
    <dimsretained> - scalar. Number of dimensions retained.
    """
    nunits, nconds, ntrials = data.shape
    mag_type = opt['mag_type']
    mag_frac = opt['mag_frac']
    denoisingtype = opt['denoisingtype']

    cv_scores = np.array([])  # Not used in magnitude thresholding
    
    # Get magnitudes based on mag_type
    if mag_type == 1:
        # Use pre-computed magnitudes from results
        magnitudes = results['mags']
    else:
        # Variance-based threshold in user basis
        # Initialize list to store signal variances
        sigvars = []

        data_reshaped = data.transpose(1, 2, 0)
        # Compute signal variance for each basis dimension
        for i in range(basis.shape[1]):
            this_eigv = basis[:, i]  # Select the i-th eigenvector
            proj_data = np.dot(data_reshaped, this_eigv)  # Project data into this eigenvector's subspace

            # Compute signal variance (using same computation as in noise ceiling)
            noisevar = np.mean(np.std(proj_data, axis=1, ddof=1) ** 2)
            datavar = np.std(np.mean(proj_data, axis=1), ddof=1) ** 2
            signalvar = np.maximum(datavar - noisevar / proj_data.shape[1], 0)  # Ensure non-negative variance
            sigvars.append(float(signalvar))

        magnitudes = np.array(sigvars)
    
    # Sort dimensions by magnitude in descending order to find cumulative variance
    sorted_indices = np.argsort(magnitudes)[::-1]  # Descending order
    sorted_magnitudes = magnitudes[sorted_indices]
    
    # Calculate cumulative variance explained
    total_variance = np.sum(sorted_magnitudes)
    cumulative_variance = np.cumsum(sorted_magnitudes)
    cumulative_fraction = cumulative_variance / total_variance
    
    # Find how many dimensions we need to reach mag_frac of total variance
    dims_needed = np.sum(cumulative_fraction < mag_frac) + 1  # +1 to include the dimension that crosses threshold
    dims_needed = min(dims_needed, len(sorted_magnitudes))  # Don't exceed total dimensions
    
    # Get the original indices of the selected dimensions (unsorted)
    best_threshold_indices = sorted_indices[:dims_needed]  # 0-indexed for Python
    dimsretained = len(best_threshold_indices)
    
    if dimsretained == 0:
        # If no dimensions selected, return zero matrices
        denoiser = np.zeros((nunits, nunits))
        denoiseddata = np.zeros((nunits, nconds)) if denoisingtype == 0 else np.zeros_like(data)
        signalsubspace = basis[:, :0]  # Empty but valid shape
        dimreduce = np.zeros((0, nconds)) if denoisingtype == 0 else np.zeros((0, nconds, ntrials))
        best_threshold = np.array([])
        return denoiser, cv_scores, best_threshold, denoiseddata, basis, signalsubspace, dimreduce, magnitudes, dimsretained

    # Create denoising matrix using retained dimensions (use 0-indexed)
    denoising_fn = np.zeros(basis.shape[1])
    denoising_fn[best_threshold_indices] = 1
    denoiser = basis @ np.diag(denoising_fn) @ basis.T

    # Calculate denoised data
    if denoisingtype == 0:
        # Trial-averaged denoising
        trial_avg = np.mean(data, axis=2)
        # Demean trial average before denoising
        trial_avg_demeaned = trial_avg - results['unit_means'][:, np.newaxis]
        denoiseddata = (trial_avg_demeaned.T @ denoiser).T
    else:
        # Single-trial denoising
        denoiseddata = np.zeros_like(data)
        for t in range(ntrials):
            # Demean each trial before denoising
            data_demeaned = data[:, :, t] - results['unit_means'][:, np.newaxis]
            denoiseddata[:, :, t] = (data_demeaned.T @ denoiser).T
            
    # add back the means
    if results is not None and 'unit_means' in results:
        if denoiseddata.ndim == 3:  # Single-trial case
            denoiseddata = denoiseddata + results['unit_means'][:, np.newaxis, np.newaxis]
        else:  # Trial-averaged case
            denoiseddata = denoiseddata + results['unit_means'][:, np.newaxis]

    # Calculate signal subspace and reduced dimensions (use 0-indexed)
    signalsubspace = basis[:, best_threshold_indices]
    if denoisingtype == 0:
        trial_avg = np.mean(data, axis=2)
        # Demean before projecting to signal subspace for consistency
        trial_avg_demeaned = trial_avg - results['unit_means'][:, np.newaxis]
        dimreduce = signalsubspace.T @ trial_avg_demeaned
    else:
        dimreduce = np.zeros((len(best_threshold_indices), nconds, ntrials))
        for t in range(ntrials):
            # Demean before projecting to signal subspace for consistency
            data_demeaned = data[:, :, t] - results['unit_means'][:, np.newaxis]
            dimreduce[:, :, t] = signalsubspace.T @ data_demeaned

    # Return 0-indexed indices for Python (MATLAB will be 1-indexed)
    best_threshold = best_threshold_indices

    return denoiser, cv_scores, best_threshold, denoiseddata, basis, signalsubspace, dimreduce, magnitudes, dimsretained


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
        - ndarray: user-supplied orthonormal basis matrix, shape (nunits, dims)
        
    cv : str or None, default='unit'
        Cross-validation strategy for threshold selection:
        - 'unit': unit-wise thresholding, separate threshold chosen per unit
        - 'population': population thresholding, one threshold for all units
        - None: magnitude thresholding, retains dimensions for specified variance fraction
        
    scoring : str or callable, default='mse'
        Scoring function for cross-validation (when cv is 'unit' or 'population'):
        - 'mse': mean squared error (default)
        - 'r2': coefficient of determination (R²)
        - callable: custom scoring function with signature score(y_true, y_pred)
        
    mag_threshold : float, default=0.95
        Proportion of variance to retain when cv=None (magnitude thresholding mode)
        
    unit_groups : array-like or None, default=None
        Integer array of shape (nunits,) specifying which units should receive the same 
        CV threshold. Only applicable when cv='unit'. Units with the same integer value 
        get the same threshold. If None, each unit gets its own threshold (for 'unit' mode)
        or all units get the same threshold (for 'population' mode).
        
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
    
    def __init__(self, basis='signal', cv='unit', scoring='mse', mag_threshold=0.95, 
                 unit_groups=None, verbose=False, wantfig=True, gsn_kwargs=None):
        self.basis = basis
        self.cv = cv
        self.scoring = scoring
        self.mag_threshold = mag_threshold
        self.unit_groups = unit_groups
        self.verbose = verbose
        self.wantfig = wantfig
        self.gsn_kwargs = gsn_kwargs
        
    def _validate_params(self):
        """Validate input parameters."""
        # Validate basis
        valid_basis_strings = ['signal', 'whitened-signal', 'noise', 'pca', 'random']
        if isinstance(self.basis, str):
            if self.basis not in valid_basis_strings:
                raise ValueError(f"basis must be one of {valid_basis_strings} or an ndarray")
        elif not isinstance(self.basis, np.ndarray):
            raise ValueError(f"basis must be one of {valid_basis_strings} or an ndarray")
            
        # Validate cv
        if self.cv not in ['unit', 'population', None]:
            raise ValueError("cv must be 'unit', 'population', or None")
            
        # Validate scoring
        valid_scoring_strings = ['mse', 'r2']
        if isinstance(self.scoring, str):
            if self.scoring not in valid_scoring_strings:
                raise ValueError(f"scoring must be one of {valid_scoring_strings} or a callable")
        elif not callable(self.scoring):
            raise ValueError(f"scoring must be one of {valid_scoring_strings} or a callable")
            
        # Validate mag_threshold
        if not isinstance(self.mag_threshold, (int, float)) or not 0 < self.mag_threshold <= 1:
            raise ValueError("mag_threshold must be a number between 0 and 1")
            
    def _convert_params_to_functional(self, data):
        """Convert sklearn-style parameters to functional PSN parameters."""
        self._validate_params()
        
        nunits = data.shape[0]
        
        # Convert basis parameter
        if isinstance(self.basis, str):
            if self.basis == 'signal':
                V = 0
            elif self.basis == 'whitened-signal':
                V = 1
            elif self.basis == 'noise':
                V = 2
            elif self.basis == 'pca':
                V = 3
            elif self.basis == 'random':
                V = 4
        else:  # numpy array
            V = self.basis
            
        # Convert cv and scoring parameters
        if self.cv is None:
            cv_mode = -1
            cv_threshold_per = 'population'  # Not used in magnitude mode
        else:
            cv_mode = 0  # Use leave-one-out cross-validation
            cv_threshold_per = self.cv
            
        # Convert scoring function
        if self.cv is not None:  # Only matters for cross-validation
            if self.scoring == 'mse':
                cv_scoring_fn = negative_mse_columns
            elif self.scoring == 'r2':
                cv_scoring_fn = r2_score_columns
            else:  # callable
                cv_scoring_fn = self.scoring
        else:
            cv_scoring_fn = negative_mse_columns  # Default, not used
            
        # Build options dictionary
        opt = {
            'cv_mode': cv_mode,
            'cv_threshold_per': cv_threshold_per,
            'cv_scoring_fn': cv_scoring_fn,
            'mag_frac': self.mag_threshold,
            'denoisingtype': 0,  # Always use trial-averaged for fitting
        }
        
        # Add unit_groups if provided
        if self.unit_groups is not None:
            opt['unit_groups'] = np.asarray(self.unit_groups, dtype=int)
        
        # Add GSN kwargs if provided
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
        # Validate input data
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("Input data must be 3-dimensional (nunits, nconds, ntrials)")
        if X.shape[2] < 2:
            raise ValueError("Data must have at least 2 trials")
        if X.shape[1] < 2:
            raise ValueError("Data must have at least 2 conditions")
            
        # Convert parameters and fit
        V, opt = self._convert_params_to_functional(X)
        
        if self.verbose:
            print("Fitting PSN denoiser...")
            
        # Call functional PSN
        results = psn(X, V=V, opt=opt, wantfig=self.wantfig)
        
        # Store fitted attributes
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
            # Trial-averaged data
            nunits, nconds = X.shape
            if nunits != self.denoiser_.shape[0]:
                raise ValueError(f"Number of units ({nunits}) doesn't match fitted denoiser "
                               f"({self.denoiser_.shape[0]})")
                               
            # Demean and denoise
            X_demeaned = X - self.unit_means_[:, np.newaxis]
            X_denoised = (X_demeaned.T @ self.denoiser_).T
            # Add back means
            X_denoised = X_denoised + self.unit_means_[:, np.newaxis]
            
        elif X.ndim == 3:
            # Single-trial data
            nunits, nconds, ntrials = X.shape
            if nunits != self.denoiser_.shape[0]:
                raise ValueError(f"Number of units ({nunits}) doesn't match fitted denoiser "
                               f"({self.denoiser_.shape[0]})")
                               
            X_denoised = np.zeros_like(X)
            for t in range(ntrials):
                # Demean and denoise each trial
                X_trial_demeaned = X[:, :, t] - self.unit_means_[:, np.newaxis]
                X_denoised[:, :, t] = (X_trial_demeaned.T @ self.denoiser_).T
            # Add back means
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
        # Return trial-averaged denoised data
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
        Return the mean noise ceiling score on the given test data.
        
        This computes the noise ceiling for the denoised data as a measure
        of denoising quality. Higher values indicate better denoising.
        
        Parameters
        ----------
        X : ndarray, shape (nunits, nconds, ntrials)
            Test data
            
        y : Ignored
            Not used, present for sklearn compatibility
            
        Returns
        -------
        score : float
            Mean noise ceiling across all units
        """
        check_is_fitted(self, 'denoiser_')
        
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("Input data must be 3-dimensional for scoring")
            
        # Transform the data
        X_denoised = self.transform(X)
        
        # Compute noise ceiling
        from .utils import compute_noise_ceiling
        noise_ceiling, *_ = compute_noise_ceiling(X_denoised)
        
        return np.mean(noise_ceiling)
        
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
            # Fallback to direct plotting
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
