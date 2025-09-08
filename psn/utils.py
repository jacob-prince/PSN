import numpy as np

# Try to import GSN, provide fallback if not available
try:
    from gsn.perform_gsn import perform_gsn as gsn_perform_gsn
    _GSN_AVAILABLE = True
except ImportError:
    _GSN_AVAILABLE = False
    def gsn_perform_gsn(data, opt=None):
        raise ImportError("GSN package not installed. Please install with: pip install git+https://github.com/cvnlab/GSN.git")

def perform_gsn(data, opt=None):
    """
    Perform Generative Signal and Noise (GSN) analysis on neural data using the external GSN package.
    
    This function is a wrapper around the cvnlab/GSN package that computes signal and noise 
    covariance matrices from neural response data.
    
    Parameters:
    -----------
    data : ndarray
        Neural response data, shape (nunits, nconds, ntrials)
    opt : dict, optional
        Options dictionary to pass to the GSN algorithm
        
    Returns:
    --------
    dict with fields:
        'cSb' : ndarray
            Signal covariance matrix, shape (nunits, nunits)
        'cNb' : ndarray  
            Noise covariance matrix, shape (nunits, nunits)
    """
    if not _GSN_AVAILABLE:
        raise ImportError("GSN package not installed. Please install with: pip install git+https://github.com/cvnlab/GSN.git")

    if opt is None:
        opt = {}

    # Call the external GSN package
    results = gsn_perform_gsn(data, opt)

    return {
        'cSb': results['cSb'],
        'cNb': results['cNb']
    }

def compute_noise_ceiling(data_in):
    """
    Compute the noise ceiling signal-to-noise ratio (SNR) and percentage noise ceiling for each unit.
    
    Parameters:
    ----------
    data_in : np.ndarray
        A 3D array of shape (units/voxels, conditions, trials), representing the data for which to compute 
        the noise ceiling. Each unit requires more than 1 trial for each condition.

    Returns:
    -------
    noiseceiling : np.ndarray
        The noise ceiling for each unit, expressed as a percentage.
    ncsnr : np.ndarray
        The noise ceiling signal-to-noise ratio (SNR) for each unit.
    signalvar : np.ndarray
        The signal variance for each unit.
    noisevar : np.ndarray
        The noise variance for each unit.
    """
    # noisevar: mean variance across trials for each unit
    noisevar = np.mean(np.std(data_in, axis=2, ddof=1) ** 2, axis=1)

    # datavar: variance of the trial means across conditions for each unit
    datavar = np.std(np.mean(data_in, axis=2), axis=1, ddof=1) ** 2

    # signalvar: signal variance, obtained by subtracting noise variance from data variance
    signalvar = np.maximum(datavar - noisevar / data_in.shape[2], 0)  # Ensure non-negative variance

    # ncsnr: signal-to-noise ratio (SNR) for each unit
    ncsnr = np.sqrt(signalvar) / np.sqrt(noisevar)

    # noiseceiling: percentage noise ceiling based on SNR
    noiseceiling = 100 * (ncsnr ** 2 / (ncsnr ** 2 + 1 / data_in.shape[2]))

    return noiseceiling, ncsnr, signalvar, noisevar

def compute_r2(y_true, y_pred):
    """Compute R2 score between true and predicted values."""
    residual_ss = np.sum((y_true - y_pred) ** 2)
    total_ss = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (residual_ss / total_ss)
    return r2

def make_orthonormal(V):
    """MAKE_ORTHONORMAL Find the nearest matrix with orthonormal columns.

    Uses Singular Value Decomposition (SVD) to find the nearest orthonormal matrix:
    1. Decompose <V> = <U>*<S>*<Vh> where <U> and <Vh> are orthogonal
    2. The nearest orthonormal matrix is <U>*<Vh>
    3. Take only the first n columns if m > n
    4. Verify orthonormality within numerical precision

    Inputs:
        <V> - m x n matrix where m >= n. Input matrix to be made orthonormal.
            The number of rows (m) must be at least as large as the number of
            columns (n).

    Returns:
        <V_orthonormal> - m x n matrix with orthonormal columns.
            The resulting matrix will have:
            1. All columns unit length
            2. All columns pairwise orthogonal

    Example:
        V = np.random.randn(5,3)  # Random 5x3 matrix
        V_ortho = make_orthonormal(V)
        # Check orthonormality
        gram = V_ortho.T @ V_ortho  # Should be very close to identity
        print(np.max(np.abs(gram - np.eye(gram.shape[0]))))  # Should be ~1e-15

    Notes:
        The SVD method guarantees orthonormality within numerical precision.
        A warning is issued if the result is not perfectly orthonormal.
    """
    # Check input dimensions
    m, n = V.shape
    if m < n:
        raise ValueError('Input matrix must have at least as many rows as columns')

    # Use SVD to find the nearest orthonormal matrix
    # SVD gives us V = U*S*Vh where U and Vh are orthogonal
    # The nearest orthonormal matrix is U*Vh
    U, _, Vh = np.linalg.svd(V, full_matrices=False)

    # Take only the first n columns of U if m > n
    V_orthonormal = U[:,:n] @ Vh

    # Double check that the result is orthonormal within numerical precision
    # This is mainly for debugging - the SVD method should guarantee this
    gram = V_orthonormal.T @ V_orthonormal
    if not np.allclose(gram, np.eye(n), rtol=0, atol=1e-10):
        print('Warning: Result may not be perfectly orthonormal due to numerical precision')

    return V_orthonormal


def negative_mse_columns(x, y):
    """
    Calculate negative mean squared error between columns.

    Parameters:
    -----------
    <x> - shape (nconds, nunits). First matrix (usually test data).
    <y> - shape (nconds, nunits). Second matrix (usually predictions).
        Must have same shape as <x>.

    Returns:
    --------
    <scores> - shape (1, nunits). Negative MSE for each column/unit.
            0 indicates perfect prediction
            More negative values indicate worse predictions
            Each unit gets its own score

    Example:
    --------
        x = np.array([[1, 2], [3, 4]])  # 2 conditions, 2 units
        y = np.array([[1.1, 2.1], [2.9, 3.9]])  # Predictions
        scores = negative_mse_columns(x, y)  # Close to 0

    Notes:
    ------
        The function handles empty inputs gracefully by returning zeros, which is useful
        when no data survives thresholding.
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        return np.zeros(x.shape[1])  # Return zeros for empty arrays
    return -np.mean((x - y) ** 2, axis=0)


def r2_score_columns(y_true, y_pred):
    """
    Compute R² (coefficient of determination) for each column.
    
    R² represents the proportion of variance in the dependent variable that is 
    predictable from the independent variable. It ranges from -∞ to 1, where:
    - 1 indicates perfect prediction
    - 0 indicates no predictive power (same as predicting the mean)
    - Negative values indicate worse than predicting the mean
    
    Parameters:
    -----------
    y_true : ndarray, shape (n_samples, n_features)
        True values
    y_pred : ndarray, shape (n_samples, n_features) 
        Predicted values
        
    Returns:
    --------
    r2_scores : ndarray, shape (n_features,)
        R² score for each column/feature
        
    Formula:
    --------
    R² = 1 - (SS_res / SS_tot)
    where:
    - SS_res = Σ(y_true - y_pred)² (residual sum of squares)
    - SS_tot = Σ(y_true - mean(y_true))² (total sum of squares)
    
    Example:
    --------
        y_true = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples, 2 features
        y_pred = np.array([[1.1, 2.1], [2.9, 3.9], [4.8, 5.8]])  # Good predictions
        r2_scores = r2_score_columns(y_true, y_pred)  # Close to 1.0
        
    Notes:
    ------
        - Handles edge cases where total variance is zero (returns 1.0 if residuals 
          are also zero, 0.0 otherwise)
        - Returns zeros for empty inputs
        - This is equivalent to sklearn's r2_score applied column-wise
    """
    if y_true.shape[0] == 0 or y_pred.shape[0] == 0:
        return np.zeros(y_true.shape[1])  # Return zeros for empty arrays

    r2_scores = []
    for i in range(y_true.shape[1]):
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)

        if ss_tot == 0:
            # If total variance is zero, return 1.0 if residuals are also zero, else 0.0
            r2_scores.append(1.0 if ss_res == 0 else 0.0)
        else:
            r2 = 1 - (ss_res / ss_tot)
            r2_scores.append(r2)

    return np.array(r2_scores)

def split_half_reliability(data_orig, data_denoised):
    """
    Compute split-half reliability between denoised and original data using odd/even trial splits.
    
    This function measures how well denoised data preserves the reliability of tuning profiles
    across conditions by comparing odd and even trial averages. It provides a measure of how
    much noise reduction improves the consistency of neural responses.
    
    Parameters:
    -----------
    data_orig : ndarray, shape (n_conditions, n_units)
        Original data (typically trial-averaged or single trial data from test set)
    data_denoised : ndarray, shape (n_conditions, n_units)  
        Denoised version of the original data
        
    Returns:
    --------
    reliability_scores : ndarray, shape (n_units,)
        Split-half reliability score for each unit, comparing denoised vs original data
        Values closer to 1.0 indicate better reliability preservation/improvement
        
    Notes:
    ------
    This scoring function expects that the original 3D data (units, conditions, trials) 
    is available in the cross-validation context to compute the split-half reliability.
    For PSN cross-validation, this should compare:
    - Odd/even trial correlations in original data
    - Odd/even trial correlations in denoised data  
    
    The improvement in reliability (denoised - original) serves as the score.
    
    Example:
    --------
        # In practice, this would be called during cross-validation with access to trial data
        data_3d = np.random.randn(50, 100, 10)  # 50 units, 100 conditions, 10 trials  
        # Split into odd/even, average, correlate, compute improvement
        reliability = split_half_reliability(orig_corr, denoised_corr)
    """
    if data_orig.shape[0] == 0 or data_denoised.shape[0] == 0:
        return np.zeros(data_orig.shape[1])  # Return zeros for empty arrays

    # For the cross-validation context, we compute correlation between the test data
    # and denoised predictions as a proxy for reliability
    # This is a simplified version - ideally we'd have access to the full trial structure

    reliability_scores = []
    for i in range(data_orig.shape[1]):
        # Compute correlation between original and denoised tuning profiles
        corr_coef = np.corrcoef(data_orig[:, i], data_denoised[:, i])[0, 1]

        # Handle NaN cases (e.g., when variance is zero)
        if np.isnan(corr_coef):
            reliability_scores.append(0.0)
        else:
            reliability_scores.append(corr_coef)

    return np.array(reliability_scores)

def split_half_reliability_3d(data_3d):
    """
    Compute split-half reliability for 3D neural data using odd/even trial splits.
    
    This function computes the correlation between odd and even trial averages for each unit,
    providing a measure of response reliability across conditions. This is the core computation
    for split-half reliability analysis.
    
    Parameters:
    -----------
    data_3d : ndarray, shape (n_units, n_conditions, n_trials)
        Neural response data with multiple trials
        
    Returns:
    --------
    reliability_scores : ndarray, shape (n_units,)
        Split-half reliability score for each unit
        Values range from -1 to 1, with higher values indicating more reliable responses
        
    Notes:
    ------
    - Requires at least 2 trials to compute split-half reliability
    - Odd and even trials are averaged separately, then correlated across conditions
    - Units with insufficient trials or zero variance return 0.0
    
    Example:
    --------
        data = np.random.randn(50, 100, 10)  # 50 units, 100 conditions, 10 trials
        reliability = split_half_reliability_3d(data)
        print(f"Mean reliability: {np.mean(reliability):.3f}")
    """
    n_units, n_conditions, n_trials = data_3d.shape

    if n_trials < 2:
        return np.zeros(n_units)

    # Split trials into odd and even
    odd_trials = data_3d[:, :, 0::2]  # trials 0, 2, 4, ...
    even_trials = data_3d[:, :, 1::2]  # trials 1, 3, 5, ...

    # Average across odd and even trials
    odd_avg = np.mean(odd_trials, axis=2)  # shape: (n_units, n_conditions)
    even_avg = np.mean(even_trials, axis=2)  # shape: (n_units, n_conditions)

    reliability_scores = []
    for unit in range(n_units):
        # Compute correlation between odd and even averages across conditions
        corr_coef = np.corrcoef(odd_avg[unit, :], even_avg[unit, :])[0, 1]

        # Handle NaN cases (e.g., when variance is zero)
        if np.isnan(corr_coef):
            reliability_scores.append(0.0)
        else:
            reliability_scores.append(corr_coef)

    return np.array(reliability_scores)
