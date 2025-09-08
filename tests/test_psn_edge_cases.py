"""Edge case tests for PSN denoiser."""

import numpy as np
import pytest
from psn import PSN

def test_nearly_singular_data():
    """Test with nearly singular data where some dimensions are almost linearly dependent."""
    nunits, nconds, ntrials = 10, 5, 3
    # Create nearly linearly dependent data
    base = np.random.randn(nunits, nconds, ntrials)
    data = base.copy()
    data[1:4] = base[0:1] + 1e-10 * np.random.randn(3, nconds, ntrials)  # Make rows nearly identical
    
    psn_model = PSN(wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert not np.any(np.isnan(psn_model.denoiser_))  # Should handle near-singularity gracefully
    assert denoised.shape == data.shape

def test_extreme_magnitude_differences():
    """Test with data having extreme differences in magnitudes across dimensions."""
    nunits, nconds, ntrials = 5, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    data[0] *= 1e6  # Make first dimension much larger
    data[-1] *= 1e-6  # Make last dimension much smaller
    
    psn_model = PSN(wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert not np.any(np.isnan(psn_model.denoiser_))
    assert denoised.shape == data.shape

def test_alternating_signal_noise():
    """Test with alternating strong signal and pure noise dimensions."""
    nunits, nconds, ntrials = 6, 5, 4
    data = np.zeros((nunits, nconds, ntrials))
    # Even indices: strong signal
    data[::2, :, :] = np.repeat(np.random.randn(nunits//2, nconds, 1), ntrials, axis=2)
    # Odd indices: pure noise
    data[1::2, :, :] = np.random.randn(nunits//2, nconds, ntrials)
    
    psn_model = PSN(wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert denoised.shape == data.shape

def test_identical_trials():
    """Test with identical trials (zero noise case)."""
    nunits, nconds = 5, 4
    single_trial = np.random.randn(nunits, nconds)
    data = np.repeat(single_trial[:, :, np.newaxis], 3, axis=2)
    
    psn_model = PSN(wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert denoised.shape == data.shape
    # Check that trial-averaged output is close to original (for 2D transform)
    denoised_2d = psn_model.transform(single_trial)
    assert np.allclose(denoised_2d, single_trial, atol=1e-1)

def test_zero_variance_dimensions():
    """Test with dimensions having zero variance."""
    nunits, nconds, ntrials = 5, 4, 3
    data = np.random.randn(nunits, nconds, ntrials)
    data[2] = 0  # Set middle dimension to zero
    
    psn_model = PSN(wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert not np.any(np.isnan(psn_model.denoiser_))
    assert denoised.shape == data.shape

def test_anti_correlated_dimensions():
    """Test with perfectly anti-correlated dimensions."""
    nunits, nconds, ntrials = 4, 5, 3
    base = np.random.randn(1, nconds, ntrials)
    data = np.zeros((nunits, nconds, ntrials))
    data[0] = base
    data[1] = -base  # Perfect anti-correlation
    data[2:] = np.random.randn(nunits-2, nconds, ntrials)
    
    psn_model = PSN(wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert denoised.shape == data.shape

def test_sparse_signal():
    """Test with very sparse signal (mostly zeros with occasional spikes)."""
    nunits, nconds, ntrials = 5, 20, 3
    data = np.zeros((nunits, nconds, ntrials))
    # Add occasional spikes
    spike_positions = np.random.choice(nconds, size=2, replace=False)
    data[:, spike_positions, :] = np.random.randn(nunits, 2, ntrials)
    
    psn_model = PSN(wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert denoised.shape == data.shape

def test_structured_noise():
    """Test with structured noise that mimics signal."""
    nunits, nconds, ntrials = 5, 10, 4
    signal = np.random.randn(nunits, nconds, 1)
    structured_noise = np.random.randn(nunits, 1, ntrials)  # Noise correlated across conditions
    data = signal + structured_noise
    
    psn_model = PSN(wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert denoised.shape == data.shape

def test_all_signal_data():
    """Test with data that is pure signal (perfectly repeatable)."""
    nunits, nconds, ntrials = 5, 5, 3
    signal = np.random.randn(nunits, nconds, 1)
    data = np.repeat(signal, ntrials, axis=2)
    
    psn_model = PSN(wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert denoised.shape == data.shape
    # Should return identity-like denoiser for pure signal
    assert np.allclose(psn_model.denoiser_, np.eye(nunits), atol=1e-1)

def test_rank_deficient_data():
    """Test with rank-deficient data."""
    nunits, nconds, ntrials = 5, 3, 3  # More units than conditions
    data = np.random.randn(2, nconds, ntrials)  # Generate low-rank data
    data = np.vstack([data, np.zeros((nunits-2, nconds, ntrials))])  # Pad with zeros
    
    psn_model = PSN(wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert not np.any(np.isnan(psn_model.denoiser_))
    assert denoised.shape == data.shape

def test_minimal_trials():
    """Test with minimum possible number of trials."""
    nunits, nconds = 5, 5
    data = np.random.randn(nunits, nconds, 2)  # Minimum 2 trials
    
    psn_model = PSN(wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert not np.any(np.isnan(psn_model.denoiser_))
    assert denoised.shape == data.shape

def test_many_trials():
    """Test with unusually large number of trials."""
    nunits, nconds = 5, 5
    data = np.random.randn(nunits, nconds, 100)  # Many trials
    
    psn_model = PSN(wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert denoised.shape == data.shape

def test_magnitude_fraction_edge_cases():
    """Test edge cases of magnitude fraction thresholding."""
    nunits, nconds, ntrials = 5, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with very small mag_threshold (should retain very few dimensions)
    psn_small = PSN(cv=None, mag_threshold=0.01, wantfig=False)
    psn_small.fit(data)
    denoised_small = psn_small.transform(data)
    
    assert psn_small.denoiser_.shape == (nunits, nunits)
    assert denoised_small.shape == data.shape
    
    # Test with mag_threshold very close to 1 (should retain almost all dimensions)
    psn_large = PSN(cv=None, mag_threshold=0.999, wantfig=False)
    psn_large.fit(data)
    denoised_large = psn_large.transform(data)
    
    assert psn_large.denoiser_.shape == (nunits, nunits)
    assert denoised_large.shape == data.shape

def test_custom_basis_edge_cases():
    """Test edge cases with custom basis."""
    nunits, nconds, ntrials = 5, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with nearly orthogonal basis
    V = np.eye(nunits) + 1e-10 * np.random.randn(nunits, nunits)
    V, _ = np.linalg.qr(V)  # Make orthogonal
    
    psn_model = PSN(basis=V, wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert denoised.shape == data.shape
    
    # Test with minimal basis (single vector)
    V_minimal = np.random.randn(nunits, 1)
    V_minimal = V_minimal / np.linalg.norm(V_minimal)
    
    psn_minimal = PSN(basis=V_minimal, wantfig=False)
    psn_minimal.fit(data)
    denoised_minimal = psn_minimal.transform(data)
    
    assert psn_minimal.denoiser_.shape == (nunits, nunits)
    assert denoised_minimal.shape == data.shape

def test_mixed_scale_data():
    """Test with data having mixed scales across units."""
    nunits, nconds, ntrials = 5, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    scales = np.logspace(-3, 3, nunits)
    data = data * scales[:, np.newaxis, np.newaxis]
    
    psn_model = PSN(wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert not np.any(np.isnan(psn_model.denoiser_))
    assert denoised.shape == data.shape

def test_outlier_trials():
    """Test with data containing outlier trials."""
    nunits, nconds, ntrials = 5, 5, 5
    data = np.random.randn(nunits, nconds, ntrials)
    # Add one outlier trial
    data[:, :, -1] = 10 * np.random.randn(nunits, nconds)
    
    psn_model = PSN(wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert not np.any(np.isnan(psn_model.denoiser_))
    assert denoised.shape == data.shape

def test_block_diagonal_structure():
    """Test with block diagonal structure in the data."""
    nunits, nconds, ntrials = 6, 10, 5
    data = np.zeros((nunits, nconds, ntrials))
    
    # Create two independent blocks with distinct patterns
    t = np.linspace(0, 6*np.pi, nconds)
    block1 = 5.0 * np.vstack([
        np.sin(t),
        np.sin(2*t),
        np.sin(4*t)
    ])[:, :, np.newaxis]
    
    t = np.linspace(0, 3, nconds)
    block2 = 5.0 * np.vstack([
        np.exp(-t),
        np.exp(-2*t),
        np.exp(-4*t)
    ])[:, :, np.newaxis]
    
    # Repeat patterns across trials and add small noise
    data[:3] = np.repeat(block1, ntrials, axis=2) + 0.01 * np.random.randn(3, nconds, ntrials)
    data[3:] = np.repeat(block2, ntrials, axis=2) + 0.01 * np.random.randn(3, nconds, ntrials)
    
    # Test only in population mode
    psn_model = PSN(cv='population', wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert denoised.shape == data.shape
    
    # Check if block structure is approximately preserved
    denoiser_upper = psn_model.denoiser_[:3, 3:]
    denoiser_lower = psn_model.denoiser_[3:, :3]
    assert np.all(np.abs(denoiser_upper) < 0.05), "Upper off-diagonal block should be close to zero"
    assert np.all(np.abs(denoiser_lower) < 0.05), "Lower off-diagonal block should be close to zero"

def test_repeated_dimensions():
    """Test with repeated dimensions in the data."""
    nunits, nconds, ntrials = 5, 5, 3
    base = np.random.randn(2, nconds, ntrials)
    data = np.vstack([base, base, base[0:1]])  # Create repeated dimensions
    
    psn_model = PSN(wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert not np.any(np.isnan(psn_model.denoiser_))
    assert denoised.shape == data.shape

def test_basis_selection_edge_cases():
    """Test edge cases in basis selection."""
    nunits, nconds, ntrials = 5, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test all basis selection modes with extreme data
    basis_types = ['signal', 'whitened-signal', 'noise', 'pca', 'random']
    
    for basis in basis_types:
        # Make data nearly singular
        data_singular = data.copy()
        data_singular[1:] = data_singular[0:1] + 1e-10 * np.random.randn(nunits-1, nconds, ntrials)
        
        psn_model = PSN(basis=basis, wantfig=False)
        psn_model.fit(data_singular)
        denoised = psn_model.transform(data_singular)
        
        assert psn_model.denoiser_.shape == (nunits, nunits)
        assert not np.any(np.isnan(psn_model.denoiser_))
        assert denoised.shape == data_singular.shape

def test_scoring_function_edge_cases():
    """Test edge cases with different scoring functions."""
    nunits, nconds, ntrials = 5, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with constant scoring function
    def constant_score(x, y):
        return np.zeros(x.shape[1])
    
    psn_model = PSN(scoring=constant_score, wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert denoised.shape == data.shape

def test_transform_different_shapes():
    """Test transforming data with different shapes than training."""
    nunits, nconds, ntrials = 5, 10, 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    psn_model = PSN(wantfig=False)
    psn_model.fit(data)
    
    # Test with different number of trials
    new_data = np.random.randn(nunits, nconds, 7)
    denoised = psn_model.transform(new_data)
    assert denoised.shape == new_data.shape
    
    # Test with different number of conditions (should work)
    new_data_conds = np.random.randn(nunits, 5, ntrials)
    denoised_conds = psn_model.transform(new_data_conds)
    assert denoised_conds.shape == new_data_conds.shape
    
    # Test 2D data
    data_2d = np.random.randn(nunits, 8)
    denoised_2d = psn_model.transform(data_2d)
    assert denoised_2d.shape == data_2d.shape

def test_extreme_mag_thresholds():
    """Test extreme magnitude threshold values."""
    nunits, nconds, ntrials = 5, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with minimum valid threshold
    psn_min = PSN(cv=None, mag_threshold=0.001, wantfig=False)
    psn_min.fit(data)
    denoised_min = psn_min.transform(data)
    
    assert psn_min.denoiser_.shape == (nunits, nunits)
    assert denoised_min.shape == data.shape
    
    # Test with maximum threshold
    psn_max = PSN(cv=None, mag_threshold=1.0, wantfig=False)
    psn_max.fit(data)
    denoised_max = psn_max.transform(data)
    
    assert psn_max.denoiser_.shape == (nunits, nunits)
    assert denoised_max.shape == data.shape

def test_custom_gsn_kwargs_edge_cases():
    """Test edge cases with custom GSN kwargs."""
    nunits, nconds, ntrials = 5, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with minimal CV thresholds
    gsn_kwargs = {'cv_thresholds': [1]}
    psn_model = PSN(gsn_kwargs=gsn_kwargs, wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert denoised.shape == data.shape
    
    # Test with all possible CV thresholds
    gsn_kwargs = {'cv_thresholds': np.arange(1, nunits + 1)}
    psn_model = PSN(gsn_kwargs=gsn_kwargs, wantfig=False)
    psn_model.fit(data)
    denoised = psn_model.transform(data)
    
    assert psn_model.denoiser_.shape == (nunits, nunits)
    assert denoised.shape == data.shape


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
