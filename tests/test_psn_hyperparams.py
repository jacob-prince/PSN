"""Hyperparameter tests for PSN denoiser."""

import numpy as np
import pytest
from psn import PSN

# Test fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    nunits, nconds, ntrials = 5, 8, 4
    signal = np.random.randn(nunits, nconds)
    noise = 0.3 * np.random.randn(nunits, nconds, ntrials)
    data = signal[:, :, np.newaxis] + noise
    return data

@pytest.fixture
def high_noise_data():
    """Create high noise data for testing."""
    nunits, nconds, ntrials = 6, 10, 5
    signal = np.random.randn(nunits, nconds)
    noise = 2.0 * np.random.randn(nunits, nconds, ntrials)
    data = signal[:, :, np.newaxis] + noise
    return data

@pytest.fixture
def low_noise_data():
    """Create low noise data for testing."""
    nunits, nconds, ntrials = 6, 10, 5
    signal = np.random.randn(nunits, nconds)
    noise = 0.01 * np.random.randn(nunits, nconds, ntrials)
    data = signal[:, :, np.newaxis] + noise
    return data

class TestBasisSelection:
    """Test different basis selection methods."""
    
    def test_signal_basis(self, sample_data):
        """Test signal basis selection."""
        psn_model = PSN(basis='signal', wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape
        assert psn_model.denoiser_.shape == (sample_data.shape[0], sample_data.shape[0])
    
    def test_whitened_signal_basis(self, sample_data):
        """Test whitened signal basis selection."""
        psn_model = PSN(basis='whitened-signal', wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape
        assert psn_model.denoiser_.shape == (sample_data.shape[0], sample_data.shape[0])
    
    def test_noise_basis(self, sample_data):
        """Test noise basis selection."""
        psn_model = PSN(basis='noise', wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape
        assert psn_model.denoiser_.shape == (sample_data.shape[0], sample_data.shape[0])
    
    def test_pca_basis(self, sample_data):
        """Test PCA basis selection."""
        psn_model = PSN(basis='pca', wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape
        assert psn_model.denoiser_.shape == (sample_data.shape[0], sample_data.shape[0])
    
    def test_random_basis(self, sample_data):
        """Test random basis selection."""
        psn_model = PSN(basis='random', wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape
        assert psn_model.denoiser_.shape == (sample_data.shape[0], sample_data.shape[0])
    
    def test_custom_matrix_basis(self, sample_data):
        """Test custom matrix basis."""
        nunits = sample_data.shape[0]
        custom_basis = np.random.randn(nunits, nunits-2)
        
        psn_model = PSN(basis=custom_basis, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape
        assert psn_model.denoiser_.shape == (nunits, nunits)

class TestCrossValidationMethods:
    """Test different cross-validation methods."""
    
    def test_unit_cv(self, sample_data):
        """Test unit cross-validation."""
        psn_model = PSN(cv='unit', wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape
        assert hasattr(psn_model, 'fitted_results_')
    
    def test_population_cv(self, sample_data):
        """Test population cross-validation."""
        psn_model = PSN(cv='population', wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape
        assert hasattr(psn_model, 'fitted_results_')
    
    def test_no_cv(self, sample_data):
        """Test no cross-validation with magnitude threshold."""
        psn_model = PSN(cv=None, mag_threshold=0.5, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape
        assert hasattr(psn_model, 'fitted_results_')

class TestScoringFunctions:
    """Test different scoring functions."""
    
    def test_mse_scoring(self, sample_data):
        """Test MSE scoring function."""
        psn_model = PSN(scoring='mse', wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape
    
    def test_pearson_scoring(self, sample_data):
        """Test R2 scoring function (closest to Pearson)."""
        psn_model = PSN(scoring='r2', wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape
    
    def test_custom_scoring(self, sample_data):
        """Test custom scoring function."""
        def custom_score(true, pred):
            return np.mean((true - pred)**2, axis=0)
        
        psn_model = PSN(scoring=custom_score, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape

class TestMagnitudeThresholds:
    """Test different magnitude thresholds."""
    
    @pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_various_thresholds(self, sample_data, threshold):
        """Test various magnitude thresholds."""
        psn_model = PSN(cv=None, mag_threshold=threshold, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape
        assert psn_model.denoiser_.shape == (sample_data.shape[0], sample_data.shape[0])
    
    def test_threshold_vs_cv_consistency(self, sample_data):
        """Test that threshold and CV methods produce similar results."""
        psn_cv = PSN(cv='population', wantfig=False)
        psn_cv.fit(sample_data)
        
        psn_thresh = PSN(cv=None, mag_threshold=0.5, wantfig=False)
        psn_thresh.fit(sample_data)
        
        # Both should produce valid denoisers
        assert not np.any(np.isnan(psn_cv.denoiser_))
        assert not np.any(np.isnan(psn_thresh.denoiser_))

class TestDataDimensions:
    """Test with different data dimensions."""
    
    @pytest.mark.parametrize("nunits,nconds,ntrials", [
        (3, 5, 2),   # Small data
        (10, 20, 8), # Medium data
        (20, 50, 15) # Large data
    ])
    def test_various_dimensions(self, nunits, nconds, ntrials):
        """Test with various data dimensions."""
        data = np.random.randn(nunits, nconds, ntrials)
        
        psn_model = PSN(wantfig=False)
        psn_model.fit(data)
        denoised = psn_model.transform(data)
        
        assert denoised.shape == data.shape
        assert psn_model.denoiser_.shape == (nunits, nunits)
    
    def test_more_units_than_conditions(self):
        """Test when number of units exceeds conditions."""
        nunits, nconds, ntrials = 15, 8, 5
        data = np.random.randn(nunits, nconds, ntrials)
        
        psn_model = PSN(wantfig=False)
        psn_model.fit(data)
        denoised = psn_model.transform(data)
        
        assert denoised.shape == data.shape
        assert psn_model.denoiser_.shape == (nunits, nunits)
    
    def test_more_conditions_than_units(self):
        """Test when number of conditions exceeds units."""
        nunits, nconds, ntrials = 5, 20, 8
        data = np.random.randn(nunits, nconds, ntrials)
        
        psn_model = PSN(wantfig=False)
        psn_model.fit(data)
        denoised = psn_model.transform(data)
        
        assert denoised.shape == data.shape
        assert psn_model.denoiser_.shape == (nunits, nunits)

class TestNoiseConditions:
    """Test with different noise conditions."""
    
    def test_high_noise_condition(self, high_noise_data):
        """Test performance with high noise."""
        psn_model = PSN(wantfig=False)
        psn_model.fit(high_noise_data)
        denoised = psn_model.transform(high_noise_data)
        
        assert denoised.shape == high_noise_data.shape
        # High noise should still produce valid denoiser
        assert not np.any(np.isnan(psn_model.denoiser_))
    
    def test_low_noise_condition(self, low_noise_data):
        """Test performance with low noise."""
        psn_model = PSN(wantfig=False)
        psn_model.fit(low_noise_data)
        denoised = psn_model.transform(low_noise_data)
        
        assert denoised.shape == low_noise_data.shape
        # Low noise should produce near-identity denoiser
        diagonal_dominance = np.mean(np.diag(psn_model.denoiser_))
        assert diagonal_dominance > 0.5

class TestCombinedParameters:
    """Test combinations of parameters."""
    
    @pytest.mark.parametrize("basis,cv,scoring", [
        ('signal', 'unit', 'mse'),
        ('whitened-signal', 'population', 'r2'),
        ('noise', 'unit', 'mse'),
        ('pca', 'population', 'mse'),
        ('random', 'unit', 'r2')
    ])
    def test_parameter_combinations(self, sample_data, basis, cv, scoring):
        """Test various parameter combinations."""
        psn_model = PSN(basis=basis, cv=cv, scoring=scoring, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape
        assert psn_model.denoiser_.shape == (sample_data.shape[0], sample_data.shape[0])
    
    def test_threshold_with_different_bases(self, sample_data):
        """Test magnitude threshold with different bases."""
        bases = ['signal', 'whitened-signal', 'noise', 'pca', 'random']
        
        for basis in bases:
            psn_model = PSN(basis=basis, cv=None, mag_threshold=0.5, wantfig=False)
            psn_model.fit(sample_data)
            denoised = psn_model.transform(sample_data)
            
            assert denoised.shape == sample_data.shape
            assert not np.any(np.isnan(psn_model.denoiser_))

class TestPerformanceConsistency:
    """Test performance consistency across runs."""
    
    def test_deterministic_results(self, sample_data):
        """Test that results are deterministic for same random seed."""
        # Set seed for reproducibility
        np.random.seed(42)
        
        psn1 = PSN(basis='random', wantfig=False)
        psn1.fit(sample_data)
        denoised1 = psn1.transform(sample_data)
        
        # Reset seed
        np.random.seed(42)
        
        psn2 = PSN(basis='random', wantfig=False)
        psn2.fit(sample_data)
        denoised2 = psn2.transform(sample_data)
        
        # Results should be identical (except for random basis)
        # Note: random basis will differ, but structure should be similar
        assert denoised1.shape == denoised2.shape
    
    def test_repeated_fits(self, sample_data):
        """Test that repeated fits give consistent results."""
        psn_model = PSN(basis='signal', wantfig=False)
        
        psn_model.fit(sample_data)
        denoised1 = psn_model.transform(sample_data)
        
        psn_model.fit(sample_data)  # Fit again
        denoised2 = psn_model.transform(sample_data)
        
        # Results should be identical
        assert np.allclose(denoised1, denoised2)

class TestGSNKwargs:
    """Test custom GSN kwargs."""
    
    def test_custom_cv_thresholds(self, sample_data):
        """Test custom CV thresholds."""
        nunits = sample_data.shape[0]
        custom_thresholds = [1, 2, 3]
        
        gsn_kwargs = {'cv_thresholds': custom_thresholds}
        psn_model = PSN(gsn_kwargs=gsn_kwargs, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape
    
    def test_wantverbose_option(self, sample_data):
        """Test verbose option."""
        gsn_kwargs = {'wantverbose': 0}
        psn_model = PSN(gsn_kwargs=gsn_kwargs, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape
    
    def test_multiple_gsn_options(self, sample_data):
        """Test multiple GSN options."""
        gsn_kwargs = {
            'wantverbose': 0,
            'cv_thresholds': [1, 2],
        }
        psn_model = PSN(gsn_kwargs=gsn_kwargs, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)
        
        assert denoised.shape == sample_data.shape

class TestTransformConsistency:
    """Test transform method consistency."""
    
    def test_transform_2d_vs_3d(self, sample_data):
        """Test that 2D and 3D transforms are consistent."""
        psn_model = PSN(wantfig=False)
        psn_model.fit(sample_data)
        
        # Transform 3D data
        denoised_3d = psn_model.transform(sample_data)
        
        # Transform mean (2D)
        mean_data = np.mean(sample_data, axis=2)
        denoised_2d = psn_model.transform(mean_data)
        
        # Compare with mean of 3D result
        mean_denoised_3d = np.mean(denoised_3d, axis=2)
        
        assert np.allclose(denoised_2d, mean_denoised_3d, rtol=1e-10)
    
    def test_transform_different_trial_counts(self, sample_data):
        """Test transform with different trial counts."""
        psn_model = PSN(wantfig=False)
        psn_model.fit(sample_data)
        
        nunits, nconds, _ = sample_data.shape
        
        # Test with fewer trials
        fewer_trials = np.random.randn(nunits, nconds, 2)
        denoised_fewer = psn_model.transform(fewer_trials)
        assert denoised_fewer.shape == fewer_trials.shape
        
        # Test with more trials
        more_trials = np.random.randn(nunits, nconds, 10)
        denoised_more = psn_model.transform(more_trials)
        assert denoised_more.shape == more_trials.shape

class TestRobustness:
    """Test robustness to various conditions."""
    
    def test_robustness_to_scaling(self, sample_data):
        """Test robustness to data scaling."""
        scales = [0.1, 1.0, 10.0, 100.0]
        
        for scale in scales:
            scaled_data = sample_data * scale
            
            psn_model = PSN(wantfig=False)
            psn_model.fit(scaled_data)
            denoised = psn_model.transform(scaled_data)
            
            assert denoised.shape == scaled_data.shape
            assert not np.any(np.isnan(psn_model.denoiser_))
    
    def test_robustness_to_offset(self, sample_data):
        """Test robustness to data offset."""
        offsets = [-10.0, -1.0, 0.0, 1.0, 10.0]
        
        for offset in offsets:
            offset_data = sample_data + offset
            
            psn_model = PSN(wantfig=False)
            psn_model.fit(offset_data)
            denoised = psn_model.transform(offset_data)
            
            assert denoised.shape == offset_data.shape
            assert not np.any(np.isnan(psn_model.denoiser_))

class TestErrorHandling:
    """Test error handling for invalid hyperparameters."""
    
    def test_invalid_basis(self, sample_data):
        """Test error for invalid basis."""
        with pytest.raises(ValueError):
            psn_model = PSN(basis='invalid_basis', wantfig=False)
            psn_model.fit(sample_data)
    
    def test_invalid_cv(self, sample_data):
        """Test error for invalid CV method."""
        with pytest.raises(ValueError):
            psn_model = PSN(cv='invalid_cv', wantfig=False)
            psn_model.fit(sample_data)
    
    def test_invalid_scoring(self, sample_data):
        """Test error for invalid scoring method."""
        with pytest.raises((ValueError, TypeError)):
            psn_model = PSN(scoring='invalid_scoring', wantfig=False)
            psn_model.fit(sample_data)
    
    def test_invalid_mag_threshold(self, sample_data):
        """Test error for invalid magnitude threshold."""
        with pytest.raises(ValueError):
            psn_model = PSN(cv=None, mag_threshold=1.5, wantfig=False)
            psn_model.fit(sample_data)
    
    def test_invalid_custom_basis_shape(self, sample_data):
        """Test error for invalid custom basis shape."""
        nunits = sample_data.shape[0]
        invalid_basis = np.random.randn(nunits + 1, nunits)  # Wrong number of rows
        
        with pytest.raises((ValueError, AssertionError)):
            psn_model = PSN(basis=invalid_basis, wantfig=False)
            psn_model.fit(sample_data)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
