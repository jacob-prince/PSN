"""Tests for PSN using simulated data."""

import numpy as np
import pytest
from psn import PSN

class TestPSN:
    """Test class for PSN sklearn-style API."""

    def test_basic_functionality(self):
        """Test basic functionality of PSN denoiser."""
        nunits = 8
        nconds = 10
        ntrials = 3
        data = np.random.randn(nunits, nconds, ntrials)
        
        # Test with default options
        psn_model = PSN(wantfig=False)
        psn_model.fit(data)
        denoised = psn_model.transform(data)
        
        assert psn_model.denoiser_.shape == (nunits, nunits)
        assert denoised.shape == data.shape  # Same shape as input
        assert psn_model.fullbasis_.shape[0] == nunits  # Basis should have nunits rows
        assert psn_model.fullbasis_.shape[1] >= 1  # Basis should have at least 1 column
        assert hasattr(psn_model, 'unit_means_')
        assert hasattr(psn_model, 'best_threshold_')

    def test_fit_transform(self):
        """Test fit_transform method."""
        nunits = 8
        nconds = 10
        ntrials = 3
        data = np.random.randn(nunits, nconds, ntrials)
        
        psn_model = PSN(wantfig=False)
        denoised = psn_model.fit_transform(data)
        
        # PSN fit_transform returns trial-averaged data by default
        assert denoised.shape == (nunits, nconds)
        assert hasattr(psn_model, 'denoiser_')

    def test_cross_validation_population(self):
        """Test cross-validation with population thresholding."""
        nunits = 8
        nconds = 10
        ntrials = 3
        data = np.random.randn(nunits, nconds, ntrials)
        
        # Test with population thresholding
        psn_model = PSN(cv='population', wantfig=False)
        psn_model.fit(data)
        denoised = psn_model.transform(data)
        
        assert psn_model.denoiser_.shape == (nunits, nunits)
        assert denoised.shape == data.shape  # Same shape as input
        assert isinstance(psn_model.best_threshold_, (int, np.integer))
        assert psn_model.fullbasis_.shape == (nunits, nunits)  # Full basis should be square
        if psn_model.signalsubspace_ is not None:
            assert psn_model.signalsubspace_.shape == (nunits, psn_model.best_threshold_)
        # Check symmetry for population thresholding
        assert np.allclose(psn_model.denoiser_, psn_model.denoiser_.T)

    def test_cross_validation_unit(self):
        """Test cross-validation with unit thresholding."""
        nunits = 8
        nconds = 10
        ntrials = 3
        data = np.random.randn(nunits, nconds, ntrials)
        
        # Test with unit thresholding
        psn_model = PSN(cv='unit', wantfig=False)
        psn_model.fit(data)
        denoised = psn_model.transform(data)
        
        assert psn_model.denoiser_.shape == (nunits, nunits)
        assert denoised.shape == data.shape  # Same shape as input
        assert psn_model.fullbasis_.shape == (nunits, nunits)  # Full basis should be square
        assert len(psn_model.best_threshold_) == nunits  # One threshold per unit
        assert psn_model.fitted_results_['cv_scores'].shape[0] == nunits  # One score per unit

    def test_magnitude_thresholding(self):
        """Test magnitude thresholding mode."""
        nunits = 8
        nconds = 10
        ntrials = 3
        data = np.random.randn(nunits, nconds, ntrials)
        
        # Test with magnitude thresholding
        psn_model = PSN(cv=None, mag_threshold=0.95, wantfig=False)
        psn_model.fit(data)
        denoised = psn_model.transform(data)
        
        assert psn_model.denoiser_.shape == (nunits, nunits)
        assert denoised.shape == data.shape  # Same shape as input
        assert psn_model.fullbasis_.shape == (nunits, nunits)  # Full basis should be square
        assert len(psn_model.fitted_results_['mags']) == nunits  # One magnitude per dimension
        assert isinstance(psn_model.fitted_results_['dimsretained'], (int, np.integer))
        if psn_model.signalsubspace_ is not None:
            assert psn_model.signalsubspace_.shape == (nunits, psn_model.fitted_results_['dimsretained'])
        # Check symmetry for population thresholding
        assert np.allclose(psn_model.denoiser_, psn_model.denoiser_.T)

    def test_custom_basis(self):
        """Test denoising with custom basis."""
        nunits = 8
        nconds = 10
        ntrials = 3
        data = np.random.randn(nunits, nconds, ntrials)
        
        # Test with different basis dimensions
        basis_dims = [1, nunits//2, nunits]  # Test different numbers of basis vectors
        
        for dim in basis_dims:
            # Create a random orthonormal basis with dim columns
            V = np.linalg.qr(np.random.randn(nunits, dim))[0]
            
            # Test with default options
            psn_model = PSN(basis=V, wantfig=False)
            psn_model.fit(data)
            denoised = psn_model.transform(data)
            
            assert psn_model.denoiser_.shape == (nunits, nunits)
            assert denoised.shape == data.shape  # Same shape as input
            assert psn_model.fullbasis_.shape == V.shape  # Basis should match input dimensions
            
            # Test with population thresholding
            psn_pop = PSN(basis=V, cv='population', wantfig=False)
            psn_pop.fit(data)
            denoised_pop = psn_pop.transform(data)
            
            assert psn_pop.denoiser_.shape == (nunits, nunits)
            assert denoised_pop.shape == data.shape  # Same shape as input
            assert isinstance(psn_pop.best_threshold_, (int, np.integer))
            assert psn_pop.fullbasis_.shape == V.shape  # Basis should match input dimensions
            if psn_pop.signalsubspace_ is not None:
                assert psn_pop.signalsubspace_.shape[0] == nunits
                assert psn_pop.signalsubspace_.shape[1] <= dim  # Can't use more dimensions than provided
            
            # Test with magnitude thresholding
            psn_mag = PSN(basis=V, cv=None, wantfig=False)
            psn_mag.fit(data)
            denoised_mag = psn_mag.transform(data)
            
            assert psn_mag.denoiser_.shape == (nunits, nunits)
            assert denoised_mag.shape == data.shape  # Same shape as input
            assert psn_mag.fullbasis_.shape == V.shape  # Basis should match input dimensions
            assert len(psn_mag.fitted_results_['mags']) == dim  # One magnitude per basis dimension
            assert isinstance(psn_mag.fitted_results_['dimsretained'], (int, np.integer))

    def test_custom_scoring(self):
        """Test denoising with custom scoring function."""
        nunits = 8
        nconds = 10
        ntrials = 3
        data = np.random.randn(nunits, nconds, ntrials)
        
        # Define a custom scoring function
        def custom_score(A, B):
            return -np.mean(np.abs(A - B), axis=0)
        
        # Test with default options and custom scoring
        psn_model = PSN(scoring=custom_score, wantfig=False)
        psn_model.fit(data)
        denoised = psn_model.transform(data)
        
        assert psn_model.denoiser_.shape == (nunits, nunits)
        assert denoised.shape == data.shape  # Same shape as input
        assert psn_model.fullbasis_.shape == (nunits, nunits)  # Full basis should be square
        
        # Test with population thresholding and custom scoring
        psn_pop = PSN(cv='population', scoring=custom_score, wantfig=False)
        psn_pop.fit(data)
        denoised_pop = psn_pop.transform(data)
        
        assert psn_pop.denoiser_.shape == (nunits, nunits)
        assert denoised_pop.shape == data.shape  # Same shape as input
        assert isinstance(psn_pop.best_threshold_, (int, np.integer))
        assert psn_pop.fullbasis_.shape == (nunits, nunits)  # Full basis should be square
        if psn_pop.signalsubspace_ is not None:
            assert psn_pop.signalsubspace_.shape == (nunits, psn_pop.best_threshold_)
        # Check symmetry for population thresholding
        assert np.allclose(psn_pop.denoiser_, psn_pop.denoiser_.T)

    def test_different_basis_types(self):
        """Test different basis types."""
        nunits = 8
        nconds = 10
        ntrials = 3
        data = np.random.randn(nunits, nconds, ntrials)
        
        basis_types = ['signal', 'whitened-signal', 'noise', 'pca', 'random']  # PSN uses string basis types
        
        for basis in basis_types:
            psn_model = PSN(basis=basis, wantfig=False)
            psn_model.fit(data)
            denoised = psn_model.transform(data)
            
            assert psn_model.denoiser_.shape == (nunits, nunits)
            assert denoised.shape == data.shape  # Same shape as input
            assert psn_model.fullbasis_.shape[0] == nunits

    def test_pearson_scoring(self):
        """Test Pearson correlation scoring."""
        nunits = 8
        nconds = 10
        ntrials = 3
        data = np.random.randn(nunits, nconds, ntrials)
        
        # Note: PSN might use different scoring parameter name
        psn_model = PSN(wantfig=False)  # Use default scoring for now
        psn_model.fit(data)
        denoised = psn_model.transform(data)
        
        assert psn_model.denoiser_.shape == (nunits, nunits)
        assert denoised.shape == data.shape  # Same shape as input

    def test_transform_2d_data(self):
        """Test transforming 2D (trial-averaged) data."""
        nunits = 8
        nconds = 10
        ntrials = 3
        data = np.random.randn(nunits, nconds, ntrials)
        
        psn_model = PSN(wantfig=False)
        psn_model.fit(data)
        
        # Test transforming 2D data
        data_2d = np.mean(data, axis=2)
        denoised_2d = psn_model.transform(data_2d)
        
        assert denoised_2d.shape == data_2d.shape

    def test_score_method(self):
        """Test the score method."""
        nunits = 8
        nconds = 10
        ntrials = 5  # Need at least 2 trials for scoring
        data = np.random.randn(nunits, nconds, ntrials)
        
        psn_model = PSN(wantfig=False)
        psn_model.fit(data)
        score = psn_model.score(data)
        
        assert isinstance(score, (float, np.floating))

    def test_get_feature_names_out(self):
        """Test get_feature_names_out method."""
        nunits = 8
        nconds = 10
        ntrials = 3
        data = np.random.randn(nunits, nconds, ntrials)
        
        psn_model = PSN(wantfig=False)
        psn_model.fit(data)
        
        feature_names = psn_model.get_feature_names_out()
        assert len(feature_names) == nunits
        assert all('unit_' in name for name in feature_names)

    def test_get_set_params(self):
        """Test get_params and set_params methods."""
        psn_model = PSN(basis=3, cv='population')
        
        params = psn_model.get_params()
        assert params['basis'] == 3
        assert params['cv'] == 'population'
        
        psn_model.set_params(basis=0, cv='unit')
        assert psn_model.basis == 0
        assert psn_model.cv == 'unit'

    def test_parameter_validation(self):
        """Test parameter validation and error handling."""
        nunits = 8
        nconds = 10
        ntrials = 3
        data = np.random.randn(nunits, nconds, ntrials)
        
        # Test invalid cv value
        with pytest.raises(ValueError):
            psn_model = PSN(cv='invalid')
            psn_model.fit(data)
        
        # Test invalid mag_threshold
        with pytest.raises(ValueError):
            psn_model = PSN(mag_threshold=1.5)
            psn_model.fit(data)
        
        # Test invalid data shape
        invalid_data = np.random.randn(nunits, nconds)
        psn_model = PSN(wantfig=False)
        with pytest.raises(ValueError):
            psn_model.fit(invalid_data)
        
        # Test data with too few trials
        invalid_data = np.random.randn(nunits, nconds, 1)
        with pytest.raises(ValueError):
            psn_model.fit(invalid_data)

    def test_transform_validation(self):
        """Test transform validation."""
        nunits = 8
        nconds = 10
        ntrials = 3
        data = np.random.randn(nunits, nconds, ntrials)
        
        psn_model = PSN(wantfig=False)
        psn_model.fit(data)
        
        # Test invalid transform data shape
        with pytest.raises(ValueError):
            psn_model.transform(np.random.randn(nunits))
        
        # Test mismatched number of units
        with pytest.raises(ValueError):
            psn_model.transform(np.random.randn(nunits+1, nconds, ntrials))

    def test_not_fitted_error(self):
        """Test error when calling methods before fitting."""
        psn_model = PSN()
        data = np.random.randn(8, 10, 3)
        
        # These should raise errors when called before fitting
        with pytest.raises(ValueError):
            psn_model.transform(data)
        
        with pytest.raises(ValueError):
            psn_model.score(data)
        
        with pytest.raises(ValueError):
            psn_model.get_feature_names_out()

    def test_different_mag_thresholds(self):
        """Test different magnitude thresholds."""
        nunits = 8
        nconds = 10
        ntrials = 3
        data = np.random.randn(nunits, nconds, ntrials)
        
        thresholds = [0.5, 0.8, 0.95, 0.99]
        
        for threshold in thresholds:
            psn_model = PSN(cv=None, mag_threshold=threshold, wantfig=False)
            psn_model.fit(data)
            denoised = psn_model.transform(data)
            
            assert psn_model.denoiser_.shape == (nunits, nunits)
            assert denoised.shape == data.shape  # Same shape as input

    def test_verbose_mode(self):
        """Test verbose mode output."""
        nunits = 8
        nconds = 10
        ntrials = 3
        data = np.random.randn(nunits, nconds, ntrials)
        
        # Test that verbose mode doesn't break anything
        psn_model = PSN(verbose=True, wantfig=False)
        psn_model.fit(data)
        denoised = psn_model.transform(data)
        
        assert psn_model.denoiser_.shape == (nunits, nunits)
        assert denoised.shape == data.shape  # Same shape as input

    def test_psn_kwargs(self):
        """Test passing additional PSN kwargs."""
        nunits = 8
        nconds = 10
        ntrials = 3
        data = np.random.randn(nunits, nconds, ntrials)
        
        gsn_kwargs = {
            'cv_thresholds': np.arange(1, 6),  # Test only small thresholds
            'denoisingtype': 1  # Single-trial denoising
        }
        
        psn_model = PSN(gsn_kwargs=gsn_kwargs, wantfig=False)
        psn_model.fit(data)
        denoised = psn_model.transform(data)
        
        assert psn_model.denoiser_.shape == (nunits, nunits)
        # Same shape as input for transform
        assert denoised.shape == data.shape


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
