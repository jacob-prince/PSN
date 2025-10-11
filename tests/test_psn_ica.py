"""Tests for ICA basis option in PSN."""

import numpy as np
import pytest
from psn import psn, PSN
from psn.simulate import generate_data


class TestICABasis:
    """Test ICA basis functionality."""

    def test_ica_basic_functionality(self):
        """Test that ICA basis runs without errors."""
        # Generate test data
        data, _, _ = generate_data(nvox=20, ncond=50, ntrial=3, random_seed=42, verbose=False)

        # Run PSN with ICA basis
        opt = {'cv_mode': 0, 'cv_threshold_per': 'population'}
        results = psn(data, V=5, opt=opt, wantfig=False)

        # Check that results have expected fields
        assert 'denoiser' in results
        assert 'best_threshold' in results
        assert 'fullbasis' in results
        assert 'denoiseddata' in results

        # Check shapes
        assert results['denoiser'].shape == (20, 20)
        assert results['fullbasis'].shape == (20, 20)
        assert results['denoiseddata'].shape == (20, 50)

        # Check basis is orthonormal
        basis = results['fullbasis']
        gram = basis.T @ basis
        assert np.allclose(gram, np.eye(20), atol=1e-10)

    def test_ica_sklearn_api(self):
        """Test ICA basis with sklearn API."""
        # Generate test data
        data, _, _ = generate_data(nvox=15, ncond=40, ntrial=3, random_seed=42, verbose=False)

        # Test with sklearn API
        denoiser = PSN(basis='ica', cv='population', verbose=False, wantfig=False)
        denoiser.fit(data)
        denoised = denoiser.transform(data)

        # Check outputs - transform preserves input shape
        assert denoised.shape == (15, 40, 3)
        assert hasattr(denoiser, 'denoiser_')
        assert hasattr(denoiser, 'best_threshold_')
        assert hasattr(denoiser, 'fullbasis_')

        # Check basis is orthonormal
        basis = denoiser.fullbasis_
        gram = basis.T @ basis
        assert np.allclose(gram, np.eye(15), atol=1e-10)

    def test_ica_magnitude_thresholding(self):
        """Test ICA with magnitude thresholding."""
        # Generate test data
        data, _, _ = generate_data(nvox=20, ncond=50, ntrial=3, random_seed=42, verbose=False)

        # Run PSN with ICA basis and magnitude thresholding
        opt = {'cv_mode': -1, 'mag_frac': 0.9}
        results = psn(data, V=5, opt=opt, wantfig=False)

        # Check results
        assert 'mags' in results
        assert 'dimsretained' in results
        assert results['denoiser'].shape == (20, 20)

    def test_ica_unit_wise_cv(self):
        """Test ICA with unit-wise cross-validation."""
        # Generate test data
        data, _, _ = generate_data(nvox=15, ncond=30, ntrial=3, random_seed=42, verbose=False)

        # Run PSN with ICA basis and unit-wise CV
        opt = {'cv_mode': 0, 'cv_threshold_per': 'unit'}
        results = psn(data, V=5, opt=opt, wantfig=False)

        # Check that we get unit-wise thresholds
        assert 'best_threshold' in results
        assert len(results['best_threshold']) == 15

    def test_ica_comparison_with_pca(self):
        """Compare ICA and PCA basis qualitatively."""
        # Generate test data
        data, _, _ = generate_data(nvox=20, ncond=50, ntrial=3, random_seed=42, verbose=False)

        # Run with PCA
        opt = {'cv_mode': 0, 'cv_threshold_per': 'population'}
        results_pca = psn(data, V=3, opt=opt, wantfig=False)

        # Run with ICA
        results_ica = psn(data, V=5, opt=opt, wantfig=False)

        # Both should produce valid results
        assert results_pca['denoiser'].shape == results_ica['denoiser'].shape
        assert results_pca['fullbasis'].shape == results_ica['fullbasis'].shape

        # Bases should be different (not equal)
        basis_similarity = np.abs(results_pca['fullbasis'].T @ results_ica['fullbasis'])
        # If bases were identical, diagonal would be all 1s
        # They should be different
        assert not np.allclose(basis_similarity, np.eye(20), atol=0.1)

    def test_ica_small_dataset(self):
        """Test ICA with small dataset (edge case)."""
        # Generate small dataset
        data, _, _ = generate_data(nvox=5, ncond=10, ntrial=2, random_seed=42, verbose=False)

        # Should still work
        opt = {'cv_mode': 0, 'cv_threshold_per': 'population'}
        results = psn(data, V=5, opt=opt, wantfig=False)

        assert results['denoiser'].shape == (5, 5)
        assert results['fullbasis'].shape == (5, 5)

    def test_ica_with_truncate(self):
        """Test ICA basis with truncate option."""
        # Generate test data
        data, _, _ = generate_data(nvox=15, ncond=40, ntrial=3, random_seed=42, verbose=False)

        # Run with truncate=2 (remove first 2 PCs)
        opt = {'cv_mode': 0, 'cv_threshold_per': 'population', 'truncate': 2}
        results = psn(data, V=5, opt=opt, wantfig=False)

        # Check that results are valid
        assert 'best_threshold' in results
        assert 'signalsubspace' in results

        # If best_threshold > 0, signal subspace should not include first 2 dims
        if results['best_threshold'] > 0:
            # Signal subspace should have fewer columns than best_threshold
            # since we truncated 2
            assert results['signalsubspace'].shape[0] == 15


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
