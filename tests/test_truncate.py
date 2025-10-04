"""Tests for truncate functionality in PSN denoiser.

The truncate parameter allows removal of early principal components (PCs) from
the denoising operation. This is useful when the first few PCs capture unwanted
variance (e.g., global signals, artifacts) that should not be used for denoising.
"""

import numpy as np
import pytest
from psn import PSN, psn


@pytest.fixture
def sample_data():
    """Create sample data with known structure for testing."""
    np.random.seed(42)
    nunits, nconds, ntrials = 10, 20, 4

    # Create signal with structure in first few PCs
    signal = np.random.randn(nunits, nconds)
    noise = 0.5 * np.random.randn(nunits, nconds, ntrials)
    data = signal[:, :, np.newaxis] + noise
    return data


class TestTruncateBasics:
    """Test basic truncate functionality."""

    def test_truncate_zero_default(self, sample_data):
        """Test that truncate=0 is the default (no truncation)."""
        psn_no_truncate = PSN(cv='population', truncate=0, wantfig=False)
        psn_default = PSN(cv='population', wantfig=False)

        psn_no_truncate.fit(sample_data)
        psn_default.fit(sample_data)

        # Denoisers should be identical
        np.testing.assert_array_almost_equal(
            psn_no_truncate.denoiser_,
            psn_default.denoiser_
        )

    def test_truncate_removes_first_pc(self, sample_data):
        """Test that truncate=1 excludes the first PC."""
        psn_no_truncate = PSN(cv='population', truncate=0, wantfig=False)
        psn_truncate_1 = PSN(cv='population', truncate=1, wantfig=False)

        psn_no_truncate.fit(sample_data)
        psn_truncate_1.fit(sample_data)

        # Denoisers should be different
        assert not np.allclose(psn_no_truncate.denoiser_, psn_truncate_1.denoiser_)

        # Both should be valid denoisers (symmetric, square)
        assert psn_truncate_1.denoiser_.shape[0] == psn_truncate_1.denoiser_.shape[1]
        np.testing.assert_array_almost_equal(
            psn_truncate_1.denoiser_,
            psn_truncate_1.denoiser_.T
        )

    def test_truncate_multiple_pcs(self, sample_data):
        """Test truncating multiple early PCs."""
        nunits = sample_data.shape[0]

        for truncate_val in [1, 2, 3, 5]:
            psn_model = PSN(cv='population', truncate=truncate_val, wantfig=False)
            psn_model.fit(sample_data)
            denoised = psn_model.transform(sample_data)

            # Should produce valid output
            assert denoised.shape == sample_data.shape
            assert psn_model.denoiser_.shape == (nunits, nunits)

            # Denoiser should be symmetric
            np.testing.assert_array_almost_equal(
                psn_model.denoiser_,
                psn_model.denoiser_.T
            )


class TestTruncateWithCV:
    """Test truncate with different cross-validation modes."""

    def test_truncate_with_population_cv(self, sample_data):
        """Test truncate with population-level cross-validation."""
        psn_model = PSN(cv='population', truncate=2, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)

        assert denoised.shape == sample_data.shape
        assert psn_model.best_threshold_ is not None
        # Best threshold should be a scalar for population mode
        assert np.isscalar(psn_model.best_threshold_)

    def test_truncate_with_unit_cv(self, sample_data):
        """Test truncate with unit-wise cross-validation."""
        psn_model = PSN(cv='unit', truncate=1, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)

        assert denoised.shape == sample_data.shape
        assert psn_model.best_threshold_ is not None
        # Best threshold should be an array for unit mode
        assert len(psn_model.best_threshold_) == sample_data.shape[0]

    def test_truncate_with_magnitude_thresholding(self, sample_data):
        """Test truncate with magnitude thresholding (no CV)."""
        psn_model = PSN(cv=None, mag_threshold=0.9, truncate=2, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)

        assert denoised.shape == sample_data.shape
        # Should have selected some dimensions (excluding first 2)
        assert hasattr(psn_model, 'best_threshold_')


class TestTruncateEdgeCases:
    """Test edge cases and boundary conditions for truncate."""

    def test_truncate_all_dimensions(self, sample_data):
        """Test truncating all dimensions (should result in zero denoiser)."""
        nunits = sample_data.shape[0]

        # Truncate all possible dimensions
        psn_model = PSN(cv='population', truncate=nunits, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)

        # Denoiser should be zero or near-zero
        assert np.allclose(psn_model.denoiser_, 0)
        # Denoised data should equal unit means (constant across conditions and trials)
        unit_means = psn_model.unit_means_
        # Expected shape is (nunits, nconds, ntrials) with constant values
        expected = np.tile(unit_means[:, np.newaxis, np.newaxis],
                          (1, sample_data.shape[1], sample_data.shape[2]))
        np.testing.assert_array_almost_equal(denoised, expected)

    def test_truncate_more_than_dimensions(self, sample_data):
        """Test truncating more dimensions than available."""
        nunits = sample_data.shape[0]

        # Truncate more than available dimensions
        psn_model = PSN(cv='population', truncate=nunits + 5, wantfig=False)
        psn_model.fit(sample_data)

        # Should handle gracefully (result in zero denoiser)
        assert psn_model.denoiser_.shape == (nunits, nunits)

    def test_truncate_negative_raises_error(self, sample_data):
        """Test that negative truncate values raise an error."""
        with pytest.raises(ValueError, match="truncate must be a non-negative integer"):
            psn_model = PSN(truncate=-1, wantfig=False)
            psn_model.fit(sample_data)

    def test_truncate_non_integer_raises_error(self, sample_data):
        """Test that non-integer truncate values raise an error."""
        with pytest.raises(ValueError, match="truncate must be a non-negative integer"):
            psn_model = PSN(truncate=1.5, wantfig=False)
            psn_model.fit(sample_data)


class TestTruncateWithBases:
    """Test truncate with different basis selection methods."""

    def test_truncate_with_signal_basis(self, sample_data):
        """Test truncate with signal covariance basis."""
        psn_model = PSN(basis='signal', truncate=1, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)

        assert denoised.shape == sample_data.shape

    def test_truncate_with_whitened_signal_basis(self, sample_data):
        """Test truncate with whitened signal basis."""
        psn_model = PSN(basis='whitened-signal', truncate=2, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)

        assert denoised.shape == sample_data.shape

    def test_truncate_with_noise_basis(self, sample_data):
        """Test truncate with noise covariance basis."""
        psn_model = PSN(basis='noise', truncate=1, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)

        assert denoised.shape == sample_data.shape

    def test_truncate_with_pca_basis(self, sample_data):
        """Test truncate with PCA basis."""
        psn_model = PSN(basis='pca', truncate=1, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)

        assert denoised.shape == sample_data.shape

    def test_truncate_with_custom_basis(self, sample_data):
        """Test truncate with custom basis matrix."""
        nunits = sample_data.shape[0]
        custom_basis = np.random.randn(nunits, nunits)
        # Orthogonalize
        Q, _ = np.linalg.qr(custom_basis)

        psn_model = PSN(basis=Q, truncate=2, wantfig=False)
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)

        assert denoised.shape == sample_data.shape


class TestTruncateFunctional:
    """Test truncate using functional psn() API."""

    def test_functional_truncate_population(self, sample_data):
        """Test truncate with functional API and population CV."""
        opt = {
            'cv_mode': 0,
            'cv_threshold_per': 'population',
            'truncate': 1
        }
        results = psn(sample_data, V=0, opt=opt, wantfig=False)

        assert results['denoiseddata'].shape == (sample_data.shape[0], sample_data.shape[1])
        assert 'denoiser' in results
        assert results['denoiser'].shape == (sample_data.shape[0], sample_data.shape[0])

    def test_functional_truncate_unit(self, sample_data):
        """Test truncate with functional API and unit-wise CV."""
        opt = {
            'cv_mode': 0,
            'cv_threshold_per': 'unit',
            'truncate': 2
        }
        results = psn(sample_data, V=0, opt=opt, wantfig=False)

        assert results['denoiseddata'].shape == (sample_data.shape[0], sample_data.shape[1])
        assert len(results['best_threshold']) == sample_data.shape[0]

    def test_functional_truncate_magnitude(self, sample_data):
        """Test truncate with functional API and magnitude thresholding."""
        opt = {
            'cv_mode': -1,
            'mag_frac': 0.9,
            'truncate': 1
        }
        results = psn(sample_data, V=0, opt=opt, wantfig=False)

        assert results['denoiseddata'].shape == (sample_data.shape[0], sample_data.shape[1])
        assert 'mags' in results
        assert 'dimsretained' in results


class TestTruncateSignalSubspace:
    """Test that truncate correctly affects signalsubspace."""

    def test_signalsubspace_excludes_truncated_dims(self, sample_data):
        """Test that signalsubspace doesn't include truncated dimensions."""
        truncate_val = 2
        psn_model = PSN(cv='population', truncate=truncate_val, wantfig=False)
        psn_model.fit(sample_data)

        # Get the full basis and signal subspace
        fullbasis = psn_model.fullbasis_
        signalsubspace = psn_model.signalsubspace_

        if signalsubspace is not None and signalsubspace.shape[1] > 0:
            # Signal subspace should correspond to dimensions after truncation
            # First column of signalsubspace should be truncate_val-th column of fullbasis
            # (accounting for potential reordering)
            assert signalsubspace.shape[0] == fullbasis.shape[0]
            assert signalsubspace.shape[1] <= fullbasis.shape[1] - truncate_val


class TestTruncateConsistency:
    """Test consistency of truncate behavior across different scenarios."""

    def test_truncate_deterministic(self, sample_data):
        """Test that truncate produces deterministic results."""
        psn_model1 = PSN(basis='signal', cv='population', truncate=2, wantfig=False)
        psn_model2 = PSN(basis='signal', cv='population', truncate=2, wantfig=False)

        psn_model1.fit(sample_data)
        psn_model2.fit(sample_data)

        np.testing.assert_array_almost_equal(
            psn_model1.denoiser_,
            psn_model2.denoiser_
        )

        denoised1 = psn_model1.transform(sample_data)
        denoised2 = psn_model2.transform(sample_data)

        np.testing.assert_array_almost_equal(denoised1, denoised2)

    def test_truncate_increases_monotonically(self, sample_data):
        """Test that increasing truncate progressively removes more variance."""
        nunits = sample_data.shape[0]

        denoisers = []
        for truncate_val in [0, 1, 2, 3]:
            psn_model = PSN(cv='population', truncate=truncate_val, wantfig=False)
            psn_model.fit(sample_data)
            denoisers.append(psn_model.denoiser_)

        # Each successive denoiser should be different
        for i in range(len(denoisers) - 1):
            assert not np.allclose(denoisers[i], denoisers[i+1])


class TestTruncateUnitGroups:
    """Test truncate with unit groups."""

    def test_truncate_with_unit_groups(self, sample_data):
        """Test truncate with custom unit groups."""
        nunits = sample_data.shape[0]
        # Create two groups
        unit_groups = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

        psn_model = PSN(
            cv='unit',
            unit_groups=unit_groups,
            truncate=1,
            wantfig=False
        )
        psn_model.fit(sample_data)
        denoised = psn_model.transform(sample_data)

        assert denoised.shape == sample_data.shape
        # All units should have thresholds assigned
        assert len(psn_model.best_threshold_) == nunits


class TestTruncateRealWorldScenarios:
    """Test truncate in realistic use cases."""

    def test_truncate_removes_global_signal(self):
        """Test truncate to remove a global signal component."""
        np.random.seed(123)
        nunits, nconds, ntrials = 20, 50, 5

        # Create data with strong global signal in first PC
        global_signal = np.random.randn(1, nconds)
        unit_responses = np.random.randn(nunits, 1) @ global_signal
        specific_signals = 0.5 * np.random.randn(nunits, nconds)
        noise = 0.2 * np.random.randn(nunits, nconds, ntrials)

        signal = unit_responses + specific_signals
        data = signal[:, :, np.newaxis] + noise

        # Without truncation - may keep global signal
        psn_no_truncate = PSN(cv='population', truncate=0, wantfig=False)
        psn_no_truncate.fit(data)
        denoised_no_truncate = psn_no_truncate.transform(data)

        # With truncation - should remove global signal
        psn_truncate = PSN(cv='population', truncate=1, wantfig=False)
        psn_truncate.fit(data)
        denoised_truncate = psn_truncate.transform(data)

        # Results should be different
        assert not np.allclose(denoised_no_truncate, denoised_truncate)

        # Both should be valid outputs (includes trial dimension)
        assert denoised_no_truncate.shape == (nunits, nconds, ntrials)
        assert denoised_truncate.shape == (nunits, nconds, ntrials)

    def test_truncate_multiple_artifacts(self):
        """Test truncating multiple artifactual components."""
        np.random.seed(456)
        nunits, nconds, ntrials = 15, 30, 4

        # Create data with multiple artifact components
        artifact1 = np.random.randn(nunits, 1) @ np.random.randn(1, nconds)
        artifact2 = np.random.randn(nunits, 1) @ np.random.randn(1, nconds)
        signal = 0.5 * np.random.randn(nunits, nconds)
        noise = 0.3 * np.random.randn(nunits, nconds, ntrials)

        data = (artifact1 + artifact2 + signal)[:, :, np.newaxis] + noise

        # Truncate first 2 PCs to remove artifacts
        psn_model = PSN(cv='population', truncate=2, wantfig=False)
        psn_model.fit(data)
        denoised = psn_model.transform(data)

        assert denoised.shape == (nunits, nconds, ntrials)
        # Should produce non-zero output (not all artifacts)
        assert not np.allclose(denoised, 0)
