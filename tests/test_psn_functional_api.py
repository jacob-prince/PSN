"""Comprehensive tests for PSN functional API - all hyperparameters.

Tests the current psn() functional API with all combinations of:
- basis: 'signal', 'difference', 'pca', 'noise', 'random', custom matrix
- criterion: 'prediction', 'variance', 'variance_eigenvalues'
- threshold_method: 'global', 'hybrid', 'unit'
- Other options: basis_ordering, variance_threshold, allowable_thresholds, unit_groups, gsn_args
"""

import numpy as np
import pytest
from psn import psn


# Test fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing: 10 units, 20 conditions, 5 trials."""
    np.random.seed(42)
    nunits, nconds, ntrials = 10, 20, 5
    signal = np.random.randn(nunits, nconds)
    noise = 0.3 * np.random.randn(nunits, nconds, ntrials)
    data = signal[:, :, np.newaxis] + noise
    return data


@pytest.fixture
def small_data():
    """Create small data for quick tests: 5 units, 10 conditions, 3 trials."""
    np.random.seed(123)
    nunits, nconds, ntrials = 5, 10, 3
    signal = np.random.randn(nunits, nconds)
    noise = 0.5 * np.random.randn(nunits, nconds, ntrials)
    data = signal[:, :, np.newaxis] + noise
    return data


@pytest.fixture
def large_data():
    """Create larger data: 20 units, 50 conditions, 8 trials."""
    np.random.seed(456)
    nunits, nconds, ntrials = 20, 50, 8
    signal = np.random.randn(nunits, nconds)
    noise = 0.4 * np.random.randn(nunits, nconds, ntrials)
    data = signal[:, :, np.newaxis] + noise
    return data


# ============================================================================
# Test Basis Types
# ============================================================================

class TestBasisTypes:
    """Test all basis types."""

    def test_basis_signal(self, sample_data):
        """Test signal basis (default and recommended)."""
        results = psn(sample_data, {'basis': 'signal', 'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['denoiseddata'].shape == sample_data.shape[:2]
        assert 'denoiser' in results
        assert results['denoiser'].shape == (sample_data.shape[0], sample_data.shape[0])
        assert results['opt_used']['basis'] == 'signal'

    def test_basis_difference(self, sample_data):
        """Test difference basis (aggressive mode)."""
        results = psn(sample_data, {'basis': 'difference', 'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['denoiseddata'].shape == sample_data.shape[:2]
        assert results['opt_used']['basis'] == 'difference'
        # Difference basis should produce a valid denoiser
        assert not np.any(np.isnan(results['denoiser']))

    def test_basis_pca(self, sample_data):
        """Test PCA basis (trial-averaged covariance eigenvectors)."""
        results = psn(sample_data, {'basis': 'pca', 'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['denoiseddata'].shape == sample_data.shape[:2]
        assert results['opt_used']['basis'] == 'pca'

    def test_basis_noise(self, sample_data):
        """Test noise basis (not recommended)."""
        results = psn(sample_data, {'basis': 'noise', 'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['denoiseddata'].shape == sample_data.shape[:2]
        assert results['opt_used']['basis'] == 'noise'

    def test_basis_random(self, sample_data):
        """Test random basis (not recommended)."""
        results = psn(sample_data, {'basis': 'random', 'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['denoiseddata'].shape == sample_data.shape[:2]
        assert results['opt_used']['basis'] == 'random'

    def test_basis_custom_matrix(self, sample_data):
        """Test custom matrix basis."""
        nunits = sample_data.shape[0]
        # Create a random orthonormal basis with fewer dimensions
        D = nunits - 2
        custom_basis = np.linalg.qr(np.random.randn(nunits, D))[0]

        results = psn(sample_data, {'basis': custom_basis, 'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['denoiseddata'].shape == sample_data.shape[:2]
        assert 'basis_viz' in results
        assert results['basis_viz'].shape == (nunits, D)

    def test_basis_custom_full_rank(self, small_data):
        """Test custom matrix basis with full rank."""
        nunits = small_data.shape[0]
        custom_basis = np.linalg.qr(np.random.randn(nunits, nunits))[0]

        results = psn(small_data, {'basis': custom_basis, 'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['basis_viz'].shape == (nunits, nunits)


# ============================================================================
# Test Criterion Types
# ============================================================================

class TestCriterionTypes:
    """Test all criterion types."""

    def test_criterion_prediction(self, sample_data):
        """Test prediction criterion (default, maximizes out-of-sample generalization)."""
        results = psn(sample_data, {'criterion': 'prediction', 'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['opt_used']['criterion'] == 'prediction'
        assert 'objective' in results
        # Objective should be computed for prediction criterion
        assert results['objective'] is not None

    def test_criterion_variance(self, sample_data):
        """Test variance criterion (retain fraction of signal variance)."""
        results = psn(sample_data, {
            'criterion': 'variance',
            'variance_threshold': 0.95,
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert results['opt_used']['criterion'] == 'variance'
        assert results['opt_used']['variance_threshold'] == 0.95

    def test_criterion_variance_eigenvalues(self, sample_data):
        """Test variance_eigenvalues criterion (retain fraction of eigenvalues)."""
        results = psn(sample_data, {
            'basis': 'signal',
            'criterion': 'variance_eigenvalues',
            'threshold_method': 'global',  # Required for variance_eigenvalues
            'variance_threshold': 0.90,
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert results['opt_used']['criterion'] == 'variance_eigenvalues'
        assert results['opt_used']['variance_threshold'] == 0.90

    def test_criterion_variance_thresholds(self, sample_data):
        """Test different variance thresholds."""
        thresholds = [0.80, 0.90, 0.95, 0.99]

        for thresh in thresholds:
            results = psn(sample_data, {
                'criterion': 'variance',
                'variance_threshold': thresh,
                'wantfig': False,
                'wantverbose': False
            })

            assert 'denoiseddata' in results
            assert results['opt_used']['variance_threshold'] == thresh


# ============================================================================
# Test Threshold Methods
# ============================================================================

class TestThresholdMethods:
    """Test all threshold methods."""

    def test_threshold_global(self, sample_data):
        """Test global threshold (single threshold for all units)."""
        results = psn(sample_data, {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert results['opt_used']['threshold_method'] == 'global'
        assert 'best_threshold' in results
        # Global threshold: all units should have same threshold
        assert len(np.unique(results['best_threshold'])) == 1

    def test_threshold_hybrid(self, sample_data):
        """Test hybrid threshold (global ordering, unit-specific thresholds)."""
        results = psn(sample_data, {
            'threshold_method': 'hybrid',
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert results['opt_used']['threshold_method'] == 'hybrid'
        assert 'best_threshold' in results
        # Hybrid: units can have different thresholds
        assert len(results['best_threshold']) == sample_data.shape[0]

    def test_threshold_unit(self, sample_data):
        """Test unit-specific threshold (unit-specific ordering and thresholds)."""
        results = psn(sample_data, {
            'threshold_method': 'unit',
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert results['opt_used']['threshold_method'] == 'unit'
        assert 'best_threshold' in results
        # Unit-specific: each unit has its own threshold
        assert len(results['best_threshold']) == sample_data.shape[0]


# ============================================================================
# Test Basis Ordering
# ============================================================================

class TestBasisOrdering:
    """Test basis ordering options."""

    def test_basis_ordering_eigenvalues(self, sample_data):
        """Test eigenvalue-based ordering (default when available)."""
        results = psn(sample_data, {
            'basis': 'signal',
            'basis_ordering': 'eigenvalues',
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert results['opt_used']['basis_ordering'] == 'eigenvalues'

    def test_basis_ordering_signalvariance(self, sample_data):
        """Test signal variance-based ordering."""
        results = psn(sample_data, {
            'basis': 'signal',
            'basis_ordering': 'signalvariance',
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert results['opt_used']['basis_ordering'] == 'signalvariance'
        assert 'signalvar' in results

    def test_basis_ordering_random_basis(self, sample_data):
        """Test basis ordering with random basis."""
        results = psn(sample_data, {
            'basis': 'random',
            'basis_ordering': 'signalvariance',  # Use signalvariance explicitly
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        # Random basis uses signalvariance ordering
        assert results['opt_used']['basis_ordering'] == 'signalvariance'


# ============================================================================
# Test Allowable Thresholds
# ============================================================================

class TestAllowableThresholds:
    """Test allowable_thresholds constraint."""

    def test_single_threshold(self, sample_data):
        """Test forcing a specific threshold."""
        forced_threshold = 5
        results = psn(sample_data, {
            'allowable_thresholds': [forced_threshold],
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert 'best_threshold' in results
        # Should use exactly the forced threshold
        assert np.all(results['best_threshold'] == forced_threshold)

    def test_multiple_allowable_thresholds(self, sample_data):
        """Test constraining to specific threshold options."""
        allowed = [2, 5, 8]
        results = psn(sample_data, {
            'allowable_thresholds': allowed,
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        # Best threshold should be one of the allowed values
        best_t = results['best_threshold']
        if np.isscalar(best_t):
            assert best_t in allowed
        else:
            assert best_t[0] in allowed

    def test_allowable_thresholds_hybrid(self, small_data):
        """Test allowable thresholds with hybrid method."""
        allowed = [1, 3, 5]
        results = psn(small_data, {
            'allowable_thresholds': allowed,
            'threshold_method': 'hybrid',
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        # All unit thresholds should be from allowed set
        assert all(t in allowed for t in results['best_threshold'])


# ============================================================================
# Test Unit Groups
# ============================================================================

class TestUnitGroups:
    """Test unit_groups option for shared thresholds."""

    def test_unit_groups_hybrid(self, sample_data):
        """Test unit groups with hybrid threshold method."""
        nunits = sample_data.shape[0]
        # Group units: [0,1] share, [2,3,4] share, rest are individual
        unit_groups = np.array([0, 0, 1, 1, 1, 2, 3, 4, 5, 6])

        results = psn(sample_data, {
            'threshold_method': 'hybrid',
            'unit_groups': unit_groups,
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        # Units in the same group should have the same threshold
        assert results['best_threshold'][0] == results['best_threshold'][1]
        assert results['best_threshold'][2] == results['best_threshold'][3] == results['best_threshold'][4]

    def test_unit_groups_unit_method(self, sample_data):
        """Test unit groups with unit-specific threshold method."""
        nunits = sample_data.shape[0]
        # Create two groups
        unit_groups = np.array([0] * 5 + [1] * 5)

        results = psn(sample_data, {
            'threshold_method': 'unit',
            'unit_groups': unit_groups,
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        # Units in the same group should have the same threshold
        assert np.all(results['best_threshold'][:5] == results['best_threshold'][0])
        assert np.all(results['best_threshold'][5:] == results['best_threshold'][5])

    def test_unit_groups_ignored_global(self, sample_data):
        """Test that unit_groups is ignored for global method."""
        unit_groups = np.arange(sample_data.shape[0])

        results = psn(sample_data, {
            'threshold_method': 'global',
            'unit_groups': unit_groups,
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        # Global method: all units get same threshold regardless of unit_groups
        assert len(np.unique(results['best_threshold'])) == 1


# ============================================================================
# Test GSN Arguments
# ============================================================================

class TestGSNArguments:
    """Test gsn_args options."""

    def test_gsn_wantverbose(self, small_data):
        """Test GSN verbose option."""
        results = psn(small_data, {
            'gsn_args': {'wantverbose': False},
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results

    def test_gsn_wantshrinkage(self, small_data):
        """Test GSN shrinkage option."""
        # Test with shrinkage
        results_shrink = psn(small_data, {
            'gsn_args': {'wantshrinkage': True},
            'wantfig': False,
            'wantverbose': False
        })

        # Test without shrinkage
        results_no_shrink = psn(small_data, {
            'gsn_args': {'wantshrinkage': False},
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results_shrink
        assert 'denoiseddata' in results_no_shrink
        # Results may differ due to shrinkage

    def test_gsn_random_seed(self, small_data):
        """Test GSN random seed for reproducibility."""
        results1 = psn(small_data, {
            'basis': 'random',
            'gsn_args': {'random_seed': 42},
            'wantfig': False,
            'wantverbose': False
        })

        results2 = psn(small_data, {
            'basis': 'random',
            'gsn_args': {'random_seed': 42},
            'wantfig': False,
            'wantverbose': False
        })

        # Results should be identical with same seed
        assert np.allclose(results1['denoiseddata'], results2['denoiseddata'])


# ============================================================================
# Test Preset Modes
# ============================================================================

class TestPresetModes:
    """Test preset modes: conservative, standard, aggressive."""

    def test_mode_conservative(self, sample_data):
        """Test conservative mode."""
        results = psn(sample_data, 'conservative', {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['opt_used']['basis'] == 'signal'
        assert results['opt_used']['criterion'] == 'variance'
        assert results['opt_used']['threshold_method'] == 'global'

    def test_mode_standard(self, sample_data):
        """Test standard mode (default)."""
        results = psn(sample_data, 'standard', {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['opt_used']['basis'] == 'signal'
        assert results['opt_used']['criterion'] == 'prediction'
        assert results['opt_used']['threshold_method'] == 'hybrid'

    def test_mode_aggressive(self, sample_data):
        """Test aggressive mode."""
        results = psn(sample_data, 'aggressive', {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['opt_used']['basis'] == 'difference'
        assert results['opt_used']['criterion'] == 'prediction'
        assert results['opt_used']['threshold_method'] == 'hybrid'

    def test_default_mode(self, sample_data):
        """Test that default (no mode specified) uses the 'auto' mode."""
        results = psn(sample_data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        # Default is now 'auto': signal-vs-difference selection at the max-tradeoff point.
        assert results['opt_used']['criterion'] == 'max-tradeoff'
        assert results['opt_used']['threshold_method'] == 'hybrid'
        # auto selects one of the two truncation bases (reported in results)
        assert results['auto_basis_selected'] in ('signal', 'difference')
        assert results['opt_used']['basis'] in ('signal', 'difference')


# ============================================================================
# Test Data Dimensions
# ============================================================================

class TestDataDimensions:
    """Test with various data dimensions."""

    @pytest.mark.parametrize("nunits,nconds,ntrials", [
        (3, 5, 2),    # Minimal
        (5, 10, 3),   # Small
        (10, 20, 5),  # Medium
        (20, 50, 8),  # Large
        (15, 10, 4),  # More units than conditions
        (5, 30, 6),   # More conditions than units
    ])
    def test_various_dimensions(self, nunits, nconds, ntrials):
        """Test with various data dimensions."""
        np.random.seed(789)
        data = np.random.randn(nunits, nconds, ntrials)

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['denoiseddata'].shape == (nunits, nconds)
        assert results['denoiser'].shape == (nunits, nunits)

    def test_two_trials_minimum(self):
        """Test with minimum number of trials (2)."""
        data = np.random.randn(5, 10, 2)

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['denoiseddata'].shape == (5, 10)


# ============================================================================
# Test Output Structure
# ============================================================================

class TestOutputStructure:
    """Test that results contain expected fields."""

    def test_required_fields(self, sample_data):
        """Test that all required fields are present."""
        results = psn(sample_data, {'wantfig': False, 'wantverbose': False})

        required_fields = [
            'denoiseddata', 'denoiser', 'residuals',
            'best_threshold', 'opt_used', 'gsn_result',
            'basis_viz', 'signalvar', 'noisevar'
        ]

        for field in required_fields:
            assert field in results, f"Missing required field: {field}"

    def test_optional_fields_prediction(self, sample_data):
        """Test fields specific to prediction criterion."""
        results = psn(sample_data, {
            'criterion': 'prediction',
            'wantfig': False,
            'wantverbose': False
        })

        assert 'objective' in results
        assert results['objective'] is not None

    def test_gsn_result_structure(self, sample_data):
        """Test that gsn_result contains expected fields."""
        results = psn(sample_data, {'wantfig': False, 'wantverbose': False})

        gsn_fields = ['cSb', 'cNb']  # ntrials_avg may not be in gsn_result
        for field in gsn_fields:
            assert field in results['gsn_result'], f"Missing GSN field: {field}"


# ============================================================================
# Test Residuals and Reconstruction
# ============================================================================

class TestResidualsAndReconstruction:
    """Test residuals and data reconstruction."""

    def test_reconstruction(self, sample_data):
        """Test that denoised + residuals = original data."""
        results = psn(sample_data, {'wantfig': False, 'wantverbose': False})

        # Get trial-averaged data
        data_avg = np.nanmean(sample_data, axis=2)

        # Reconstruction should match
        reconstructed = results['denoiseddata'] + np.nanmean(results['residuals'], axis=2)

        assert np.allclose(data_avg, reconstructed, rtol=1e-10)

    def test_residuals_shape(self, sample_data):
        """Test that residuals have correct shape."""
        results = psn(sample_data, {'wantfig': False, 'wantverbose': False})

        assert results['residuals'].shape == sample_data.shape

    def test_denoiser_application(self, sample_data):
        """Test that denoiser applied to data gives denoised result."""
        results = psn(sample_data, {'wantfig': False, 'wantverbose': False})

        data_avg = np.nanmean(sample_data, axis=2)
        manual_denoised = results['denoiser'] @ data_avg

        assert np.allclose(results['denoiseddata'], manual_denoised, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
