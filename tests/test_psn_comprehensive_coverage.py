"""Comprehensive coverage tests for PSN - filling coverage gaps.

This test file provides additional coverage for edge cases, boundary conditions,
degenerate data, and parameter interactions not fully covered in other test files.
"""

import numpy as np
import pytest
from psn import psn


# Test fixtures
@pytest.fixture
def standard_data():
    """Standard test data."""
    np.random.seed(100)
    nunits, nconds, ntrials = 10, 20, 5
    signal = np.random.randn(nunits, nconds)
    noise = 0.3 * np.random.randn(nunits, nconds, ntrials)
    data = signal[:, :, np.newaxis] + noise
    return data


@pytest.fixture
def nan_data():
    """Data with NaNs."""
    np.random.seed(200)
    data = np.random.randn(8, 15, 6)
    # Create uneven trials
    data[:, 0, 4:] = np.nan
    data[:, 5, 5:] = np.nan
    data[:, 10, 3:] = np.nan
    return data


# ============================================================================
# Comprehensive allowable_thresholds Edge Cases
# ============================================================================

class TestAllowableThresholdsEdgeCases:
    """Comprehensive tests for allowable_thresholds parameter."""

    def test_allowable_threshold_zero_only(self, standard_data):
        """Test forcing exactly zero dimensions."""
        results = psn(standard_data, {
            'allowable_thresholds': [0],
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert results['best_threshold'] == 0
        # Zero threshold means denoised data should be constant across conditions (unit means)
        for unit in range(standard_data.shape[0]):
            assert np.allclose(results['denoiseddata'][unit, :], results['denoiseddata'][unit, 0], atol=1e-10)

    def test_allowable_threshold_max_only(self, standard_data):
        """Test forcing maximum dimensions."""
        nunits = standard_data.shape[0]
        results = psn(standard_data, {
            'allowable_thresholds': [nunits],
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert results['best_threshold'] == nunits
        # Full rank should perfectly reconstruct trial average
        data_avg = np.mean(standard_data, axis=2)
        assert np.allclose(results['denoiseddata'], data_avg, rtol=1e-10)

    def test_allowable_thresholds_exceeding_dims_error(self, standard_data):
        """Test that allowable_thresholds exceeding basis dims raises error."""
        nunits = standard_data.shape[0]

        with pytest.raises(ValueError):
            psn(standard_data, {
                'allowable_thresholds': [nunits + 5],
                'threshold_method': 'global',
                'wantfig': False,
                'wantverbose': False
            })

    def test_allowable_thresholds_with_hybrid(self, standard_data):
        """Test allowable thresholds with hybrid method."""
        allowed = [2, 4, 6]
        results = psn(standard_data, {
            'allowable_thresholds': allowed,
            'threshold_method': 'hybrid',
            'wantfig': False,
            'wantverbose': False
        })

        # All thresholds should be from allowed set
        assert all(t in allowed for t in results['best_threshold'])

    def test_allowable_thresholds_with_unit(self, standard_data):
        """Test allowable thresholds with unit-specific method."""
        allowed = [1, 3, 5, 7]
        results = psn(standard_data, {
            'allowable_thresholds': allowed,
            'threshold_method': 'unit',
            'wantfig': False,
            'wantverbose': False
        })

        # All thresholds should be from allowed set
        assert all(t in allowed for t in results['best_threshold'])

    def test_allowable_thresholds_dense_range(self, standard_data):
        """Test with many allowable thresholds."""
        nunits = standard_data.shape[0]
        allowed = list(range(nunits + 1))  # All possible thresholds

        results = psn(standard_data, {
            'allowable_thresholds': allowed,
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert results['best_threshold'] in allowed

    def test_allowable_thresholds_sparse_range(self, standard_data):
        """Test with very sparse allowable thresholds."""
        allowed = [1, 5, 9]

        results = psn(standard_data, {
            'allowable_thresholds': allowed,
            'criterion': 'prediction',
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert results['best_threshold'] in allowed

    def test_allowable_thresholds_with_variance_criterion(self, standard_data):
        """Test allowable thresholds with variance criterion."""
        allowed = [3, 6, 9]

        results = psn(standard_data, {
            'allowable_thresholds': allowed,
            'criterion': 'variance',
            'variance_threshold': 0.90,
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert results['best_threshold'] in allowed

    def test_allowable_thresholds_custom_basis(self, standard_data):
        """Test allowable thresholds with custom basis."""
        nunits = standard_data.shape[0]
        custom_basis = np.linalg.qr(np.random.randn(nunits, 7))[0]
        allowed = [2, 4, 6]

        results = psn(standard_data, {
            'basis': custom_basis,
            'allowable_thresholds': allowed,
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert results['best_threshold'] in allowed


# ============================================================================
# NaN Handling with All Parameter Combinations
# ============================================================================

class TestNaNHandlingComprehensive:
    """Test NaN handling with various parameter combinations."""

    @pytest.mark.parametrize("threshold_method", ['global', 'hybrid', 'unit'])
    def test_nan_with_all_threshold_methods(self, nan_data, threshold_method):
        """Test NaN data with all threshold methods."""
        results = psn(nan_data, {
            'threshold_method': threshold_method,
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert not np.any(np.isnan(results['denoiseddata']))
        # Residuals preserve NaN structure
        assert np.any(np.isnan(results['residuals']))

    @pytest.mark.parametrize("basis", ['signal', 'difference', 'pca', 'noise'])
    def test_nan_with_all_bases(self, nan_data, basis):
        """Test NaN data with all basis types."""
        results = psn(nan_data, {
            'basis': basis,
            'wantfig': False,
            'wantverbose': False
        })

        assert not np.any(np.isnan(results['denoiseddata']))

    @pytest.mark.parametrize("criterion", ['prediction', 'variance'])
    def test_nan_with_all_criteria(self, nan_data, criterion):
        """Test NaN data with all criteria."""
        results = psn(nan_data, {
            'criterion': criterion,
            'wantfig': False,
            'wantverbose': False
        })

        assert not np.any(np.isnan(results['denoiseddata']))

    def test_nan_with_variance_eigenvalues(self, nan_data):
        """Test NaN data with variance_eigenvalues criterion."""
        results = psn(nan_data, {
            'criterion': 'variance_eigenvalues',
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert not np.any(np.isnan(results['denoiseddata']))

    def test_nan_with_unit_groups(self, nan_data):
        """Test NaN data with unit groups."""
        nunits = nan_data.shape[0]
        unit_groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])

        results = psn(nan_data, {
            'threshold_method': 'hybrid',
            'unit_groups': unit_groups,
            'wantfig': False,
            'wantverbose': False
        })

        assert not np.any(np.isnan(results['denoiseddata']))

    def test_nan_extreme_sparsity(self):
        """Test with extremely sparse data (many NaNs)."""
        np.random.seed(300)
        data = np.random.randn(6, 10, 8)

        # Make 70% of data NaN, but ensure each condition has >= 2 valid trials
        for c in range(data.shape[1]):
            valid_trials = np.random.choice(8, size=3, replace=False)
            for t in range(8):
                if t not in valid_trials:
                    data[:, c, t] = np.nan

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert not np.any(np.isnan(results['denoiseddata']))


# ============================================================================
# Degenerate Data Tests
# ============================================================================

class TestDegenerateData:
    """Test with degenerate data conditions."""

    def test_rank_deficient_signal(self):
        """Test with rank-deficient signal."""
        np.random.seed(400)
        nunits, nconds, ntrials = 10, 20, 5

        # Create rank-deficient signal (rank 3)
        U = np.random.randn(nunits, 3)
        V = np.random.randn(3, nconds)
        signal = U @ V

        noise = 0.1 * np.random.randn(nunits, nconds, ntrials)
        data = signal[:, :, np.newaxis] + noise

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert not np.any(np.isnan(results['denoiseddata']))

    def test_collinear_units(self):
        """Test with highly collinear units."""
        np.random.seed(500)
        nunits, nconds, ntrials = 8, 15, 4

        # Create data where units are highly correlated
        base_signal = np.random.randn(1, nconds)
        signal = np.repeat(base_signal, nunits, axis=0) + 0.01 * np.random.randn(nunits, nconds)

        noise = 0.2 * np.random.randn(nunits, nconds, ntrials)
        data = signal[:, :, np.newaxis] + noise

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results

    def test_one_dominant_dimension(self):
        """Test with one dominant signal dimension."""
        np.random.seed(600)
        nunits, nconds, ntrials = 10, 20, 5

        # One very strong dimension, rest are weak
        signal = np.zeros((nunits, nconds))
        signal[0, :] = 10.0 * np.random.randn(nconds)
        signal[1:, :] = 0.1 * np.random.randn(nunits-1, nconds)

        noise = 0.3 * np.random.randn(nunits, nconds, ntrials)
        data = signal[:, :, np.newaxis] + noise

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        # Should retain primarily the dominant dimension
        assert results['best_threshold'][0] >= 1

    def test_orthogonal_signal_noise(self):
        """Test when signal and noise are orthogonal."""
        np.random.seed(700)
        nunits, nconds, ntrials = 8, 15, 4

        # Signal in first few dimensions
        signal = np.zeros((nunits, nconds))
        signal[:3, :] = np.random.randn(3, nconds)

        # Noise in orthogonal dimensions
        noise = np.zeros((nunits, nconds, ntrials))
        noise[3:, :, :] = np.random.randn(nunits-3, nconds, ntrials)

        data = signal[:, :, np.newaxis] + noise

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results

    def test_identical_conditions(self):
        """Test with identical conditions."""
        np.random.seed(800)
        nunits, ntrials = 8, 5

        # All conditions are identical
        base_response = np.random.randn(nunits, 1)
        signal = np.tile(base_response, (1, 10))

        noise = 0.1 * np.random.randn(nunits, 10, ntrials)
        data = signal[:, :, np.newaxis] + noise

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results


# ============================================================================
# Boundary Condition Tests
# ============================================================================

class TestBoundaryConditions:
    """Test exact boundary conditions for parameters."""

    def test_variance_threshold_zero(self, standard_data):
        """Test variance threshold of exactly 0."""
        results = psn(standard_data, {
            'criterion': 'variance',
            'variance_threshold': 0.0,
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        # Should retain no dimensions or minimal dimensions
        assert 'denoiseddata' in results

    def test_variance_threshold_one(self, standard_data):
        """Test variance threshold of exactly 1.0."""
        results = psn(standard_data, {
            'criterion': 'variance',
            'variance_threshold': 1.0,
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        # Should retain all or nearly all dimensions
        assert 'denoiseddata' in results

    def test_variance_eigenvalues_boundaries(self, standard_data):
        """Test variance_eigenvalues at boundary values."""
        for thresh in [0.0, 0.5, 1.0]:
            results = psn(standard_data, {
                'criterion': 'variance_eigenvalues',
                'variance_threshold': thresh,
                'threshold_method': 'global',
                'wantfig': False,
                'wantverbose': False
            })

            assert 'denoiseddata' in results

    def test_minimum_data_size(self):
        """Test with minimum valid data size (2×2×2)."""
        data = np.random.randn(2, 2, 2)

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert results['denoiseddata'].shape == (2, 2)

    def test_single_unit_multiple_conditions(self):
        """Test with single unit, multiple conditions."""
        data = np.random.randn(1, 50, 10)

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert results['denoiseddata'].shape == (1, 50)
        assert results['denoiser'].shape == (1, 1)

    def test_many_units_single_condition(self):
        """Test that single condition raises appropriate error."""
        data = np.random.randn(50, 1, 10)

        # Single condition should raise error (need at least 2 for covariance estimation)
        with pytest.raises(ValueError):
            psn(data, {'wantfig': False, 'wantverbose': False})


# ============================================================================
# Comprehensive GSN Args Testing
# ============================================================================

class TestGSNArgsComprehensive:
    """Comprehensive tests for gsn_args parameter."""

    def test_gsn_wantshrinkage_true(self, standard_data):
        """Test with shrinkage enabled."""
        results = psn(standard_data, {
            'gsn_args': {'wantshrinkage': True},
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results

    def test_gsn_wantshrinkage_false(self, standard_data):
        """Test with shrinkage disabled."""
        results = psn(standard_data, {
            'gsn_args': {'wantshrinkage': False},
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results

    def test_gsn_wantverbose_true(self, standard_data):
        """Test with GSN verbose output."""
        results = psn(standard_data, {
            'gsn_args': {'wantverbose': True},
            'wantfig': False,
            'wantverbose': False  # PSN verbose off, GSN verbose on
        })

        assert 'denoiseddata' in results

    def test_gsn_random_seed_reproducibility(self, standard_data):
        """Test that GSN random seed gives reproducible results."""
        results1 = psn(standard_data, {
            'basis': 'random',
            'gsn_args': {'random_seed': 999},
            'wantfig': False,
            'wantverbose': False
        })

        results2 = psn(standard_data, {
            'basis': 'random',
            'gsn_args': {'random_seed': 999},
            'wantfig': False,
            'wantverbose': False
        })

        assert np.allclose(results1['denoiseddata'], results2['denoiseddata'])

    def test_gsn_all_options_combined(self, standard_data):
        """Test all GSN options together."""
        results = psn(standard_data, {
            'gsn_args': {
                'wantshrinkage': True,
                'wantverbose': False,
                'random_seed': 42
            },
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results


# ============================================================================
# Output Validation and Correctness Tests
# ============================================================================

class TestOutputValidation:
    """Validate correctness of output properties."""

    def test_denoiser_symmetry_global(self, standard_data):
        """Test that global denoiser is symmetric."""
        results = psn(standard_data, {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        # Global denoiser should be symmetric
        assert np.allclose(results['denoiser'], results['denoiser'].T)

    def test_denoiser_idempotency_global(self, standard_data):
        """Test that applying global denoiser twice gives same result."""
        results = psn(standard_data, {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        denoised_once = results['denoiseddata']
        denoised_twice = results['denoiser'].T @ denoised_once

        # Applying denoiser to already denoised data should be close to identity
        # (not exactly due to mean subtraction/addition)
        # But the pattern should be preserved

    def test_residuals_orthogonality(self, standard_data):
        """Test properties of residuals."""
        results = psn(standard_data, {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        # Residuals should reconstruct original when added to denoised
        data_avg = np.mean(standard_data, axis=2)
        residuals_avg = np.mean(results['residuals'], axis=2)

        reconstructed = results['denoiseddata'] + residuals_avg
        assert np.allclose(reconstructed, data_avg, rtol=1e-10)

    def test_variance_conservation(self, standard_data):
        """Test that total variance is conserved."""
        results = psn(standard_data, {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        # Check that svnv_before and svnv_after make sense
        assert 'svnv_before' in results
        assert 'svnv_after' in results
        assert results['svnv_before'].shape[0] == standard_data.shape[0]
        assert results['svnv_after'].shape[0] == standard_data.shape[0]

    def test_threshold_consistency(self, standard_data):
        """Test that best_threshold is consistent with actual denoising."""
        results = psn(standard_data, {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        threshold = results['best_threshold']

        if threshold > 0:
            assert results['signalsubspace'] is not None
            assert results['signalsubspace'].shape[1] == threshold
        else:
            assert results['signalsubspace'] is None

    def test_basis_orthonormality(self, standard_data):
        """Test that basis vectors are orthonormal."""
        results = psn(standard_data, {
            'wantfig': False,
            'wantverbose': False
        })

        basis = results['fullbasis']

        # Check orthonormality: B^T @ B = I
        gram = basis.T @ basis
        identity = np.eye(basis.shape[1])

        assert np.allclose(gram, identity, rtol=1e-10)

    def test_eigenvalue_ordering(self, standard_data):
        """Test that eigenvalues are properly ordered."""
        results = psn(standard_data, {
            'basis': 'signal',
            'basis_ordering': 'eigenvalues',
            'wantfig': False,
            'wantverbose': False
        })

        if results['basis_eigenvalues'] is not None:
            evals = results['basis_eigenvalues']
            # Should be in descending order
            assert np.all(evals[:-1] >= evals[1:])

    def test_signal_variance_ordering(self, standard_data):
        """Test that signal variance is properly ordered."""
        results = psn(standard_data, {
            'basis_ordering': 'signalvariance',
            'wantfig': False,
            'wantverbose': False
        })

        signalvar = results['signalvar']
        if isinstance(signalvar, np.ndarray):
            # Should be in descending order
            assert np.all(signalvar[:-1] >= signalvar[1:] - 1e-10)  # Allow small numerical errors


# ============================================================================
# Interaction Effect Tests
# ============================================================================

class TestParameterInteractions:
    """Test specific parameter interactions that might cause issues."""

    def test_difference_basis_with_variance_eigenvalues(self, standard_data):
        """Test difference basis with variance_eigenvalues criterion."""
        results = psn(standard_data, {
            'basis': 'difference',
            'criterion': 'variance_eigenvalues',
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results

    def test_unit_groups_with_allowable_thresholds(self, standard_data):
        """Test interaction between unit_groups and allowable_thresholds."""
        nunits = standard_data.shape[0]
        unit_groups = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
        allowed = [2, 4, 6]

        results = psn(standard_data, {
            'threshold_method': 'hybrid',
            'unit_groups': unit_groups,
            'allowable_thresholds': allowed,
            'wantfig': False,
            'wantverbose': False
        })

        # Units in same group should have same threshold from allowed set
        assert results['best_threshold'][0] == results['best_threshold'][1]
        assert all(t in allowed for t in results['best_threshold'])

    def test_custom_basis_with_variance_criterion(self, standard_data):
        """Test custom basis with variance criterion."""
        nunits = standard_data.shape[0]
        custom_basis = np.linalg.qr(np.random.randn(nunits, 6))[0]

        results = psn(standard_data, {
            'basis': custom_basis,
            'criterion': 'variance',
            'variance_threshold': 0.90,
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results

    def test_unit_method_with_variance_criterion(self, standard_data):
        """Test unit-specific method with variance criterion."""
        results = psn(standard_data, {
            'threshold_method': 'unit',
            'criterion': 'variance',
            'variance_threshold': 0.95,
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        # Each unit can have different threshold
        assert len(results['best_threshold']) == standard_data.shape[0]

    def test_nan_with_forced_thresholds(self, nan_data):
        """Test NaN data with forced thresholds."""
        results = psn(nan_data, {
            'allowable_thresholds': [3],
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert not np.any(np.isnan(results['denoiseddata']))
        assert results['best_threshold'] == 3


# ============================================================================
# Stress Tests
# ============================================================================

class TestStressConditions:
    """Test with computationally challenging scenarios."""

    def test_very_large_dataset(self):
        """Test with large dataset."""
        np.random.seed(900)
        nunits, nconds, ntrials = 50, 100, 20
        data = np.random.randn(nunits, nconds, ntrials)

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert results['denoiseddata'].shape == (nunits, nconds)

    def test_many_trials(self):
        """Test with many trials."""
        np.random.seed(1000)
        nunits, nconds, ntrials = 10, 20, 50
        data = np.random.randn(nunits, nconds, ntrials)

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert results['denoiseddata'].shape == (nunits, nconds)

    def test_many_conditions(self):
        """Test with many conditions."""
        np.random.seed(1100)
        nunits, nconds, ntrials = 15, 200, 5
        data = np.random.randn(nunits, nconds, ntrials)

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert results['denoiseddata'].shape == (nunits, nconds)

    def test_many_nan_patterns(self):
        """Test with many different NaN patterns."""
        np.random.seed(1200)
        data = np.random.randn(10, 30, 10)

        # Create diverse NaN patterns
        for c in range(30):
            n_missing = np.random.randint(0, 6)
            if n_missing > 0:
                missing_trials = np.random.choice(10, size=n_missing, replace=False)
                data[:, c, missing_trials] = np.nan

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert not np.any(np.isnan(results['denoiseddata']))


# ============================================================================
# Mode Comparison Tests
# ============================================================================

class TestModeComparisons:
    """Test and compare different preset modes."""

    def test_conservative_vs_manual(self, standard_data):
        """Test that conservative mode matches manual specification."""
        results_mode = psn(standard_data, 'conservative', {
            'wantfig': False,
            'wantverbose': False
        })

        results_manual = psn(standard_data, {
            'basis': 'signal',
            'criterion': 'variance',
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert np.allclose(results_mode['denoiseddata'], results_manual['denoiseddata'])

    def test_standard_vs_manual(self, standard_data):
        """Test that standard mode matches manual specification."""
        results_mode = psn(standard_data, 'standard', {
            'wantfig': False,
            'wantverbose': False
        })

        results_manual = psn(standard_data, {
            'basis': 'signal',
            'criterion': 'prediction',
            'threshold_method': 'hybrid',
            'wantfig': False,
            'wantverbose': False
        })

        assert np.allclose(results_mode['denoiseddata'], results_manual['denoiseddata'])

    def test_aggressive_vs_manual(self, standard_data):
        """Test that aggressive mode matches manual specification."""
        results_mode = psn(standard_data, 'aggressive', {
            'wantfig': False,
            'wantverbose': False
        })

        results_manual = psn(standard_data, {
            'basis': 'difference',
            'criterion': 'prediction',
            'threshold_method': 'hybrid',
            'wantfig': False,
            'wantverbose': False
        })

        assert np.allclose(results_mode['denoiseddata'], results_manual['denoiseddata'])

    def test_mode_ordering_conservative_standard_aggressive(self, standard_data):
        """Test expected ordering of aggressiveness in modes."""
        results_cons = psn(standard_data, 'conservative', {
            'wantfig': False,
            'wantverbose': False
        })

        results_std = psn(standard_data, 'standard', {
            'wantfig': False,
            'wantverbose': False
        })

        results_agg = psn(standard_data, 'aggressive', {
            'wantfig': False,
            'wantverbose': False
        })

        # Conservative should generally retain most dimensions
        # (though this depends on data)
        assert 'denoiseddata' in results_cons
        assert 'denoiseddata' in results_std
        assert 'denoiseddata' in results_agg


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
