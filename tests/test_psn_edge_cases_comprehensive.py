"""Comprehensive edge case tests for PSN functional API.

Tests edge cases, NaN handling, extreme data conditions, and error scenarios.
"""

import numpy as np
import pytest
from psn import psn


# ============================================================================
# Test NaN Handling
# ============================================================================

class TestNaNHandling:
    """Test NaN handling in input data."""

    def test_uneven_trials_across_conditions(self):
        """Test data with different numbers of trials per condition."""
        np.random.seed(111)
        data = np.random.randn(5, 10, 8)

        # Make some trials NaN for different conditions
        data[:, 0, 5:] = np.nan  # Condition 0: only 5 trials
        data[:, 3, 6:] = np.nan  # Condition 3: only 6 trials
        data[:, 7, 4:] = np.nan  # Condition 7: only 4 trials

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        # Denoised data should not contain NaN
        assert not np.any(np.isnan(results['denoiseddata']))
        # Residuals should preserve NaN structure
        assert np.isnan(results['residuals'][:, 0, 5:]).all()

    def test_many_nans_sparse_data(self):
        """Test with sparse data (many NaNs)."""
        np.random.seed(222)
        data = np.random.randn(8, 20, 6)

        # Make 50% of trials NaN randomly
        mask = np.random.rand(*data.shape) > 0.5
        data[mask] = np.nan

        # Ensure each condition has at least 2 valid trials
        for c in range(data.shape[1]):
            # Find first two complete trials
            count = 0
            for t in range(data.shape[2]):
                if count < 2:
                    data[:, c, t] = np.random.randn(data.shape[0])
                    count += 1

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert not np.any(np.isnan(results['denoiseddata']))

    def test_minimum_valid_trials_per_condition(self):
        """Test with exactly 2 valid trials per condition."""
        np.random.seed(333)
        data = np.random.randn(6, 12, 5)

        # Keep only first 2 trials valid for each condition
        data[:, :, 2:] = np.nan

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert not np.any(np.isnan(results['denoiseddata']))

    def test_single_unit_has_nans(self):
        """Test when individual units have NaN patterns."""
        np.random.seed(444)
        data = np.random.randn(8, 15, 4)

        # Unit 2 has many NaNs
        data[2, :, 2:] = np.nan
        # Unit 5 has some NaNs
        data[5, ::2, 3] = np.nan

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert not np.any(np.isnan(results['denoiseddata']))


# ============================================================================
# Test Extreme Data Conditions
# ============================================================================

class TestExtremeDataConditions:
    """Test with extreme data conditions."""

    def test_zero_data(self):
        """Test with all-zero data."""
        data = np.zeros((5, 10, 3))

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        # Zero data should remain zero
        assert np.allclose(results['denoiseddata'], 0)

    def test_constant_data(self):
        """Test with constant data (no variance)."""
        data = np.ones((5, 10, 3)) * 5.0

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        # Constant data should be preserved
        assert np.allclose(results['denoiseddata'], 5.0)

    def test_very_high_noise(self):
        """Test with extremely high noise."""
        np.random.seed(555)
        signal = np.random.randn(8, 15)
        noise = 100.0 * np.random.randn(8, 15, 4)
        data = signal[:, :, np.newaxis] + noise

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert not np.any(np.isnan(results['denoiseddata']))
        # High noise should result in aggressive denoising (low threshold)
        # Most dimensions should be removed

    def test_very_low_noise(self):
        """Test with extremely low noise."""
        np.random.seed(666)
        signal = np.random.randn(8, 15)
        noise = 0.0001 * np.random.randn(8, 15, 4)
        data = signal[:, :, np.newaxis] + noise

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert not np.any(np.isnan(results['denoiseddata']))
        # Low noise: denoised should be very close to trial-averaged data
        data_avg = np.mean(data, axis=2)
        assert np.allclose(results['denoiseddata'], data_avg, rtol=0.1)

    def test_perfect_data_no_noise(self):
        """Test with perfect data (identical trials, no noise)."""
        np.random.seed(777)
        signal = np.random.randn(6, 12)
        # All trials identical
        data = np.tile(signal[:, :, np.newaxis], (1, 1, 5))

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        # Perfect data should be preserved exactly
        assert np.allclose(results['denoiseddata'], signal, rtol=1e-10)

    def test_very_large_values(self):
        """Test with very large data values."""
        np.random.seed(888)
        data = 1e10 * np.random.randn(5, 10, 3)

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert not np.any(np.isnan(results['denoiseddata']))
        assert not np.any(np.isinf(results['denoiseddata']))

    def test_very_small_values(self):
        """Test with very small data values."""
        np.random.seed(999)
        data = 1e-10 * np.random.randn(5, 10, 3)

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert not np.any(np.isnan(results['denoiseddata']))

    def test_negative_data(self):
        """Test with all negative data."""
        np.random.seed(1010)
        data = -np.abs(np.random.randn(6, 12, 4))

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        # Should handle negative data correctly
        assert not np.any(np.isnan(results['denoiseddata']))


# ============================================================================
# Test Minimal Dimensions
# ============================================================================

class TestMinimalDimensions:
    """Test with minimal data dimensions."""

    def test_two_units_minimum(self):
        """Test with minimum number of units."""
        np.random.seed(1111)
        data = np.random.randn(2, 10, 3)

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['denoiseddata'].shape == (2, 10)

    def test_two_conditions_minimum(self):
        """Test with minimum number of conditions."""
        np.random.seed(1212)
        data = np.random.randn(5, 2, 3)

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['denoiseddata'].shape == (5, 2)

    def test_two_trials_minimum(self):
        """Test with minimum number of trials."""
        np.random.seed(1313)
        data = np.random.randn(5, 10, 2)

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['denoiseddata'].shape == (5, 10)

    def test_smallest_valid_data(self):
        """Test with smallest valid data: 2×2×2."""
        np.random.seed(1414)
        data = np.random.randn(2, 2, 2)

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['denoiseddata'].shape == (2, 2)


# ============================================================================
# Test Single Unit/Condition Edge Cases
# ============================================================================

class TestSingleDimensionCases:
    """Test edge cases with single units or conditions."""

    def test_single_unit(self):
        """Test with single unit (should still work)."""
        np.random.seed(1515)
        data = np.random.randn(1, 20, 5)

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert results['denoiseddata'].shape == (1, 20)
        assert results['denoiser'].shape == (1, 1)

    def test_single_condition(self):
        """Test with single condition - should raise ValueError."""
        np.random.seed(1616)
        data = np.random.randn(10, 1, 5)

        # PSN requires at least 2 conditions to estimate covariance
        with pytest.raises(ValueError, match='at least 2 conditions'):
            psn(data, {'wantfig': False, 'wantverbose': False})


# ============================================================================
# Test Threshold Edge Cases
# ============================================================================

class TestThresholdEdgeCases:
    """Test edge cases for threshold selection."""

    def test_force_zero_dimensions(self):
        """Test forcing zero dimensions (complete removal)."""
        np.random.seed(1717)
        data = np.random.randn(8, 15, 4)

        results = psn(data, {
            'allowable_thresholds': [0],
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert results['best_threshold'] == 0
        # Zero dimensions: denoised should equal unit means (constant per unit across conditions)
        expected = results['unit_means'][:, np.newaxis]
        assert np.allclose(results['denoiseddata'], expected)

    def test_force_full_dimensions(self):
        """Test forcing full rank (no denoising)."""
        np.random.seed(1818)
        data = np.random.randn(6, 12, 4)
        nunits = data.shape[0]

        results = psn(data, {
            'allowable_thresholds': [nunits],
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert np.all(results['best_threshold'] == nunits)
        # Full rank: denoised should equal trial average
        data_avg = np.mean(data, axis=2)
        assert np.allclose(results['denoiseddata'], data_avg, rtol=1e-10)

    def test_single_intermediate_threshold(self):
        """Test forcing a single intermediate threshold."""
        np.random.seed(1919)
        data = np.random.randn(10, 20, 5)

        forced_threshold = 4
        results = psn(data, {
            'allowable_thresholds': [forced_threshold],
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert np.all(results['best_threshold'] == forced_threshold)

    def test_allowable_thresholds_rounding(self):
        """Test that optimal threshold rounds to nearest allowed."""
        np.random.seed(2020)
        data = np.random.randn(10, 20, 5)

        # Only allow specific thresholds
        allowed = [2, 5, 8]
        results = psn(data, {
            'allowable_thresholds': allowed,
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        # With global method, best_threshold is a scalar
        assert results['best_threshold'] in allowed


# ============================================================================
# Test Custom Basis Edge Cases
# ============================================================================

class TestCustomBasisEdgeCases:
    """Test edge cases for custom basis matrices."""

    def test_custom_basis_single_dimension(self):
        """Test custom basis with single dimension."""
        np.random.seed(2121)
        data = np.random.randn(8, 15, 4)
        nunits = data.shape[0]

        # Single basis vector
        custom_basis = np.random.randn(nunits, 1)
        custom_basis = custom_basis / np.linalg.norm(custom_basis)

        results = psn(data, {
            'basis': custom_basis,
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert results['fullbasis'].shape == (nunits, 1)

    def test_custom_basis_nearly_full_rank(self):
        """Test custom basis with nearly full rank."""
        np.random.seed(2222)
        data = np.random.randn(10, 20, 5)
        nunits = data.shape[0]

        # Full rank minus 1
        custom_basis = np.linalg.qr(np.random.randn(nunits, nunits-1))[0]

        results = psn(data, {
            'basis': custom_basis,
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert results['fullbasis'].shape == (nunits, nunits-1)

    def test_custom_basis_with_forced_threshold(self):
        """Test custom basis with forced threshold."""
        np.random.seed(2323)
        data = np.random.randn(8, 15, 4)
        nunits = data.shape[0]

        custom_basis = np.linalg.qr(np.random.randn(nunits, 6))[0]

        results = psn(data, {
            'basis': custom_basis,
            'allowable_thresholds': [3],
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert np.all(results['best_threshold'] == 3)


# ============================================================================
# Test Unit Groups Edge Cases
# ============================================================================

class TestUnitGroupsEdgeCases:
    """Test edge cases for unit_groups parameter."""

    def test_all_units_one_group(self):
        """Test when all units are in one group."""
        np.random.seed(2424)
        data = np.random.randn(10, 20, 5)

        unit_groups = np.zeros(10, dtype=int)

        results = psn(data, {
            'threshold_method': 'hybrid',
            'unit_groups': unit_groups,
            'wantfig': False,
            'wantverbose': False
        })

        # All units should have same threshold
        assert len(np.unique(results['best_threshold'])) == 1

    def test_each_unit_separate_group(self):
        """Test when each unit is its own group (default behavior)."""
        np.random.seed(2525)
        data = np.random.randn(8, 15, 4)

        unit_groups = np.arange(8)

        results = psn(data, {
            'threshold_method': 'hybrid',
            'unit_groups': unit_groups,
            'wantfig': False,
            'wantverbose': False
        })

        # Each unit can have different threshold
        assert len(results['best_threshold']) == 8

    def test_unit_groups_non_contiguous_ids(self):
        """Test unit groups with non-contiguous group IDs."""
        np.random.seed(2626)
        data = np.random.randn(10, 20, 5)

        # Use non-contiguous IDs: 0, 5, 10, 15, 20
        unit_groups = np.array([0, 0, 5, 5, 10, 10, 15, 15, 20, 20])

        results = psn(data, {
            'threshold_method': 'hybrid',
            'unit_groups': unit_groups,
            'wantfig': False,
            'wantverbose': False
        })

        # Check that units in same group have same threshold
        assert results['best_threshold'][0] == results['best_threshold'][1]
        assert results['best_threshold'][2] == results['best_threshold'][3]

    def test_unit_groups_single_member_groups(self):
        """Test unit groups where some groups have single member."""
        np.random.seed(2727)
        data = np.random.randn(8, 15, 4)

        unit_groups = np.array([0, 0, 1, 2, 2, 2, 3, 4])

        results = psn(data, {
            'threshold_method': 'hybrid',
            'unit_groups': unit_groups,
            'wantfig': False,
            'wantverbose': False
        })

        # Group 0: units 0,1 should match
        assert results['best_threshold'][0] == results['best_threshold'][1]
        # Group 2: units 3,4,5 should match
        assert results['best_threshold'][3] == results['best_threshold'][4] == results['best_threshold'][5]


# ============================================================================
# Test Consistency and Reproducibility
# ============================================================================

class TestConsistencyAndReproducibility:
    """Test consistency and reproducibility."""

    def test_repeated_calls_identical(self):
        """Test that repeated calls with same data give identical results."""
        np.random.seed(2828)
        data = np.random.randn(8, 15, 4)

        results1 = psn(data, {'wantfig': False, 'wantverbose': False})
        results2 = psn(data, {'wantfig': False, 'wantverbose': False})

        assert np.allclose(results1['denoiseddata'], results2['denoiseddata'])
        assert np.allclose(results1['denoiser'], results2['denoiser'])
        assert np.array_equal(results1['best_threshold'], results2['best_threshold'])

    def test_random_basis_with_seed(self):
        """Test that random basis with seed gives reproducible results."""
        np.random.seed(2929)
        data = np.random.randn(6, 12, 4)

        results1 = psn(data, {
            'basis': 'random',
            'gsn_args': {'random_seed': 42},
            'wantfig': False,
            'wantverbose': False
        })

        results2 = psn(data, {
            'basis': 'random',
            'gsn_args': {'random_seed': 42},
            'wantfig': False,
            'wantverbose': False
        })

        # Should be identical with same seed
        assert np.allclose(results1['denoiseddata'], results2['denoiseddata'])

    def test_mode_overrides_consistent(self):
        """Test that mode overrides are applied consistently."""
        np.random.seed(3030)
        data = np.random.randn(8, 15, 4)

        # Conservative mode explicit
        results1 = psn(data, 'conservative', {'wantfig': False, 'wantverbose': False})

        # Conservative mode manual
        results2 = psn(data, {
            'basis': 'signal',
            'criterion': 'variance',
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        # Should produce same results
        assert np.allclose(results1['denoiseddata'], results2['denoiseddata'])


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test proper error handling for invalid inputs."""

    def test_single_trial_error(self):
        """Test that single trial raises appropriate error."""
        data = np.random.randn(5, 10, 1)

        with pytest.raises((ValueError, AssertionError)):
            psn(data, {'wantfig': False, 'wantverbose': False})

    def test_2d_input_error(self):
        """Test that 2D input raises appropriate error."""
        data = np.random.randn(5, 10)

        with pytest.raises((ValueError, AssertionError, IndexError)):
            psn(data, {'wantfig': False, 'wantverbose': False})

    def test_invalid_basis_string(self):
        """Test that invalid basis string raises error."""
        data = np.random.randn(5, 10, 3)

        with pytest.raises((ValueError, KeyError)):
            psn(data, {
                'basis': 'invalid_basis_type',
                'wantfig': False,
                'wantverbose': False
            })

    def test_invalid_criterion(self):
        """Test that invalid criterion raises error."""
        data = np.random.randn(5, 10, 3)

        with pytest.raises((ValueError, KeyError)):
            psn(data, {
                'criterion': 'invalid_criterion',
                'wantfig': False,
                'wantverbose': False
            })

    def test_invalid_threshold_method(self):
        """Test that invalid threshold method raises error."""
        data = np.random.randn(5, 10, 3)

        with pytest.raises((ValueError, KeyError)):
            psn(data, {
                'threshold_method': 'invalid_method',
                'wantfig': False,
                'wantverbose': False
            })

    def test_removed_unit_threshold_method(self):
        """The 'unit' threshold method was removed; it should now raise."""
        data = np.random.randn(5, 10, 3)

        with pytest.raises(ValueError, match='threshold_method'):
            psn(data, {
                'threshold_method': 'unit',
                'wantfig': False,
                'wantverbose': False
            })

    def test_variance_threshold_out_of_range(self):
        """Test that variance threshold outside [0,1] raises error."""
        data = np.random.randn(5, 10, 3)

        with pytest.raises((ValueError, AssertionError)):
            psn(data, {
                'criterion': 'variance',
                'variance_threshold': 1.5,
                'wantfig': False,
                'wantverbose': False
            })

    def test_custom_basis_wrong_dimensions(self):
        """Test that custom basis with wrong dimensions raises error."""
        data = np.random.randn(8, 15, 4)

        # Wrong number of rows
        wrong_basis = np.random.randn(6, 5)

        with pytest.raises((ValueError, AssertionError)):
            psn(data, {
                'basis': wrong_basis,
                'wantfig': False,
                'wantverbose': False
            })

    def test_unit_groups_wrong_length(self):
        """Test that unit_groups with wrong length raises error."""
        data = np.random.randn(8, 15, 4)

        # Wrong length
        wrong_groups = np.array([0, 1, 2])

        with pytest.raises((ValueError, AssertionError, IndexError)):
            psn(data, {
                'unit_groups': wrong_groups,
                'threshold_method': 'hybrid',
                'wantfig': False,
                'wantverbose': False
            })


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
