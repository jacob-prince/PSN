"""Systematic tests of all valid PSN hyperparameter combinations.

This test file systematically exercises all valid combinations of PSN parameters
to ensure comprehensive coverage and detect any interaction issues.
"""

import numpy as np
import pytest

from psn import psn


# Test fixtures
@pytest.fixture
def test_data():
    """Create standard test data for combination tests."""
    np.random.seed(999)
    nunits, nconds, ntrials = 8, 15, 4
    signal = np.random.randn(nunits, nconds)
    noise = 0.4 * np.random.randn(nunits, nconds, ntrials)
    data = signal[:, :, np.newaxis] + noise
    return data


# ============================================================================
# Systematic Combination Tests
# ============================================================================

class TestBasisCriterionCombinations:
    """Test all valid basis × criterion combinations."""

    @pytest.mark.parametrize("basis,criterion", [
        # Signal basis with all criteria
        ('signal', 'prediction'),
        ('signal', 'variance'),
        ('signal', 'variance_eigenvalues'),

        # Difference basis with all criteria
        ('difference', 'prediction'),
        ('difference', 'variance'),
        ('difference', 'variance_eigenvalues'),

        # PCA basis (variance_eigenvalues not compatible without global)
        ('pca', 'prediction'),
        ('pca', 'variance'),

        # Noise basis
        ('noise', 'prediction'),
        ('noise', 'variance'),
        ('noise', 'variance_eigenvalues'),

        # Random basis (variance_eigenvalues not compatible)
        ('random', 'prediction'),
        ('random', 'variance'),
    ])
    def test_basis_criterion_valid(self, test_data, basis, criterion):
        """Test all valid basis-criterion combinations."""
        opt = {
            'basis': basis,
            'criterion': criterion,
            'threshold_method': 'global',  # Use global for variance_eigenvalues compatibility
            'wantfig': False,
            'wantverbose': False
        }

        results = psn(test_data, opt)

        assert 'denoiseddata' in results
        assert results['opt_used']['basis'] == basis
        assert results['opt_used']['criterion'] == criterion
        assert not np.any(np.isnan(results['denoiseddata']))


class TestBasisThresholdMethodCombinations:
    """Test all basis × threshold_method combinations."""

    @pytest.mark.parametrize("basis,threshold_method", [
        # Signal basis with all threshold methods
        ('signal', 'global'),
        ('signal', 'hybrid'),

        # Difference basis with all threshold methods
        ('difference', 'global'),
        ('difference', 'hybrid'),

        # PCA basis
        ('pca', 'global'),
        ('pca', 'hybrid'),

        # Noise basis
        ('noise', 'global'),
        ('noise', 'hybrid'),

        # Random basis
        ('random', 'global'),
        ('random', 'hybrid'),
    ])
    def test_basis_threshold_method_valid(self, test_data, basis, threshold_method):
        """Test all valid basis-threshold_method combinations."""
        opt = {
            'basis': basis,
            'threshold_method': threshold_method,
            'wantfig': False,
            'wantverbose': False
        }

        results = psn(test_data, opt)

        assert 'denoiseddata' in results
        assert results['opt_used']['basis'] == basis
        assert results['opt_used']['threshold_method'] == threshold_method


class TestCriterionThresholdMethodCombinations:
    """Test all criterion × threshold_method combinations."""

    @pytest.mark.parametrize("criterion,threshold_method", [
        # Prediction with all methods
        ('prediction', 'global'),
        ('prediction', 'hybrid'),

        # Variance with all methods
        ('variance', 'global'),
        ('variance', 'hybrid'),

        # Variance eigenvalues only compatible with global
        ('variance_eigenvalues', 'global'),
    ])
    def test_criterion_threshold_method_valid(self, test_data, criterion, threshold_method):
        """Test all valid criterion-threshold_method combinations."""
        opt = {
            'criterion': criterion,
            'threshold_method': threshold_method,
            'wantfig': False,
            'wantverbose': False
        }

        results = psn(test_data, opt)

        assert 'denoiseddata' in results
        assert results['opt_used']['criterion'] == criterion
        assert results['opt_used']['threshold_method'] == threshold_method


class TestThreeWayCombinations:
    """Test all valid basis × criterion × threshold_method combinations."""

    @pytest.mark.parametrize("basis,criterion,threshold_method", [
        # Signal basis - comprehensive
        ('signal', 'prediction', 'global'),
        ('signal', 'prediction', 'hybrid'),
        ('signal', 'variance', 'global'),
        ('signal', 'variance', 'hybrid'),
        ('signal', 'variance_eigenvalues', 'global'),

        # Difference basis - comprehensive
        ('difference', 'prediction', 'global'),
        ('difference', 'prediction', 'hybrid'),
        ('difference', 'variance', 'global'),
        ('difference', 'variance', 'hybrid'),
        ('difference', 'variance_eigenvalues', 'global'),

        # PCA basis - no variance_eigenvalues with hybrid
        ('pca', 'prediction', 'global'),
        ('pca', 'prediction', 'hybrid'),
        ('pca', 'variance', 'global'),
        ('pca', 'variance', 'hybrid'),

        # Noise basis - comprehensive
        ('noise', 'prediction', 'global'),
        ('noise', 'prediction', 'hybrid'),
        ('noise', 'variance', 'global'),
        ('noise', 'variance', 'hybrid'),
        ('noise', 'variance_eigenvalues', 'global'),

        # Random basis - no variance_eigenvalues
        ('random', 'prediction', 'global'),
        ('random', 'prediction', 'hybrid'),
        ('random', 'variance', 'global'),
        ('random', 'variance', 'hybrid'),
    ])
    def test_three_way_combinations(self, test_data, basis, criterion, threshold_method):
        """Test all valid three-way combinations."""
        opt = {
            'basis': basis,
            'criterion': criterion,
            'threshold_method': threshold_method,
            'wantfig': False,
            'wantverbose': False
        }

        results = psn(test_data, opt)

        assert 'denoiseddata' in results
        assert results['opt_used']['basis'] == basis
        assert results['opt_used']['criterion'] == criterion
        assert results['opt_used']['threshold_method'] == threshold_method
        assert not np.any(np.isnan(results['denoiseddata']))


class TestBasisOrderingCombinations:
    """Test basis_ordering with different bases."""

    @pytest.mark.parametrize("basis,basis_ordering", [
        # Signal basis with both orderings
        ('signal', 'eigenvalues'),
        ('signal', 'signalvariance'),

        # Difference basis with both orderings
        ('difference', 'eigenvalues'),
        ('difference', 'signalvariance'),

        # PCA basis with both orderings
        ('pca', 'eigenvalues'),
        ('pca', 'signalvariance'),

        # Noise basis with both orderings
        ('noise', 'eigenvalues'),
        ('noise', 'signalvariance'),

        # Random basis (will fall back to signalvariance)
        ('random', 'eigenvalues'),
        ('random', 'signalvariance'),
    ])
    def test_basis_ordering_combinations(self, test_data, basis, basis_ordering):
        """Test all basis-ordering combinations."""
        opt = {
            'basis': basis,
            'basis_ordering': basis_ordering,
            'wantfig': False,
            'wantverbose': False
        }

        results = psn(test_data, opt)

        assert 'denoiseddata' in results
        assert results['opt_used']['basis'] == basis
        # Random basis uses eigenvalues ordering (random basis generates eigenvalues)
        assert results['opt_used']['basis_ordering'] == basis_ordering


class TestVarianceThresholdCombinations:
    """Test variance_threshold with different criteria."""

    @pytest.mark.parametrize("criterion,variance_threshold", [
        ('variance', 0.80),
        ('variance', 0.90),
        ('variance', 0.95),
        ('variance', 0.99),
        ('variance_eigenvalues', 0.85),
        ('variance_eigenvalues', 0.95),
        ('variance_eigenvalues', 0.99),
    ])
    def test_variance_threshold_combinations(self, test_data, criterion, variance_threshold):
        """Test variance thresholds with variance criteria."""
        opt = {
            'criterion': criterion,
            'variance_threshold': variance_threshold,
            'threshold_method': 'global',  # Required for variance_eigenvalues
            'wantfig': False,
            'wantverbose': False
        }

        results = psn(test_data, opt)

        assert 'denoiseddata' in results
        assert results['opt_used']['criterion'] == criterion
        assert results['opt_used']['variance_threshold'] == variance_threshold


# ============================================================================
# Test Invalid Combinations (Should Fail or Warn)
# ============================================================================

class TestInvalidCombinations:
    """Test that invalid combinations are properly handled."""

    def test_variance_eigenvalues_with_custom_basis(self, test_data):
        """Test that variance_eigenvalues is not compatible with custom basis."""
        nunits = test_data.shape[0]
        custom_basis = np.linalg.qr(np.random.randn(nunits, nunits-2))[0]

        with pytest.raises((ValueError, AssertionError, KeyError)):
            psn(test_data, {
                'basis': custom_basis,
                'criterion': 'variance_eigenvalues',
                'threshold_method': 'global',
                'wantfig': False,
                'wantverbose': False
            })

    def test_variance_eigenvalues_with_random_basis(self, test_data):
        """Test that variance_eigenvalues is not compatible with random basis."""
        with pytest.raises((ValueError, AssertionError, KeyError)):
            psn(test_data, {
                'basis': 'random',
                'criterion': 'variance_eigenvalues',
                'threshold_method': 'global',
                'wantfig': False,
                'wantverbose': False
            })

    def test_variance_eigenvalues_with_hybrid(self, test_data):
        """Test that variance_eigenvalues is not compatible with hybrid method."""
        with pytest.raises((ValueError, AssertionError, KeyError)):
            psn(test_data, {
                'basis': 'signal',
                'criterion': 'variance_eigenvalues',
                'threshold_method': 'hybrid',
                'wantfig': False,
                'wantverbose': False
            })


# ============================================================================
# Test Consistency Across Combinations
# ============================================================================

class TestConsistencyAcrossCombinations:
    """Test that certain combinations produce consistent results."""

    def test_global_prediction_consistent_across_bases(self, test_data):
        """Test that global prediction gives consistent structure across bases."""
        bases = ['signal', 'difference', 'pca', 'noise']
        results_list = []

        for basis in bases:
            results = psn(test_data, {
                'basis': basis,
                'criterion': 'prediction',
                'threshold_method': 'global',
                'wantfig': False,
                'wantverbose': False
            })
            results_list.append(results)

        # All should produce valid denoisers with same shape
        for results in results_list:
            assert not np.any(np.isnan(results['denoiser']))
            assert results['denoiser'].shape == (test_data.shape[0], test_data.shape[0])

    def test_variance_criterion_honors_threshold(self, test_data):
        """Test that variance criterion respects the variance threshold."""
        thresholds = [0.80, 0.90, 0.95]
        best_thresholds = []

        for thresh in thresholds:
            results = psn(test_data, {
                'criterion': 'variance',
                'variance_threshold': thresh,
                'threshold_method': 'global',
                'wantfig': False,
                'wantverbose': False
            })
            best_t = results['best_threshold']
            if np.isscalar(best_t):
                best_thresholds.append(best_t)
            else:
                best_thresholds.append(best_t[0])

        # Higher variance threshold should generally retain more dimensions
        # (though not guaranteed in all cases due to discrete dimension selection)
        assert best_thresholds[-1] >= best_thresholds[0]

    def test_threshold_methods_ordering(self, test_data):
        """Test relationship between threshold methods."""
        results_global = psn(test_data, {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        results_hybrid = psn(test_data, {
            'threshold_method': 'hybrid',
            'wantfig': False,
            'wantverbose': False
        })

        # Global: all units same threshold
        assert len(np.unique(results_global['best_threshold'])) == 1

        # Hybrid: can have different thresholds per unit
        assert len(results_hybrid['best_threshold']) == test_data.shape[0]


# ============================================================================
# Test Edge Cases with Different Combinations
# ============================================================================

class TestEdgeCasesWithCombinations:
    """Test edge cases with different parameter combinations."""

    def test_minimal_data_all_methods(self):
        """Test minimal data (2 trials) with all threshold methods."""
        data = np.random.randn(5, 10, 2)

        for method in ['global', 'hybrid']:
            results = psn(data, {
                'threshold_method': method,
                'wantfig': False,
                'wantverbose': False
            })

            assert 'denoiseddata' in results
            assert results['denoiseddata'].shape == (5, 10)

    def test_forced_zero_threshold_all_bases(self, test_data):
        """Test forcing zero dimensions across all bases."""
        bases = ['signal', 'difference', 'pca', 'noise']

        for basis in bases:
            results = psn(test_data, {
                'basis': basis,
                'allowable_thresholds': [0],
                'threshold_method': 'global',
                'wantfig': False,
                'wantverbose': False
            })

            # Zero threshold should give unit means (constant per unit across conditions)
            best_t = results['best_threshold']
            if np.isscalar(best_t):
                assert best_t == 0
            else:
                assert np.all(best_t == 0)
            # Denoised data should equal unit means (broadcasted across conditions)
            expected = results['unit_means'][:, np.newaxis]
            assert np.allclose(results['denoiseddata'], expected)

    def test_forced_full_rank_all_bases(self, test_data):
        """Test forcing full rank denoising across all bases."""
        nunits = test_data.shape[0]
        bases = ['signal', 'difference', 'pca', 'noise']

        for basis in bases:
            results = psn(test_data, {
                'basis': basis,
                'allowable_thresholds': [nunits],
                'threshold_method': 'global',
                'wantfig': False,
                'wantverbose': False
            })

            # Full rank threshold
            assert np.all(results['best_threshold'] == nunits)


# ============================================================================
# Performance and Scaling Tests
# ============================================================================

class TestPerformanceWithCombinations:
    """Test that all combinations scale reasonably."""

    @pytest.mark.parametrize("basis,criterion,threshold_method", [
        ('signal', 'prediction', 'global'),
        ('difference', 'prediction', 'hybrid'),
        ('signal', 'variance', 'hybrid'),
    ])
    def test_combinations_with_large_data(self, basis, criterion, threshold_method):
        """Test that combinations work with larger datasets."""
        # Larger dataset
        np.random.seed(123)
        data = np.random.randn(30, 100, 10)

        results = psn(data, {
            'basis': basis,
            'criterion': criterion,
            'threshold_method': threshold_method,
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert results['denoiseddata'].shape == (30, 100)
        assert not np.any(np.isnan(results['denoiseddata']))


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
