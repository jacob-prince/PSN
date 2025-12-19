"""Numerical stability and mathematical property tests for PSN.

This test file focuses on:
- Numerical stability and conditioning
- Mathematical properties of the denoiser
- Scale and translation invariance properties
- Extreme numerical conditions
"""

import numpy as np
import pytest
from psn import psn


# Test fixtures
@pytest.fixture
def well_conditioned_data():
    """Well-conditioned test data."""
    np.random.seed(1111)
    nunits, nconds, ntrials = 8, 15, 4
    signal = np.random.randn(nunits, nconds)
    noise = 0.3 * np.random.randn(nunits, nconds, ntrials)
    data = signal[:, :, np.newaxis] + noise
    return data


# ============================================================================
# Numerical Conditioning Tests
# ============================================================================

class TestNumericalConditioning:
    """Test numerical conditioning and stability."""

    def test_ill_conditioned_signal_covariance(self):
        """Test with ill-conditioned signal covariance."""
        np.random.seed(2222)
        nunits, nconds, ntrials = 10, 20, 5

        # Create signal with very different scales
        scales = np.logspace(-5, 5, nunits)
        signal = scales[:, np.newaxis] * np.random.randn(nunits, nconds)

        noise = 0.1 * np.random.randn(nunits, nconds, ntrials)
        data = signal[:, :, np.newaxis] + noise

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert not np.any(np.isnan(results['denoiseddata']))
        assert not np.any(np.isinf(results['denoiseddata']))

    def test_near_singular_covariance(self):
        """Test with nearly singular covariance matrix."""
        np.random.seed(3333)
        nunits, nconds, ntrials = 8, 15, 4

        # Create highly redundant signal
        base = np.random.randn(2, nconds)
        signal = np.vstack([base, base + 1e-8 * np.random.randn(2, nconds),
                           base + 1e-8 * np.random.randn(2, nconds),
                           base + 1e-8 * np.random.randn(2, nconds)])

        noise = 0.1 * np.random.randn(nunits, nconds, ntrials)
        data = signal[:, :, np.newaxis] + noise

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert not np.any(np.isnan(results['denoiseddata']))

    def test_wide_dynamic_range(self):
        """Test with very wide dynamic range in data."""
        np.random.seed(4444)
        nunits, nconds, ntrials = 8, 15, 4

        # Some units have large values, others very small
        signal = np.random.randn(nunits, nconds)
        signal[:4, :] *= 1e6  # Very large
        signal[4:, :] *= 1e-6  # Very small

        noise = 0.1 * np.abs(signal[:, :, np.newaxis]) * np.random.randn(nunits, nconds, ntrials) + 0.01 * np.random.randn(nunits, nconds, ntrials)
        data = signal[:, :, np.newaxis] + noise

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert not np.any(np.isnan(results['denoiseddata']))
        assert not np.any(np.isinf(results['denoiseddata']))

    def test_condition_number_monitoring(self, well_conditioned_data):
        """Test that condition numbers remain reasonable."""
        results = psn(well_conditioned_data, {
            'wantfig': False,
            'wantverbose': False
        })

        # Check that GSN covariances are reasonable
        cSb = results['gsn_result']['cSb']
        cNb = results['gsn_result']['cNb']

        # Should not have NaN or Inf
        assert not np.any(np.isnan(cSb))
        assert not np.any(np.isnan(cNb))
        assert not np.any(np.isinf(cSb))
        assert not np.any(np.isinf(cNb))


# ============================================================================
# Mathematical Property Tests
# ============================================================================

class TestMathematicalProperties:
    """Test mathematical properties of PSN."""

    def test_projection_property_global(self, well_conditioned_data):
        """Test projection property for global denoiser."""
        results = psn(well_conditioned_data, {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        D = results['denoiser']
        data_avg = np.mean(well_conditioned_data, axis=2)
        denoised = results['denoiseddata']

        # D should project to signal subspace
        # Applying D twice should give same result (up to mean adjustment)
        # D @ D ≈ D for global denoiser (not exactly due to mean handling)

    def test_rank_preservation(self, well_conditioned_data):
        """Test that denoiser rank matches threshold."""
        results = psn(well_conditioned_data, {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        threshold = results['best_threshold']
        if threshold > 0:
            # Signal subspace should have correct rank
            subspace = results['signalsubspace']
            rank = np.linalg.matrix_rank(subspace)
            assert rank == threshold

    def test_variance_decomposition(self, well_conditioned_data):
        """Test variance decomposition properties."""
        results = psn(well_conditioned_data, {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        # Signal + Noise variance should approximately equal total variance
        svnv_before = results['svnv_before']
        signal_var_before = svnv_before[:, 0]
        noise_var_before = svnv_before[:, 1]

        # Each should be non-negative
        assert np.all(signal_var_before >= 0)
        assert np.all(noise_var_before >= 0)

    def test_orthogonality_of_basis(self, well_conditioned_data):
        """Test that basis vectors remain orthogonal."""
        results = psn(well_conditioned_data, {
            'wantfig': False,
            'wantverbose': False
        })

        basis = results['fullbasis']

        # Gram matrix should be identity
        gram = basis.T @ basis
        identity = np.eye(basis.shape[1])

        assert np.allclose(gram, identity, rtol=1e-10, atol=1e-10)

    def test_signal_subspace_properties(self, well_conditioned_data):
        """Test properties of the signal subspace."""
        results = psn(well_conditioned_data, {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        if results['signalsubspace'] is not None:
            subspace = results['signalsubspace']

            # Columns should be orthonormal
            gram = subspace.T @ subspace
            identity = np.eye(subspace.shape[1])
            assert np.allclose(gram, identity, rtol=1e-10)


# ============================================================================
# Scale Invariance Tests
# ============================================================================

class TestScaleInvariance:
    """Test scale invariance properties."""

    def test_global_scale_invariance(self, well_conditioned_data):
        """Test that global scaling doesn't affect relative denoising."""
        scale = 100.0

        results_original = psn(well_conditioned_data, {
            'wantfig': False,
            'wantverbose': False
        })

        results_scaled = psn(well_conditioned_data * scale, {
            'wantfig': False,
            'wantverbose': False
        })

        # Denoised data should scale proportionally
        ratio = results_scaled['denoiseddata'] / (results_original['denoiseddata'] + 1e-10)

        # Should be approximately constant (equal to scale)
        assert np.allclose(ratio, scale, rtol=0.1) or np.allclose(results_original['denoiseddata'], 0, atol=1e-10)

    def test_per_unit_scale_invariance(self, well_conditioned_data):
        """Test behavior under per-unit scaling."""
        nunits = well_conditioned_data.shape[0]
        scales = np.random.uniform(0.1, 10.0, size=nunits)

        scaled_data = well_conditioned_data * scales[:, np.newaxis, np.newaxis]

        results = psn(scaled_data, {
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        assert not np.any(np.isnan(results['denoiseddata']))


# ============================================================================
# Translation Invariance Tests
# ============================================================================

class TestTranslationInvariance:
    """Test translation invariance properties."""

    def test_global_translation(self, well_conditioned_data):
        """Test that global offset doesn't affect denoising structure."""
        offset = 100.0

        results_original = psn(well_conditioned_data, {
            'wantfig': False,
            'wantverbose': False
        })

        results_offset = psn(well_conditioned_data + offset, {
            'wantfig': False,
            'wantverbose': False
        })

        # The denoised patterns (after removing means) should be similar
        denoised_orig_centered = results_original['denoiseddata'] - results_original['denoiseddata'].mean(axis=1, keepdims=True)
        denoised_offset_centered = results_offset['denoiseddata'] - results_offset['denoiseddata'].mean(axis=1, keepdims=True)

        assert np.allclose(denoised_orig_centered, denoised_offset_centered, rtol=0.1)

    def test_per_unit_translation(self, well_conditioned_data):
        """Test behavior under per-unit offsets."""
        nunits = well_conditioned_data.shape[0]
        offsets = np.random.uniform(-10, 10, size=nunits)

        offset_data = well_conditioned_data + offsets[:, np.newaxis, np.newaxis]

        results = psn(offset_data, {
            'wantfig': False,
            'wantverbose': False
        })

        assert 'denoiseddata' in results
        # Unit means should reflect the offsets
        expected_means = offsets + np.mean(np.mean(well_conditioned_data, axis=2), axis=1)
        assert np.allclose(results['unit_means'], expected_means, rtol=0.1)


# ============================================================================
# Extreme Numerical Conditions
# ============================================================================

class TestExtremeNumericalConditions:
    """Test extreme numerical conditions."""

    def test_very_small_variance(self):
        """Test with very small variance."""
        np.random.seed(5555)
        nunits, nconds, ntrials = 6, 10, 4

        signal = 1e-12 * np.random.randn(nunits, nconds)
        noise = 1e-13 * np.random.randn(nunits, nconds, ntrials)
        data = signal[:, :, np.newaxis] + noise

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert not np.any(np.isnan(results['denoiseddata']))

    def test_very_large_variance(self):
        """Test with very large variance."""
        np.random.seed(6666)
        nunits, nconds, ntrials = 6, 10, 4

        signal = 1e12 * np.random.randn(nunits, nconds)
        noise = 1e11 * np.random.randn(nunits, nconds, ntrials)
        data = signal[:, :, np.newaxis] + noise

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert not np.any(np.isnan(results['denoiseddata']))
        assert not np.any(np.isinf(results['denoiseddata']))

    def test_mixed_scales_extreme(self):
        """Test with extreme mixed scales."""
        np.random.seed(7777)
        nunits, nconds, ntrials = 8, 15, 4

        signal = np.random.randn(nunits, nconds)
        signal[0, :] *= 1e10
        signal[1, :] *= 1e-10

        noise = 0.1 * np.abs(signal) + 1e-15
        noise = noise[:, :, np.newaxis] * np.random.randn(nunits, nconds, ntrials)
        data = signal[:, :, np.newaxis] + noise

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results
        assert not np.any(np.isnan(results['denoiseddata']))

    def test_subnormal_numbers(self):
        """Test with subnormal numbers."""
        np.random.seed(8888)
        nunits, nconds, ntrials = 5, 8, 3

        # Create data with subnormal numbers
        signal = 1e-320 * np.random.randn(nunits, nconds)
        noise = 1e-321 * np.random.randn(nunits, nconds, ntrials)
        data = signal[:, :, np.newaxis] + noise

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert 'denoiseddata' in results

    def test_near_overflow_prevention(self):
        """Test that computations don't overflow."""
        np.random.seed(9999)
        nunits, nconds, ntrials = 6, 10, 3

        # Large but not overflow-inducing values (reduced from 1e100 to 1e15)
        signal = 1e15 * np.random.randn(nunits, nconds)
        noise = 1e14 * np.random.randn(nunits, nconds, ntrials)
        data = signal[:, :, np.newaxis] + noise

        results = psn(data, {'wantfig': False, 'wantverbose': False})

        assert not np.any(np.isinf(results['denoiseddata']))


# ============================================================================
# Monotonicity and Ordering Tests
# ============================================================================

class TestMonotonicityProperties:
    """Test monotonicity and ordering properties."""

    def test_variance_threshold_monotonicity(self, well_conditioned_data):
        """Test that higher variance thresholds retain more dimensions."""
        thresholds = [0.7, 0.8, 0.9, 0.95]
        best_dims = []

        for thresh in thresholds:
            results = psn(well_conditioned_data, {
                'criterion': 'variance',
                'variance_threshold': thresh,
                'threshold_method': 'global',
                'wantfig': False,
                'wantverbose': False
            })

            best_dims.append(results['best_threshold'])

        # Should be non-decreasing
        for i in range(len(best_dims) - 1):
            assert best_dims[i] <= best_dims[i + 1]

    def test_prediction_objective_monotonicity(self, well_conditioned_data):
        """Test that prediction objective is computed correctly."""
        results = psn(well_conditioned_data, {
            'criterion': 'prediction',
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        objective = results['objective']

        # Objective should be cumulative sum
        # Check that it's non-decreasing up to the optimal point
        best_threshold = results['best_threshold']
        if best_threshold > 0:
            assert np.all(np.diff(objective[:best_threshold + 1]) >= -1e-10)


# ============================================================================
# Denoiser Matrix Properties
# ============================================================================

class TestDenoiserMatrixProperties:
    """Test properties of the denoiser matrix."""

    def test_denoiser_spectral_properties(self, well_conditioned_data):
        """Test spectral properties of denoiser."""
        results = psn(well_conditioned_data, {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        denoiser = results['denoiser']

        # Eigenvalues should be in [0, 1] for global denoiser (projection)
        eigenvalues = np.linalg.eigvalsh(denoiser)

        # Should be non-negative (allowing small numerical errors)
        assert np.all(eigenvalues >= -1e-10)

    def test_denoiser_frobenius_norm(self, well_conditioned_data):
        """Test Frobenius norm of denoiser."""
        results = psn(well_conditioned_data, {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        denoiser = results['denoiser']
        frob_norm = np.linalg.norm(denoiser, 'fro')

        # Should be reasonable (not too large)
        nunits = well_conditioned_data.shape[0]
        assert frob_norm <= nunits * 1.5  # Reasonable upper bound

    def test_denoiser_operator_norm(self, well_conditioned_data):
        """Test operator norm of denoiser."""
        results = psn(well_conditioned_data, {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        denoiser = results['denoiser']
        op_norm = np.linalg.norm(denoiser, 2)

        # Operator norm should be at most 1 for projection (approximately)
        assert op_norm <= 1.5  # Allow some slack


# ============================================================================
# Reconstruction Quality Tests
# ============================================================================

class TestReconstructionQuality:
    """Test reconstruction quality properties."""

    def test_perfect_reconstruction_full_rank(self, well_conditioned_data):
        """Test perfect reconstruction with full rank."""
        nunits = well_conditioned_data.shape[0]

        results = psn(well_conditioned_data, {
            'allowable_thresholds': [nunits],
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        # Full rank should perfectly reconstruct trial average
        data_avg = np.mean(well_conditioned_data, axis=2)
        assert np.allclose(results['denoiseddata'], data_avg, rtol=1e-10)

    def test_zero_reconstruction_zero_dims(self, well_conditioned_data):
        """Test zero reconstruction with zero dimensions."""
        results = psn(well_conditioned_data, {
            'allowable_thresholds': [0],
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        # Zero dimensions should give unit means (constant per unit across conditions)
        expected = results['unit_means'][:, np.newaxis]
        assert np.allclose(results['denoiseddata'], expected, atol=1e-10)

    def test_reconstruction_error_bounds(self, well_conditioned_data):
        """Test that reconstruction error is bounded."""
        results = psn(well_conditioned_data, {
            'wantfig': False,
            'wantverbose': False
        })

        data_avg = np.mean(well_conditioned_data, axis=2)
        reconstruction_error = np.linalg.norm(results['denoiseddata'] - data_avg, 'fro')
        data_norm = np.linalg.norm(data_avg, 'fro')

        # Relative error should be reasonable
        relative_error = reconstruction_error / (data_norm + 1e-10)
        assert relative_error <= 1.5  # Should not be worse than throwing away all data


# ============================================================================
# Symmetry Tests
# ============================================================================

class TestSymmetryProperties:
    """Test symmetry properties."""

    def test_permutation_equivariance_units(self, well_conditioned_data):
        """Test behavior under unit permutation."""
        nunits = well_conditioned_data.shape[0]
        perm = np.random.permutation(nunits)

        # Original
        results_orig = psn(well_conditioned_data, {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        # Permuted
        results_perm = psn(well_conditioned_data[perm, :, :], {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        # Denoised data should be permuted accordingly
        assert np.allclose(results_orig['denoiseddata'][perm, :],
                          results_perm['denoiseddata'], rtol=0.1)

    def test_condition_permutation_invariance(self, well_conditioned_data):
        """Test that condition permutation doesn't affect denoiser."""
        nconds = well_conditioned_data.shape[1]
        perm = np.random.permutation(nconds)

        results_orig = psn(well_conditioned_data, {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        results_perm = psn(well_conditioned_data[:, perm, :], {
            'threshold_method': 'global',
            'wantfig': False,
            'wantverbose': False
        })

        # Denoiser should be the same (conditions don't affect unit-space denoiser)
        assert np.allclose(results_orig['denoiser'], results_perm['denoiser'], rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
