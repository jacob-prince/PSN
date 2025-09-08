"""Tests for simulate_data.py"""

import numpy as np
import pytest
from psn.simulate import generate_data, _adjust_alignment_gradient_descent


class TestSimulateData:
    """Test class for PSN simulate data functionality."""

    def test_basic_alignment(self):
        """Test basic alignment properties for a simple case."""
        nvox = 10
        k = 3
        rng = np.random.RandomState(42)
        U_signal = np.linalg.qr(rng.randn(nvox, nvox))[0]
        U_noise = np.linalg.qr(rng.randn(nvox, nvox))[0]
        
        # Test perfect alignment (alpha = 1)
        U_aligned = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha=1.0, k=k, verbose=False)
        alignments = [np.abs(np.dot(U_signal[:, i], U_aligned[:, i])) for i in range(k)]
        assert np.mean(alignments) > 0.8, "Failed perfect alignment"
        
        # Test perfect orthogonality (alpha = 0)
        U_orthogonal = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha=0.0, k=k, verbose=False)
        alignments = [np.abs(np.dot(U_signal[:, i], U_orthogonal[:, i])) for i in range(k)]
        assert np.mean(alignments) < 0.2, "Failed perfect orthogonality"
        
        # Test partial alignment (alpha = 0.5)
        U_partial = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha=0.5, k=k, verbose=False)
        alignments = [np.abs(np.dot(U_signal[:, i], U_partial[:, i])) for i in range(k)]
        assert abs(np.mean(alignments) - 0.5) < 0.2, "Failed partial alignment"

    def test_orthonormality_preservation(self):
        """Test that the adjusted basis remains orthonormal."""
        nvox_values = [5, 10, 20]
        k_values = [1, 3, 5, 10]
        alpha_values = [0.0, 0.3, 0.7, 1.0]
        
        rng = np.random.RandomState(42)
        
        for nvox in nvox_values:
            U_signal = np.linalg.qr(rng.randn(nvox, nvox))[0]
            U_noise = np.linalg.qr(rng.randn(nvox, nvox))[0]
            
            for k in k_values:
                if k > nvox:
                    continue
                    
                for alpha in alpha_values:
                    U_adjusted = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha, k, verbose=False)
                    
                    # Check orthonormality
                    product = U_adjusted.T @ U_adjusted
                    np.testing.assert_allclose(
                        product, np.eye(nvox), 
                        rtol=1e-3, atol=1e-3,  # More lenient tolerance for gradient descent
                        err_msg=f"Failed orthonormality for nvox={nvox}, k={k}, alpha={alpha}"
                    )

    def test_extreme_cases(self):
        """Test alignment behavior in extreme cases."""
        nvox = 10
        rng = np.random.RandomState(42)
        U_signal = np.linalg.qr(rng.randn(nvox, nvox))[0]
        U_noise = np.linalg.qr(rng.randn(nvox, nvox))[0]
        
        # Test k=0
        U_adjusted = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha=0.5, k=0, verbose=False)
        np.testing.assert_allclose(U_adjusted, U_noise)
        
        # Test k=1
        U_adjusted = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha=0.5, k=1, verbose=False)
        alignment = np.abs(np.dot(U_signal[:, 0], U_adjusted[:, 0]))
        assert abs(alignment - 0.5) < 0.2
        
        # Test k=nvox with different alphas
        for alpha in [0.0, 0.5, 1.0]:
            U_adjusted = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha=alpha, k=nvox, verbose=False)
            alignments = [np.abs(np.dot(U_signal[:, i], U_adjusted[:, i])) for i in range(nvox)]
            avg_alignment = np.mean(alignments)
            if alpha == 0.0:
                assert avg_alignment < 0.2
            elif alpha == 1.0:
                assert avg_alignment > 0.8
            else:
                assert abs(avg_alignment - alpha) < 0.2

    def test_monotonicity(self):
        """Test that alignment increases monotonically with alpha."""
        nvox = 10
        k = 3
        alpha_values = np.linspace(0, 1, 11)
        
        rng = np.random.RandomState(42)
        U_signal = np.linalg.qr(rng.randn(nvox, nvox))[0]
        U_noise = np.linalg.qr(rng.randn(nvox, nvox))[0]
        
        avg_alignments = []
        for alpha in alpha_values:
            U_adjusted = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha, k, verbose=False)
            alignments = [np.abs(np.dot(U_signal[:, i], U_adjusted[:, i])) for i in range(k)]
            avg_alignments.append(np.mean(alignments))
        
        # Check monotonicity with some tolerance for numerical issues
        diffs = np.diff(avg_alignments)
        assert np.all(diffs >= -0.1), "Alignment not monotonic with alpha"
        
        # Check endpoints
        assert avg_alignments[0] < 0.2, "Initial alignment too high"
        assert avg_alignments[-1] > 0.8, "Final alignment too low"

    def test_stability(self):
        """Test stability across different random initializations and dimensions."""
        nvox_values = [5, 10, 20]
        k_values = [1, 3, 5]
        alpha = 0.5
        n_repeats = 5
        
        for nvox in nvox_values:
            for k in k_values:
                if k > nvox:
                    continue
                    
                avg_alignments = []
                for seed in range(n_repeats):
                    rng = np.random.RandomState(seed)
                    U_signal = np.linalg.qr(rng.randn(nvox, nvox))[0]
                    U_noise = np.linalg.qr(rng.randn(nvox, nvox))[0]
                    
                    U_adjusted = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha, k, verbose=False)
                    alignments = [np.abs(np.dot(U_signal[:, i], U_adjusted[:, i])) for i in range(k)]
                    avg_alignments.append(np.mean(alignments))
                
                # Check consistency across random initializations
                assert abs(np.mean(avg_alignments) - alpha) < 0.2, \
                    f"Failed stability test for nvox={nvox}, k={k}"
                assert np.std(avg_alignments) < 0.1, \
                    f"Alignment too variable for nvox={nvox}, k={k}"

    def test_full_pipeline(self):
        """Test alignment in the context of the full data generation pipeline."""
        nvox = 20
        ncond = 10
        ntrial = 5
        k = 3
        
        for alpha in [0.0, 0.5, 1.0]:
            # Generate data
            _, _, ground_truth = generate_data(
                nvox=nvox,
                ncond=ncond,
                ntrial=ntrial,
                signal_decay=1.0,
                noise_decay=1.0,
                align_alpha=alpha,
                align_k=k,
                random_seed=42,
                verbose=False
            )
            
            # Check alignment properties
            U_signal = ground_truth['U_signal']
            U_noise = ground_truth['U_noise']
            
            # Verify alignment
            alignments = [np.abs(np.dot(U_signal[:, i], U_noise[:, i])) for i in range(k)]
            avg_alignment = np.mean(alignments)
            
            if alpha == 0.0:
                assert avg_alignment < 0.2
            elif alpha == 1.0:
                assert avg_alignment > 0.8
            else:
                assert abs(avg_alignment - alpha) < 0.2
            
            # Verify orthonormality
            np.testing.assert_allclose(
                U_noise.T @ U_noise,
                np.eye(nvox),
                rtol=1e-3, atol=1e-3  # More lenient for integration test
            )

    def test_numerical_stability(self):
        """Test behavior with numerically challenging inputs."""
        nvox = 10
        k = 3
        rng = np.random.RandomState(42)
        
        # Test with nearly identical signal and noise bases
        U_signal = np.linalg.qr(rng.randn(nvox, nvox))[0]
        U_noise = U_signal + 1e-10 * rng.randn(nvox, nvox)
        U_noise = np.linalg.qr(U_noise)[0]
        
        for alpha in [0.0, 0.5, 1.0]:
            U_adjusted = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha, k, verbose=False)
            
            # Check orthonormality
            np.testing.assert_allclose(
                U_adjusted.T @ U_adjusted,
                np.eye(nvox),
                rtol=1e-5, atol=1e-5
            )
            
            # Check alignment - be more lenient for the challenging case
            alignments = [np.abs(np.dot(U_signal[:, i], U_adjusted[:, i])) for i in range(k)]
            avg_alignment = np.mean(alignments)
            
            if alpha == 0.0:
                # For orthogonality, allow more tolerance in this edge case
                assert avg_alignment < 0.3
            elif alpha == 1.0:
                assert avg_alignment > 0.8
            else:
                assert abs(avg_alignment - alpha) < 0.3  # More tolerant for edge case 

    def test_signal_noise_ratio(self):
        """Test that signal-to-noise ratio is controlled by decay parameters."""
        nvox = 20
        ncond = 10
        ntrial = 5
        
        # Test that decreasing signal decay relative to noise decay increases signal power
        train_data_low_snr, _, _ = generate_data(
            nvox=nvox,
            ncond=ncond,
            ntrial=ntrial,
            signal_decay=1.0,  # Fast decay = less signal
            noise_decay=0.1,   # Slow decay = more noise
            random_seed=42,
            verbose=False
        )
        
        train_data_high_snr, _, _ = generate_data(
            nvox=nvox,
            ncond=ncond,
            ntrial=ntrial,
            signal_decay=0.1,  # Slow decay = more signal
            noise_decay=1.0,   # Fast decay = less noise
            random_seed=42,
            verbose=False
        )
        
        # Calculate signal and noise power for both cases
        trial_means_low = np.mean(train_data_low_snr, axis=2)
        noise_low = train_data_low_snr - trial_means_low[:, :, np.newaxis]
        signal_power_low = np.var(trial_means_low)
        noise_power_low = np.var(noise_low)
        snr_low = signal_power_low / noise_power_low
        
        trial_means_high = np.mean(train_data_high_snr, axis=2)
        noise_high = train_data_high_snr - trial_means_high[:, :, np.newaxis]
        signal_power_high = np.var(trial_means_high)
        noise_power_high = np.var(noise_high)
        snr_high = signal_power_high / noise_power_high
        
        # Test that SNR increases when signal decay decreases relative to noise decay
        assert snr_high > snr_low, "SNR should increase when signal decay decreases relative to noise decay"

    def test_trial_independence(self):
        """Test that trials are independently generated."""
        nvox = 20
        ncond = 10
        ntrial = 10
        
        train_data, test_data, _ = generate_data(
            nvox=nvox,
            ncond=ncond,
            ntrial=ntrial,
            random_seed=42,
            verbose=False
        )
        
        # Check correlations between trials after removing condition means
        trial_means = np.mean(train_data, axis=2)
        noise = train_data - trial_means[:, :, np.newaxis]
        
        # Check correlations between noise components
        for i in range(ntrial):
            for j in range(i+1, ntrial):
                trial_i = noise[:, :, i].flatten()
                trial_j = noise[:, :, j].flatten()
                correlation = np.corrcoef(trial_i, trial_j)[0, 1]
                assert abs(correlation) < 0.7, f"Trial noise components {i} and {j} are too correlated"

    def test_condition_structure(self):
        """Test that condition structure is preserved across trials."""
        nvox = 20
        ncond = 10
        ntrial = 5
        
        train_data, test_data, ground_truth = generate_data(
            nvox=nvox,
            ncond=ncond,
            ntrial=ntrial,
            signal_decay=1.0,
            noise_decay=0.1,  # Low noise to see condition structure
            random_seed=42,
            verbose=False
        )
        
        # Calculate mean pattern for each condition
        condition_means = np.mean(train_data, axis=2)  # Average across trials
        
        # Check that conditions are distinct
        for i in range(ncond):
            for j in range(i+1, ncond):
                pattern_i = condition_means[:, i]
                pattern_j = condition_means[:, j]
                correlation = np.corrcoef(pattern_i, pattern_j)[0, 1]
                assert abs(correlation) < 0.9, f"Conditions {i} and {j} are too similar"

    def test_dimensionality_scaling(self):
        """Test behavior with different dimensionality ratios."""
        test_configs = [
            (10, 5, 3),    # More units than conditions
            (5, 10, 3),    # More conditions than units
            (10, 10, 3),   # Equal units and conditions
            (3, 3, 10),    # Many trials
            (50, 5, 2)     # High-dimensional units
        ]
        
        for nvox, ncond, ntrial in test_configs:
            train_data, test_data, ground_truth = generate_data(
                nvox=nvox,
                ncond=ncond,
                ntrial=ntrial,
                random_seed=42,
                verbose=False
            )
            
            assert train_data.shape == (nvox, ncond, ntrial)
            assert test_data.shape == (nvox, ncond, ntrial)
            assert ground_truth['U_signal'].shape == (nvox, nvox)
            assert ground_truth['U_noise'].shape == (nvox, nvox)

    def test_random_seed_reproducibility(self):
        """Test that random seed controls reproducibility."""
        nvox = 20
        ncond = 10
        ntrial = 5
        
        # Generate two datasets with same seed
        data1, _, _ = generate_data(nvox=nvox, ncond=ncond, ntrial=ntrial, random_seed=42, verbose=False)
        data2, _, _ = generate_data(nvox=nvox, ncond=ncond, ntrial=ntrial, random_seed=42, verbose=False)
        
        # Generate dataset with different seed
        data3, _, _ = generate_data(nvox=nvox, ncond=ncond, ntrial=ntrial, random_seed=43, verbose=False)
        
        # Same seed should give identical results
        np.testing.assert_array_equal(data1, data2)
        
        # Different seeds should give different results
        assert not np.array_equal(data1, data3)

    def test_basis_properties(self):
        """Test properties of signal and noise bases."""
        nvox = 20
        ncond = 10
        ntrial = 5
        
        _, _, ground_truth = generate_data(
            nvox=nvox,
            ncond=ncond,
            ntrial=ntrial,
            random_seed=42,
            verbose=False
        )
        
        U_signal = ground_truth['U_signal']
        U_noise = ground_truth['U_noise']
        
        # Test orthonormality with appropriate tolerance
        np.testing.assert_allclose(U_signal.T @ U_signal, np.eye(nvox), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(U_noise.T @ U_noise, np.eye(nvox), rtol=1e-5, atol=1e-5)
        
        # Test span
        assert np.linalg.matrix_rank(U_signal) == nvox
        assert np.linalg.matrix_rank(U_noise) == nvox

    def test_train_test_independence(self):
        """Test that train and test datasets are independently generated."""
        nvox = 20
        ncond = 10
        ntrial = 5
        
        train_data, test_data, _ = generate_data(
            nvox=nvox,
            ncond=ncond,
            ntrial=ntrial,
            random_seed=42,
            verbose=False
        )
        
        # Remove condition means before checking correlations
        train_means = np.mean(train_data, axis=2)
        test_means = np.mean(test_data, axis=2)
        train_noise = train_data - train_means[:, :, np.newaxis]
        test_noise = test_data - test_means[:, :, np.newaxis]
        
        # Check correlations between train and test noise components
        for i in range(ntrial):
            for j in range(ntrial):
                train_trial = train_noise[:, :, i].flatten()
                test_trial = test_noise[:, :, j].flatten()
                correlation = np.corrcoef(train_trial, test_trial)[0, 1]
                assert abs(correlation) < 0.7, f"Train and test noise components {i} and {j} are too correlated"

    def test_edge_case_dimensions(self):
        """Test edge cases for data dimensions."""
        test_configs = [
            (2, 2, 2),     # Minimum valid dimensions
            (100, 2, 2),   # Very high dimensional units
            (2, 100, 2),   # Very high dimensional conditions
            (2, 2, 100),   # Very high dimensional trials
            (50, 50, 2)    # Equal high dimensions
        ]
        
        for nvox, ncond, ntrial in test_configs:
            train_data, test_data, ground_truth = generate_data(
                nvox=nvox,
                ncond=ncond,
                ntrial=ntrial,
                random_seed=42,
                verbose=False
            )
            
            assert train_data.shape == (nvox, ncond, ntrial)
            assert test_data.shape == (nvox, ncond, ntrial)
            assert ground_truth['U_signal'].shape == (nvox, nvox)
            assert ground_truth['U_noise'].shape == (nvox, nvox)
            
            # Check basic properties are maintained even in edge cases
            assert np.all(np.isfinite(train_data))
            assert np.all(np.isfinite(test_data))
            assert np.allclose(ground_truth['U_signal'].T @ ground_truth['U_signal'], np.eye(nvox))
            assert np.allclose(ground_truth['U_noise'].T @ ground_truth['U_noise'], np.eye(nvox))


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
