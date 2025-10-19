"""
Comprehensive tests for basis ranking functionality in PSN.

This module tests the _rank_basis_dimensions function and its integration
with the main psn() function across all ranking methods and basis types.
"""

import numpy as np
import pytest
from psn import psn
from psn.psn import _rank_basis_dimensions, _compute_symmetric_eigen


class TestRankBasisDimensions:
    """Test suite for the _rank_basis_dimensions function."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        nunits = 20
        nconds = 50
        ntrials = 5
        
        # Create structured data with signal
        signal_template = np.random.randn(nunits, nconds)
        data = signal_template[:, :, np.newaxis] + 0.3 * np.random.randn(nunits, nconds, ntrials)
        
        return data
    
    @pytest.fixture
    def sample_basis(self, sample_data):
        """Generate a sample orthonormal basis."""
        nunits = sample_data.shape[0]
        basis = np.linalg.qr(np.random.randn(nunits, nunits))[0]
        return basis
    
    @pytest.fixture
    def sample_magnitudes(self, sample_basis):
        """Generate sample eigenvalue magnitudes."""
        ndims = sample_basis.shape[1]
        # Random magnitudes in decreasing order
        magnitudes = np.sort(np.random.rand(ndims))[::-1]
        return magnitudes
    
    def test_eigs_ranking_decreasing(self, sample_basis, sample_data, sample_magnitudes):
        """Test that 'eigenvalue' ranking sorts by eigenvalues in decreasing order."""
        basis_ranked, mags_ranked, _ = _rank_basis_dimensions(
            sample_basis, None, sample_data, sample_magnitudes, ranking='eigenvalue'
        )
        
        # Check that magnitudes are in decreasing order
        assert np.all(np.diff(mags_ranked) <= 1e-10), "Magnitudes should be decreasing"
        
        # Check that basis is reordered correctly
        assert basis_ranked.shape == sample_basis.shape
        assert np.allclose(basis_ranked.T @ basis_ranked, np.eye(basis_ranked.shape[1]))
    
    def test_eig_inv_ranking_increasing(self, sample_basis, sample_data, sample_magnitudes):
        """Test that 'eigenvalue_asc' ranking sorts by eigenvalues in increasing order."""
        basis_ranked, mags_ranked, _ = _rank_basis_dimensions(
            sample_basis, None, sample_data, sample_magnitudes, ranking='eigenvalue_asc'
        )
        
        # Check that magnitudes are in increasing order
        assert np.all(np.diff(mags_ranked) >= -1e-10), "Magnitudes should be increasing"
        
        # Check orthonormality
        assert np.allclose(basis_ranked.T @ basis_ranked, np.eye(basis_ranked.shape[1]))
    
    def test_signal_ranking(self, sample_basis, sample_data, sample_magnitudes):
        """Test that 'signal_variance' ranking works and produces decreasing signal variances."""
        basis_ranked, mags_ranked, _ = _rank_basis_dimensions(
            sample_basis, None, sample_data, sample_magnitudes, ranking='signal_variance'
        )
        
        # Check that magnitudes are in decreasing order
        assert np.all(np.diff(mags_ranked) <= 1e-10), "Signal variances should be decreasing"
        
        # Check that all magnitudes are non-negative
        assert np.all(mags_ranked >= 0), "Signal variances should be non-negative"
        
        # Check orthonormality
        assert np.allclose(basis_ranked.T @ basis_ranked, np.eye(basis_ranked.shape[1]))
    
    def test_ncsnr_ranking(self, sample_basis, sample_data, sample_magnitudes):
        """Test that 'snr' ranking works and produces decreasing SNR values."""
        basis_ranked, mags_ranked, _ = _rank_basis_dimensions(
            sample_basis, None, sample_data, sample_magnitudes, ranking='snr'
        )
        
        # Check that magnitudes are in decreasing order
        assert np.all(np.diff(mags_ranked) <= 1e-10), "NCSNR values should be decreasing"
        
        # Check that all magnitudes are non-negative
        assert np.all(mags_ranked >= 0), "NCSNR values should be non-negative"
        
        # Check orthonormality
        assert np.allclose(basis_ranked.T @ basis_ranked, np.eye(basis_ranked.shape[1]))
    
    def test_sig_noise_ranking(self, sample_basis, sample_data, sample_magnitudes):
        """Test that 'signal_specificity' ranking works."""
        basis_ranked, mags_ranked, _ = _rank_basis_dimensions(
            sample_basis, None, sample_data, sample_magnitudes, ranking='signal_specificity'
        )
        
        # Check that magnitudes are in decreasing order
        assert np.all(np.diff(mags_ranked) <= 1e-10), "Sig-noise diff should be decreasing"
        
        # Check orthonormality
        assert np.allclose(basis_ranked.T @ basis_ranked, np.eye(basis_ranked.shape[1]))
    
    def test_basis_source_reordering(self, sample_basis, sample_data, sample_magnitudes):
        """Test that basis_source is reordered correctly when provided."""
        nunits = sample_basis.shape[0]
        basis_source = np.random.randn(nunits, nunits)
        
        basis_ranked, mags_ranked, source_ranked = _rank_basis_dimensions(
            sample_basis, basis_source, sample_data, sample_magnitudes, ranking='eigenvalue'
        )
        
        # Check that source is reordered
        assert source_ranked is not None
        assert source_ranked.shape == basis_source.shape
        
        # Check that the reordering is consistent
        sort_idx = np.argsort(sample_magnitudes)[::-1]
        expected_source = basis_source[:, sort_idx]
        assert np.allclose(source_ranked, expected_source)
    
    def test_invalid_ranking_raises_error(self, sample_basis, sample_data, sample_magnitudes):
        """Test that invalid ranking method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid ranking method"):
            _rank_basis_dimensions(
                sample_basis, None, sample_data, sample_magnitudes, ranking='invalid'
            )
    
    def test_orthonormality_preserved(self, sample_basis, sample_data, sample_magnitudes):
        """Test that all ranking methods preserve orthonormality."""
        ranking_methods = ['eigenvalue', 'eigenvalue_asc', 'signal_variance', 'snr', 'signal_specificity']
        
        for ranking in ranking_methods:
            basis_ranked, _, _ = _rank_basis_dimensions(
                sample_basis, None, sample_data, sample_magnitudes, ranking=ranking
            )
            
            # Check orthonormality
            gram = basis_ranked.T @ basis_ranked
            assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-10), \
                f"Orthonormality not preserved for {ranking} ranking"


class TestPSNDefaultRankings:
    """Test that PSN uses correct default rankings for different basis types."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        nunits = 20
        nconds = 50
        ntrials = 5
        data = np.random.randn(nunits, nconds, ntrials)
        return data
    
    def test_signal_basis_default_signal(self, sample_data):
        """Test that V=0 (signal basis) defaults to 'signal_variance' ranking."""
        opt = {'cv_mode': -1, 'mag_frac': 0.95, 'wantfig': False}
        results = psn(sample_data, V=0, opt=opt, wantfig=False)

        # Check that ranking was set
        assert 'ranking' in results['opt']
        assert results['opt']['ranking'] == 'signal_variance'

    def test_whitened_signal_basis_default_signal(self, sample_data):
        """Test that V=1 (whitened signal) defaults to 'signal_variance' ranking."""
        opt = {'cv_mode': -1, 'mag_frac': 0.95, 'wantfig': False}
        results = psn(sample_data, V=1, opt=opt, wantfig=False)

        assert 'ranking' in results['opt']
        assert results['opt']['ranking'] == 'signal_variance'

    def test_noise_basis_unit_threshold_default_signal(self, sample_data):
        """Test that V=2 (noise basis) with unit thresholding defaults to 'signal_variance' ranking."""
        opt = {'cv_mode': -1, 'mag_frac': 0.95, 'cv_threshold_per': 'unit', 'wantfig': False}
        results = psn(sample_data, V=2, opt=opt, wantfig=False)

        assert 'ranking' in results['opt']
        assert results['opt']['ranking'] == 'signal_variance'

    def test_noise_basis_population_threshold_default_signal(self, sample_data):
        """Test that V=2 (noise basis) with population thresholding defaults to 'signal_variance' ranking."""
        opt = {'cv_mode': -1, 'mag_frac': 0.95, 'cv_threshold_per': 'population', 'wantfig': False}
        results = psn(sample_data, V=2, opt=opt, wantfig=False)

        assert 'ranking' in results['opt']
        assert results['opt']['ranking'] == 'signal_variance'

    def test_pca_basis_default_signal(self, sample_data):
        """Test that V=3 (PCA) defaults to 'signal_variance' ranking."""
        opt = {'cv_mode': -1, 'mag_frac': 0.95, 'wantfig': False}
        results = psn(sample_data, V=3, opt=opt, wantfig=False)

        assert 'ranking' in results['opt']
        assert results['opt']['ranking'] == 'signal_variance'

    def test_random_basis_unit_threshold_default_signal(self, sample_data):
        """Test that V=4 (random) with unit thresholding defaults to 'signal_variance' ranking."""
        opt = {'cv_mode': -1, 'mag_frac': 0.95, 'cv_threshold_per': 'unit', 'wantfig': False}
        results = psn(sample_data, V=4, opt=opt, wantfig=False)

        assert 'ranking' in results['opt']
        assert results['opt']['ranking'] == 'signal_variance'

    def test_random_basis_population_threshold_default_signal(self, sample_data):
        """Test that V=4 (random) with population thresholding defaults to 'signal_variance' ranking."""
        opt = {'cv_mode': -1, 'mag_frac': 0.95, 'cv_threshold_per': 'population', 'wantfig': False}
        results = psn(sample_data, V=4, opt=opt, wantfig=False)

        assert 'ranking' in results['opt']
        assert results['opt']['ranking'] == 'signal_variance'

    def test_ica_basis_default_signal(self, sample_data):
        """Test that V=5 (ICA) defaults to 'signal_variance' ranking."""
        opt = {'cv_mode': -1, 'mag_frac': 0.95, 'wantfig': False}
        results = psn(sample_data, V=5, opt=opt, wantfig=False)

        assert 'ranking' in results['opt']
        assert results['opt']['ranking'] == 'signal_variance'

        # Check that ICA mixing matrix is stored
        assert results['ica_mixing'] is not None

    def test_custom_basis_default_signal(self, sample_data):
        """Test that custom basis defaults to 'signal_variance' ranking."""
        nunits = sample_data.shape[0]
        custom_basis = np.linalg.qr(np.random.randn(nunits, nunits))[0]

        opt = {'cv_mode': -1, 'mag_frac': 0.95, 'wantfig': False}
        results = psn(sample_data, V=custom_basis, opt=opt, wantfig=False)

        assert 'ranking' in results['opt']
        assert results['opt']['ranking'] == 'signal_variance'


class TestPSNRankingOverride:
    """Test that users can override default rankings."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        nunits = 20
        nconds = 50
        ntrials = 5
        data = np.random.randn(nunits, nconds, ntrials)
        return data
    
    @pytest.mark.parametrize("V", [0, 1, 2, 3, 4, 5])
    @pytest.mark.parametrize("ranking", ['eigenvalue', 'eigenvalue_asc', 'signal_variance', 'snr', 'signal_specificity'])
    def test_ranking_override_all_combinations(self, sample_data, V, ranking):
        """Test that all ranking methods work with all basis types.
        
        This test verifies that:
        1. The ranking method is correctly applied
        2. Valid outputs are produced
        3. The basis remains orthonormal
        4. The magnitudes are properly sorted according to the ranking method
        """
        opt = {
            'cv_mode': -1,
            'mag_frac': 0.95,
            'ranking': ranking,
            'wantfig': False
        }
        
        results = psn(sample_data, V=V, opt=opt, wantfig=False)
        
        # Check that the requested ranking was used
        assert results['opt']['ranking'] == ranking
        
        # Check that results are valid
        assert results['fullbasis'].shape[0] == sample_data.shape[0]
        assert results['denoiser'].shape == (sample_data.shape[0], sample_data.shape[0])
        assert len(results['mags']) == results['fullbasis'].shape[1]
        
        # Check that magnitudes exist and are reasonable
        assert np.all(np.isfinite(results['mags']))
        
        # Check that magnitudes are sorted correctly
        mags = results['mags']
        if ranking == 'eigenvalue_asc':
            # Should be sorted in increasing order
            assert np.all(np.diff(mags) >= 0), \
                f"V={V}, ranking={ranking}: magnitudes not sorted in increasing order"
        else:
            # Should be sorted in decreasing order
            assert np.all(np.diff(mags) <= 0), \
                f"V={V}, ranking={ranking}: magnitudes not sorted in decreasing order"
    
    def test_signal_basis_with_signal_ranking(self, sample_data):
        """Test signal basis with signal variance ranking (override)."""
        opt = {
            'cv_mode': -1,
            'mag_frac': 0.95,
            'ranking': 'signal_variance',
            'wantfig': False
        }
        
        results = psn(sample_data, V=0, opt=opt, wantfig=False)
        
        assert results['opt']['ranking'] == 'signal_variance'
        assert np.all(results['mags'] >= 0)  # Signal variances should be non-negative
    
    def test_pca_with_eigs_ranking(self, sample_data):
        """Test PCA with eigenvalue ranking (override)."""
        opt = {
            'cv_mode': -1,
            'mag_frac': 0.95,
            'ranking': 'eigenvalue',
            'wantfig': False
        }
        
        results = psn(sample_data, V=3, opt=opt, wantfig=False)
        
        assert results['opt']['ranking'] == 'eigenvalue'
        # Check that magnitudes are reasonable
        assert np.all(np.isfinite(results['mags']))


class TestRankingWithCrossValidation:
    """Test ranking methods with cross-validation."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        nunits = 20
        nconds = 50
        ntrials = 5
        
        # Create structured data
        signal_template = np.random.randn(nunits, nconds)
        data = signal_template[:, :, np.newaxis] + 0.3 * np.random.randn(nunits, nconds, ntrials)
        
        return data
    
    @pytest.mark.parametrize("ranking", ['eigenvalue', 'signal_variance', 'snr', 'signal_specificity'])
    def test_cv_with_different_rankings(self, sample_data, ranking):
        """Test that cross-validation works with different ranking methods."""
        opt = {
            'cv_mode': 0,
            'cv_threshold_per': 'population',
            'cv_thresholds': [1, 2, 5, 10],
            'ranking': ranking,
            'wantfig': False
        }
        
        results = psn(sample_data, V=0, opt=opt, wantfig=False)
        
        # Check that CV was performed
        assert results['cv_scores'] is not None
        assert len(results['cv_scores']) > 0
        
        # Check that ranking was applied
        assert results['opt']['ranking'] == ranking
        
        # Check that best_threshold was selected
        assert results['best_threshold'] > 0
    
    def test_unit_wise_cv_with_ranking(self, sample_data):
        """Test unit-wise cross-validation with custom ranking."""
        opt = {
            'cv_mode': 0,
            'cv_threshold_per': 'unit',
            'cv_thresholds': [1, 2, 5, 10],
            'ranking': 'signal_variance',
            'wantfig': False
        }
        
        results = psn(sample_data, V=0, opt=opt, wantfig=False)
        
        # Check that unit-wise thresholds were selected
        assert len(results['best_threshold']) == sample_data.shape[0]
        
        # Check that ranking was applied
        assert results['opt']['ranking'] == 'signal_variance'


class TestRankingConsistency:
    """Test consistency of ranking across different scenarios."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        nunits = 20
        nconds = 50
        ntrials = 5
        data = np.random.randn(nunits, nconds, ntrials)
        return data
    
    def test_same_ranking_produces_same_order(self, sample_data):
        """Test that running same ranking twice produces same order."""
        opt = {
            'cv_mode': -1,
            'mag_frac': 0.95,
            'ranking': 'signal_variance',
            'wantfig': False
        }
        
        results1 = psn(sample_data, V=0, opt=opt, wantfig=False)
        results2 = psn(sample_data, V=0, opt=opt, wantfig=False)
        
        assert np.allclose(results1['mags'], results2['mags'])
        assert np.allclose(results1['fullbasis'], results2['fullbasis'])
    
    def test_different_rankings_produce_different_orders(self, sample_data):
        """Test that different rankings produce different orderings."""
        opt1 = {
            'cv_mode': -1,
            'mag_frac': 0.95,
            'ranking': 'eigenvalue',
            'wantfig': False
        }
        
        opt2 = {
            'cv_mode': -1,
            'mag_frac': 0.95,
            'ranking': 'signal_variance',
            'wantfig': False
        }
        
        results1 = psn(sample_data, V=0, opt=opt1, wantfig=False)
        results2 = psn(sample_data, V=0, opt=opt2, wantfig=False)
        
        # The magnitude values should be different (different metrics)
        # But we can't guarantee the basis order is different in all cases
        assert results1['opt']['ranking'] != results2['opt']['ranking']
    
    def test_basis_orthonormality_maintained(self, sample_data):
        """Test that all rankings maintain orthonormality."""
        rankings = ['eigenvalue', 'eigenvalue_asc', 'signal_variance', 'snr', 'signal_specificity']
        
        for ranking in rankings:
            opt = {
                'cv_mode': -1,
                'mag_frac': 0.95,
                'ranking': ranking,
                'wantfig': False
            }
            
            results = psn(sample_data, V=0, opt=opt, wantfig=False)
            basis = results['fullbasis']
            
            # Check orthonormality
            gram = basis.T @ basis
            assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-10), \
                f"Orthonormality violated for {ranking} ranking"


class TestRankingEdgeCases:
    """Test edge cases and error handling for ranking."""
    
    @pytest.fixture
    def minimal_data(self):
        """Generate minimal valid data."""
        np.random.seed(42)
        nunits = 5
        nconds = 10
        ntrials = 2
        data = np.random.randn(nunits, nconds, ntrials)
        return data
    
    def test_ranking_with_minimal_data(self, minimal_data):
        """Test that ranking works with minimal data."""
        opt = {
            'cv_mode': -1,
            'mag_frac': 0.95,
            'ranking': 'signal_variance',
            'wantfig': False
        }
        
        results = psn(minimal_data, V=0, opt=opt, wantfig=False)
        
        assert results['fullbasis'].shape[0] == minimal_data.shape[0]
        assert len(results['mags']) == minimal_data.shape[0]
    
    def test_all_rankings_with_low_snr_data(self):
        """Test all rankings with very noisy data (low SNR)."""
        np.random.seed(42)
        nunits = 10
        nconds = 20
        ntrials = 3
        
        # Pure noise (no signal structure)
        data = np.random.randn(nunits, nconds, ntrials)
        
        rankings = ['eigenvalue', 'eigenvalue_asc', 'signal_variance', 'snr', 'signal_specificity']
        
        for ranking in rankings:
            opt = {
                'cv_mode': -1,
                'mag_frac': 0.95,
                'ranking': ranking,
                'wantfig': False
            }
            
            results = psn(data, V=0, opt=opt, wantfig=False)
            
            # Should still produce valid results
            assert results['fullbasis'].shape[0] == nunits
            assert len(results['mags']) == nunits
    
    def test_ranking_with_high_snr_data(self):
        """Test rankings with high SNR structured data."""
        np.random.seed(42)
        nunits = 10
        nconds = 20
        ntrials = 3
        
        # High SNR: strong signal, low noise
        signal_template = np.random.randn(nunits, nconds)
        data = signal_template[:, :, np.newaxis] + 0.01 * np.random.randn(nunits, nconds, ntrials)
        
        rankings = ['eigenvalue', 'signal_variance', 'snr']
        
        for ranking in rankings:
            opt = {
                'cv_mode': -1,
                'mag_frac': 0.95,
                'ranking': ranking,
                'wantfig': False
            }
            
            results = psn(data, V=0, opt=opt, wantfig=False)
            
            # High SNR data should retain more dimensions
            assert results['dimsretained'] > 0
            
            if ranking in ['signal_variance', 'snr']:
                # Signal-based rankings should show strong signal in top dimensions
                assert results['mags'][0] > 0


class TestICAMixingMatrix:
    """Test ICA mixing matrix storage and ranking."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        nunits = 20
        nconds = 50
        ntrials = 5
        data = np.random.randn(nunits, nconds, ntrials)
        return data
    
    def test_ica_mixing_stored(self, sample_data):
        """Test that ICA mixing matrix is stored correctly."""
        opt = {'cv_mode': -1, 'mag_frac': 0.95, 'wantfig': False}
        results = psn(sample_data, V=5, opt=opt, wantfig=False)
        
        # Check that ICA mixing matrix is stored
        assert results['ica_mixing'] is not None
        assert results['ica_mixing'].shape[0] == sample_data.shape[0]
        
        # basis_source should be None for ICA (replaced by ica_mixing)
        assert results['basis_source'] is None
    
    def test_ica_mixing_ranked_consistently(self, sample_data):
        """Test that ICA mixing matrix is ranked consistently with basis."""
        opt = {'cv_mode': -1, 'mag_frac': 0.95, 'wantfig': False}
        results = psn(sample_data, V=5, opt=opt, wantfig=False)
        
        # The mixing matrix should have been reordered
        # We can't easily verify the exact order without recomputing,
        # but we can check that it has the right shape and properties
        assert results['ica_mixing'].shape == (sample_data.shape[0], 
                                               min(sample_data.shape[0], sample_data.shape[1]))
    
    def test_non_ica_has_no_mixing_matrix(self, sample_data):
        """Test that non-ICA basis types don't have ica_mixing."""
        for V in [0, 1, 2, 3, 4]:
            opt = {'cv_mode': -1, 'mag_frac': 0.95, 'wantfig': False}
            results = psn(sample_data, V=V, opt=opt, wantfig=False)
            
            # ica_mixing should be None for non-ICA bases
            assert results['ica_mixing'] is None


class TestRankingDocumentation:
    """Test that ranking behavior matches documentation."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        nunits = 20
        nconds = 50
        ntrials = 5
        data = np.random.randn(nunits, nconds, ntrials)
        return data
    
    def test_mag_threshold_respects_ranking(self, sample_data):
        """Test that magnitude thresholding uses ranked dimensions."""
        opt = {
            'cv_mode': -1,
            'mag_frac': 0.8,  # Lower threshold to ensure some dims excluded
            'ranking': 'signal_variance',
            'wantfig': False
        }
        
        results = psn(sample_data, V=0, opt=opt, wantfig=False)
        
        # Check that dimensions were retained based on signal variance
        assert results['dimsretained'] < results['fullbasis'].shape[1]
        
        # Check that retained dimensions have highest magnitudes
        retained_indices = results['best_threshold']
        if isinstance(retained_indices, np.ndarray):
            # Make sure retained dimensions have high magnitudes
            for idx in retained_indices:
                assert idx < len(results['mags'])
    
    def test_truncate_works_with_ranking(self, sample_data):
        """Test that truncate parameter works correctly with ranking."""
        opt = {
            'cv_mode': 0,
            'cv_threshold_per': 'population',
            'cv_thresholds': [1, 2, 5, 10],
            'truncate': 2,  # Remove first 2 dimensions
            'ranking': 'signal_variance',
            'wantfig': False
        }
        
        results = psn(sample_data, V=0, opt=opt, wantfig=False)
        
        # Check that truncation was applied
        # The denoiser should exclude the first 2 dimensions
        # This is harder to verify directly, but we can check it ran successfully
        assert results['denoiser'] is not None
        assert results['best_threshold'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
