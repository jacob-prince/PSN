"""
Extended tests for PSN sklearn interface covering all parameters.

These tests ensure that the sklearn interface properly supports all
parameters available in the functional psn() interface.
"""

import numpy as np
import pytest
from psn import PSN


class TestPSNRankingParameter:
    """Test the ranking parameter in sklearn interface."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return np.random.randn(20, 50, 5)

    @pytest.mark.parametrize("ranking", [
        'eigenvalue', 'eigenvalue_asc', 'signal_variance', 'snr', 'signal_specificity'
    ])
    def test_all_ranking_methods(self, sample_data, ranking):
        """Test that all ranking methods work through sklearn interface."""
        psn = PSN(basis='signal', cv='unit', ranking=ranking, wantfig=False)
        psn.fit(sample_data)

        # Check that ranking was applied
        assert psn.fitted_results_['opt']['ranking'] == ranking

        # Check that denoiser was created
        assert psn.denoiser_ is not None
        assert psn.denoiser_.shape == (sample_data.shape[0], sample_data.shape[0])

    def test_default_ranking(self, sample_data):
        """Test that default ranking is applied when not specified."""
        psn = PSN(basis='signal', cv='unit', wantfig=False)
        psn.fit(sample_data)

        # Default should be signal_variance
        assert psn.fitted_results_['opt']['ranking'] == 'signal_variance'

    @pytest.mark.parametrize("basis,ranking", [
        ('signal', 'eigenvalue'),
        ('pca', 'signal_variance'),
        ('ica', 'snr'),
        ('noise', 'signal_specificity'),
    ])
    def test_ranking_with_different_bases(self, sample_data, basis, ranking):
        """Test that ranking works with different basis types."""
        psn = PSN(basis=basis, cv='population', ranking=ranking, wantfig=False)
        psn.fit(sample_data)

        assert psn.fitted_results_['opt']['ranking'] == ranking
        # Transform adapts to input shape - give it trial-averaged data for 2D output
        trial_avg = np.mean(sample_data, axis=2)
        denoised = psn.transform(trial_avg)
        assert denoised.shape == (sample_data.shape[0], sample_data.shape[1])


class TestPSNCVThresholds:
    """Test the cv_thresholds parameter in sklearn interface."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return np.random.randn(15, 30, 5)

    def test_custom_cv_thresholds_list(self, sample_data):
        """Test cv_thresholds with a custom list."""
        custom_thresholds = [1, 2, 5, 10]
        psn = PSN(basis='pca', cv='population', cv_thresholds=custom_thresholds, wantfig=False)
        psn.fit(sample_data)

        # Check that custom thresholds were used
        np.testing.assert_array_equal(
            psn.fitted_results_['opt']['cv_thresholds'],
            custom_thresholds
        )

        # Best threshold should be one of the tested values
        assert psn.best_threshold_ in custom_thresholds

    def test_cv_thresholds_array(self, sample_data):
        """Test cv_thresholds with numpy array."""
        custom_thresholds = np.array([2, 4, 6, 8])
        psn = PSN(basis='signal', cv='unit', cv_thresholds=custom_thresholds, wantfig=False)
        psn.fit(sample_data)

        np.testing.assert_array_equal(
            psn.fitted_results_['opt']['cv_thresholds'],
            custom_thresholds
        )

    def test_cv_thresholds_sparse_sampling(self, sample_data):
        """Test cv_thresholds with sparse sampling for efficiency."""
        # Test only a few dimensions instead of all
        sparse_thresholds = [1, 3, 5, 7, 10]
        psn = PSN(basis='signal', cv='population', cv_thresholds=sparse_thresholds, wantfig=False)
        psn.fit(sample_data)

        assert len(psn.cv_scores_) == len(sparse_thresholds)

    def test_cv_thresholds_with_truncate(self, sample_data):
        """Test cv_thresholds combined with truncate parameter."""
        psn = PSN(
            basis='pca',
            cv='population',
            cv_thresholds=[1, 2, 3, 5],
            truncate=2,
            wantfig=False
        )
        psn.fit(sample_data)

        # Truncate should have been applied
        assert psn.fitted_results_['opt']['truncate'] == 2


class TestPSNCVMode:
    """Test the cv_mode parameter in sklearn interface."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return np.random.randn(10, 25, 6)

    def test_cv_mode_0_leave_one_out(self, sample_data):
        """Test cv_mode=0 (leave-one-trial-out)."""
        psn = PSN(basis='signal', cv_mode=0, cv='unit', wantfig=False)
        psn.fit(sample_data)

        assert psn.fitted_results_['opt']['cv_mode'] == 0
        # CV scores should have shape (n_thresholds, ntrials, nunits)
        assert psn.cv_scores_.shape[1] == sample_data.shape[2]

    def test_cv_mode_1_keep_one_in(self, sample_data):
        """Test cv_mode=1 (keep-one-trial-in)."""
        psn = PSN(basis='pca', cv_mode=1, cv='population', wantfig=False)
        psn.fit(sample_data)

        assert psn.fitted_results_['opt']['cv_mode'] == 1
        assert psn.cv_scores_.shape[1] == sample_data.shape[2]

    def test_cv_mode_minus1_magnitude_thresholding(self, sample_data):
        """Test cv_mode=-1 (magnitude thresholding)."""
        psn = PSN(basis='signal', cv_mode=-1, mag_threshold=0.90, wantfig=False)
        psn.fit(sample_data)

        assert psn.fitted_results_['opt']['cv_mode'] == -1
        assert psn.fitted_results_['dimsretained'] > 0

    def test_cv_mode_overrides_cv_parameter(self, sample_data):
        """Test that cv_mode takes priority over cv parameter."""
        # cv=None normally means magnitude thresholding
        # But cv_mode=0 should override this
        psn = PSN(basis='signal', cv=None, cv_mode=0, wantfig=False)
        psn.fit(sample_data)

        # Should use cross-validation, not magnitude thresholding
        assert psn.fitted_results_['opt']['cv_mode'] == 0
        assert psn.cv_scores_ is not None


class TestPSNDenoisingType:
    """Test the denoisingtype parameter in sklearn interface."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return np.random.randn(12, 20, 8)

    def test_denoisingtype_0_trial_averaged(self, sample_data):
        """Test denoisingtype=0 (trial-averaged denoising)."""
        psn = PSN(basis='pca', cv='unit', denoisingtype=0, wantfig=False)
        psn.fit(sample_data)

        # transform() adapts to input shape - give 2D input for 2D output
        trial_avg = np.mean(sample_data, axis=2)
        denoised = psn.transform(trial_avg)

        # Should return trial-averaged data (nunits, nconds)
        assert denoised.shape == (sample_data.shape[0], sample_data.shape[1])
        assert denoised.ndim == 2

    def test_denoisingtype_1_single_trial(self, sample_data):
        """Test denoisingtype=1 (single-trial denoising)."""
        psn = PSN(basis='signal', cv='population', denoisingtype=1, wantfig=False)
        psn.fit(sample_data)

        # transform() adapts to input shape - 3D input gives 3D output
        denoised = psn.transform(sample_data)

        # Should return single-trial data (nunits, nconds, ntrials)
        assert denoised.shape == sample_data.shape
        assert denoised.ndim == 3

    def test_denoisingtype_with_magnitude_thresholding(self, sample_data):
        """Test denoisingtype with magnitude thresholding."""
        psn = PSN(
            basis='pca',
            cv=None,
            denoisingtype=1,
            mag_threshold=0.85,
            wantfig=False
        )
        psn.fit(sample_data)
        denoised = psn.transform(sample_data)

        # Should still return single-trial data
        assert denoised.shape == sample_data.shape


class TestPSNParameterCombinations:
    """Test combinations of parameters to ensure they work together."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return np.random.randn(15, 40, 5)

    def test_all_parameters_together(self, sample_data):
        """Test using all extended parameters together."""
        psn = PSN(
            basis='ica',
            cv='population',
            ranking='signal_variance',
            cv_thresholds=[1, 2, 3, 5, 8],
            cv_mode=0,
            denoisingtype=0,
            truncate=1,
            mag_threshold=0.90,
            scoring='mse',
            wantfig=False
        )
        psn.fit(sample_data)

        # Verify all parameters were applied
        assert psn.fitted_results_['opt']['ranking'] == 'signal_variance'
        assert psn.fitted_results_['opt']['cv_mode'] == 0
        assert psn.fitted_results_['opt']['denoisingtype'] == 0
        assert psn.fitted_results_['opt']['truncate'] == 1

        # Should be able to transform - use trial-averaged input for 2D output
        trial_avg = np.mean(sample_data, axis=2)
        denoised = psn.transform(trial_avg)
        assert denoised.shape == (sample_data.shape[0], sample_data.shape[1])

    def test_ranking_with_different_cv_modes(self, sample_data):
        """Test that ranking works with different cv_modes."""
        for cv_mode in [0, 1, -1]:
            psn = PSN(
                basis='signal',
                ranking='snr',
                cv_mode=cv_mode,
                cv='population' if cv_mode >= 0 else None,
                wantfig=False
            )
            psn.fit(sample_data)

            assert psn.fitted_results_['opt']['ranking'] == 'snr'
            assert psn.fitted_results_['opt']['cv_mode'] == cv_mode

    def test_custom_thresholds_with_ranking(self, sample_data):
        """Test custom cv_thresholds with custom ranking."""
        psn = PSN(
            basis='pca',
            cv='population',
            ranking='eigenvalue',
            cv_thresholds=[2, 4, 6],
            wantfig=False
        )
        psn.fit(sample_data)

        assert psn.fitted_results_['opt']['ranking'] == 'eigenvalue'
        assert len(psn.fitted_results_['opt']['cv_thresholds']) == 3

    def test_single_trial_with_unit_cv(self, sample_data):
        """Test single-trial denoising with unit-wise CV."""
        psn = PSN(
            basis='signal',
            cv='unit',
            denoisingtype=1,
            cv_thresholds=[1, 2, 3, 5],
            wantfig=False
        )
        psn.fit(sample_data)
        denoised = psn.transform(sample_data)

        # Should return single-trial data
        assert denoised.shape == sample_data.shape

        # Should have unit-wise thresholds
        assert len(psn.best_threshold_) == sample_data.shape[0]

    def test_truncate_with_ranking(self, sample_data):
        """Test that truncate works properly with custom ranking."""
        psn = PSN(
            basis='pca',
            cv='population',
            ranking='signal_variance',
            truncate=2,
            cv_thresholds=[1, 3, 5],
            wantfig=False
        )
        psn.fit(sample_data)

        # Truncate should skip first 2 dimensions
        assert psn.fitted_results_['opt']['truncate'] == 2
        assert psn.fitted_results_['opt']['ranking'] == 'signal_variance'


class TestPSNEdgeCases:
    """Test edge cases and boundary conditions with extended parameters."""

    @pytest.fixture
    def small_data(self):
        """Generate small dataset for edge case testing."""
        np.random.seed(42)
        return np.random.randn(5, 10, 3)

    def test_cv_thresholds_exceeding_max_dims(self, small_data):
        """Test cv_thresholds with values exceeding number of dimensions."""
        # Dataset has 5 units, so max dims is 5
        # Request testing up to 10
        psn = PSN(
            basis='pca',
            cv='population',
            cv_thresholds=list(range(0, 11)),
            wantfig=False
        )
        psn.fit(small_data)

        # Should still work, just won't test impossible dimensions
        assert psn.denoiser_ is not None

    def test_truncate_with_small_dataset(self, small_data):
        """Test truncate with dataset that has few dimensions."""
        psn = PSN(
            basis='pca',
            cv='population',
            truncate=2,
            cv_thresholds=[1, 2],
            wantfig=False
        )
        psn.fit(small_data)

        # Should still work even though we're truncating a significant portion
        assert psn.fitted_results_['opt']['truncate'] == 2
        assert psn.denoiser_ is not None

    def test_single_trial_denoising_minimal_trials(self, small_data):
        """Test single-trial denoising with minimal number of trials."""
        # Dataset has only 3 trials
        psn = PSN(
            basis='signal',
            cv='unit',
            denoisingtype=1,
            wantfig=False
        )
        psn.fit(small_data)
        denoised = psn.transform(small_data)

        assert denoised.shape == small_data.shape

    def test_ranking_with_low_dimensional_data(self, small_data):
        """Test all ranking methods with low-dimensional data."""
        rankings = ['eigenvalue', 'signal_variance', 'snr', 'signal_specificity']

        for ranking in rankings:
            psn = PSN(
                basis='pca',
                cv='population',
                ranking=ranking,
                wantfig=False
            )
            psn.fit(small_data)

            assert psn.fitted_results_['opt']['ranking'] == ranking
            assert psn.denoiser_ is not None


class TestPSNGetSetParams:
    """Test get_params and set_params with extended parameters."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return np.random.randn(10, 20, 5)

    def test_get_params_includes_new_parameters(self):
        """Test that get_params returns all new parameters."""
        psn = PSN(
            ranking='snr',
            cv_thresholds=[1, 2, 3],
            cv_mode=1,
            denoisingtype=1
        )
        params = psn.get_params()

        assert 'ranking' in params
        assert params['ranking'] == 'snr'
        assert 'cv_thresholds' in params
        assert 'cv_mode' in params
        assert 'denoisingtype' in params

    def test_set_params_with_new_parameters(self, sample_data):
        """Test that set_params works with new parameters."""
        psn = PSN()

        # Set new parameters
        psn.set_params(
            ranking='signal_variance',
            cv_thresholds=[2, 4, 6],
            cv_mode=0,
            denoisingtype=1
        )

        assert psn.ranking == 'signal_variance'
        assert psn.cv_thresholds == [2, 4, 6]
        assert psn.cv_mode == 0
        assert psn.denoisingtype == 1

        # Should work when fitted
        psn.fit(sample_data)
        assert psn.fitted_results_['opt']['ranking'] == 'signal_variance'

    def test_clone_with_new_parameters(self, sample_data):
        """Test that sklearn's clone works with new parameters."""
        from sklearn.base import clone

        psn_original = PSN(
            basis='ica',
            ranking='snr',
            cv_thresholds=[1, 3, 5],
            cv_mode=1,
            denoisingtype=0,
            truncate=2
        )

        psn_cloned = clone(psn_original)

        # Check that all parameters were cloned
        assert psn_cloned.ranking == psn_original.ranking
        assert psn_cloned.cv_thresholds == psn_original.cv_thresholds
        assert psn_cloned.cv_mode == psn_original.cv_mode
        assert psn_cloned.denoisingtype == psn_original.denoisingtype
        assert psn_cloned.truncate == psn_original.truncate


class TestPSNConsistencyWithFunctional:
    """Test that sklearn interface produces same results as functional interface."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return np.random.randn(10, 20, 5)

    def test_consistency_basic_usage(self, sample_data):
        """Test that sklearn and functional interfaces give same results."""
        from psn import psn

        # Sklearn interface
        psn_sklearn = PSN(basis='signal', cv='population', wantfig=False)
        psn_sklearn.fit(sample_data)
        # Use trial-averaged input to match functional interface output
        trial_avg = np.mean(sample_data, axis=2)
        denoised_sklearn = psn_sklearn.transform(trial_avg)

        # Functional interface
        opt = {'cv_threshold_per': 'population', 'denoisingtype': 0, 'wantfig': False}
        results_functional = psn(sample_data, V=0, opt=opt, wantfig=False)
        denoised_functional = results_functional['denoiseddata']

        # Should be very close (may differ slightly due to randomness)
        np.testing.assert_allclose(denoised_sklearn, denoised_functional, rtol=1e-10)

    def test_consistency_with_ranking(self, sample_data):
        """Test consistency when using custom ranking."""
        from psn import psn

        # Sklearn interface
        psn_sklearn = PSN(
            basis='pca',
            cv='population',
            ranking='signal_variance',
            wantfig=False
        )
        psn_sklearn.fit(sample_data)

        # Functional interface
        opt = {
            'cv_threshold_per': 'population',
            'ranking': 'signal_variance',
            'denoisingtype': 0,
            'wantfig': False
        }
        results_functional = psn(sample_data, V=3, opt=opt, wantfig=False)

        # Check that same ranking was used
        assert psn_sklearn.fitted_results_['opt']['ranking'] == \
               results_functional['opt']['ranking']

    def test_consistency_magnitude_thresholding(self, sample_data):
        """Test consistency for magnitude thresholding."""
        from psn import psn

        # Sklearn interface
        psn_sklearn = PSN(
            basis='signal',
            cv=None,
            mag_threshold=0.90,
            wantfig=False
        )
        psn_sklearn.fit(sample_data)
        # Use trial-averaged input to match functional interface output
        trial_avg = np.mean(sample_data, axis=2)
        denoised_sklearn = psn_sklearn.transform(trial_avg)

        # Functional interface
        opt = {
            'cv_mode': -1,
            'mag_frac': 0.90,
            'denoisingtype': 0,
            'wantfig': False
        }
        results_functional = psn(sample_data, V=0, opt=opt, wantfig=False)
        denoised_functional = results_functional['denoiseddata']

        # Should be identical
        np.testing.assert_allclose(denoised_sklearn, denoised_functional, rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
