"""Tests for sklearn-style PSN API.

Tests the PSN class implementation including fit, transform, fit_transform,
plot_diagnostics, pickling, and various edge cases.
"""

import numpy as np
import pickle
import tempfile
import pytest

from psn import PSN, psn


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    nunits, nconds, ntrials = 15, 25, 6
    return np.random.randn(nunits, nconds, ntrials)


@pytest.fixture
def small_data():
    """Small data for fast tests."""
    np.random.seed(123)
    return np.random.randn(5, 10, 3)


@pytest.fixture
def data_with_nans():
    """Data with NaN values (uneven trials)."""
    np.random.seed(42)
    nunits, nconds, ntrials = 10, 15, 5
    data = np.random.randn(nunits, nconds, ntrials)
    # Set some trials to NaN
    data[:, 0, 3:] = np.nan  # condition 0 has only 3 trials
    data[:, 5, 4:] = np.nan  # condition 5 has only 4 trials
    data[:, 10, 2:] = np.nan  # condition 10 has only 2 trials
    return data


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestBasicFunctionality:
    """Test basic fit/transform/fit_transform functionality."""

    def test_instantiation_default(self):
        """Test default instantiation."""
        model = PSN()
        assert model.mode is None          # None == library defaults (= 'standard')
        assert model._is_fitted is False
        assert 'PSN(' in repr(model)

    def test_instantiation_with_mode(self):
        """Test instantiation with different modes."""
        for mode in ['conservative', 'standard', 'aggressive']:
            model = PSN(mode=mode)
            assert model.mode == mode

    def test_instantiation_with_none_mode(self):
        """Test instantiation with mode=None (custom options only)."""
        model = PSN(mode=None, basis='signal', criterion='prediction')
        assert model.mode is None

    def test_instantiation_with_options(self):
        """Test instantiation with custom options stored as explicit params."""
        model = PSN(
            mode='standard',
            basis='difference',
            criterion='variance',
            threshold_method='global',
            variance_threshold=0.95,
            wantverbose=False
        )
        assert model.basis == 'difference'
        assert model.criterion == 'variance'
        assert model.threshold_method == 'global'

    def test_fit_returns_self(self, small_data):
        """Test that fit returns self for chaining."""
        model = PSN(wantverbose=False, wantfig=False)
        result = model.fit(small_data)
        assert result is model

    def test_fit_sets_is_fitted(self, small_data):
        """Test that fit sets _is_fitted flag."""
        model = PSN(wantverbose=False, wantfig=False)
        assert model._is_fitted is False
        model.fit(small_data)
        assert model._is_fitted is True

    def test_fit_creates_all_attributes(self, small_data):
        """Test that fit creates all expected attributes."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(small_data)

        # Core attributes
        assert hasattr(model, 'denoiseddata_')
        assert hasattr(model, 'residuals_')
        assert hasattr(model, 'unit_means_')
        assert hasattr(model, 'denoiser_')

        # Diagnostic attributes
        assert hasattr(model, 'svnv_before_')
        assert hasattr(model, 'svnv_after_')
        assert hasattr(model, 'best_threshold_')
        assert hasattr(model, 'fullbasis_')

        # GSN and variance attributes
        assert hasattr(model, 'gsn_result_')
        assert hasattr(model, 'signalvar_')
        assert hasattr(model, 'noisevar_')
        assert hasattr(model, 'objective_')

        # Visualization attributes
        assert hasattr(model, 'basis_viz_')
        assert hasattr(model, 'signal_proj_viz_')
        assert hasattr(model, 'noise_proj_viz_')
        assert hasattr(model, 'basis_eigenvalues_viz_')
        assert hasattr(model, 'basis_eigenvalues_')

        # Input and options
        assert hasattr(model, 'input_data_')
        assert hasattr(model, 'opt_used_')

    def test_fit_attribute_shapes(self, sample_data):
        """Test that fitted attributes have correct shapes."""
        nunits, nconds, ntrials = sample_data.shape
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(sample_data)

        assert model.denoiseddata_.shape == (nunits, nconds)
        assert model.residuals_.shape == (nunits, nconds, ntrials)
        assert model.unit_means_.shape == (nunits,)
        assert model.denoiser_.shape == (nunits, nunits)
        assert model.svnv_before_.shape == (nunits, 2)
        assert model.svnv_after_.shape == (nunits, 2)
        assert model.fullbasis_.shape[0] == nunits
        assert model.input_data_.shape == sample_data.shape

    def test_transform_3d_input(self, small_data):
        """Test transform with 3D input (should trial-average)."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(small_data)

        new_data = np.random.randn(*small_data.shape)
        result = model.transform(new_data)

        nunits, nconds, _ = small_data.shape
        assert result.shape == (nunits, nconds)

    def test_transform_2d_input(self, small_data):
        """Test transform with 2D input (already trial-averaged)."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(small_data)

        nunits, nconds, _ = small_data.shape
        new_data_2d = np.random.randn(nunits, nconds)
        result = model.transform(new_data_2d)

        assert result.shape == (nunits, nconds)

    def test_fit_transform_returns_denoised(self, small_data):
        """Test that fit_transform returns denoiseddata_."""
        model = PSN(wantverbose=False, wantfig=False)
        result = model.fit_transform(small_data)

        assert np.allclose(result, model.denoiseddata_)

    def test_fit_transform_equivalent_to_fit_then_access(self, small_data):
        """Test fit_transform gives same result as fit + denoiseddata_."""
        model1 = PSN(wantverbose=False, wantfig=False)
        result1 = model1.fit_transform(small_data)

        model2 = PSN(wantverbose=False, wantfig=False)
        model2.fit(small_data)
        result2 = model2.denoiseddata_

        assert np.allclose(result1, result2)

    def test_repr_shows_nondefault_params(self):
        """Test sklearn repr shows non-default constructor params."""
        model = PSN(mode='aggressive')
        assert "mode='aggressive'" in repr(model)

    def test_repr_fitted(self, small_data):
        """Test repr (sklearn style) after fit."""
        model = PSN(mode='standard', wantverbose=False, wantfig=False)
        model.fit(small_data)
        r = repr(model)
        assert "mode='standard'" in r
        assert "PSN(" in r


# =============================================================================
# Transform Behavior Tests
# =============================================================================

class TestTransformBehavior:
    """Test transform method behavior in detail."""

    def test_transform_uses_learned_means(self, small_data):
        """Test that transform uses unit_means_ from fit, not new data."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(small_data)

        # Create new data with very different means
        new_data = small_data.copy() + 1000
        result = model.transform(new_data)

        # The denoised output should not have the +1000 offset fully preserved
        # because we subtract learned means and re-add them
        trial_avg_new = np.mean(new_data, axis=2)

        # Verify transform formula: denoiser.T @ (data - learned_means) + learned_means
        expected = (
            model.denoiser_.T @ (trial_avg_new - model.unit_means_[:, np.newaxis])
            + model.unit_means_[:, np.newaxis]
        )
        assert np.allclose(result, expected)

    def test_transform_with_nans_in_new_data(self, small_data):
        """Test transform handles NaNs in new data via nanmean."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(small_data)

        # Create new data with NaNs
        new_data = np.random.randn(*small_data.shape)
        new_data[:, 0, -1] = np.nan  # Last trial of first condition is NaN

        result = model.transform(new_data)

        # Should not have NaNs in output
        assert not np.any(np.isnan(result))

    def test_transform_consistency(self, sample_data):
        """Test that multiple transforms give same result."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(sample_data)

        new_data = np.random.randn(*sample_data.shape)
        result1 = model.transform(new_data)
        result2 = model.transform(new_data)

        assert np.allclose(result1, result2)


# =============================================================================
# Mode and Options Tests
# =============================================================================

class TestModesAndOptions:
    """Test different modes and option combinations."""

    @pytest.mark.parametrize("mode", ['conservative', 'standard', 'aggressive'])
    def test_all_modes_work(self, small_data, mode):
        """Test that all preset modes work."""
        model = PSN(mode=mode, wantverbose=False, wantfig=False)
        model.fit(small_data)
        assert model._is_fitted

    def test_mode_none_with_custom_options(self, small_data):
        """Test mode=None with custom options."""
        model = PSN(
            mode=None,
            basis='signal',
            criterion='prediction',
            threshold_method='global',
            wantverbose=False,
            wantfig=False
        )
        model.fit(small_data)
        assert model._is_fitted
        assert model.opt_used_['threshold_method'] == 'global'

    def test_invalid_mode_raises(self, small_data):
        """Test that invalid mode raises ValueError."""
        model = PSN(mode='invalid_mode', wantverbose=False, wantfig=False)
        with pytest.raises(ValueError, match="mode must be"):
            model.fit(small_data)

    @pytest.mark.parametrize("basis", ['signal', 'difference', 'pca'])
    def test_different_basis_types(self, small_data, basis):
        """Test different basis types."""
        model = PSN(basis=basis, wantverbose=False, wantfig=False)
        model.fit(small_data)
        assert model._is_fitted

    @pytest.mark.parametrize("threshold_method", ['global', 'hybrid'])
    def test_different_threshold_methods(self, small_data, threshold_method):
        """Test different threshold methods."""
        model = PSN(
            threshold_method=threshold_method,
            wantverbose=False,
            wantfig=False
        )
        model.fit(small_data)
        assert model._is_fitted

        # Global should have scalar threshold, others should have array
        if threshold_method == 'global':
            assert np.isscalar(model.best_threshold_)
        else:
            assert hasattr(model.best_threshold_, '__len__')

    @pytest.mark.parametrize("criterion", ['prediction', 'variance'])
    def test_different_criteria(self, small_data, criterion):
        """Test different threshold criteria."""
        model = PSN(
            criterion=criterion,
            threshold_method='global',
            wantverbose=False,
            wantfig=False
        )
        model.fit(small_data)
        assert model._is_fitted

    def test_custom_basis_matrix(self, small_data):
        """Test with custom orthonormal basis matrix."""
        nunits = small_data.shape[0]
        # Create random orthonormal basis
        random_matrix = np.random.randn(nunits, nunits)
        custom_basis, _ = np.linalg.qr(random_matrix)

        model = PSN(basis=custom_basis, wantverbose=False, wantfig=False)
        model.fit(small_data)
        assert model._is_fitted

    def test_allowable_thresholds(self, small_data):
        """Test allowable_thresholds constraint."""
        model = PSN(
            threshold_method='global',
            allowable_thresholds=[1, 3, 5],
            wantverbose=False,
            wantfig=False
        )
        model.fit(small_data)
        assert model.best_threshold_ in [1, 3, 5]

    def test_variance_threshold(self, small_data):
        """Test variance_threshold option."""
        model = PSN(
            criterion='variance',
            variance_threshold=0.80,
            threshold_method='global',
            wantverbose=False,
            wantfig=False
        )
        model.fit(small_data)
        assert model._is_fitted


# =============================================================================
# NaN Handling Tests
# =============================================================================

class TestNaNHandling:
    """Test handling of NaN values (uneven trials)."""

    def test_fit_with_nans(self, data_with_nans):
        """Test that fit works with NaN data."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(data_with_nans)
        assert model._is_fitted

    def test_denoised_has_no_nans(self, data_with_nans):
        """Test that denoised output has no NaNs."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(data_with_nans)
        assert not np.any(np.isnan(model.denoiseddata_))

    def test_residuals_preserve_nans(self, data_with_nans):
        """Test that residuals preserve NaN positions."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(data_with_nans)

        # NaN positions in original should be NaN in residuals
        original_nans = np.isnan(data_with_nans)
        residual_nans = np.isnan(model.residuals_)
        assert np.array_equal(original_nans, residual_nans)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_transform_before_fit_raises(self):
        """Test that transform before fit raises RuntimeError."""
        model = PSN()
        data = np.random.randn(5, 10, 3)
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.transform(data)

    def test_plot_diagnostics_before_fit_raises(self):
        """Test that plot_diagnostics before fit raises RuntimeError."""
        model = PSN()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.plot_diagnostics()

    def test_transform_wrong_nunits_raises(self, small_data):
        """Test that transform with wrong nunits raises ValueError."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(small_data)

        wrong_nunits_data = np.random.randn(small_data.shape[0] + 5, 10, 3)
        with pytest.raises(ValueError, match="units"):
            model.transform(wrong_nunits_data)

    def test_transform_1d_input_raises(self, small_data):
        """Test that transform with 1D input raises ValueError."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(small_data)

        with pytest.raises(ValueError, match="2D or 3D"):
            model.transform(np.random.randn(10))

    def test_transform_4d_input_raises(self, small_data):
        """Test that transform with 4D input raises ValueError."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(small_data)

        with pytest.raises(ValueError, match="2D or 3D"):
            model.transform(np.random.randn(5, 10, 3, 2))


# =============================================================================
# Pickling Tests
# =============================================================================

class TestPickling:
    """Test pickling and unpickling of fitted models."""

    def test_pickle_unfitted_model(self):
        """Test pickling unfitted model."""
        model = PSN(mode='aggressive', basis='difference')

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(model, f)
            temp_path = f.name

        with open(temp_path, 'rb') as f:
            loaded = pickle.load(f)

        assert loaded.mode == 'aggressive'
        assert loaded.basis == 'difference'
        assert loaded._is_fitted is False

    def test_pickle_fitted_model(self, sample_data):
        """Test pickling fitted model."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(sample_data)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(model, f)
            temp_path = f.name

        with open(temp_path, 'rb') as f:
            loaded = pickle.load(f)

        assert loaded._is_fitted is True
        assert np.allclose(model.denoiseddata_, loaded.denoiseddata_)
        assert np.allclose(model.denoiser_, loaded.denoiser_)
        assert np.allclose(model.unit_means_, loaded.unit_means_)

    def test_pickled_model_transform_works(self, sample_data):
        """Test that transform works on unpickled model."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(sample_data)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(model, f)
            temp_path = f.name

        with open(temp_path, 'rb') as f:
            loaded = pickle.load(f)

        new_data = np.random.randn(*sample_data.shape)
        result_original = model.transform(new_data)
        result_loaded = loaded.transform(new_data)

        assert np.allclose(result_original, result_loaded)

    def test_pickle_removes_lambdas(self, small_data):
        """Test that pickling removes lambda functions from opt_used_."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(small_data)

        # Get pickled state
        state = model.__getstate__()

        # Check that gsn_args doesn't have callable items
        if 'gsn_args' in state.get('opt_used_', {}):
            gsn_args = state['opt_used_']['gsn_args']
            for k, v in gsn_args.items():
                assert not callable(v), f"Callable found in gsn_args: {k}"


# =============================================================================
# Consistency with Functional API Tests
# =============================================================================

class TestConsistencyWithFunctionalAPI:
    """Test that class API gives same results as functional API."""

    def test_standard_mode_matches_functional(self, sample_data):
        """Test that PSN class matches psn() function for standard mode."""
        # Class API
        model = PSN(mode='standard', wantverbose=False, wantfig=False)
        model.fit(sample_data)

        # Functional API
        results = psn(sample_data, 'standard', {'wantverbose': False, 'wantfig': False})

        assert np.allclose(model.denoiseddata_, results['denoiseddata'])
        assert np.allclose(model.residuals_, results['residuals'])
        assert np.allclose(model.denoiser_, results['denoiser'])
        assert np.allclose(model.unit_means_, results['unit_means'])

    def test_conservative_mode_matches_functional(self, sample_data):
        """Test that PSN class matches psn() function for conservative mode."""
        model = PSN(mode='conservative', wantverbose=False, wantfig=False)
        model.fit(sample_data)

        results = psn(sample_data, 'conservative', {'wantverbose': False, 'wantfig': False})

        assert np.allclose(model.denoiseddata_, results['denoiseddata'])
        assert np.allclose(model.denoiser_, results['denoiser'])

    def test_aggressive_mode_matches_functional(self, sample_data):
        """Test that PSN class matches psn() function for aggressive mode."""
        model = PSN(mode='aggressive', wantverbose=False, wantfig=False)
        model.fit(sample_data)

        results = psn(sample_data, 'aggressive', {'wantverbose': False, 'wantfig': False})

        assert np.allclose(model.denoiseddata_, results['denoiseddata'])
        assert np.allclose(model.denoiser_, results['denoiser'])

    def test_custom_options_match_functional(self, sample_data):
        """Test custom options give same results."""
        opts = {
            'basis': 'difference',
            'criterion': 'variance',
            'threshold_method': 'global',
            'variance_threshold': 0.90,
            'wantverbose': False,
            'wantfig': False
        }

        model = PSN(mode=None, **opts)
        model.fit(sample_data)

        results = psn(sample_data, opts)

        assert np.allclose(model.denoiseddata_, results['denoiseddata'])
        assert np.allclose(model.denoiser_, results['denoiser'])


# =============================================================================
# Visualization Tests
# =============================================================================

class TestVisualization:
    """Test plot_diagnostics functionality."""

    def test_plot_diagnostics_returns_figure(self, small_data):
        """Test that plot_diagnostics returns a matplotlib figure."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt

        model = PSN(wantverbose=False, wantfig=False)
        model.fit(small_data)

        fig = model.plot_diagnostics()

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_fit_with_visualize_true(self, small_data):
        """Test fit with visualize=True."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        model = PSN(wantverbose=False)
        model.fit(small_data, visualize=True)

        assert model._is_fitted
        plt.close('all')

    def test_fit_transform_with_visualize_true(self, small_data):
        """Test fit_transform with visualize=True."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        model = PSN(wantverbose=False)
        result = model.fit_transform(small_data, visualize=True)

        assert model._is_fitted
        assert result is not None
        plt.close('all')


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_data_size(self):
        """Test with minimum viable data size (2 conditions, 2 trials)."""
        np.random.seed(42)
        data = np.random.randn(5, 2, 2)

        model = PSN(wantverbose=False, wantfig=False)
        model.fit(data)
        assert model._is_fitted

    def test_many_units_few_conditions(self):
        """Test with many units but few conditions."""
        np.random.seed(42)
        data = np.random.randn(50, 5, 4)

        model = PSN(wantverbose=False, wantfig=False)
        model.fit(data)
        assert model._is_fitted

    def test_few_units_many_conditions(self):
        """Test with few units but many conditions."""
        np.random.seed(42)
        data = np.random.randn(3, 100, 4)

        model = PSN(wantverbose=False, wantfig=False)
        model.fit(data)
        assert model._is_fitted

    def test_many_trials(self):
        """Test with many trials."""
        np.random.seed(42)
        data = np.random.randn(10, 20, 50)

        model = PSN(wantverbose=False, wantfig=False)
        model.fit(data)
        assert model._is_fitted

    def test_transform_different_nconds(self, small_data):
        """Test transform with different number of conditions."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(small_data)

        nunits = small_data.shape[0]
        # Transform with different number of conditions
        new_data = np.random.randn(nunits, 50, 3)
        result = model.transform(new_data)

        assert result.shape == (nunits, 50)

    def test_transform_different_ntrials(self, small_data):
        """Test transform with different number of trials."""
        model = PSN(wantverbose=False, wantfig=False)
        model.fit(small_data)

        nunits, nconds, _ = small_data.shape
        # Transform with different number of trials
        new_data = np.random.randn(nunits, nconds, 20)
        result = model.transform(new_data)

        assert result.shape == (nunits, nconds)

    def test_chaining(self, small_data):
        """Test method chaining."""
        new_data = np.random.randn(*small_data.shape)

        # fit returns self, so we can chain
        result = PSN(wantverbose=False, wantfig=False).fit(small_data).transform(new_data)

        assert result.shape == (small_data.shape[0], small_data.shape[1])

    def test_refit_overwrites_attributes(self, small_data):
        """Test that refitting overwrites previous attributes."""
        model = PSN(wantverbose=False, wantfig=False)

        # First fit
        model.fit(small_data)
        first_denoised = model.denoiseddata_.copy()

        # Second fit with different data
        different_data = small_data * 2 + 10
        model.fit(different_data)

        assert not np.allclose(first_denoised, model.denoiseddata_)

    def test_global_mode_has_signalsubspace(self, small_data):
        """Test that global threshold method has signalsubspace attribute."""
        model = PSN(threshold_method='global', wantverbose=False, wantfig=False)
        model.fit(small_data)

        # signalsubspace_ should exist (may be None if threshold is 0)
        assert hasattr(model, 'signalsubspace_')
        assert hasattr(model, 'dimreduce_')

    def test_unit_mode_has_unit_specific_outputs(self, small_data):
        """Test that unit threshold methods have unit-specific outputs."""
        model = PSN(threshold_method='hybrid', wantverbose=False, wantfig=False)
        model.fit(small_data)

        assert hasattr(model, 'unit_signal_vars_')
        assert hasattr(model, 'unit_noise_vars_')
        assert hasattr(model, 'unit_objectives_')


# =============================================================================
# Stress Tests
# =============================================================================

class TestStress:
    """Stress tests with larger data."""

    @pytest.mark.slow
    def test_large_data(self):
        """Test with larger dataset."""
        np.random.seed(42)
        data = np.random.randn(100, 200, 10)

        model = PSN(wantverbose=False, wantfig=False)
        model.fit(data)

        assert model._is_fitted
        assert model.denoiseddata_.shape == (100, 200)

    @pytest.mark.slow
    def test_multiple_fit_transform_cycles(self, sample_data):
        """Test multiple fit/transform cycles."""
        model = PSN(wantverbose=False, wantfig=False)

        for i in range(5):
            # Generate new data each cycle
            np.random.seed(i)
            data = np.random.randn(*sample_data.shape)
            model.fit(data)

            new_data = np.random.randn(*sample_data.shape)
            result = model.transform(new_data)

            assert result.shape == (sample_data.shape[0], sample_data.shape[1])


# =============================================================================
# Scikit-learn protocol compatibility
# =============================================================================

class TestSklearnProtocol:
    """PSN follows the scikit-learn estimator protocol (get/set_params, clone)."""

    def test_get_params_roundtrip(self):
        model = PSN(mode='aggressive', criterion='prediction', wantverbose=False)
        params = model.get_params()
        assert params['mode'] == 'aggressive'
        assert params['criterion'] == 'prediction'
        assert params['wantverbose'] is False
        # Every reported param is accepted back by the constructor.
        PSN(**params)

    def test_set_params(self):
        model = PSN(wantverbose=False)
        out = model.set_params(criterion='variance', threshold_method='global')
        assert out is model
        assert model.criterion == 'variance'
        assert model.threshold_method == 'global'

    def test_clone(self, small_data):
        from sklearn.base import clone
        model = PSN(mode='conservative', wantverbose=False, wantfig=False)
        cloned = clone(model)
        assert cloned is not model
        assert cloned.mode == 'conservative'
        assert cloned._is_fitted is False
        cloned.fit(small_data)
        assert cloned._is_fitted is True
        # Cloning a fitted estimator yields an unfitted one with the same params.
        assert clone(cloned)._is_fitted is False

    def test_check_is_fitted(self, small_data):
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted
        model = PSN(wantverbose=False, wantfig=False)
        with pytest.raises(NotFittedError):
            check_is_fitted(model)
        model.fit(small_data)
        check_is_fitted(model)  # must not raise

    def test_pipeline_single_step(self, small_data):
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([('psn', PSN(wantverbose=False, wantfig=False))])
        out = pipe.fit_transform(small_data)
        nunits, nconds, _ = small_data.shape
        assert out.shape == (nunits, nconds)

    def test_clone_set_params_sweep(self, sample_data):
        from sklearn.base import clone
        base = PSN(wantverbose=False, wantfig=False)
        for crit in ('max-tradeoff', 'prediction', 'variance'):
            est = clone(base).set_params(criterion=crit, threshold_method='global')
            est.fit(sample_data)
            assert est.opt_used_['criterion'] == crit


# =============================================================================
# Compare / Wiener modes
# =============================================================================

class TestCompareAndWienerModes:
    """The 'compare' and 'wiener' modes are reachable through the class."""

    @pytest.mark.parametrize('mode', ['compare', 'wiener'])
    def test_mode_fits(self, sample_data, mode):
        model = PSN(mode=mode, wantverbose=False, wantfig=False)
        model.fit(sample_data)
        assert model._is_fitted
        assert model.denoiseddata_.shape == sample_data.shape[:2]

    def test_compare_records_selection(self, sample_data):
        model = PSN(mode='compare', wantverbose=False, wantfig=False)
        model.fit(sample_data)
        assert model.threshold_selection_ is not None
        assert model.threshold_selection_['mode'] == 'compare'
        assert model.threshold_selection_['basis'] in ('signal', 'difference')

    def test_wiener_rejects_conflicting_option(self, small_data):
        model = PSN(mode='wiener', threshold_method='hybrid',
                    wantverbose=False, wantfig=False)
        with pytest.raises(ValueError, match='Wiener'):
            model.fit(small_data)

    def test_wiener_via_criterion_kwarg(self, sample_data):
        model = PSN(criterion='wiener', wantverbose=False, wantfig=False)
        model.fit(sample_data)
        assert model.opt_used_['criterion'] == 'wiener'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
