"""Tests for the alpha interpolation parameter in PSN.

Tests that the alpha parameter correctly interpolates between the prediction
peak (alpha=0) and variance retention threshold (alpha=1) in signal variance
space. Covers:
- Boundary equivalence (alpha=0 == prediction, alpha=1 == variance)
- Monotonicity of thresholds with increasing alpha
- All threshold methods (global, hybrid, unit)
- Multiple basis types (signal, difference)
- Edge cases (degenerate data, alpha has no effect)
- Results dict contains alpha_info when alpha is set
- Visualization runs without error
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')

from psn import psn
from psn.utilities.threshold.select_threshold_analytic import select_threshold_analytic


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lowrank_data():
    """Data with a clear low-rank signal so prediction and variance thresholds diverge.

    50 units, 20 conditions, 4 trials. True signal rank ~8.
    """
    np.random.seed(42)
    nunits, nconds, ntrials = 50, 20, 4
    true_rank = 8
    U = np.linalg.qr(np.random.randn(nunits, true_rank))[0]
    coeffs = np.random.randn(true_rank, nconds) * np.array(
        [5, 4, 3, 2, 1.5, 1, 0.5, 0.3])[:, None]
    signal = U @ coeffs
    data = (np.repeat(signal[:, :, np.newaxis], ntrials, axis=2)
            + np.random.randn(nunits, nconds, ntrials) * 3)
    return data


@pytest.fixture
def small_data():
    """Small dataset for quick tests: 10 units, 15 conditions, 5 trials."""
    np.random.seed(123)
    nunits, nconds, ntrials = 10, 15, 5
    signal = np.random.randn(nunits, nconds)
    data = signal[:, :, np.newaxis] + 0.5 * np.random.randn(nunits, nconds, ntrials)
    return data


# ---------------------------------------------------------------------------
# Unit-level tests on select_threshold_analytic
# ---------------------------------------------------------------------------

class TestSelectThresholdAlpha:
    """Test alpha logic in select_threshold_analytic directly."""

    def _make_data(self):
        """Signal that decays; noise that increases — creates a prediction/variance gap."""
        signal = np.array([20, 15, 8, 3, 1.5, 0.8, 0.4, 0.2, 0.1, 0.05])
        noise = np.array([1, 1, 1, 5, 8, 10, 12, 14, 16, 18])
        return signal, noise

    def test_alpha0_matches_prediction(self):
        signal, noise = self._make_data()
        ntrials = 2
        k_pred, obj_pred = select_threshold_analytic(
            signal, noise, None, ntrials,
            {'criterion': 'prediction', 'variance_threshold': 0.99, 'alpha': None})
        k_a0, obj_a0 = select_threshold_analytic(
            signal, noise, None, ntrials,
            {'criterion': 'prediction', 'variance_threshold': 0.99, 'alpha': 0})
        assert k_pred == k_a0
        np.testing.assert_array_equal(obj_pred, obj_a0)

    def test_alpha1_matches_variance(self):
        signal, noise = self._make_data()
        ntrials = 2
        k_var, _ = select_threshold_analytic(
            signal, noise, None, ntrials,
            {'criterion': 'variance', 'variance_threshold': 0.99})
        k_a1, _ = select_threshold_analytic(
            signal, noise, None, ntrials,
            {'criterion': 'prediction', 'variance_threshold': 0.99, 'alpha': 1})
        assert k_var == k_a1

    def test_monotonicity(self):
        signal, noise = self._make_data()
        ntrials = 2
        ks = []
        for a in [0, 0.1, 0.3, 0.5, 0.7, 1.0]:
            k, _ = select_threshold_analytic(
                signal, noise, None, ntrials,
                {'criterion': 'prediction', 'variance_threshold': 0.99, 'alpha': a})
            ks.append(k)
        for i in range(len(ks) - 1):
            assert ks[i] <= ks[i + 1], f'alpha monotonicity violated: {ks}'

    def test_returns_prediction_objective(self):
        """Alpha branch should return the prediction-style objective curve."""
        signal, noise = self._make_data()
        ntrials = 2
        _, obj_alpha = select_threshold_analytic(
            signal, noise, None, ntrials,
            {'criterion': 'prediction', 'variance_threshold': 0.99, 'alpha': 0.5})
        _, obj_pred = select_threshold_analytic(
            signal, noise, None, ntrials,
            {'criterion': 'prediction', 'variance_threshold': 0.99, 'alpha': None})
        np.testing.assert_array_equal(obj_alpha, obj_pred)

    def test_spred_geq_svar_uses_kpred(self):
        """When prediction peak already exceeds variance target, alpha has no effect."""
        # High signal, low noise — prediction keeps everything already
        signal = np.array([10, 8, 5, 3, 2])
        noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        ntrials = 10
        k_a0, _ = select_threshold_analytic(
            signal, noise, None, ntrials,
            {'criterion': 'prediction', 'variance_threshold': 0.5, 'alpha': 0})
        k_a1, _ = select_threshold_analytic(
            signal, noise, None, ntrials,
            {'criterion': 'prediction', 'variance_threshold': 0.5, 'alpha': 1})
        # Both should be equal because S_pred >= S_var
        assert k_a0 == k_a1

    def test_zero_signal(self):
        """All-zero signal should yield k=0."""
        signal = np.zeros(5)
        noise = np.ones(5)
        k, obj = select_threshold_analytic(
            signal, noise, None, 3,
            {'criterion': 'prediction', 'variance_threshold': 0.99, 'alpha': 0.5})
        assert k == 0


# ---------------------------------------------------------------------------
# Integration tests: full PSN pipeline
# ---------------------------------------------------------------------------

class TestAlphaGlobal:
    """Test alpha with threshold_method='global'."""

    OPT_BASE = {'threshold_method': 'global', 'wantfig': False, 'wantverbose': False}

    def test_alpha0_matches_prediction(self, lowrank_data):
        r_pred = psn(lowrank_data, {**self.OPT_BASE, 'criterion': 'prediction'})
        r_a0 = psn(lowrank_data, {**self.OPT_BASE, 'alpha': 0})
        assert r_pred['best_threshold'] == r_a0['best_threshold']

    def test_alpha1_matches_variance(self, lowrank_data):
        r_var = psn(lowrank_data, {**self.OPT_BASE, 'criterion': 'variance',
                                    'variance_threshold': 0.99})
        r_a1 = psn(lowrank_data, {**self.OPT_BASE, 'alpha': 1,
                                   'variance_threshold': 0.99})
        assert r_var['best_threshold'] == r_a1['best_threshold']

    def test_monotonicity(self, lowrank_data):
        ks = []
        for a in [0, 0.25, 0.5, 0.75, 1.0]:
            r = psn(lowrank_data, {**self.OPT_BASE, 'alpha': a})
            ks.append(r['best_threshold'])
        for i in range(len(ks) - 1):
            assert ks[i] <= ks[i + 1], f'Global monotonicity: {ks}'

    def test_alpha_info_in_results(self, lowrank_data):
        r = psn(lowrank_data, {**self.OPT_BASE, 'alpha': 0.3})
        assert 'alpha_info' in r
        ai = r['alpha_info']
        assert 'k_pred' in ai and 'k_var' in ai and 'alpha' in ai
        assert ai['alpha'] == 0.3
        assert ai['k_pred'] <= r['best_threshold'] <= ai['k_var'] or ai['k_pred'] >= ai['k_var']

    def test_alpha_none_no_alpha_info(self, lowrank_data):
        r = psn(lowrank_data, {**self.OPT_BASE, 'criterion': 'prediction'})
        assert 'alpha_info' not in r

    def test_denoised_shape(self, lowrank_data):
        r = psn(lowrank_data, {**self.OPT_BASE, 'alpha': 0.5})
        assert r['denoiseddata'].shape == lowrank_data.shape[:2]
        assert not np.any(np.isnan(r['denoiseddata']))


class TestAlphaHybrid:
    """Test alpha with threshold_method='hybrid'."""

    OPT_BASE = {'threshold_method': 'hybrid', 'wantfig': False, 'wantverbose': False}

    def test_alpha0_matches_prediction(self, lowrank_data):
        r_pred = psn(lowrank_data, {**self.OPT_BASE, 'criterion': 'prediction'})
        r_a0 = psn(lowrank_data, {**self.OPT_BASE, 'alpha': 0})
        np.testing.assert_array_equal(r_pred['best_threshold'], r_a0['best_threshold'])

    def test_alpha1_geq_variance(self, lowrank_data):
        """alpha=1 thresholds should be >= variance thresholds (max with prediction peak)."""
        r_var = psn(lowrank_data, {**self.OPT_BASE, 'criterion': 'variance',
                                    'variance_threshold': 0.99})
        r_a1 = psn(lowrank_data, {**self.OPT_BASE, 'alpha': 1,
                                   'variance_threshold': 0.99})
        # alpha=1 uses max(k_var, k_pred), so it's >= pure variance threshold
        assert np.all(r_a1['best_threshold'] >= r_var['best_threshold'])

    def test_monotonicity_mean(self, lowrank_data):
        means = []
        for a in [0, 0.5, 1.0]:
            r = psn(lowrank_data, {**self.OPT_BASE, 'alpha': a})
            means.append(np.mean(r['best_threshold']))
        for i in range(len(means) - 1):
            assert means[i] <= means[i + 1], f'Hybrid monotonicity: {means}'

    def test_unit_objectives_present(self, lowrank_data):
        r = psn(lowrank_data, {**self.OPT_BASE, 'alpha': 0.3})
        assert 'unit_objectives' in r
        assert len(r['unit_objectives']) == lowrank_data.shape[0]


# ---------------------------------------------------------------------------
# Basis types
# ---------------------------------------------------------------------------

class TestAlphaBasisTypes:
    """Test alpha with different basis types."""

    OPT_BASE = {'threshold_method': 'global', 'wantfig': False, 'wantverbose': False}

    def test_signal_basis(self, lowrank_data):
        r = psn(lowrank_data, {**self.OPT_BASE, 'basis': 'signal', 'alpha': 0.5})
        assert np.isscalar(r['best_threshold'])

    def test_difference_basis(self, lowrank_data):
        """Difference basis with alpha should bypass the eigenvalue fast path."""
        r_pred = psn(lowrank_data, {**self.OPT_BASE, 'basis': 'difference',
                                     'criterion': 'prediction'})
        r_a0 = psn(lowrank_data, {**self.OPT_BASE, 'basis': 'difference', 'alpha': 0})
        assert r_pred['best_threshold'] == r_a0['best_threshold']

    def test_difference_basis_monotonicity(self, lowrank_data):
        ks = []
        for a in [0, 0.5, 1.0]:
            r = psn(lowrank_data, {**self.OPT_BASE, 'basis': 'difference', 'alpha': a})
            ks.append(r['best_threshold'])
        for i in range(len(ks) - 1):
            assert ks[i] <= ks[i + 1], f'Difference basis monotonicity: {ks}'

    def test_difference_hybrid(self, lowrank_data):
        r = psn(lowrank_data, {'basis': 'difference', 'threshold_method': 'hybrid',
                                'alpha': 0.3, 'wantfig': False, 'wantverbose': False})
        assert r['denoiseddata'].shape == lowrank_data.shape[:2]


# ---------------------------------------------------------------------------
# Composability with variance_threshold
# ---------------------------------------------------------------------------

class TestAlphaVarianceThreshold:
    """Test that alpha composes with different variance_threshold values."""

    OPT_BASE = {'threshold_method': 'global', 'wantfig': False, 'wantverbose': False}

    def test_lower_vt_yields_fewer_dims(self, lowrank_data):
        r_high = psn(lowrank_data, {**self.OPT_BASE, 'alpha': 1, 'variance_threshold': 0.99})
        r_low = psn(lowrank_data, {**self.OPT_BASE, 'alpha': 1, 'variance_threshold': 0.5})
        assert r_low['best_threshold'] <= r_high['best_threshold']

    def test_alpha1_vt050_geq_variance(self, lowrank_data):
        """alpha=1 with low vt: prediction peak may exceed variance target,
        so alpha=1 threshold >= variance threshold (clamped by k_pred)."""
        r_var = psn(lowrank_data, {**self.OPT_BASE, 'criterion': 'variance',
                                    'variance_threshold': 0.5})
        r_a1 = psn(lowrank_data, {**self.OPT_BASE, 'alpha': 1,
                                   'variance_threshold': 0.5})
        assert r_a1['best_threshold'] >= r_var['best_threshold']


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

class TestAlphaVisualization:
    """Test that diagnostic figures render without error when alpha is active."""

    def test_global_viz(self, lowrank_data, tmp_path):
        fig_path = str(tmp_path / 'alpha_global.png')
        r = psn(lowrank_data, {'alpha': 0.3, 'threshold_method': 'global',
                                'wantfig': True, 'wantverbose': False,
                                'figurepath': fig_path})
        import os
        assert os.path.exists(fig_path)

    def test_hybrid_viz(self, lowrank_data, tmp_path):
        fig_path = str(tmp_path / 'alpha_hybrid.png')
        r = psn(lowrank_data, {'alpha': 0.3, 'threshold_method': 'hybrid',
                                'wantfig': True, 'wantverbose': False,
                                'figurepath': fig_path})
        import os
        assert os.path.exists(fig_path)

    def test_global_viz_alpha0(self, small_data, tmp_path):
        """Boundary: alpha=0 visualization should also work."""
        fig_path = str(tmp_path / 'alpha0_global.png')
        psn(small_data, {'alpha': 0, 'threshold_method': 'global',
                          'wantfig': True, 'wantverbose': False,
                          'figurepath': fig_path})
        import os
        assert os.path.exists(fig_path)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestAlphaEdgeCases:
    """Edge cases for alpha parameter."""

    OPT_BASE = {'threshold_method': 'global', 'wantfig': False, 'wantverbose': False}

    def test_alpha_none_is_default(self, small_data):
        """alpha=None should behave identically to not setting it."""
        r1 = psn(small_data, {**self.OPT_BASE, 'criterion': 'prediction'})
        r2 = psn(small_data, {**self.OPT_BASE, 'criterion': 'prediction', 'alpha': None})
        assert r1['best_threshold'] == r2['best_threshold']
        assert 'alpha_info' not in r1
        assert 'alpha_info' not in r2

    def test_alpha_overrides_criterion(self, lowrank_data):
        """When alpha is set, the criterion field is ignored."""
        r1 = psn(lowrank_data, {**self.OPT_BASE, 'criterion': 'prediction', 'alpha': 0.5})
        r2 = psn(lowrank_data, {**self.OPT_BASE, 'criterion': 'variance', 'alpha': 0.5})
        assert r1['best_threshold'] == r2['best_threshold']

    def test_opt_used_contains_alpha(self, small_data):
        r = psn(small_data, {**self.OPT_BASE, 'alpha': 0.3})
        assert r['opt_used']['alpha'] == 0.3

    def test_alpha_with_unit_groups(self, lowrank_data):
        """Alpha should work with custom unit_groups in hybrid mode."""
        nunits = lowrank_data.shape[0]
        groups = np.repeat(np.arange(5), nunits // 5)
        r = psn(lowrank_data, {'alpha': 0.5, 'threshold_method': 'hybrid',
                                'unit_groups': groups, 'wantfig': False,
                                'wantverbose': False})
        # Units in the same group should share a threshold
        for g in range(5):
            mask = groups == g
            vals = r['best_threshold'][mask]
            assert np.all(vals == vals[0])
