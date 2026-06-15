"""Tests for GSN result caching in PSN.

Tests that passing opt['gsn_result'] from a previous call correctly skips
the GSN estimation while producing identical results. Covers:
- Correctness: cached results are bitwise identical to fresh runs
- Speed: cached run is faster than fresh run
- Validation: wrong-sized or missing-key gsn_result raises ValueError
- Compatibility: works with all basis types, criterion, threshold_method, alpha
- Reuse across configs: same gsn_result works with different hyperparameters
"""

import numpy as np
import pytest
import time
import matplotlib
matplotlib.use('Agg')

from psn import psn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sim_data():
    """Simple simulated data: 30 units, 15 conditions, 6 trials."""
    np.random.seed(42)
    nunits, nconds, ntrials = 30, 15, 6
    true_rank = 5
    U = np.linalg.qr(np.random.randn(nunits, true_rank))[0]
    coeffs = np.random.randn(true_rank, nconds) * np.array([5, 3, 2, 1, 0.5])[:, None]
    signal = U @ coeffs
    data = np.stack([signal + np.random.randn(nunits, nconds) * 2
                     for _ in range(ntrials)], axis=2)
    return data


@pytest.fixture
def base_result(sim_data):
    """Run PSN once to get a gsn_result for reuse."""
    return psn(sim_data, {'wantfig': False, 'wantverbose': False})


# ---------------------------------------------------------------------------
# Correctness: cached == fresh
# ---------------------------------------------------------------------------

class TestCorrectness:
    """Cached gsn_result produces identical output to a fresh run."""

    def test_identical_denoiseddata(self, sim_data, base_result):
        cached = psn(sim_data, {
            'wantfig': False, 'wantverbose': False,
            'gsn_result': base_result['gsn_result']
        })
        np.testing.assert_array_equal(
            cached['denoiseddata'], base_result['denoiseddata'])

    def test_identical_threshold(self, sim_data, base_result):
        cached = psn(sim_data, {
            'wantfig': False, 'wantverbose': False,
            'gsn_result': base_result['gsn_result']
        })
        np.testing.assert_array_equal(
            cached['best_threshold'], base_result['best_threshold'])

    def test_identical_denoiser(self, sim_data, base_result):
        cached = psn(sim_data, {
            'wantfig': False, 'wantverbose': False,
            'gsn_result': base_result['gsn_result']
        })
        np.testing.assert_array_equal(
            cached['denoiser'], base_result['denoiser'])

    def test_gsn_result_passthrough(self, sim_data, base_result):
        """The returned gsn_result should be the same object passed in."""
        cached = psn(sim_data, {
            'wantfig': False, 'wantverbose': False,
            'gsn_result': base_result['gsn_result']
        })
        np.testing.assert_array_equal(
            cached['gsn_result']['cSb'], base_result['gsn_result']['cSb'])
        np.testing.assert_array_equal(
            cached['gsn_result']['cNb'], base_result['gsn_result']['cNb'])


# ---------------------------------------------------------------------------
# Speed: cached is faster
# ---------------------------------------------------------------------------

class TestSpeed:

    def test_cached_faster(self, sim_data, base_result):
        """Cached run should be noticeably faster (no GSN)."""
        gsn_res = base_result['gsn_result']
        opt_base = {'wantfig': False, 'wantverbose': False}
        opt_cached = {**opt_base, 'gsn_result': gsn_res}

        # Time fresh run
        t0 = time.perf_counter()
        for _ in range(3):
            psn(sim_data, opt_base)
        t_fresh = time.perf_counter() - t0

        # Time cached run
        t0 = time.perf_counter()
        for _ in range(3):
            psn(sim_data, opt_cached)
        t_cached = time.perf_counter() - t0

        assert t_cached < t_fresh, (
            f"Cached ({t_cached:.3f}s) should be faster than fresh ({t_fresh:.3f}s)")


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:

    def test_missing_cSb_key(self, sim_data):
        with pytest.raises(ValueError, match="must contain 'cSb' and 'cNb'"):
            psn(sim_data, {
                'wantfig': False, 'wantverbose': False,
                'gsn_result': {'cNb': np.eye(30)}
            })

    def test_missing_cNb_key(self, sim_data):
        with pytest.raises(ValueError, match="must contain 'cSb' and 'cNb'"):
            psn(sim_data, {
                'wantfig': False, 'wantverbose': False,
                'gsn_result': {'cSb': np.eye(30)}
            })

    def test_wrong_size(self, sim_data):
        wrong = {'cSb': np.eye(10), 'cNb': np.eye(10)}
        with pytest.raises(ValueError, match="does not match data"):
            psn(sim_data, {
                'wantfig': False, 'wantverbose': False,
                'gsn_result': wrong
            })


# ---------------------------------------------------------------------------
# Compatibility with all basis / criterion / threshold combos
# ---------------------------------------------------------------------------

class TestCompatibility:

    @pytest.mark.parametrize("basis", ['signal', 'difference'])
    @pytest.mark.parametrize("threshold_method", ['global', 'hybrid'])
    def test_basis_threshold_combos(self, sim_data, base_result, basis, threshold_method):
        """Same gsn_result works across basis/threshold combos."""
        fresh = psn(sim_data, {
            'basis': basis, 'threshold_method': threshold_method,
            'wantfig': False, 'wantverbose': False,
        })
        cached = psn(sim_data, {
            'basis': basis, 'threshold_method': threshold_method,
            'wantfig': False, 'wantverbose': False,
            'gsn_result': base_result['gsn_result'],
        })
        np.testing.assert_array_equal(
            cached['denoiseddata'], fresh['denoiseddata'])

    def test_wiener_basis(self, sim_data, base_result):
        """Cached gsn_result works with basis='wiener'."""
        fresh = psn(sim_data, {
            'basis': 'wiener', 'wantfig': False, 'wantverbose': False,
        })
        cached = psn(sim_data, {
            'basis': 'wiener', 'wantfig': False, 'wantverbose': False,
            'gsn_result': base_result['gsn_result'],
        })
        np.testing.assert_array_equal(
            cached['denoiseddata'], fresh['denoiseddata'])

    @pytest.mark.parametrize("alpha", [0, 0.3, 0.5, 1.0])
    def test_alpha_values(self, sim_data, base_result, alpha):
        """Cached gsn_result works with various alpha values."""
        fresh = psn(sim_data, {
            'alpha': alpha, 'wantfig': False, 'wantverbose': False,
        })
        cached = psn(sim_data, {
            'alpha': alpha, 'wantfig': False, 'wantverbose': False,
            'gsn_result': base_result['gsn_result'],
        })
        np.testing.assert_array_equal(
            cached['denoiseddata'], fresh['denoiseddata'])

    @pytest.mark.parametrize("criterion", ['prediction', 'variance'])
    def test_criterion_values(self, sim_data, base_result, criterion):
        """Cached gsn_result works with different criteria."""
        fresh = psn(sim_data, {
            'criterion': criterion, 'threshold_method': 'global',
            'wantfig': False, 'wantverbose': False,
        })
        cached = psn(sim_data, {
            'criterion': criterion, 'threshold_method': 'global',
            'wantfig': False, 'wantverbose': False,
            'gsn_result': base_result['gsn_result'],
        })
        np.testing.assert_array_equal(
            cached['denoiseddata'], fresh['denoiseddata'])


# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------

class TestVerbose:

    def test_verbose_cached_message(self, sim_data, base_result, capsys):
        psn(sim_data, {
            'wantfig': False, 'wantverbose': True,
            'gsn_result': base_result['gsn_result'],
        })
        captured = capsys.readouterr()
        assert 'Using provided GSN result' in captured.out
        assert 'Running GSN' not in captured.out

    def test_verbose_fresh_message(self, sim_data, capsys):
        psn(sim_data, {'wantfig': False, 'wantverbose': True})
        captured = capsys.readouterr()
        assert 'Running GSN' in captured.out
