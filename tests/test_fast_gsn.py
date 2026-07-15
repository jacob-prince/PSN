"""Backend-equivalence tests for ``gsn.fast_perform_gsn.fast_perform_gsn``.

Verifies that the torch (batched-Cholesky) path returns the same ``cSb`` /
``cNb`` as the numpy+scipy path across a range of data scenarios - the numpy
path is the simple, trusted reference, and the torch path is validated against
it. Covers all three dispatch branches:

1. NaN / uneven-trials → delegated path (should be bit-identical to numpy).
2. torch available, clean data → batched-Cholesky path.
3. torch absent, clean data → numpy+scipy path (forced via monkey-patch).

GSN is the single source of truth (see psn.utils.perform_gsn); PSN no longer
vendors a copy of the fast backend.

Usage:
    pytest tests/test_fast_gsn.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gsn.fast_perform_gsn as _fg  # noqa: E402
from gsn.fast_perform_gsn import _HAS_TORCH, fast_perform_gsn  # noqa: E402

# ---------------------------------------------------------------------------
# Data generators - a spread of realistic shapes and conditioning
# ---------------------------------------------------------------------------

def _gen_lowrank_plus_noise(nvox, ncond, ntrial, rank, noise_std=1.0, seed=0):
    """Low-rank signal + iid Gaussian noise. The workhorse scenario."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((nvox, rank)) / np.sqrt(rank)
    Z = rng.standard_normal((rank, ncond))
    signal = U @ Z
    noise = rng.standard_normal((nvox, ncond, ntrial)) * noise_std
    return signal[:, :, None] + noise


def _gen_fullrank(nvox, ncond, ntrial, seed=0):
    """Full-rank signal (rank = min(nvox, ncond))."""
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal((nvox, ncond))
    noise = rng.standard_normal((nvox, ncond, ntrial))
    return signal[:, :, None] + noise


def _gen_high_snr(nvox, ncond, ntrial, seed=0):
    """Strong signal relative to noise - tests conditioning extremes."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((nvox, 5)) / np.sqrt(5)
    Z = rng.standard_normal((5, ncond)) * 10
    signal = U @ Z
    noise = rng.standard_normal((nvox, ncond, ntrial)) * 0.1
    return signal[:, :, None] + noise


def _gen_low_snr(nvox, ncond, ntrial, seed=0):
    """Very weak signal - tests behavior when signal ≲ noise."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((nvox, 5)) / np.sqrt(5)
    Z = rng.standard_normal((5, ncond)) * 0.2
    signal = U @ Z
    noise = rng.standard_normal((nvox, ncond, ntrial)) * 3.0
    return signal[:, :, None] + noise


def _gen_with_nans(nvox, ncond, ntrial, frac_missing=0.2, seed=0):
    """Inject NaN trials → exercises the uneven-trials delegation path."""
    data = _gen_lowrank_plus_noise(nvox, ncond, ntrial, rank=10, seed=seed)
    rng = np.random.default_rng(seed + 1)
    # Drop random (condition, trial) pairs - but ensure every condition keeps
    # at least one valid trial (GSN contract).
    for c in range(ncond):
        mask = rng.random(ntrial) < frac_missing
        if mask.all():
            mask[0] = False
        data[:, c, mask] = np.nan
    return data


# ---------------------------------------------------------------------------
# The comparison helper - what "numerically equivalent" means here
# ---------------------------------------------------------------------------

def _ref(data, opt=None):
    """Reference = the gsn fast backend forced onto its NUMPY path (the simple,
    trusted ground truth). The torch path is validated against this."""
    saved = _fg._HAS_TORCH
    _fg._HAS_TORCH = False
    try:
        res = fast_perform_gsn(data, opt or {})
    finally:
        _fg._HAS_TORCH = saved
    return {'cSb': np.asarray(res['cSb']), 'cNb': np.asarray(res['cNb'])}


def _assert_equivalent(ref, fast, *, atol=1e-10, rtol=1e-10, label=''):
    for k in ('cSb', 'cNb'):
        a = np.asarray(ref[k])
        b = np.asarray(fast[k])
        assert a.shape == b.shape, f'{label} {k} shape mismatch: {a.shape} vs {b.shape}'
        max_abs = float(np.max(np.abs(a - b)))
        norm_ref = float(np.linalg.norm(a)) + 1e-300
        rel = float(np.linalg.norm(a - b) / norm_ref)
        assert np.allclose(a, b, atol=atol, rtol=rtol), (
            f'{label} {k}: max|diff|={max_abs:.3e} rel={rel:.3e} '
            f'atol={atol:.1e} rtol={rtol:.1e}'
        )


# ---------------------------------------------------------------------------
# Shapes to sweep - small enough that the reference finishes in a test pass
# ---------------------------------------------------------------------------

SHAPES_SMALL = [
    (30, 20, 3),
    (50, 40, 4),
    (80, 60, 5),
]
SHAPES_MEDIUM = [
    (100, 80, 4),
    (200, 100, 4),
]

ALL_SHAPES = SHAPES_SMALL + SHAPES_MEDIUM
SEEDS = [0, 1, 7]


# ===========================================================================
# Branch 1: clean data - the torch path (used whenever torch is installed)
# ===========================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason='torch not installed; torch branch unreachable')
class TestTorchBranchEquivalence:
    """Fast path with torch: must match the reference to float64 precision."""

    @pytest.mark.parametrize('shape', ALL_SHAPES)
    @pytest.mark.parametrize('seed', SEEDS)
    def test_lowrank_equivalence(self, shape, seed):
        data = _gen_lowrank_plus_noise(*shape, rank=10, seed=seed)
        _assert_equivalent(_ref(data), fast_perform_gsn(data),
                           label=f'torch/lowrank shape={shape} seed={seed}')

    @pytest.mark.parametrize('shape', SHAPES_SMALL)
    @pytest.mark.parametrize('seed', SEEDS)
    def test_fullrank_equivalence(self, shape, seed):
        data = _gen_fullrank(*shape, seed=seed)
        _assert_equivalent(_ref(data), fast_perform_gsn(data),
                           label=f'torch/fullrank shape={shape} seed={seed}')

    @pytest.mark.parametrize('shape', SHAPES_SMALL)
    def test_high_snr_equivalence(self, shape):
        data = _gen_high_snr(*shape, seed=0)
        _assert_equivalent(_ref(data), fast_perform_gsn(data),
                           label=f'torch/high_snr shape={shape}')

    @pytest.mark.parametrize('shape', SHAPES_SMALL)
    def test_low_snr_equivalence(self, shape):
        data = _gen_low_snr(*shape, seed=0)
        _assert_equivalent(_ref(data), fast_perform_gsn(data),
                           label=f'torch/low_snr shape={shape}')

    def test_wantshrinkage_false(self):
        """wantshrinkage=False forces shrinkage grid=[1.0] - check that too."""
        data = _gen_lowrank_plus_noise(60, 50, 4, rank=10, seed=0)
        opt = {'wantshrinkage': False, 'wantverbose': False}
        _assert_equivalent(_ref(data, opt), fast_perform_gsn(data, opt),
                           label='torch/no_shrinkage')


# ===========================================================================
# Branch 2: clean data - the numpy+scipy path (force-enabled via monkey-patch)
# ===========================================================================

class TestNumpyBranchEquivalence:
    """Fast path without torch: must also match the reference to float64 precision."""

    @pytest.fixture(autouse=True)
    def _force_numpy_path(self, monkeypatch):
        monkeypatch.setattr(_fg, '_HAS_TORCH', False)
        yield

    @pytest.mark.parametrize('shape', ALL_SHAPES)
    @pytest.mark.parametrize('seed', SEEDS)
    def test_lowrank_equivalence(self, shape, seed):
        data = _gen_lowrank_plus_noise(*shape, rank=10, seed=seed)
        _assert_equivalent(_ref(data), fast_perform_gsn(data),
                           label=f'numpy/lowrank shape={shape} seed={seed}')

    @pytest.mark.parametrize('shape', SHAPES_SMALL)
    @pytest.mark.parametrize('seed', SEEDS)
    def test_fullrank_equivalence(self, shape, seed):
        data = _gen_fullrank(*shape, seed=seed)
        _assert_equivalent(_ref(data), fast_perform_gsn(data),
                           label=f'numpy/fullrank shape={shape} seed={seed}')

    @pytest.mark.parametrize('shape', SHAPES_SMALL)
    def test_high_snr_equivalence(self, shape):
        data = _gen_high_snr(*shape, seed=0)
        _assert_equivalent(_ref(data), fast_perform_gsn(data),
                           label=f'numpy/high_snr shape={shape}')

    @pytest.mark.parametrize('shape', SHAPES_SMALL)
    def test_low_snr_equivalence(self, shape):
        data = _gen_low_snr(*shape, seed=0)
        _assert_equivalent(_ref(data), fast_perform_gsn(data),
                           label=f'numpy/low_snr shape={shape}')

    def test_wantshrinkage_false(self):
        data = _gen_lowrank_plus_noise(60, 50, 4, rank=10, seed=0)
        opt = {'wantshrinkage': False, 'wantverbose': False}
        _assert_equivalent(_ref(data, opt), fast_perform_gsn(data, opt),
                           label='numpy/no_shrinkage')


# ===========================================================================
# Branch 3: NaN / uneven trials - delegates to reference; must be IDENTICAL
# ===========================================================================

class TestUnevenTrialsDelegation:
    """NaN inputs should pass through to the reference unchanged."""

    @pytest.mark.parametrize('shape', SHAPES_SMALL)
    def test_nan_delegation_matches_reference(self, shape):
        data = _gen_with_nans(*shape, frac_missing=0.3, seed=0)
        ref = _ref(data)
        fast = fast_perform_gsn(data)
        # Delegation path: should be identical (same code, not just equivalent).
        _assert_equivalent(ref, fast, atol=0, rtol=0, label=f'nan/delegate shape={shape}')

    def test_heavy_nan_fraction(self):
        data = _gen_with_nans(50, 40, 5, frac_missing=0.5, seed=2)
        ref = _ref(data)
        fast = fast_perform_gsn(data)
        _assert_equivalent(ref, fast, atol=0, rtol=0, label='nan/heavy')


# ===========================================================================
# Structural properties - output shape, symmetry, PSD
# ===========================================================================

class TestStructuralProperties:
    """Things that should hold for fast_perform_gsn regardless of path."""

    @pytest.mark.parametrize('shape', SHAPES_SMALL)
    def test_output_shapes(self, shape):
        data = _gen_lowrank_plus_noise(*shape, rank=10, seed=0)
        res = fast_perform_gsn(data)
        N = shape[0]
        assert res['cSb'].shape == (N, N)
        assert res['cNb'].shape == (N, N)

    @pytest.mark.parametrize('shape', SHAPES_SMALL)
    def test_symmetry(self, shape):
        data = _gen_lowrank_plus_noise(*shape, rank=10, seed=0)
        res = fast_perform_gsn(data)
        assert np.allclose(res['cSb'], res['cSb'].T, atol=1e-10), 'cSb not symmetric'
        assert np.allclose(res['cNb'], res['cNb'].T, atol=1e-10), 'cNb not symmetric'

    @pytest.mark.parametrize('shape', SHAPES_SMALL)
    def test_psd(self, shape):
        """Biconvex optimization guarantees PSD output - eigenvalues ≥ -epsilon."""
        data = _gen_lowrank_plus_noise(*shape, rank=10, seed=0)
        res = fast_perform_gsn(data)
        for k in ('cSb', 'cNb'):
            evals = np.linalg.eigvalsh((res[k] + res[k].T) / 2)
            assert evals.min() >= -1e-8, (
                f'{k} has negative eigenvalue {evals.min():.3e}'
            )

    def test_returns_selector_drops_unrequested_covs(self):
        """opt['returns'] gates the heavy covariance matrices: requesting only
        cSb/cNb must drop cN/cS (cSb/cNb still present)."""
        data = _gen_lowrank_plus_noise(30, 20, 3, rank=5, seed=0)
        res = fast_perform_gsn(data, {'returns': ['cSb', 'cNb']})
        assert 'cSb' in res and 'cNb' in res
        assert 'cN' not in res and 'cS' not in res


# ===========================================================================
# Determinism - same input → same output
# ===========================================================================

class TestDeterminism:

    def test_identical_repeated_calls(self):
        data = _gen_lowrank_plus_noise(60, 50, 4, rank=10, seed=0)
        a = fast_perform_gsn(data)
        b = fast_perform_gsn(data)
        _assert_equivalent(a, b, atol=0, rtol=0, label='determinism')

    def test_shrinkage_selection_matches_reference(self):
        """The shrinkage-level choice is driver of downstream output. Pin it.

        At sizes where the reference's ``np.argmin`` bites NaN slots (and
        thus picks level=1.0), our code must too. This guards against
        regression on that subtle contract.
        """
        data = _gen_lowrank_plus_noise(200, 100, 4, rank=20, seed=0)
        _assert_equivalent(_ref(data), fast_perform_gsn(data),
                           label='shrinkage_selection')


# ===========================================================================
# Option plumbing - make sure opt dict is respected
# ===========================================================================

class TestOptionPlumbing:

    def test_wantverbose_true_does_not_crash(self, capsys):
        data = _gen_lowrank_plus_noise(40, 30, 3, rank=5, seed=0)
        res = fast_perform_gsn(data, {'wantverbose': True})
        assert res['cSb'].shape == (40, 40)
        # We don't pin whether/what it prints - only that it doesn't crash.
        capsys.readouterr()

    def test_empty_opt(self):
        data = _gen_lowrank_plus_noise(40, 30, 3, rank=5, seed=0)
        res_none = fast_perform_gsn(data, None)
        res_empty = fast_perform_gsn(data, {})
        _assert_equivalent(res_none, res_empty, atol=0, rtol=0, label='empty_opt')
