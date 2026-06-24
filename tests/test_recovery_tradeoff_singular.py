"""Tests for the singular-matrix fallback in the recovery-tradeoff Wiener point.

`compute_recovery_tradeoff` builds a Wiener reference filter
    D = Sigma_S @ (Sigma_S + Sigma_N / t)^{-1}
by solving M @ D.T = cSb with M = cSb + cNb / t. M is PSD, but it can be
numerically singular: GSN floors cSb's eigenvalues to a 1e-10 ridge, and when a
direction carries ~0 signal AND ~0 noise (e.g. collinear units), M's smallest
eigenvalue sits at that floor. `torch.linalg.solve` hard-errors on such a matrix
(numpy's solver merely limps through), which previously crashed psn() entirely,
even in the default 'standard' mode, because the recovery-tradeoff diagnostic is
always computed.

The fix wraps the solve in a try/except that falls back to the pseudo-inverse,
using the same operand order as solve so the convention is identical. These tests
exercise that fallback:

* the pinv expression is mathematically identical to the solve on a non-singular M
  (guards against a transpose bug),
* psn() completes with a finite Wiener reference on rank-deficient data, and
* when torch.linalg.solve is forced to raise, the fallback runs and reproduces the
  primary-path result.

Run headless and per-file (torch + MKL can deadlock the full suite):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
        pytest tests/test_recovery_tradeoff_singular.py
"""

import numpy as np
import pytest

from psn import psn


def _well_conditioned_data(seed=11):
    """Ordinary full-rank data; M is well-conditioned (primary path)."""
    rng = np.random.RandomState(seed)
    nunits, nconds, ntrials = 10, 30, 4
    signal = rng.randn(nunits, nconds)
    data = signal[:, :, None] + 0.4 * rng.randn(nunits, nconds, ntrials)
    return data


def _rank_deficient_data(seed=0):
    """Data confined to a low-dim subspace plus an exactly duplicated unit, so the
    signal/noise covariances are rank-deficient and M = cSb + cNb/t is numerically
    singular (its smallest eigenvalue sits at the cSb 1e-10 floor)."""
    rng = np.random.RandomState(seed)
    nunits, nconds, ntrials, latent = 8, 40, 4, 3
    W = rng.randn(nunits, latent)                       # rank-3 mixing -> 5 null dirs
    signal = W @ rng.randn(latent, nconds)
    noise = np.einsum('ul,lct->uct', W, 0.3 * rng.randn(latent, nconds, ntrials))
    data = signal[:, :, None] + noise
    data[1] = data[0]                                   # exact linear dependency
    return data


QUIET = {'wantfig': False, 'wantverbose': False}


class TestRecoveryTradeoffWienerFallback:
    """Exercise the pinv fallback for the Wiener reference point."""

    def test_pinv_matches_solve_convention(self):
        """(pinv(M) @ cSb).T must equal solve(M, cSb).T on a non-singular M.

        This pins the exact expression used in the fallback (same operand order
        as the primary solve) and would fail if a transpose were dropped.
        """
        rng = np.random.RandomState(7)
        n = 12
        A = rng.randn(n, n)
        M = A @ A.T + np.eye(n)        # symmetric positive-definite
        cSb = rng.randn(n, n)
        cSb = cSb @ cSb.T              # symmetric positive-definite

        direct = np.linalg.solve(M, cSb).T
        fallback = (np.linalg.pinv(M) @ cSb).T
        assert np.allclose(direct, fallback, atol=1e-8)

    def test_rank_deficient_data_completes(self):
        """psn() on rank-deficient data yields a finite Wiener reference and output.

        Before the fix this could raise torch._C._LinAlgError from the always-on
        recovery-tradeoff diagnostic.
        """
        results = psn(_rank_deficient_data(), QUIET)

        rt = results['recovery_tradeoff']
        assert 'wiener' in rt
        assert np.isfinite(rt['wiener']['sv_frac'])
        assert np.isfinite(rt['wiener']['split_half_r'])
        assert np.all(np.isfinite(results['denoiseddata']))

    def test_pinv_fallback_runs_when_torch_solve_raises(self, monkeypatch):
        """Forcing torch.linalg.solve to raise must trigger the fallback, which
        reproduces the primary-path Wiener reference.

        recovery_tradeoff is the only torch.linalg.solve caller in the default
        pipeline, so patching it is safe and deterministically covers the except
        branch even though no real data reliably trips torch's singularity check.
        """
        import torch

        data = _well_conditioned_data()
        primary = psn(data, QUIET)['recovery_tradeoff']['wiener']

        lin_alg_error = getattr(torch._C, '_LinAlgError', RuntimeError)

        def _raise_singular(*args, **kwargs):
            raise lin_alg_error('forced singular matrix (test)')

        monkeypatch.setattr(torch.linalg, 'solve', _raise_singular)

        fallback = psn(data, QUIET)['recovery_tradeoff']['wiener']

        assert np.isfinite(fallback['sv_frac'])
        assert np.isfinite(fallback['split_half_r'])
        assert fallback['sv_frac'] == pytest.approx(primary['sv_frac'], abs=1e-6)
        assert fallback['split_half_r'] == pytest.approx(primary['split_half_r'], abs=1e-6)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
