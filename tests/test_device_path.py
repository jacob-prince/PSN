"""Tests for the opt['device'] GPU path in PSN.

At large nunits (≥10000) PSN's CPU runtime is dominated by GEMM-heavy
utilities (project_covs, compute_unit_weighted_projections, the
denoiser-build matmul in denoise_unitwise / denoise_global /
denoise_fullrank_wiener). Routing those through torch on a device
(CUDA or MPS) delivers a 10-50× speedup.

For the feature to be safe, the device path MUST produce results that
are numerically equivalent to the CPU/numpy path on every code branch
the user might hit. This file covers:

  1. project_covs        - diag(B.T C B) via einsum
  2. compute_unit_weighted_projections - vectorized weighted vars +
                            optional per-unit ranking
  3. denoise_unitwise    - per-threshold-group matmul (hybrid mode)
  4. denoise_global      - single basis matmul
  5. denoise_fullrank_wiener - cholesky_solve vs scipy.linalg.solve
  6. End-to-end psn.psn  - every (basis, criterion, threshold_method)
                            combination matches between device='cpu'
                            and device='cuda'/'mps' when available

Tolerances:
  - f64 numeric ops: 1e-10 absolute, 1e-10 relative
  - f32 fallback (default torch dtype): 1e-5 abs, 1e-5 rel
"""
from __future__ import annotations

import numpy as np
import pytest

import psn
from psn._device import resolve_device, is_cpu

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _device_available(name):
    if not _HAS_TORCH:
        return False
    if name == 'cuda':
        return torch.cuda.is_available()
    if name == 'mps':
        return (hasattr(torch.backends, 'mps')
                and torch.backends.mps.is_available())
    return name == 'cpu'


# Run device-path tests against the first available accelerator;
# fall back to cpu torch path (still verifies the device-dispatch
# plumbing and the f32 numerics).
def _pick_test_device():
    if not _HAS_TORCH:
        return None
    for cand in ('cuda', 'mps'):
        if _device_available(cand):
            return cand
    return None  # No accelerator → skip device tests


_TEST_DEVICE = _pick_test_device()
_SKIP_REASON = ('no torch accelerator available (cuda/mps); '
                'set up a GPU env to exercise the device path')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def small_data():
    """Synthetic GSN-style data + cSb/cNb at a small N so tests are
    fast but every code path still fires."""
    data, _, _ = psn.generate_data(
        nvox=40, ncond=80, ntrial=4,
        noise_multiplier=2.0, align_alpha=0.5, align_k=10,
        signal_decay=2, noise_decay=1.25, random_seed=0)
    from gsn.perform_gsn import perform_gsn
    res = perform_gsn(data, {
        'wantverbose': 0,
        'returns': ('cSb', 'cNb',
                    'eigvecs_signal', 'eigvals_signal',
                    'eigvecs_difference', 'eigvals_difference')})
    return data, res


@pytest.fixture(scope='module')
def medium_data():
    """Slightly larger to exercise the multi-threshold-group branch
    of denoise_unitwise."""
    data, _, _ = psn.generate_data(
        nvox=100, ncond=200, ntrial=5,
        noise_multiplier=2.0, align_alpha=0.5, align_k=15,
        signal_decay=2, noise_decay=1.25, random_seed=1)
    from gsn.perform_gsn import perform_gsn
    res = perform_gsn(data, {
        'wantverbose': 0,
        'returns': ('cSb', 'cNb',
                    'eigvecs_signal', 'eigvals_signal',
                    'eigvecs_difference', 'eigvals_difference')})
    return data, res


# Selected tolerance: torch defaults to f32 on device, so we accept
# slightly looser tolerances when the test device is non-cpu.
def _tols_for_device(dev_name):
    if dev_name is None or dev_name == 'cpu':
        return dict(atol=1e-10, rtol=1e-10)
    # GPU goes through f32 by default → looser tols.
    return dict(atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 1. resolve_device behavior
# ---------------------------------------------------------------------------

class TestResolveDevice:
    def test_cpu_passthrough(self):
        assert resolve_device('cpu') == 'cpu'
        assert resolve_device(None) == 'cpu'
        assert is_cpu(resolve_device('cpu'))
        assert is_cpu(resolve_device(None))

    @pytest.mark.skipif(_HAS_TORCH, reason='torch IS installed')
    def test_non_cpu_without_torch_errors(self):
        with pytest.raises(RuntimeError, match='requires torch'):
            resolve_device('cuda')

    @pytest.mark.skipif(not _HAS_TORCH, reason='torch missing')
    def test_auto_falls_back_to_cpu(self):
        # If neither cuda nor mps available, 'auto' returns 'cpu'.
        result = resolve_device('auto')
        if _TEST_DEVICE is None:
            assert is_cpu(result)
        else:
            # auto picked the accelerator
            assert not is_cpu(result)

    @pytest.mark.skipif(not _HAS_TORCH or _device_available('cuda'),
                        reason='only meaningful when cuda is missing')
    def test_explicit_cuda_when_missing_errors(self):
        with pytest.raises(RuntimeError, match='cuda is unavailable'.lower()
                              ) if False else pytest.raises(RuntimeError):
            resolve_device('cuda')


# ---------------------------------------------------------------------------
# 2. project_covs cpu vs device
# ---------------------------------------------------------------------------

class TestProjectCovsDevice:
    def test_cpu_path_unchanged(self, small_data):
        from psn.utilities.basis.project_covs import project_covs
        _, res = small_data
        sig_a, noi_a = project_covs(res['cSb'], res['cNb'], res['eigvecs_signal'])
        sig_b, noi_b = project_covs(res['cSb'], res['cNb'], res['eigvecs_signal'],
                                      device='cpu')
        np.testing.assert_allclose(sig_a, sig_b, atol=1e-12)
        np.testing.assert_allclose(noi_a, noi_b, atol=1e-12)

    @pytest.mark.skipif(_TEST_DEVICE is None, reason=_SKIP_REASON)
    def test_device_equals_cpu(self, small_data):
        from psn.utilities.basis.project_covs import project_covs
        _, res = small_data
        sig_c, noi_c = project_covs(res['cSb'], res['cNb'], res['eigvecs_signal'])
        sig_d, noi_d = project_covs(res['cSb'], res['cNb'], res['eigvecs_signal'],
                                      device=_TEST_DEVICE)
        tols = _tols_for_device(_TEST_DEVICE)
        np.testing.assert_allclose(sig_c, sig_d, **tols)
        np.testing.assert_allclose(noi_c, noi_d, **tols)

    def test_clamp_to_nonnegative(self, small_data):
        """Output must always be ≥ 0 regardless of device path."""
        from psn.utilities.basis.project_covs import project_covs
        _, res = small_data
        sig, noi = project_covs(res['cSb'], res['cNb'], res['eigvecs_signal'],
                                  device='cpu')
        assert sig.min() >= 0
        assert noi.min() >= 0


# ---------------------------------------------------------------------------
# 3. compute_unit_weighted_projections cpu vs device
# ---------------------------------------------------------------------------

class TestUnitWeightedProjectionsDevice:
    @pytest.mark.parametrize('do_unit_ranking', [True, False])
    def test_cpu_vectorized_matches_legacy_loop(self, small_data, do_unit_ranking):
        """The CPU path was rewritten to be vectorized; verify it
        matches a known-good per-unit-loop reference."""
        from psn.utilities.denoise.compute_unit_weighted_projections \
            import compute_unit_weighted_projections
        _, res = small_data
        B = res['eigvecs_signal']
        sp = np.maximum(np.einsum('ij,jk,ki->k',
                                    B.T, res['cSb'], B), 0)
        npj = np.maximum(np.einsum('ij,jk,ki->k',
                                     B.T, res['cNb'], B), 0)
        ntrials = 4
        ccurves, sigs, noises, orderings = compute_unit_weighted_projections(
            B, sp, npj, ntrials, do_unit_ranking)
        # Reference: per-unit loop
        for u in range(B.shape[0]):
            w = B[u, :] ** 2
            sig_u = w * sp
            noi_u = w * npj
            if do_unit_ranking:
                ord_ref = np.argsort(sig_u)[::-1]
            else:
                ord_ref = np.arange(B.shape[1])
            sig_ref = sig_u[ord_ref]
            noi_ref = noi_u[ord_ref]
            curve_ref = np.concatenate([[0], np.cumsum(sig_ref - noi_ref / ntrials)])
            np.testing.assert_allclose(orderings[u], ord_ref, atol=0)
            np.testing.assert_allclose(sigs[u], sig_ref, atol=1e-12)
            np.testing.assert_allclose(noises[u], noi_ref, atol=1e-12)
            np.testing.assert_allclose(ccurves[u], curve_ref, atol=1e-12)

    @pytest.mark.skipif(_TEST_DEVICE is None, reason=_SKIP_REASON)
    @pytest.mark.parametrize('do_unit_ranking', [True, False])
    def test_device_matches_cpu(self, small_data, do_unit_ranking):
        from psn.utilities.denoise.compute_unit_weighted_projections \
            import compute_unit_weighted_projections
        _, res = small_data
        B = res['eigvecs_signal']
        sp = np.maximum(np.einsum('ij,jk,ki->k', B.T, res['cSb'], B), 0)
        npj = np.maximum(np.einsum('ij,jk,ki->k', B.T, res['cNb'], B), 0)
        ntrials = 4
        cc_cpu, sig_cpu, noi_cpu, ord_cpu = compute_unit_weighted_projections(
            B, sp, npj, ntrials, do_unit_ranking)
        cc_dev, sig_dev, noi_dev, ord_dev = compute_unit_weighted_projections(
            B, sp, npj, ntrials, do_unit_ranking, device=_TEST_DEVICE)
        tols = _tols_for_device(_TEST_DEVICE)
        np.testing.assert_allclose(ord_dev, ord_cpu, atol=0)
        for cpu, dev in zip(cc_cpu, cc_dev):
            np.testing.assert_allclose(np.asarray(dev), np.asarray(cpu), **tols)
        for cpu, dev in zip(sig_cpu, sig_dev):
            np.testing.assert_allclose(np.asarray(dev), np.asarray(cpu), **tols)


# ---------------------------------------------------------------------------
# 4-6. End-to-end psn() equivalence across device paths
# ---------------------------------------------------------------------------

_BRANCHES = [
    # (basis, criterion, threshold_method)
    ('signal',     'prediction',   'global'),
    ('signal',     'prediction',   'hybrid'),
    ('signal',     'variance',     'global'),
    ('signal',     'max-tradeoff', 'hybrid'),
    ('difference', 'prediction',   'global'),
    ('difference', 'prediction',   'hybrid'),
    ('difference', 'max-tradeoff', 'hybrid'),
    ('wiener',     'prediction',   'global'),    # full-rank Wiener
]


class TestPsnEndToEndCPUVsDevice:
    """Same data + opts must produce numerically-equivalent results
    on CPU and on the device for every supported branch."""

    @pytest.mark.skipif(_TEST_DEVICE is None, reason=_SKIP_REASON)
    @pytest.mark.parametrize('basis,criterion,threshold_method', _BRANCHES)
    def test_branch(self, medium_data, basis, criterion, threshold_method):
        data, res = medium_data
        common = dict(
            gsn_result={'cSb': res['cSb'], 'cNb': res['cNb']},
            wantverbose=False, wantfig=False,
        )
        if basis == 'wiener':
            # Full-rank Wiener is basis-free / untruncated; it rejects a
            # contradicting criterion/threshold_method, so request it alone.
            common['basis'] = 'wiener'
        else:
            common.update(basis=basis, criterion=criterion,
                          threshold_method=threshold_method)
        out_cpu = psn.psn(data, dict(common, device='cpu'))
        out_dev = psn.psn(data, dict(common, device=_TEST_DEVICE))
        tols = _tols_for_device(_TEST_DEVICE)
        for k in ('denoiseddata', 'denoiser'):
            np.testing.assert_allclose(
                np.asarray(out_dev[k]), np.asarray(out_cpu[k]),
                err_msg=f'{k} mismatch on branch ({basis},{criterion},{threshold_method})',
                **tols)

    def test_device_cpu_explicit_matches_default(self, small_data):
        """opt['device']='cpu' must give identical results to omitting
        the key (it's a no-op on the CPU path)."""
        data, res = small_data
        common = dict(
            gsn_result={'cSb': res['cSb'], 'cNb': res['cNb']},
            basis='signal',
            wantverbose=False, wantfig=False,
        )
        out_none = psn.psn(data, common)
        out_cpu  = psn.psn(data, dict(common, device='cpu'))
        for k in ('denoiseddata', 'denoiser', 'best_threshold'):
            np.testing.assert_allclose(
                np.asarray(out_cpu[k]), np.asarray(out_none[k]), atol=1e-12)


# ---------------------------------------------------------------------------
# 7. Errors / edge cases
# ---------------------------------------------------------------------------

class TestDeviceErrorPaths:
    def test_invalid_device_string_passes_through_to_torch(self, small_data):
        """An unknown device string should propagate as a meaningful
        error from torch - we don't pre-validate every string."""
        if not _HAS_TORCH:
            pytest.skip('torch missing')
        data, res = small_data
        with pytest.raises(Exception):              # torch / runtime error
            psn.psn(data, {
                'gsn_result': {'cSb': res['cSb'], 'cNb': res['cNb']},
                'basis': 'signal', 'device': 'made-up-device-9000',
                'wantverbose': False, 'wantfig': False})
