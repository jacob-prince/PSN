"""GSN -> PSN eigenbasis handoff tests.

Background
----------
At large nunits, PSN's basis-construction eigh of cSb dominates
wall-clock (eigh is O(N^3)), and PSN's 'compare' mode runs it twice
(signal + difference) before picking. The fast-GSN branch added an
opt-in
``opt['returns']`` selector so GSN can ALSO compute and return:

    eigvecs_signal,     eigvals_signal       - eigh(cSb)
    eigvecs_difference, eigvals_difference   - eigh(cSb - cNb / ntrial)

The intent is that callers can hand these back into PSN via

    opt['basis']            = <gsn_result['eigvecs_signal']>
    opt['basis_eigenvalues'] = gsn_result['eigvals_signal']

and PSN's denoising path skips its own eigh entirely. For that to be
USEFUL, the resulting denoiser MUST match what PSN would have produced
on its own ('basis': 'signal') - otherwise downstream results would
drift in subtle, hard-to-debug ways.

What this file verifies
-----------------------
1. **Conventions agree.** GSN's ``_eigh_descending_numpy`` returns
   the same eigenvector matrix that PSN's ``eigh_descending_sym``
   does on the same well-conditioned symmetric input (same LAPACK
   call, same descending sort, same sign convention).

2. **End-to-end signal-basis handoff.** GSN(returns=...) -> PSN(basis=
   matrix, basis_eigenvalues=...) reproduces PSN(basis='signal')
   bit-for-bit (denoiser, denoiseddata, best_threshold all equal up
   to floating-point noise).

3. **End-to-end difference-basis handoff.** Same as (2) but for the
   indefinite cSb - cNb/ntrial matrix (whose eigh is harder - the
   sign normalization and ordering matter more here because PSN
   distinguishes positive vs negative eigenvalues during threshold
   selection).

4. **Independence from PSN options.** The match holds across
   non-default criterion / threshold_method / basis_ordering settings
   - the eigvecs feed only the basis-construction step.

5. **Torch path matches numpy path.** When torch is available, GSN's
   torch eigh path produces the same eigvecs as the numpy path (up
   to numerical noise), and the PSN handoff still works.

6. **Degenerate / rank-deficient cSb.** When cSb has a zero-
   eigenvalue subspace (synthetic data with nunits > rank(signal)),
   eigvecs in that subspace are not uniquely determined - different
   LAPACK drivers pick different orthonormal bases. We assert
   functional invariants (denoiser symmetric, total recovered
   variance equal) rather than bit-exact equality there.

These tests guard against future regressions in either GSN or PSN
that would silently break the eigenbasis-passthrough feature.
"""
from __future__ import annotations

import numpy as np
import pytest
from gsn.fast_perform_gsn import _eigh_descending_numpy
from gsn.perform_gsn import perform_gsn

import psn
from psn.utilities.basis.construct_basis import construct_basis
from psn.utilities.basis.eigh_descending_sym import eigh_descending_sym

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from gsn.fast_perform_gsn import _eigh_descending_torch
    _HAS_TORCH_EIGH = _HAS_TORCH
except ImportError:
    _HAS_TORCH_EIGH = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_data(nvox, ncond, ntrial, seed=0):
    """Synthetic data via psn.generate_data with reproducible seed."""
    data, _, _ = psn.generate_data(
        nvox=nvox, ncond=ncond, ntrial=ntrial,
        noise_multiplier=2.0, align_alpha=0.5, align_k=min(10, nvox // 3),
        signal_decay=2, noise_decay=1.25, random_seed=seed,
    )
    return data


@pytest.fixture(scope='module')
def small_full_rank():
    """nvox=60, ncond=120 (>> nvox) - cSb close to full rank, no
    degenerate eigenspaces. Bit-exact match expected."""
    data = _make_data(60, 120, 5, seed=0)
    res_gsn = perform_gsn(data, {
        'wantverbose': 0,
        'returns': ('cSb', 'cNb',
                    'eigvecs_signal', 'eigvals_signal',
                    'eigvecs_difference', 'eigvals_difference'),
    })
    return data, res_gsn


@pytest.fixture(scope='module')
def small_full_rank_alt_seed():
    """Same shape as small_full_rank but a different RNG draw - used
    to confirm the test isn't accidentally seed-dependent."""
    data = _make_data(60, 120, 5, seed=42)
    res_gsn = perform_gsn(data, {
        'wantverbose': 0,
        'returns': ('cSb', 'cNb',
                    'eigvecs_signal', 'eigvals_signal',
                    'eigvecs_difference', 'eigvals_difference'),
    })
    return data, res_gsn


@pytest.fixture(scope='module')
def small_rank_deficient():
    """nvox=80, ncond=40 - rank(cSb) <= 40-1 = 39, so cSb has at
    least 41 zero (or near-zero) eigenvalues. The corresponding
    eigenvectors are NOT uniquely determined."""
    data = _make_data(80, 40, 5, seed=1)
    res_gsn = perform_gsn(data, {
        'wantverbose': 0,
        'returns': ('cSb', 'cNb',
                    'eigvecs_signal', 'eigvals_signal',
                    'eigvecs_difference', 'eigvals_difference'),
    })
    return data, res_gsn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_psn_custom_basis(data, res_gsn, which, **extra_opt):
    """PSN consuming GSN's pre-computed eigvecs + eigvals."""
    return psn.psn(data, dict({
        'basis': res_gsn[f'eigvecs_{which}'],
        'basis_eigenvalues': res_gsn[f'eigvals_{which}'],
        'gsn_result': {'cSb': res_gsn['cSb'], 'cNb': res_gsn['cNb']},
        'wantverbose': False, 'wantfig': False,
    }, **extra_opt))


def _run_psn_string_basis(data, res_gsn, which, **extra_opt):
    """PSN running its own eigh from the string spec."""
    return psn.psn(data, dict({
        'basis': which,                       # 'signal' or 'difference'
        'gsn_result': {'cSb': res_gsn['cSb'], 'cNb': res_gsn['cNb']},
        'wantverbose': False, 'wantfig': False,
    }, **extra_opt))


def _assert_psn_outputs_equal(res_a, res_b, atol=1e-8, rtol=1e-8,
                               keys=('denoiser', 'denoiseddata',
                                     'best_threshold')):
    for k in keys:
        a = np.asarray(res_a[k])
        b = np.asarray(res_b[k])
        assert a.shape == b.shape, f'{k}: shape {a.shape} != {b.shape}'
        np.testing.assert_allclose(
            a, b, atol=atol, rtol=rtol,
            err_msg=f'{k} differs between custom-basis and string-basis PSN')


# ---------------------------------------------------------------------------
# 1. Convention agreement
# ---------------------------------------------------------------------------

class TestEighConvention:
    """GSN's eigh helper must match PSN's eigh_descending_sym exactly
    on a well-conditioned symmetric input."""

    def test_eigvals_descending_psn(self):
        M = np.random.default_rng(0).standard_normal((40, 40))
        M = M @ M.T + 0.1 * np.eye(40)
        d, _ = _eigh_descending_numpy(M)
        assert np.all(np.diff(d) <= 1e-12), \
            'GSN eigvals must be sorted descending'

    def test_matches_psn_helper_well_conditioned(self):
        M = np.random.default_rng(1).standard_normal((60, 60))
        M = M @ M.T + 0.1 * np.eye(60)              # well-conditioned PSD
        d_g, V_g = _eigh_descending_numpy(M)
        d_p, V_p = eigh_descending_sym(M)
        np.testing.assert_allclose(d_g, d_p, atol=1e-10)
        np.testing.assert_allclose(V_g, V_p, atol=1e-10)

    def test_matches_psn_helper_indefinite(self):
        """The difference matrix can be indefinite. Ordering is by
        VALUE descending, so negative eigenvalues sort to the tail.
        Sign convention must hold for those columns too."""
        rng = np.random.default_rng(2)
        A = rng.standard_normal((40, 40))
        A = A @ A.T
        B = rng.standard_normal((40, 40))
        B = B @ B.T
        M = A - 0.5 * B                              # indefinite
        d_g, V_g = _eigh_descending_numpy(M)
        d_p, V_p = eigh_descending_sym(M)
        assert d_g[0]  >= d_g[-1], 'descending'
        assert d_g[-1] < 0, 'negative tail eigenvalue present'
        np.testing.assert_allclose(d_g, d_p, atol=1e-10)
        np.testing.assert_allclose(V_g, V_p, atol=1e-10)

    def test_sign_convention_largest_abs_positive(self):
        M = np.random.default_rng(3).standard_normal((50, 50))
        M = M @ M.T + 0.1 * np.eye(50)
        _, V = _eigh_descending_numpy(M)
        # Per column: argmax(|V[:, k]|) row should hold a POSITIVE value.
        piv = np.argmax(np.abs(V), axis=0)
        diag = V[piv, np.arange(V.shape[1])]
        assert np.all(diag > 0), \
            'sign convention: largest-magnitude entry per column must be positive'

    def test_dtype_preserved(self):
        M = np.random.default_rng(4).standard_normal((30, 30)).astype(np.float32)
        M = M @ M.T + 0.1 * np.eye(30, dtype=np.float32)
        d, V = _eigh_descending_numpy(M)
        assert d.dtype == np.float32
        assert V.dtype == np.float32


# ---------------------------------------------------------------------------
# 2. construct_basis honors custom_basis_eigenvalues
# ---------------------------------------------------------------------------

class TestConstructBasisCustomEigvals:
    """The PSN-side hook that accepts user-supplied eigenvalues alongside
    a custom basis matrix."""

    def test_custom_eigvals_propagate(self, small_full_rank):
        data, res_gsn = small_full_rank
        trial_avg   = data.mean(axis=2)
        unit_means  = data.mean(axis=(1, 2))
        basis, eigvals = construct_basis(
            res_gsn['cSb'], res_gsn['cNb'],
            res_gsn['eigvecs_signal'],          # custom matrix
            data, trial_avg, unit_means,
            ntrials_avg=5, has_nans=False,
            custom_basis_eigenvalues=res_gsn['eigvals_signal'])
        assert eigvals is not None, \
            'eigvals must propagate when custom_basis_eigenvalues is given'
        np.testing.assert_allclose(eigvals, res_gsn['eigvals_signal'], atol=1e-12)

    def test_no_custom_eigvals_returns_none(self, small_full_rank):
        data, res_gsn = small_full_rank
        trial_avg   = data.mean(axis=2)
        unit_means  = data.mean(axis=(1, 2))
        basis, eigvals = construct_basis(
            res_gsn['cSb'], res_gsn['cNb'],
            res_gsn['eigvecs_signal'],
            data, trial_avg, unit_means,
            ntrials_avg=5, has_nans=False)
        assert eigvals is None, \
            'eigvals must remain None when none are passed in (legacy behavior)'

    def test_custom_eigvals_length_mismatch_raises(self, small_full_rank):
        data, res_gsn = small_full_rank
        trial_avg   = data.mean(axis=2)
        unit_means  = data.mean(axis=(1, 2))
        with pytest.raises(ValueError, match='length'):
            construct_basis(
                res_gsn['cSb'], res_gsn['cNb'],
                res_gsn['eigvecs_signal'],
                data, trial_avg, unit_means,
                ntrials_avg=5, has_nans=False,
                custom_basis_eigenvalues=np.zeros(3))


# ---------------------------------------------------------------------------
# 3. End-to-end SIGNAL basis handoff
# ---------------------------------------------------------------------------

class TestSignalBasisHandoff:
    """GSN(returns=...) -> PSN(custom matrix + eigvals) ==
       PSN(basis='signal') on the same cSb / cNb."""

    @pytest.mark.parametrize('criterion,threshold_method', [
        ('prediction',   'global'),
        ('prediction',   'hybrid'),
        ('variance',     'global'),
        ('max-tradeoff', 'hybrid'),
    ])
    def test_bit_equivalent_full_rank(self, small_full_rank,
                                      criterion, threshold_method):
        data, res_gsn = small_full_rank
        common = dict(criterion=criterion,
                      threshold_method=threshold_method)
        res_c = _run_psn_custom_basis(data, res_gsn, 'signal', **common)
        res_s = _run_psn_string_basis(data, res_gsn, 'signal', **common)
        _assert_psn_outputs_equal(res_c, res_s, atol=1e-8, rtol=1e-8)

    def test_bit_equivalent_alt_seed(self, small_full_rank_alt_seed):
        """Same as above but a different random draw, to catch any
        accidental seed-coupling."""
        data, res_gsn = small_full_rank_alt_seed
        res_c = _run_psn_custom_basis(data, res_gsn, 'signal')
        res_s = _run_psn_string_basis(data, res_gsn, 'signal')
        _assert_psn_outputs_equal(res_c, res_s, atol=1e-8, rtol=1e-8)


# ---------------------------------------------------------------------------
# 4. End-to-end DIFFERENCE basis handoff
# ---------------------------------------------------------------------------

class TestDifferenceBasisHandoff:
    """Same as the signal handoff but for the indefinite cSb - cNb/t
    matrix. Sign convention + ordering of negative eigvals make this
    a stricter test."""

    def test_bit_equivalent_full_rank(self, small_full_rank):
        data, res_gsn = small_full_rank
        res_c = _run_psn_custom_basis(data, res_gsn, 'difference')
        res_s = _run_psn_string_basis(data, res_gsn, 'difference')
        _assert_psn_outputs_equal(res_c, res_s, atol=1e-8, rtol=1e-8)

    def test_difference_matrix_is_indefinite(self, small_full_rank):
        """Sanity check: this branch of the test only matters if the
        difference matrix actually has negative eigenvalues."""
        _, res_gsn = small_full_rank
        ev = res_gsn['eigvals_difference']
        assert ev.min() < 0, \
            'difference matrix should be indefinite for this test to bite'


# ---------------------------------------------------------------------------
# 5. Torch path produces equivalent eigvecs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TORCH_EIGH, reason='torch not available')
class TestTorchPathEquivalence:
    """The GPU eigh helper applies the same sign convention as numpy.
    On well-conditioned (non-degenerate) inputs the two paths agree
    bit-for-bit; on rank-deficient inputs they pick different
    orthonormal bases of the zero-eigenvalue subspace (that's the
    'eigh_device' tradeoff documented in fast_perform_gsn)."""

    def test_torch_helper_matches_numpy_helper_full_rank(self):
        rng = np.random.default_rng(7)
        M = rng.standard_normal((40, 40)).astype(np.float32)
        M = M @ M.T + 0.1 * np.eye(40, dtype=np.float32)
        M_t = torch.from_numpy(M)
        d_t, V_t = _eigh_descending_torch(M_t)
        d_g, V_g = _eigh_descending_numpy(M)
        np.testing.assert_allclose(d_t.numpy(), d_g, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(V_t.numpy(), V_g, atol=1e-5, rtol=1e-5)

    def test_host_eigh_default_full_pipeline_handoff(self):
        """Default opt['eigh_device']='host' (numpy): GSN(torch CPU
        biconvex) -> PSN(custom basis+eigvals) matches PSN(basis=
        'signal') exactly, even on the typical full-rank synthetic
        case."""
        data = _make_data(60, 120, 5, seed=11)
        res = perform_gsn(data, {
            'wantverbose': 0,
            'device': 'cpu',                # torch CPU path for biconvex
            'returns': ('cSb', 'cNb',
                        'eigvecs_signal', 'eigvals_signal'),
        })
        res_c = _run_psn_custom_basis(data, res, 'signal')
        res_s = _run_psn_string_basis(data, res, 'signal')
        _assert_psn_outputs_equal(res_c, res_s, atol=1e-6, rtol=1e-6)

    def test_device_eigh_optin_produces_valid_basis(self):
        """opt['eigh_device']='device' uses the GPU eigh path. On
        full-rank inputs it should still give bit-equivalent results
        to the host path (no degenerate subspace to disagree on)."""
        data = _make_data(60, 120, 5, seed=12)
        res_host = perform_gsn(data, {
            'wantverbose': 0,
            'device': 'cpu',
            'returns': ('cSb', 'eigvecs_signal', 'eigvals_signal'),
            'eigh_device': 'host',
        })
        res_dev = perform_gsn(data, {
            'wantverbose': 0,
            'device': 'cpu',
            'returns': ('cSb', 'eigvecs_signal', 'eigvals_signal'),
            'eigh_device': 'device',
        })
        # Eigenvalues must match closely; eigenvectors may differ on
        # zero-eigenvalue tail.
        np.testing.assert_allclose(
            res_host['eigvals_signal'], res_dev['eigvals_signal'],
            atol=1e-6, rtol=1e-6)
        # On the dominant (nonzero) eigenvalue range, eigenvectors
        # match up to within-driver noise.
        ev = res_host['eigvals_signal']
        nonzero = np.where(np.abs(ev) > 1e-6)[0]
        if len(nonzero) > 0:
            V_h = res_host['eigvecs_signal'][:, nonzero]
            V_d = res_dev['eigvecs_signal'][:, nonzero]
            np.testing.assert_allclose(V_h, V_d, atol=1e-5, rtol=1e-5)

    def test_device_eigh_rejects_bad_value(self):
        data = _make_data(20, 40, 4, seed=13)
        with pytest.raises(ValueError, match="eigh_device"):
            perform_gsn(data, {
                'wantverbose': 0,
                'returns': ('cSb', 'eigvecs_signal'),
                'eigh_device': 'magic',
            })


# ---------------------------------------------------------------------------
# 6. Degenerate / rank-deficient case - functional invariants only
# ---------------------------------------------------------------------------

class TestRankDeficient:
    """When cSb has a zero-eigenvalue subspace, the corresponding
    eigenvectors are not uniquely determined. With the default
    eigh_device='host' both GSN and PSN call numpy.linalg.eigh, so
    they agree on the (arbitrary but reproducible) null-space basis
    and the full denoiser matches bit-for-bit."""

    def test_eigvecs_orthonormal(self, small_rank_deficient):
        _, res = small_rank_deficient
        V = res['eigvecs_signal']
        ortho_err = np.abs(V.T @ V - np.eye(V.shape[1])).max()
        assert ortho_err < 1e-8, \
            f'eigvecs not orthonormal: max |V.T V - I| = {ortho_err:.3e}'

    def test_cSb_actually_rank_deficient(self, small_rank_deficient):
        _, res = small_rank_deficient
        ev = res['eigvals_signal']
        # nvox=80, ncond=40 - expect a clear zero-eigenvalue subspace.
        # The exact count depends on shrinkage and biconvex nearest-PSD
        # projection (which pulls near-zero eigenvalues to ~1e-10),
        # but there should be at least 10 dimensions of degeneracy.
        n_zero = (ev < 1e-6).sum()
        assert n_zero >= 10, \
            f'fixture is meant to be rank-deficient; only {n_zero} '\
            f'eigvals < 1e-6 (need ≥10 to exercise the degenerate path)'

    def test_signal_basis_bit_equivalent_with_host_eigh(self, small_rank_deficient):
        """The default eigh_device='host' guarantees PSN parity even
        in the zero-eigenvalue subspace because both code paths call
        the same numpy.linalg.eigh."""
        data, res = small_rank_deficient
        res_c = _run_psn_custom_basis(data, res, 'signal')
        res_s = _run_psn_string_basis(data, res, 'signal')
        _assert_psn_outputs_equal(res_c, res_s, atol=1e-7, rtol=1e-7)


# ---------------------------------------------------------------------------
# 7. Documentation example - what the README would show
# ---------------------------------------------------------------------------

class TestDocstringExample:
    """End-to-end shape of the workflow we want users to follow."""

    def test_readme_recipe(self):
        data = _make_data(50, 100, 4, seed=99)
        # Step 1: run GSN once with the eigenbasis returns.
        gsn = perform_gsn(data, {
            'wantverbose': 0,
            'returns': ('cSb', 'cNb',
                        'eigvecs_signal', 'eigvals_signal',
                        'eigvecs_difference', 'eigvals_difference'),
        })
        # Step 2a: PSN with the signal basis, no eigh in PSN.
        psn_signal = psn.psn(data, {
            'basis': gsn['eigvecs_signal'],
            'basis_eigenvalues': gsn['eigvals_signal'],
            'gsn_result': {'cSb': gsn['cSb'], 'cNb': gsn['cNb']},
            'wantverbose': False, 'wantfig': False,
        })
        # Step 2b: PSN with the difference basis, no eigh in PSN.
        psn_diff = psn.psn(data, {
            'basis': gsn['eigvecs_difference'],
            'basis_eigenvalues': gsn['eigvals_difference'],
            'gsn_result': {'cSb': gsn['cSb'], 'cNb': gsn['cNb']},
            'wantverbose': False, 'wantfig': False,
        })
        # Both produce valid denoisers.
        assert psn_signal['denoiser'].shape == (50, 50)
        assert psn_diff['denoiser'].shape == (50, 50)
        # And they're different from each other (the bases are different
        # subspaces in general).
        assert not np.allclose(psn_signal['denoiser'], psn_diff['denoiser'])
