"""Wiring tests for select_threshold dispatch in psn().

Locks the behavior-preserving guarantees of the compare/wiener refactor:
  - criterion='wiener' is exactly the legacy basis='wiener' (full-rank filter)
  - basis='compare' selects signal or difference and matches that explicit run
  - explicit signal/difference are untouched (no threshold_selection record)
A shared gsn_result is passed where an exact denoiser match is asserted, so GSN
estimation is identical across the runs being compared.
"""
import numpy as np
import pytest

from psn.psn import psn

OPT = dict(wantfig=False, wantverbose=False)


def _data(seed=0, nunits=15, nconds=30, ntrials=4):
    rng = np.random.RandomState(seed)
    signal = rng.randn(nunits, nconds)
    return signal[:, :, None] + 0.6 * rng.randn(nunits, nconds, ntrials)


def test_wiener_criterion_equals_legacy_basis():
    d = _data()
    ref = psn(d, {**OPT, 'basis': 'signal'})       # any run to obtain a gsn_result
    g = ref['gsn_result']
    by_crit = psn(d, {**OPT, 'criterion': 'wiener', 'gsn_result': g})
    by_basis = psn(d, {**OPT, 'basis': 'wiener', 'gsn_result': g})
    assert by_crit['threshold_selection']['criterion'] == 'wiener'
    np.testing.assert_allclose(by_crit['denoiser'], by_basis['denoiser'], rtol=1e-10, atol=1e-12)


def test_compare_matches_chosen_candidate():
    d = _data()
    ref_sig = psn(d, {**OPT, 'basis': 'signal', 'criterion': 'max-tradeoff',
                      'threshold_method': 'hybrid'})
    g = ref_sig['gsn_result']
    comp = psn(d, {**OPT, 'basis': 'compare', 'criterion': 'max-tradeoff',
                   'threshold_method': 'hybrid', 'gsn_result': g})
    chosen = comp['threshold_selection']['basis']
    assert chosen in ('signal', 'difference')
    ref = psn(d, {**OPT, 'basis': chosen, 'criterion': 'max-tradeoff',
                  'threshold_method': 'hybrid', 'gsn_result': g})
    np.testing.assert_allclose(comp['denoiser'], ref['denoiser'], rtol=1e-10, atol=1e-12)
    assert set(comp['diagnostics']['candidates']) == {'signal', 'difference'}


def test_compare_reports_both_recovery_curves():
    d = _data()
    comp = psn(d, 'compare', OPT)
    rt = comp['recovery_tradeoff']
    for key in ('signal_basis', 'difference_basis'):
        assert key in rt
        assert rt[key].get('split_half_r') is not None        # empirical (validation)
        assert rt[key].get('analytic_recovery') is not None   # analytic (decision)


def test_explicit_basis_has_no_threshold_selection():
    d = _data()
    for b in ('signal', 'difference'):
        r = psn(d, {**OPT, 'basis': b})
        assert 'threshold_selection' not in r          # only auto/compare/wiener set it
        assert r['denoiser'].shape == (d.shape[0], d.shape[0])


def test_compare_tiebreak_prefers_signal(monkeypatch):
    """When the two bases' analytic recovery tie, 'compare' keeps signal."""
    import psn.utilities.threshold.select_threshold as st
    d = _data()
    g = psn(d, {**OPT, 'basis': 'signal'})['gsn_result']
    # force identical recovery for both candidates -> tie -> signal wins
    orig = st._select_lowrank
    def _equal(cSb, cNb, t, basis_key, criterion, opt, device='cpu'):
        rec = orig(cSb, cNb, t, basis_key, criterion, opt, device=device)
        rec['recovery'] = 0.5
        return rec
    monkeypatch.setattr(st, '_select_lowrank', _equal)
    comp = psn(d, {**OPT, 'basis': 'compare', 'criterion': 'max-tradeoff',
                   'threshold_method': 'hybrid', 'gsn_result': g})
    assert comp['threshold_selection']['basis'] == 'signal'
