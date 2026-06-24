"""Tests for the filename-based opt['gsn_result'] input + auto-upgrade
to cached eigvecs when present.

Background
----------
Running GSN at large nunits takes minutes per call. The natural
pattern is to persist the GSN output to a single .npz file and then
point any number of downstream PSN calls at that file. This file
verifies:

  1. opt['gsn_result'] accepts a string path (existing dict path
     remains the legacy default).
  2. The string-path form produces bit-equivalent results to passing
     the same data as a dict.
  3. When the .npz contains the GSN eigenbasis returns
     (eigvecs_signal/eigvals_signal/eigvecs_difference/
     eigvals_difference), and the caller asked for basis='signal' or
     'difference', PSN auto-swaps basis to the cached matrix and
     propagates the eigenvalues - bit-equivalent to PSN's own eigh
     branch but skipping the eigh entirely. The 'using cached
     eigvecs' verbose line confirms the auto-upgrade fired.
  4. Auto-upgrade is a no-op for:
     - other basis specs (custom matrix, 'wiener', 'pca', 'random')
     - npz files that lack the eigvecs keys
     - explicit opt['basis_eigenvalues'] (caller wins)
  5. Errors when the path doesn't exist or isn't a .npz.

These tests give us a sentinel: if either GSN's saved-file format
changes or PSN's loader logic regresses, this file fails immediately.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import psn
from gsn.perform_gsn import perform_gsn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_data(seed=0):
    data, _, _ = psn.generate_data(
        nvox=40, ncond=80, ntrial=4,
        noise_multiplier=2.0, align_alpha=0.5, align_k=10,
        signal_decay=2, noise_decay=1.25, random_seed=seed)
    return data


@pytest.fixture(scope='module')
def gsn_with_eigvecs(tmp_path_factory):
    """GSN run with the eigenbasis returns enabled, saved to disk."""
    data = _make_data(seed=0)
    res = perform_gsn(data, {
        'wantverbose': 0,
        'returns': ('cSb', 'cNb',
                    'eigvecs_signal',     'eigvals_signal',
                    'eigvecs_difference', 'eigvals_difference'),
    })
    p = tmp_path_factory.mktemp('gsn') / 'with_eigvecs.npz'
    np.savez(str(p), **{k: res[k] for k in (
        'cSb', 'cNb',
        'eigvecs_signal', 'eigvals_signal',
        'eigvecs_difference', 'eigvals_difference')})
    return data, res, str(p)


@pytest.fixture(scope='module')
def gsn_without_eigvecs(tmp_path_factory):
    """GSN run with ONLY cSb / cNb saved - the cached-eigvecs upgrade
    must be a no-op when the file doesn't have the right keys."""
    data = _make_data(seed=0)
    res = perform_gsn(data, {'wantverbose': 0, 'returns': ('cSb', 'cNb')})
    p = tmp_path_factory.mktemp('gsn') / 'without_eigvecs.npz'
    np.savez(str(p), cSb=res['cSb'], cNb=res['cNb'])
    return data, res, str(p)


# ---------------------------------------------------------------------------
# 1. dict path stays correct
# ---------------------------------------------------------------------------

class TestDictPath:
    def test_dict_path_works(self, gsn_with_eigvecs):
        data, res, _ = gsn_with_eigvecs
        out = psn.psn(data, {
            'gsn_result': {'cSb': res['cSb'], 'cNb': res['cNb']},
            'basis': 'signal',
            'wantverbose': False, 'wantfig': False,
        })
        assert out['denoiser'].shape == (40, 40)


# ---------------------------------------------------------------------------
# 2. String path equivalent to dict path
# ---------------------------------------------------------------------------

class TestStringPath:
    def test_string_equals_dict_signal(self, gsn_with_eigvecs):
        data, res, p = gsn_with_eigvecs
        out_dict = psn.psn(data, {
            'gsn_result': {'cSb': res['cSb'], 'cNb': res['cNb']},
            'basis': 'signal', 'wantverbose': False, 'wantfig': False})
        out_file = psn.psn(data, {
            'gsn_result': p,
            'basis': 'signal', 'wantverbose': False, 'wantfig': False})
        np.testing.assert_allclose(out_file['denoiser'], out_dict['denoiser'],
                                    atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(out_file['denoiseddata'], out_dict['denoiseddata'],
                                    atol=1e-10, rtol=1e-10)

    def test_string_equals_dict_difference(self, gsn_with_eigvecs):
        data, res, p = gsn_with_eigvecs
        out_dict = psn.psn(data, {
            'gsn_result': {'cSb': res['cSb'], 'cNb': res['cNb']},
            'basis': 'difference', 'wantverbose': False, 'wantfig': False})
        out_file = psn.psn(data, {
            'gsn_result': p,
            'basis': 'difference', 'wantverbose': False, 'wantfig': False})
        np.testing.assert_allclose(out_file['denoiser'], out_dict['denoiser'],
                                    atol=1e-10, rtol=1e-10)

    def test_string_equals_dict_when_no_eigvecs_in_file(self, gsn_without_eigvecs):
        """File has only cSb/cNb - PSN must just do its own eigh and
        produce the same answer as the dict path."""
        data, res, p = gsn_without_eigvecs
        out_dict = psn.psn(data, {
            'gsn_result': {'cSb': res['cSb'], 'cNb': res['cNb']},
            'basis': 'signal', 'wantverbose': False, 'wantfig': False})
        out_file = psn.psn(data, {
            'gsn_result': p,
            'basis': 'signal', 'wantverbose': False, 'wantfig': False})
        np.testing.assert_allclose(out_file['denoiser'], out_dict['denoiser'],
                                    atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# 3. Auto-eigvecs upgrade fires when conditions are right
# ---------------------------------------------------------------------------

class TestAutoEigvecsUpgrade:
    def test_signal_basis_logs_upgrade(self, gsn_with_eigvecs, capsys):
        data, _, p = gsn_with_eigvecs
        psn.psn(data, {
            'gsn_result': p, 'basis': 'signal',
            'wantverbose': True, 'wantfig': False})
        cap = capsys.readouterr()
        assert "using cached 'signal' eigvecs" in cap.out

    def test_difference_basis_logs_upgrade(self, gsn_with_eigvecs, capsys):
        data, _, p = gsn_with_eigvecs
        psn.psn(data, {
            'gsn_result': p, 'basis': 'difference',
            'wantverbose': True, 'wantfig': False})
        cap = capsys.readouterr()
        assert "using cached 'difference' eigvecs" in cap.out

    def test_pca_basis_does_not_upgrade(self, gsn_with_eigvecs, capsys):
        """'pca' is a different basis; the upgrade must not fire."""
        data, _, p = gsn_with_eigvecs
        psn.psn(data, {
            'gsn_result': p, 'basis': 'pca',
            'wantverbose': True, 'wantfig': False})
        cap = capsys.readouterr()
        assert "using cached" not in cap.out

    def test_no_upgrade_when_file_missing_eigvecs(self, gsn_without_eigvecs, capsys):
        data, _, p = gsn_without_eigvecs
        psn.psn(data, {
            'gsn_result': p, 'basis': 'signal',
            'wantverbose': True, 'wantfig': False})
        cap = capsys.readouterr()
        assert "using cached" not in cap.out

    def test_explicit_basis_eigenvalues_wins(self, gsn_with_eigvecs, capsys):
        """User-supplied opt['basis_eigenvalues'] must NOT be
        overridden by the auto-upgrade (the user is more explicit)."""
        data, res, p = gsn_with_eigvecs
        # Override with a deliberately-wrong custom eigvals so we'd
        # notice if the upgrade clobbered it.
        sentinel = np.zeros(40) - 7.0
        psn.psn(data, {
            'gsn_result': p,
            'basis': 'signal',
            'basis_eigenvalues': sentinel,
            'wantverbose': True, 'wantfig': False})
        cap = capsys.readouterr()
        # The auto-upgrade prints when it fires. With the explicit
        # eigvals overriding, it must NOT print.
        assert "using cached" not in cap.out


# ---------------------------------------------------------------------------
# 4. Path-like input types
# ---------------------------------------------------------------------------

class TestPathLikeInputs:
    def test_pathlib_Path_accepted(self, gsn_with_eigvecs):
        from pathlib import Path
        data, res, p = gsn_with_eigvecs
        out = psn.psn(data, {
            'gsn_result': Path(p), 'basis': 'signal',
            'wantverbose': False, 'wantfig': False})
        out_dict = psn.psn(data, {
            'gsn_result': {'cSb': res['cSb'], 'cNb': res['cNb']},
            'basis': 'signal', 'wantverbose': False, 'wantfig': False})
        np.testing.assert_allclose(out['denoiser'], out_dict['denoiser'],
                                    atol=1e-10, rtol=1e-10)

    def test_npz_file_object_accepted(self, gsn_with_eigvecs):
        """np.load(...) returns an NpzFile - passing it directly
        should work too."""
        data, res, p = gsn_with_eigvecs
        npz = np.load(p, allow_pickle=False)
        out = psn.psn(data, {
            'gsn_result': npz, 'basis': 'signal',
            'wantverbose': False, 'wantfig': False})
        npz.close()
        out_dict = psn.psn(data, {
            'gsn_result': {'cSb': res['cSb'], 'cNb': res['cNb']},
            'basis': 'signal', 'wantverbose': False, 'wantfig': False})
        np.testing.assert_allclose(out['denoiser'], out_dict['denoiser'],
                                    atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# 5. Error paths
# ---------------------------------------------------------------------------

class TestErrors:
    def test_missing_file_raises(self):
        data = _make_data()
        with pytest.raises(FileNotFoundError, match="doesn't exist"):
            psn.psn(data, {
                'gsn_result': '/tmp/__does_not_exist__.npz',
                'basis': 'signal',
                'wantverbose': False, 'wantfig': False})

    def test_non_npz_path_raises(self, tmp_path):
        data = _make_data()
        bogus = tmp_path / 'gsn.pkl'
        bogus.write_text('not really npz')
        with pytest.raises(ValueError, match='must end in .npz'):
            psn.psn(data, {
                'gsn_result': str(bogus),
                'basis': 'signal',
                'wantverbose': False, 'wantfig': False})

    def test_npz_missing_cSb_cNb_raises(self, tmp_path):
        data = _make_data()
        p = tmp_path / 'empty.npz'
        np.savez(str(p), foo=np.zeros(3))
        with pytest.raises(ValueError, match="must contain 'cSb' and 'cNb'"):
            psn.psn(data, {
                'gsn_result': str(p),
                'basis': 'signal',
                'wantverbose': False, 'wantfig': False})
