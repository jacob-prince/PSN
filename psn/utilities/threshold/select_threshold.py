"""Basis and threshold selection for PSN.

Chooses which eigenbasis to use and how many dimensions to retain for a single
PSN run, covering the 'compare', 'signal', and 'difference' bases, and routing
the full-rank Wiener filter back to the caller. See select_threshold.
"""
import numpy as np

from ..basis.eigh_descending_sym import eigh_descending_sym
from ..diagnostics.split_half import split_half_r
from .max_tradeoff import max_tradeoff_threshold
from .select_threshold_analytic import select_threshold_analytic

_HARD_CRITERIA = ('max-tradeoff', 'prediction', 'variance', 'variance_eigenvalues')


def format_threshold(best_threshold):
    """Compact verbose string for the retained-dimension threshold(s).

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <best_threshold> - scalar or [nunits] array. Number of dimensions retained:
        a single value (global mode) or one threshold per unit (hybrid mode).

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <s> - string. The integer threshold for global mode, or 'mean ± std across
        units' for per-unit (hybrid) mode, never the full per-unit list.
    """
    bt = np.atleast_1d(np.asarray(best_threshold))
    if bt.size == 1:
        return f"{int(bt.flat[0])}"
    return f"{np.mean(bt):.1f} ± {np.std(bt):.1f} (mean ± std across units)"


def _basis_eigvecs(cSb, cNb, t, basis_key, device='cpu'):
    """Descending-eigenvalue basis for the requested key (matches recovery_tradeoff)."""
    if basis_key == 'signal':
        M = cSb
    elif basis_key == 'difference':
        M = cSb - cNb / t
    else:
        raise ValueError(f"_basis_eigvecs: unknown basis '{basis_key}'")
    evals, V = eigh_descending_sym(M, do_symmetrize=(basis_key == 'difference'), device=device)
    return V, evals


def _per_dim_var(cov, V):
    """Per-dimension variance of `cov` along the columns of V: diag(V^T cov V)."""
    return np.sum((cov @ V) * V, axis=0)


def _analytic_recovery_curve(sig, noi, t, total_S):
    """sv_frac(K) and normalized analytic recovery(K) on the K=0..ndims grid."""
    sv = np.concatenate([[0.0], np.cumsum(sig)]) / total_S
    rec = np.concatenate([[0.0], np.cumsum(sig - noi / t)]) / total_S
    return sv, rec


def _select_lowrank(cSb, cNb, t, basis_key, criterion, opt, device='cpu'):
    """Build basis, pick K for the given criterion, return a threshold record + analytic curve."""
    V, evals = _basis_eigvecs(cSb, cNb, t, basis_key, device=device)
    sig = _per_dim_var(cSb, V)
    noi = _per_dim_var(cNb, V)
    total_S = float(np.sum(sig))
    # Honor allowable_thresholds so each candidate's K is the one the denoiser
    # would actually use; compare must not rank bases at a disallowed threshold.
    allowable = opt.get('allowable_thresholds')
    if criterion == 'max-tradeoff':
        k = max_tradeoff_threshold(sig, noi, t, allowable=allowable)
    else:                                   # 'prediction' / 'variance' etc.
        k, _ = select_threshold_analytic(sig, noi, evals, t, {**opt, 'criterion': criterion})
    sv_curve, rec_curve = _analytic_recovery_curve(sig, noi, t, total_S)
    k = int(np.clip(k, 0, len(sig)))
    return {
        'basis': basis_key, 'criterion': criterion, 'family': 'lowrank',
        'best_threshold': k,
        'sv_frac': float(sv_curve[k]),
        'recovery': float(rec_curve[k]),               # analytic recovery AT the chosen K
        'eigvecs': V,                                  # reuse downstream (avoid re-eigh)
        'eigvals': evals,                              # eigenvalues for the reused basis (keep ordering)
        'curve': {'sv_frac': sv_curve, 'recovery': rec_curve},
    }


def select_threshold(cSb, cNb, ntrials, opt, device='cpu',
                     data=None, unit_means=None, has_nans=False):
    """SELECT_THRESHOLD  Choose the basis and truncation threshold for PSN.

    record = select_threshold(cSb, cNb, ntrials, opt) returns the basis and the
    number of dimensions to retain for one PSN run. The threshold within a basis
    is set by <criterion>; when more than one basis is in play ('compare') the
    basis is chosen by empirical split-half r evaluated at each basis's
    max-tradeoff threshold (requires <data>/<unit_means>).

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <cSb> - [nunits x nunits] signal covariance (GSN cSb).
    <cNb> - [nunits x nunits] noise covariance (GSN cNb).
    <ntrials> - scalar. Average number of trials per condition (handles NaNs).
    <opt> - dict. Reads <basis> and <criterion> (see psn for full descriptions):
        <basis>
          'compare'    -> build both the signal and difference bases, each at its
                          own 'max-tradeoff' threshold, and keep whichever has the
                          higher empirical split-half r evaluated at that threshold.
                          Per-candidate analytic recovery is also recorded under
                          record['diagnostics'].
          'signal'     -> signal basis with <criterion>.
          'difference' -> difference basis with <criterion>.
        <criterion> - 'max-tradeoff', 'prediction', 'variance', 'variance_eigenvalues',
          or 'wiener'. The 'wiener' criterion (also reachable through the legacy
          basis='wiener') selects the full-rank matrix Wiener filter, which is
          basis-free and has no truncation threshold: select_threshold returns a
          'fullrank' record and the caller applies that filter directly.
    <device> - 'cpu' (numpy) or a GPU device for the eigendecomposition.
    <data>, <unit_means>, <has_nans> - required for basis='compare': the trial data
        and per-unit means used to evaluate split-half r at each candidate threshold.

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    record - dict with fields:
      'family'         -> 'lowrank' (basis + threshold) or 'fullrank' (Wiener).
      'basis'          -> chosen basis name, or None for 'fullrank'.
      'criterion'      -> criterion used.
      'best_threshold' -> number of dimensions to retain (None for 'fullrank').
      'sv_frac'        -> fraction of signal variance retained at best_threshold.
      'recovery'       -> analytic recovery at best_threshold.
      'eigvecs'        -> basis vectors, reused downstream to avoid a second eigh.
      'curve'          -> {'sv_frac', 'recovery'}, the analytic recovery curve.
      'diagnostics'    -> selection metadata; for 'compare', the per-candidate
                          recovery and (caller-supplied) split-half reliability.
    """
    basis = opt.get('basis')
    criterion = opt.get('criterion', 'max-tradeoff')
    t = float(ntrials)

    # --- full-rank Wiener: a CRITERION, not a basis (basis ignored) -----------
    if criterion == 'wiener' or basis == 'wiener':
        return {'family': 'fullrank', 'basis': None, 'criterion': 'wiener',
                'best_threshold': None, 'sv_frac': None, 'recovery': None,
                'eigvecs': None, 'curve': None, 'diagnostics': {}}

    # --- compare: signal vs difference. Within each basis the threshold is the
    #     analytic max-tradeoff point; the BASIS is then chosen by empirical
    #     split-half r evaluated AT those two thresholds (higher wins). -----------
    if basis == 'compare':
        if data is None or unit_means is None:
            raise ValueError("basis='compare' requires <data> and <unit_means> "
                             "to evaluate split-half r")
        cands = {b: _select_lowrank(cSb, cNb, t, b, criterion, opt, device=device)
                 for b in ('signal', 'difference')}
        shr = {}
        for b in ('signal', 'difference'):
            V = cands[b]['eigvecs']
            K = int(cands[b]['best_threshold'])
            D = V[:, :K] @ V[:, :K].T if K > 0 else np.zeros((V.shape[0], V.shape[0]))
            shr[b] = split_half_r(data, D, unit_means, has_nans)
        winner = 'difference' if shr['difference'] > shr['signal'] else 'signal'
        rec = dict(cands[winner])
        rec['split_half_r'] = shr[winner]
        rec['diagnostics'] = {
            'mode': 'compare', 'picked_by': 'split_half_r',
            'candidates': {b: {'best_threshold': cands[b]['best_threshold'],
                               'recovery': cands[b]['recovery'],
                               'sv_frac': cands[b]['sv_frac'],
                               'split_half_r': shr[b]} for b in cands},
            # both candidate bases, so the figure reuses them instead of re-eigh-ing
            'eigvecs_by_basis': {b: cands[b]['eigvecs'] for b in cands},
        }
        return rec

    # --- explicit signal / difference -----------------------------------------
    if basis in ('signal', 'difference'):
        rec = _select_lowrank(cSb, cNb, t, basis, criterion, opt, device=device)
        rec['diagnostics'] = {'mode': 'explicit'}
        return rec

    raise ValueError(f"select_threshold: unknown basis '{basis}' "
                     f"(expected compare|signal|difference|wiener)")
