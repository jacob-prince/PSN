"""Recovery / bias-variance tradeoff curve - COMPUTE (no plotting).

For each requested basis we trace the MEDIAN per-unit 'TAvg vs Denoised'
split-half reliability as the basis is truncated to K dimensions, plus
reference points (trial-average, full-rank Wiener) and the operating point of
the denoiser PSN actually applied. This is the data behind the diagnostic
"Split-half reliability vs. signal retained" panel.

This lives in psn's compute layer (NOT the plotting layer) so it ALWAYS runs
as part of psn() - independent of whether a figure is generated. The figure
just reads results['recovery_tradeoff'] and draws it.

Cost: each curve is O(n^2 * nconds) (incremental rank-K reconstruction on a
capped K-grid); the dots are O(n^2 * nconds) reductions plus, for Wiener, an
O(n^3) solve. All of it runs in torch when importable (≈2x faster on CPU,
GPU when device is cuda/mps), numpy otherwise / for NaN data.
"""

import numpy as np

from psn._device import _torch_available, is_cpu


def recovery_curve_policy(nunits, basis):
    """Decide which recovery curves and reference points to compute for a basis.

    Only the requested basis's curve is drawn, except 'compare' which shows both
    signal and difference for the comparison.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <nunits> - scalar. Number of units; gates the Wiener point (included when
        nunits <= 1000, where the O(n^3) solve is cheap).

    <basis> - str or matrix. The requested basis.

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <which_curves> - set[str]. Subset of {'signal','difference'} to trace:
        'compare' -> both; 'signal'/'difference' -> that one; 'wiener'/other -> {}.

    <include_wiener> - bool. Whether to add the full-rank Wiener reference point
        (True when nunits <= 1000 or basis == 'wiener').
    """
    include_wiener = (nunits <= 1000) or (basis == 'wiener')
    if isinstance(basis, str):
        if basis == 'compare':
            return {'signal', 'difference'}, include_wiener
        if basis == 'signal':
            return {'signal'}, include_wiener
        if basis == 'difference':
            return {'difference'}, include_wiener
        if basis == 'wiener':
            return set(), include_wiener
    return set(), include_wiener


def reusable_recovery_inputs(results):
    """Pull already-computed eigenbases / Wiener filter out of a results dict.

    Lets the curve computation skip the two O(n^3) eighs (and the O(n^3) Wiener
    solve) PSN/GSN may already have done. Sources, in priority order:
    gsn_result's cached eigvecs_signal/eigvecs_difference (GSN eigenbasis-returns
    feature), results['fullbasis'] for the basis PSN used, then results['denoiser']
    when PSN ran the full-rank Wiener filter.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <results> - dict of PSN results. Reads 'gsn_result', 'fullbasis',
        'opt_used'/'auto_basis_selected' (the basis used), and 'denoiser'.

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <V_signal> - [n x n] signal eigenbasis, or None if no reusable copy exists.

    <V_difference> - [n x n] difference eigenbasis, or None.

    <D_wiener> - [n x n] full-rank Wiener filter (denoiser.T), or None.
    """
    gsn = results.get('gsn_result', {}) or {}
    V_signal = gsn.get('eigvecs_signal')
    V_difference = gsn.get('eigvecs_difference')

    chosen = results.get('auto_basis_selected') \
        or (results.get('opt_used') or {}).get('basis')
    fb = results.get('fullbasis')
    if isinstance(chosen, str) and fb is not None:
        fb = np.asarray(fb)
        if fb.ndim == 2 and fb.shape[0] == fb.shape[1]:
            if chosen == 'signal' and V_signal is None:
                V_signal = fb
            elif chosen == 'difference' and V_difference is None:
                V_difference = fb

    D_wiener = None
    used_basis = (results.get('opt_used') or {}).get('basis')
    if isinstance(used_basis, str) and used_basis == 'wiener' \
            and results.get('denoiser') is not None:
        D_wiener = np.asarray(results['denoiser']).T

    return (np.asarray(V_signal) if V_signal is not None else None,
            np.asarray(V_difference) if V_difference is not None else None,
            D_wiener)


def _threshold_label(best_threshold):
    if best_threshold is None:
        return 'chosen'
    bt = np.asarray(best_threshold, dtype=float)
    if bt.size == 1:
        kv = float(bt)
        return f'K={kv:.0f}' if abs(kv - round(kv)) < 1e-6 else f'K={kv:.1f}'
    return f'K≈{bt.mean():.1f}'


def compute_recovery_tradeoff(cSb, cNb, t, data, unit_means, has_nans, *,
                              which=('signal', 'difference'),
                              include_wiener=True, include_trial_avg=True,
                              denoiser=None, best_threshold=None,
                              chosen_sv_frac=None,
                              V_signal=None, V_difference=None, D_wiener=None,
                              unit_idx=None, unit_K=None, unit_groups=None,
                              chosen_basis_key=None, device='cpu',
                              skip_split_half=False):
    """Compute the recovery-tradeoff data dict (no matplotlib).

    Traces, over truncation K, the analytic recovery curve and the empirical
    split-half reliability for the requested bases, plus the trial-average,
    full-rank Wiener, and chosen operating-point markers. Reuses any supplied
    eigenbases / Wiener filter to avoid redundant O(n^3) work.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <cSb>, <cNb> - [n x n] GSN signal and noise covariances.

    <t> - scalar. Average number of trials per condition (handles NaNs).

    <data> - [nunits x nconds x ntrials]. Used for the split-half curves.

    <unit_means> - [nunits]. Per-unit mean, re-added after denoising.

    <has_nans> - bool. Whether <data> has NaNs (selects nan-aware reductions).

    Keyword-only:

    <which> - subset of {'signal','difference'} to trace. Default: both.

    <include_wiener>, <include_trial_avg> - bool. Toggle those reference points.

    <denoiser> - [n x n] or None. When given, adds the 'chosen' operating point.

    <best_threshold> - scalar/array or None. Labels the chosen point.

    <chosen_sv_frac> - float or None. Precomputed x (signal-var fraction) for the
        chosen point, e.g. sum(svnv_after[:,0]) / tr(cSb); skips an O(n^3) trace.

    <V_signal>, <V_difference>, <D_wiener> - precomputed bases / Wiener filter to
        reuse instead of recomputing; None recomputes as needed.

    <unit_idx>, <unit_K>, <unit_groups>, <chosen_basis_key> - per-unit faint-trace
        data for hybrid mode (unit indices, per-unit K, group labels, basis key).

    <device> - 'cpu' (numpy) or a GPU device for the eigendecomposition.

    <skip_split_half> - bool. When True, compute ONLY the analytic recovery
        curve(s) and skip every empirical split-half computation: the per-K
        split-half truncation curves (the O(n^2 * nconds * nKs) per-unit
        reliability loop that dominates runtime, especially on NaN data where
        it falls to the numpy per-unit path), plus the trial-average, Wiener,
        and chosen-point split-half markers. The analytic recovery trace, its
        prediction peak, the max-tradeoff chord/inset, and the chosen point's
        analytic marker are all preserved. Default False.

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <rt> - dict of curves and marker points ('signal_basis'/'difference_basis',
        'trial_average', 'wiener', 'chosen'), or None when there is no signal
        variance (tr(cSb) <= 0).
    """
    which = set(which)
    n = cSb.shape[0]
    total_S = float(np.trace(cSb))
    if total_S <= 0:
        return None

    def _eig_desc(A):
        A = (A + A.T) / 2.0
        w, V = np.linalg.eigh(A)
        return V[:, np.argsort(w)[::-1]]

    ntrials = data.shape[2]
    A = data[:, :, np.arange(0, ntrials, 2)]
    B = data[:, :, np.arange(1, ntrials, 2)]
    tavg_A = np.nanmean(A, axis=2) if has_nans else np.mean(A, axis=2)
    tavg_B = np.nanmean(B, axis=2) if has_nans else np.mean(B, axis=2)
    um = unit_means[:, np.newaxis]

    if not has_nans:
        tAc = tavg_A - tavg_A.mean(1, keepdims=True)
        tBc = tavg_B - tavg_B.mean(1, keepdims=True)
        nA = np.sqrt((tAc ** 2).sum(1))
        nB = np.sqrt((tBc ** 2).sum(1))

    use_torch = _torch_available() and not has_nans
    if use_torch:
        import torch
        if isinstance(device, torch.device):
            _tdev = device
        elif is_cpu(device):
            _tdev = torch.device('cpu')
        else:
            _tdev = torch.device(str(device))
        _tdt = torch.float32

        def _to_t(a):
            return torch.as_tensor(np.asarray(a), dtype=_tdt, device=_tdev)

        cSb_t = _to_t(cSb)
        um_t = _to_t(um)
        tavg_A_t = _to_t(tavg_A)
        tavg_B_t = _to_t(tavg_B)
        aA_t = tavg_A_t - um_t
        aB_t = tavg_B_t - um_t
        tAc_t = tavg_A_t - tavg_A_t.mean(1, keepdim=True)
        tBc_t = tavg_B_t - tavg_B_t.mean(1, keepdim=True)
        nA_t = torch.sqrt((tAc_t ** 2).sum(1))
        nB_t = torch.sqrt((tBc_t ** 2).sum(1))

        def _t_nanmedian(x):
            # Averaged median to match np.nanmedian. torch.nanmedian returns the
            # LOWER of the two middle values for even counts, which would make the
            # torch path's markers/curve disagree with the numpy path (and ax13).
            v = x[~torch.isnan(x)]
            if v.numel() == 0:
                return float('nan')
            s, _ = torch.sort(v)
            m = s.numel()
            if m % 2 == 1:
                return float(s[m // 2])
            return float(0.5 * (s[m // 2 - 1] + s[m // 2]))

        def _both_t(R_A, R_B):
            rBc = R_B - R_B.mean(1, keepdim=True)
            rAc = R_A - R_A.mean(1, keepdim=True)
            dB = nA_t * torch.sqrt((rBc ** 2).sum(1))
            dA = nB_t * torch.sqrt((rAc ** 2).sum(1))
            c1 = (tAc_t * R_B).sum(1) / dB
            c2 = (tBc_t * R_A).sum(1) / dA
            nan = torch.tensor(float('nan'), device=_tdev, dtype=_tdt)
            c1 = torch.where(dB > 0, c1, nan)
            c2 = torch.where(dA > 0, c2, nan)
            return 0.5 * (c1 + c2)            # per-unit reliabilities

        def _shr_fast_t(R_A, R_B):
            return _t_nanmedian(_both_t(R_A, R_B))

        def _trunc_curve_t(V, unit_idx=None):
            V_t = _to_t(V)
            coefA = V_t.T @ aA_t
            coefB = V_t.T @ aB_t
            s = (torch.matmul(cSb_t, V_t) * V_t).sum(0)
            svf = torch.cat([torch.zeros(1, device=_tdev, dtype=_tdt),
                             torch.cumsum(s, 0)]) / total_S
            Ks = (np.arange(n + 1) if n <= 256
                  else np.unique(np.linspace(0, n, 256).round().astype(int)))
            R_A = torch.zeros_like(coefA)
            R_B = torch.zeros_like(coefB)
            ys_t = torch.full((len(Ks),), float('nan'), device=_tdev, dtype=_tdt)
            uidx_t = (torch.as_tensor(np.asarray(unit_idx), device=_tdev, dtype=torch.long)
                      if unit_idx is not None else None)
            unit_ys_t = (torch.full((len(unit_idx), len(Ks)), float('nan'),
                                    device=_tdev, dtype=_tdt)
                         if unit_idx is not None else None)
            prev = 0
            for i, K in enumerate(Ks):
                K = int(K)
                if K > prev:
                    R_A += V_t[:, prev:K] @ coefA[prev:K, :]
                    R_B += V_t[:, prev:K] @ coefB[prev:K, :]
                    prev = K
                both = _both_t(R_A, R_B)
                ys_t[i] = _t_nanmedian(both)
                if uidx_t is not None:
                    unit_ys_t[:, i] = both.index_select(0, uidx_t)
            Ks_t = torch.as_tensor(Ks, device=_tdev, dtype=torch.long)
            svf_sub = svf.index_select(0, Ks_t).cpu().numpy()
            if unit_idx is None:
                return svf_sub, ys_t.cpu().numpy()
            return (svf_sub, ys_t.cpu().numpy(), unit_ys_t.cpu().numpy(),
                    svf.cpu().numpy(), Ks)

    def _row_corr(X, Y):
        if has_nans:
            out = np.full(X.shape[0], np.nan)
            for u in range(X.shape[0]):
                m = ~(np.isnan(X[u]) | np.isnan(Y[u]))
                if m.sum() > 1 and np.std(X[u, m]) > 0 and np.std(Y[u, m]) > 0:
                    out[u] = np.corrcoef(X[u, m], Y[u, m])[0, 1]
            return out
        Xc = X - X.mean(1, keepdims=True)
        Yc = Y - Y.mean(1, keepdims=True)
        den = np.sqrt((Xc ** 2).sum(1) * (Yc ** 2).sum(1))
        out = np.full(X.shape[0], np.nan)
        nz = den > 0
        out[nz] = (Xc * Yc).sum(1)[nz] / den[nz]
        return out

    def _shr_from_dn(dn_A, dn_B):
        both = 0.5 * (_row_corr(tavg_A, dn_B) + _row_corr(dn_A, tavg_B))
        return float(np.nanmedian(both)) if np.any(~np.isnan(both)) else np.nan

    def _shr_for_D(D):
        if use_torch:
            D_t = _to_t(D)
            return float(_shr_fast_t(D_t @ aA_t, D_t @ aB_t))
        return _shr_from_dn(D @ (tavg_A - um) + um, D @ (tavg_B - um) + um)

    def _xfrac(D):
        if use_torch:
            D_t = _to_t(D)
            return float(((D_t @ cSb_t) * D_t).sum() / total_S)
        return float(np.sum((D @ cSb) * D) / total_S)

    def _both(R_A, R_B):
        rBc = R_B - R_B.mean(1, keepdims=True)
        rAc = R_A - R_A.mean(1, keepdims=True)
        dB = nA * np.sqrt((rBc ** 2).sum(1))
        dA = nB * np.sqrt((rAc ** 2).sum(1))
        c1 = np.full(R_B.shape[0], np.nan)
        c2 = np.full(R_A.shape[0], np.nan)
        m1 = dB > 0
        m2 = dA > 0
        c1[m1] = (tAc * R_B).sum(1)[m1] / dB[m1]
        c2[m2] = (tBc * R_A).sum(1)[m2] / dA[m2]
        return 0.5 * (c1 + c2)               # per-unit reliabilities

    def _med(both):
        return float(np.nanmedian(both)) if np.any(~np.isnan(both)) else np.nan

    def _trunc_curve(V, unit_idx=None):
        if use_torch:
            return _trunc_curve_t(V, unit_idx)
        s = np.sum((cSb @ V) * V, axis=0)
        svf = np.concatenate([[0.0], np.cumsum(s)]) / total_S
        coefA = V.T @ (tavg_A - um)
        coefB = V.T @ (tavg_B - um)
        Ks = (np.arange(n + 1) if n <= 256
              else np.unique(np.linspace(0, n, 256).round().astype(int)))
        R_A = np.zeros_like(coefA)
        R_B = np.zeros_like(coefB)
        ys = np.empty(Ks.size)
        unit_ys = (np.full((len(unit_idx), Ks.size), np.nan)
                   if unit_idx is not None else None)
        prev = 0
        for i, K in enumerate(Ks):
            if K > prev:
                R_A += V[:, prev:K] @ coefA[prev:K, :]
                R_B += V[:, prev:K] @ coefB[prev:K, :]
                prev = K
            if has_nans:
                both = 0.5 * (_row_corr(tavg_A, R_B + um) + _row_corr(R_A + um, tavg_B))
            else:
                both = _both(R_A, R_B)
            ys[i] = _med(both)
            if unit_ys is not None:
                unit_ys[:, i] = both[unit_idx]
        if unit_idx is None:
            return svf[Ks], ys
        return svf[Ks], ys, unit_ys, svf, Ks

    out = {
        'description': ('Split-half reliability (median per-unit TAvg-vs-Denoised '
                        'correlation) vs. fraction of signal variance retained, '
                        'as the basis is truncated to K dimensions.'),
        'xlabel': 'frac. signal var. retained',
        'ylabel': 'median split-half r (TAvg vs Denoised)',
    }

    # Per-unit faint traces are computed for the CHOSEN basis only, and only
    # when PSN used unit-specific thresholds (each unit has its own K).
    want_units = (unit_idx is not None and unit_K is not None
                  and chosen_basis_key in which and not skip_split_half)

    def _unit_traces(svf_sub, unit_ys, svf_full, Ks, basis_key):
        mx, my = [], []
        for j, u in enumerate(unit_idx):
            ku = int(np.clip(unit_K[u], 0, n))
            mx.append(float(svf_full[ku]))
            my.append(float(np.interp(ku, Ks, unit_ys[j])))
        grp = (np.asarray(unit_groups)[np.asarray(unit_idx)]
               if unit_groups is not None else None)
        return {
            'basis': basis_key,
            'sv_frac': svf_sub,                 # shared x-grid
            'split_half_r': unit_ys,            # (n_sub, nKs)
            'unit_idx': np.asarray(unit_idx),
            'markers': {'sv_frac': np.array(mx), 'split_half_r': np.array(my),
                        'group': grp},
        }

    def _analytic_recovery(V):
        # ANALYTIC recovery curve (no CV): cumsum(signal - noise/t), the SAME raw
        # cumulative objective the objective-function panel shows (so the two
        # panels peak at the same level). x = fraction of signal variance retained.
        sig = np.sum((cSb @ V) * V, axis=0)
        noi = np.sum((cNb @ V) * V, axis=0)
        sv = np.concatenate([[0.0], np.cumsum(sig)]) / total_S
        rec = np.concatenate([[0.0], np.cumsum(sig - noi / t)])
        return sv, rec

    if 'signal' in which:
        Vsig = np.asarray(V_signal) if V_signal is not None else _eig_desc(cSb)
        if skip_split_half:
            xs = ys = None
        elif want_units and chosen_basis_key == 'signal':
            xs, ys, u_ys, svf_full, Ks = _trunc_curve(Vsig, unit_idx)
            out['unit_traces'] = _unit_traces(xs, u_ys, svf_full, Ks, 'signal')
        else:
            xs, ys = _trunc_curve(Vsig)
        sv_a, rec_a = _analytic_recovery(Vsig)
        out['signal_basis'] = {'sv_frac': xs, 'split_half_r': ys,
                               'analytic_sv_frac': sv_a, 'analytic_recovery': rec_a}

    if 'difference' in which:
        Vdif = (np.asarray(V_difference) if V_difference is not None
                else _eig_desc(cSb - cNb / t))
        if skip_split_half:
            xd = yd = None
        elif want_units and chosen_basis_key == 'difference':
            xd, yd, u_ys, svf_full, Ks = _trunc_curve(Vdif, unit_idx)
            out['unit_traces'] = _unit_traces(xd, u_ys, svf_full, Ks, 'difference')
        else:
            xd, yd = _trunc_curve(Vdif)
        sv_a, rec_a = _analytic_recovery(Vdif)
        out['difference_basis'] = {'sv_frac': xd, 'split_half_r': yd,
                                   'analytic_sv_frac': sv_a, 'analytic_recovery': rec_a}

    if include_trial_avg and not skip_split_half:
        y_ta = (float(_shr_fast_t(aA_t, aB_t)) if use_torch
                else _shr_from_dn(tavg_A, tavg_B))
        out['trial_average'] = {'sv_frac': 1.0, 'split_half_r': y_ta}

    if include_wiener and not skip_split_half:
        if D_wiener is not None:
            Dw = np.asarray(D_wiener)
        elif use_torch:
            M = cSb_t + _to_t(cNb) / float(t)
            M = M + (1e-10 * float(torch.trace(M)) / n) * torch.eye(n, device=_tdev, dtype=_tdt)
            try:
                Dw = torch.linalg.solve(M, cSb_t).T.cpu().numpy()
            except Exception:
                # M is PSD but can be numerically singular: cSb is rank-deficient
                # (its null space is floored to the 1e-10 ridge) and cNb/t is ~0 in
                # that same null space, so M's smallest eigenvalue sits at the floor
                # and torch.linalg.solve raises. pinv gives the minimum-norm Wiener
                # solution; cSb shares the null space, so the filter stays bounded.
                # Same operand order as solve, so the convention is identical.
                Dw = (torch.linalg.pinv(M) @ cSb_t).T.cpu().numpy()
        else:
            M = cSb + cNb / t
            M = M + 1e-10 * np.trace(M) / n * np.eye(n)
            try:
                Dw = np.linalg.solve(M, cSb).T
            except np.linalg.LinAlgError:
                Dw = (np.linalg.pinv(M) @ cSb).T
        out['wiener'] = {'sv_frac': _xfrac(Dw), 'split_half_r': _shr_for_D(Dw)}

    out['chosen'] = None
    if denoiser is not None:
        D = np.asarray(denoiser).T
        x_ch = float(chosen_sv_frac) if chosen_sv_frac is not None else _xfrac(D)
        shr = None if skip_split_half else _shr_for_D(D)
        out['chosen'] = {'sv_frac': x_ch, 'split_half_r': shr,
                         'label': _threshold_label(best_threshold)}

    return out


def attach_recovery_tradeoff(results, cSb, cNb, t, data, unit_means, has_nans,
                             orig_basis, nunits, device='cpu', extra_bases=None,
                             skip_split_half=False):
    """Compute the recovery-tradeoff data and store it on results.

    Applies the size+basis policy, reuses any already-computed bases / Wiener
    filter, then calls compute_recovery_tradeoff and writes the result to
    results['recovery_tradeoff']. Called by psn() on every run so the data is
    present regardless of wantfig; the figure just reads it.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <results> - dict of PSN results. Read for fullbasis/denoiser/svnv_after/
        opt_used/etc.; the 'recovery_tradeoff' field is written in place.

    <cSb>, <cNb> - [n x n] GSN signal and noise covariances.

    <t> - scalar. Average number of trials per condition (handles NaNs).

    <data> - [nunits x nconds x ntrials]. Used for the split-half curves.

    <unit_means> - [nunits]. Per-unit mean.

    <has_nans> - bool. Whether <data> has NaNs.

    <orig_basis> - str or matrix. The requested basis; drives which curves run.

    <nunits> - scalar. Gates the Wiener reference point.

    <device> - 'cpu' (numpy) or a GPU device for any eigendecomposition.

    <extra_bases> - optional {'signal': V, 'difference': V} to reuse bases already
        computed elsewhere (e.g. both 'compare' candidates). Default: None.

    <skip_split_half> - bool. Forwarded to compute_recovery_tradeoff: keep the
        analytic recovery curve(s) but skip all empirical split-half computation
        (the runtime-dominating per-unit reliability loop). Default: False.

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <results> - the same dict, with results['recovery_tradeoff'] set to the data
        dict (or left unset when there is no signal variance).
    """
    which, include_wiener = recovery_curve_policy(nunits, orig_basis)
    V_signal, V_difference, D_wiener = reusable_recovery_inputs(results)
    if extra_bases:
        V_signal = extra_bases.get('signal', V_signal)
        V_difference = extra_bases.get('difference', V_difference)

    # Chosen operating point's x (fraction of signal var retained) = tr(D cSb D^T)
    # / tr(cSb). diag(D cSb D^T) is already in results['svnv_after'][:, 0], so
    # reuse its sum and skip an O(n^3) trace.
    chosen_sv_frac = None
    sv = results.get('svnv_after')
    total_S = float(np.trace(cSb))
    if sv is not None and total_S > 0:
        sv = np.asarray(sv)
        if sv.ndim == 2 and sv.shape[1] >= 1:
            chosen_sv_frac = float(np.sum(sv[:, 0]) / total_S)

    # Per-unit faint traces when PSN used unit-specific thresholds (each unit has
    # its own K). The basis those thresholds live on is the one PSN used.
    unit_idx = unit_K = unit_groups = chosen_basis_key = None
    bt = results.get('best_threshold')
    opt_used = results.get('opt_used') or {}
    tmethod = opt_used.get('threshold_method')
    bt_arr = np.atleast_1d(np.asarray(bt)) if bt is not None else None
    if (tmethod == 'hybrid' and bt_arr is not None
            and bt_arr.size == nunits and bt_arr.size > 1):
        unit_K = bt_arr.astype(float)
        unit_groups = opt_used.get('unit_groups')
        chosen_basis_key = (results.get('threshold_selection') or {}).get('basis')
        if chosen_basis_key is None and isinstance(orig_basis, str) \
                and orig_basis in ('signal', 'difference'):
            chosen_basis_key = orig_basis
        if chosen_basis_key in which:
            # Subsample units (match the figure's convention: 100 when > 500).
            if nunits > 500:
                rng = np.random.RandomState(42)
                unit_idx = np.sort(rng.choice(nunits, 100, replace=False))
            else:
                unit_idx = np.arange(nunits)
        else:
            unit_K = None  # chosen basis not among computed curves; skip traces

    rec = compute_recovery_tradeoff(
        cSb, cNb, t, data, unit_means, has_nans,
        which=which, include_wiener=include_wiener, include_trial_avg=True,
        denoiser=results.get('denoiser'),
        best_threshold=results.get('best_threshold'),
        chosen_sv_frac=chosen_sv_frac,
        V_signal=V_signal, V_difference=V_difference, D_wiener=D_wiener,
        unit_idx=unit_idx, unit_K=unit_K, unit_groups=unit_groups,
        chosen_basis_key=chosen_basis_key, device=device,
        skip_split_half=skip_split_half)
    # Analytic recovery at the chosen operating point (for the figure's star on
    # the analytic trajectory), interpolated on the chosen basis's curve.
    if rec is not None and rec.get('chosen') is not None:
        cb = (results.get('threshold_selection') or {}).get('basis')
        if cb is None and isinstance(orig_basis, str) and orig_basis in ('signal', 'difference'):
            cb = orig_basis
        bdat = rec.get(f'{cb}_basis') if cb in ('signal', 'difference') else None
        if (bdat is not None and bdat.get('analytic_recovery') is not None
                and rec['chosen'].get('sv_frac') is not None):
            rec['chosen']['recovery'] = float(np.interp(
                rec['chosen']['sv_frac'],
                np.asarray(bdat['analytic_sv_frac'], float),
                np.asarray(bdat['analytic_recovery'], float)))
        rec['chosen']['basis'] = cb   # basis the operating point was chosen on (inset)

    if rec is not None:
        # Criterion drives the max-tradeoff inset in the figure (only drawn then).
        rec['criterion'] = opt_used.get('criterion')
        results['recovery_tradeoff'] = rec
    return results
