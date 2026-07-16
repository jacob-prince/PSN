function results = attach_recovery_tradeoff(results, cSb, cNb, t, data, unit_means, has_nans, orig_basis, nunits, extra_bases)
% ATTACH_RECOVERY_TRADEOFF  Compute results.recovery_tradeoff for the diagnostic figure.
%
%   results = attach_recovery_tradeoff(results, cSb, cNb, t, data, unit_means,
%   has_nans, orig_basis, nunits, extra_bases) mirrors the Python
%   attach_recovery_tradeoff (numpy path). For the chosen basis (and both
%   candidates under 'compare') it computes the analytic recovery curve
%   cumsum(signal - noise/t) and the empirical split-half reliability curve
%   (median per-unit TAvg-vs-Denoised correlation) over truncation thresholds,
%   plus the trial-average, full-rank Wiener, and chosen operating-point markers.
%
%   In hybrid mode the chosen basis also carries faint per-unit reliability
%   traces (each unit's curve + its own-K marker), stored under .units.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <results> - struct of PSN results; read for fullbasis/denoiser/svnv_after/etc.
%   and written with the new .recovery_tradeoff field.
%
% <cSb> - [N x N] signal covariance matrix from GSN.
%
% <cNb> - [N x N] noise covariance matrix from GSN.
%
% <t> - scalar, number of trials (or average if NaNs present).
%
% <data> - [nunits x nconds x ntrials] measured data (used for the split-half
%   reliability curves).
%
% <unit_means> - [nunits x 1] per-unit mean, added back after denoising.
%
% <has_nans> - logical, whether <data> contains NaNs (selects nan-aware paths).
%
% <orig_basis> - char or matrix, the requested basis ('signal', 'difference',
%   'compare', 'wiener', or a custom matrix); drives which curves are computed.
%
% <nunits> - scalar, number of units (gates whether the Wiener marker is drawn).
%
% <extra_bases> (optional) - struct with precomputed candidate eigenvector
%   matrices (.signal, .difference) to reuse instead of re-running eigh.
%   Default: [].
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <results> - the input struct with field .recovery_tradeoff added: a struct
%   holding per-basis curves (sv_frac, split_half_r, analytic_sv_frac,
%   analytic_recovery) plus .trial_average, .wiener, and .chosen marker structs.
%   Empty ([]) when total signal variance is non-positive.

    total_S = trace(cSb);
    if total_S <= 0
        results.recovery_tradeoff = [];
        return;
    end
    n = size(cSb, 1);

    [which_sig, which_dif, include_wiener] = local_policy(nunits, orig_basis);

    % Reusable bases / Wiener filter from the results struct (avoid extra eighs).
    V_signal = [];
    V_difference = [];
    D_wiener = [];
    used_basis = '';
    if isfield(results, 'opt_used') && isfield(results.opt_used, 'basis') ...
            && (ischar(results.opt_used.basis) || isstring(results.opt_used.basis))
        used_basis = char(results.opt_used.basis);
    end
    if isfield(results, 'fullbasis')
        fb = results.fullbasis;
        if ismatrix(fb) && size(fb, 1) == size(fb, 2)
            if strcmp(used_basis, 'signal'); V_signal = fb; end
            if strcmp(used_basis, 'difference'); V_difference = fb; end
        end
    end
    if strcmp(used_basis, 'wiener') && isfield(results, 'denoiser')
        D_wiener = results.denoiser';
    end

    % Caller-provided precomputed candidate bases (mirrors Python's extra_bases):
    % under 'compare' this carries both signal and difference eigvecs from
    % select_compare_basis, so the figure reuses the basis the pipeline did NOT
    % keep instead of re-running its eigendecomposition.
    if nargin >= 10 && ~isempty(extra_bases) && isstruct(extra_bases)
        if isempty(V_signal) && isfield(extra_bases, 'signal') && ~isempty(extra_bases.signal)
            V_signal = extra_bases.signal;
        end
        if isempty(V_difference) && isfield(extra_bases, 'difference') && ~isempty(extra_bases.difference)
            V_difference = extra_bases.difference;
        end
    end

    % Trial split (odd/even interleaved, matching the figure's split-half panel).
    ntrials = size(data, 3);
    A = data(:, :, 1:2:ntrials);
    B = data(:, :, 2:2:ntrials);
    if has_nans
        tavg_A = mean(A, 3, 'omitnan');
        tavg_B = mean(B, 3, 'omitnan');
    else
        tavg_A = mean(A, 3);
        tavg_B = mean(B, 3);
    end
    um = unit_means(:);

    rt = struct();

    % Per-unit faint traces (hybrid mode only): which units to trace, each unit's
    % own threshold K, group labels, and which basis carries them. Mirrors Python.
    [unit_idx, unit_K, unit_groups, chosen_basis_key] = ...
        unit_trace_setup(results, orig_basis, nunits, which_sig, which_dif);

    if which_sig
        if isempty(V_signal); V_signal = eig_desc(cSb); end
        uidx_s = []; if strcmp(chosen_basis_key, 'signal'); uidx_s = unit_idx; end
        [xs, ys, unit_ys, Ks, svf_full] = trunc_curve(V_signal, cSb, total_S, tavg_A, tavg_B, um, has_nans, uidx_s);
        [sv_a, rec_a] = analytic_recovery(V_signal, cSb, cNb, t, total_S);
        rt.signal_basis = struct('sv_frac', xs, 'split_half_r', ys, ...
                                 'analytic_sv_frac', sv_a, 'analytic_recovery', rec_a);
        if ~isempty(uidx_s)
            rt.signal_basis.units = build_unit_traces(xs, unit_ys, Ks, svf_full, unit_idx, unit_K, unit_groups, n);
        end
    end
    if which_dif
        if isempty(V_difference); V_difference = eig_desc(cSb - cNb / t); end
        uidx_d = []; if strcmp(chosen_basis_key, 'difference'); uidx_d = unit_idx; end
        [xd, yd, unit_ys, Ks, svf_full] = trunc_curve(V_difference, cSb, total_S, tavg_A, tavg_B, um, has_nans, uidx_d);
        [sv_a, rec_a] = analytic_recovery(V_difference, cSb, cNb, t, total_S);
        rt.difference_basis = struct('sv_frac', xd, 'split_half_r', yd, ...
                                     'analytic_sv_frac', sv_a, 'analytic_recovery', rec_a);
        if ~isempty(uidx_d)
            rt.difference_basis.units = build_unit_traces(xd, unit_ys, Ks, svf_full, unit_idx, unit_K, unit_groups, n);
        end
    end

    % Trial-average reference point.
    y_ta = shr_from_dn(tavg_A, tavg_B, tavg_A, tavg_B, has_nans);
    rt.trial_average = struct('sv_frac', 1.0, 'split_half_r', y_ta);

    % Full-rank Wiener reference point.
    if include_wiener
        if isempty(D_wiener)
            M = cSb + cNb / t;
            M = M + 1e-10 * trace(M) / n * eye(n);
            Dw = (M \ cSb)';                 % = solve(M, cSb)'
        else
            Dw = D_wiener;
        end
        rt.wiener = struct('sv_frac', xfrac(Dw, cSb, total_S), ...
                           'split_half_r', shr_for_D(Dw, tavg_A, tavg_B, um, has_nans));
    end

    % Chosen operating point.
    rt.chosen = [];
    if isfield(results, 'denoiser')
        D = results.denoiser';
        chosen_sv_frac = [];
        if isfield(results, 'svnv_after') && size(results.svnv_after, 2) >= 1
            chosen_sv_frac = sum(results.svnv_after(:, 1)) / total_S;
        end
        if isempty(chosen_sv_frac); chosen_sv_frac = xfrac(D, cSb, total_S); end
        rt.chosen = struct('sv_frac', chosen_sv_frac, ...
                           'split_half_r', shr_for_D(D, tavg_A, tavg_B, um, has_nans), ...
                           'label', threshold_label(results.best_threshold), ...
                           'recovery', NaN);
        % Analytic recovery at the chosen point, interpolated on the chosen basis.
        ck = '';
        if isfield(results, 'threshold_selection') && isfield(results.threshold_selection, 'basis') ...
                && ischar(results.threshold_selection.basis)
            ck = results.threshold_selection.basis;
        end
        if isempty(ck) && (ischar(orig_basis) || isstring(orig_basis)); ck = char(orig_basis); end
        cb = [];
        if strcmp(ck, 'signal') && isfield(rt, 'signal_basis'); cb = rt.signal_basis;
        elseif strcmp(ck, 'difference') && isfield(rt, 'difference_basis'); cb = rt.difference_basis;
        elseif isfield(rt, 'signal_basis'); cb = rt.signal_basis;
        elseif isfield(rt, 'difference_basis'); cb = rt.difference_basis;
        end
        if ~isempty(cb)
            rt.chosen.recovery = np_interp(chosen_sv_frac, cb.analytic_sv_frac, cb.analytic_recovery);
        end
        rt.chosen.basis = ck;   % basis the operating point was chosen on (for the inset)
    end

    % Criterion drives the max-tradeoff inset in the figure (only drawn then).
    rt.criterion = '';
    if isfield(results, 'opt_used') && isfield(results.opt_used, 'criterion') ...
            && (ischar(results.opt_used.criterion) || isstring(results.opt_used.criterion))
        rt.criterion = char(results.opt_used.criterion);
    end

    results.recovery_tradeoff = rt;
end


% ------------------------------------------------------------------------- %
function [which_sig, which_dif, include_wiener] = local_policy(nunits, basis)
    include_wiener = (nunits <= 1000);
    which_sig = false;
    which_dif = false;
    if ischar(basis) || isstring(basis)
        switch char(basis)
            case 'compare';    which_sig = true; which_dif = true;
            case 'signal';     which_sig = true;
            case 'difference'; which_dif = true;
            case 'wiener';     include_wiener = true;
        end
    end
end


function V = eig_desc(A)
    [~, V] = eigh_descending_sym((A + A') / 2, false);
end


function [sv, rec] = analytic_recovery(V, cSb, cNb, t, total_S)
    sig = sum((cSb * V) .* V, 1)';
    noi = sum((cNb * V) .* V, 1)';
    sv = [0; cumsum(sig)] / total_S;
    rec = [0; cumsum(sig - noi / t)];
end


function [svf_sub, ys, unit_ys, Ks, svf_full] = trunc_curve(V, cSb, total_S, tavg_A, tavg_B, um, has_nans, unit_idx)
    if nargin < 8, unit_idx = []; end
    n = size(V, 2);
    sig = sum((cSb * V) .* V, 1)';
    svf = [0; cumsum(sig)] / total_S;             % length n+1
    coefA = V' * (tavg_A - um);
    coefB = V' * (tavg_B - um);
    if n <= 256
        Ks = (0:n)';
    else
        Ks = unique(round(linspace(0, n, 256)))';
    end
    R_A = zeros(size(coefA));
    R_B = zeros(size(coefB));
    ys = zeros(numel(Ks), 1);
    if ~isempty(unit_idx)
        unit_ys = nan(numel(unit_idx), numel(Ks));   % per-unit reliability (hybrid traces)
    else
        unit_ys = [];
    end
    prev = 0;
    for i = 1:numel(Ks)
        K = Ks(i);
        if K > prev
            R_A = R_A + V(:, prev+1:K) * coefA(prev+1:K, :);
            R_B = R_B + V(:, prev+1:K) * coefB(prev+1:K, :);
            prev = K;
        end
        both = 0.5 * (row_corr(tavg_A, R_B + um, has_nans) + row_corr(R_A + um, tavg_B, has_nans));
        ys(i) = nanmedian_(both);
        if ~isempty(unit_idx)
            unit_ys(:, i) = both(unit_idx);
        end
    end
    svf_sub = svf(Ks + 1);
    svf_full = svf;
end


function [unit_idx, unit_K, unit_groups, chosen_basis_key] = unit_trace_setup(results, orig_basis, nunits, which_sig, which_dif)
% Decide which units get faint per-unit recovery traces (hybrid mode only),
% their per-unit thresholds, group labels, and which basis carries them.
    unit_idx = []; unit_K = []; unit_groups = []; chosen_basis_key = '';
    if ~isfield(results, 'opt_used'); return; end
    opt_used = results.opt_used;
    if ~isfield(opt_used, 'threshold_method') || ~strcmp(opt_used.threshold_method, 'hybrid'); return; end
    if ~isfield(results, 'best_threshold'); return; end
    bt = results.best_threshold(:);
    if numel(bt) ~= nunits || nunits <= 1; return; end
    unit_K = double(bt);
    if isfield(opt_used, 'unit_groups'); unit_groups = opt_used.unit_groups; end
    if isfield(results, 'threshold_selection') && isfield(results.threshold_selection, 'basis') ...
            && ischar(results.threshold_selection.basis)
        chosen_basis_key = results.threshold_selection.basis;
    elseif (ischar(orig_basis) || isstring(orig_basis)) && any(strcmp(char(orig_basis), {'signal', 'difference'}))
        chosen_basis_key = char(orig_basis);
    end
    drawn = (strcmp(chosen_basis_key, 'signal') && which_sig) || ...
            (strcmp(chosen_basis_key, 'difference') && which_dif);
    if ~drawn
        unit_idx = []; unit_K = []; return;        % chosen basis not drawn; skip
    end
    if nunits > 500                                 % subsample to match the figure
        rng(42); unit_idx = sort(randperm(nunits, 100));
    else
        unit_idx = 1:nunits;
    end
end


function u = build_unit_traces(svf_sub, unit_ys, Ks, svf_full, unit_idx, unit_K, unit_groups, n)
% Package per-unit reliability curves + each unit's own-K marker for the figure.
    nsub = numel(unit_idx);
    mx = zeros(nsub, 1); my = zeros(nsub, 1);
    for j = 1:nsub
        ku = min(max(round(unit_K(unit_idx(j))), 0), n);
        mx(j) = svf_full(ku + 1);
        my(j) = np_interp(ku, Ks, unit_ys(j, :)');
    end
    if ~isempty(unit_groups)
        grp = unit_groups(unit_idx); grp = grp(:);
    else
        grp = [];
    end
    u = struct('sv_frac', svf_sub(:), 'split_half_r', unit_ys, 'unit_idx', unit_idx(:), ...
               'markers', struct('sv_frac', mx, 'split_half_r', my, 'group', grp));
end


function out = row_corr(X, Y, has_nans)
    m = size(X, 1);
    out = nan(m, 1);
    if has_nans
        for u = 1:m
            mask = ~(isnan(X(u, :)) | isnan(Y(u, :)));
            if sum(mask) > 1 && std(X(u, mask)) > 0 && std(Y(u, mask)) > 0
                c = corrcoef(X(u, mask), Y(u, mask));
                out(u) = c(1, 2);
            end
        end
    else
        Xc = X - mean(X, 2);
        Yc = Y - mean(Y, 2);
        den = sqrt(sum(Xc.^2, 2) .* sum(Yc.^2, 2));
        nz = den > 0;
        out(nz) = sum(Xc(nz, :) .* Yc(nz, :), 2) ./ den(nz);
    end
end


function v = shr_from_dn(dn_A, dn_B, tavg_A, tavg_B, has_nans)
    both = 0.5 * (row_corr(tavg_A, dn_B, has_nans) + row_corr(dn_A, tavg_B, has_nans));
    v = nanmedian_(both);
end


function v = shr_for_D(D, tavg_A, tavg_B, um, has_nans)
    v = shr_from_dn(D * (tavg_A - um) + um, D * (tavg_B - um) + um, tavg_A, tavg_B, has_nans);
end


function v = xfrac(D, cSb, total_S)
    v = sum(sum((D * cSb) .* D)) / total_S;
end


function v = nanmedian_(x)
    x = x(~isnan(x));
    if isempty(x); v = NaN; else; v = median(x); end
end


function v = np_interp(xq, xp, fp)
% Piecewise-linear interpolation matching numpy.interp for non-decreasing xp
% (clamps at the ends; tolerates duplicate xp values, unlike interp1).
    xp = xp(:); fp = fp(:);
    if xq <= xp(1); v = fp(1); return; end
    if xq >= xp(end); v = fp(end); return; end
    idx = find(xp <= xq, 1, 'last');
    x0 = xp(idx); x1 = xp(idx + 1); y0 = fp(idx); y1 = fp(idx + 1);
    if x1 == x0
        v = y1;
    else
        v = y0 + (y1 - y0) * (xq - x0) / (x1 - x0);
    end
end


function s = threshold_label(bt)
    bt = double(bt);
    if isscalar(bt)
        if abs(bt - round(bt)) < 1e-6
            s = sprintf('K=%.0f', bt);
        else
            s = sprintf('K=%.1f', bt);
        end
    else
        s = sprintf('K~%.1f', mean(bt(:)));
    end
end
