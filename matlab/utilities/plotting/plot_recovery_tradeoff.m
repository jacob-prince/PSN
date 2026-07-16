function plot_recovery_tradeoff(ax, results)
% PLOT_RECOVERY_TRADEOFF  Draw the recovery / bias-variance tradeoff panel.
%
%   plot_recovery_tradeoff(ax, results) reads results.recovery_tradeoff (built
%   in psn() by attach_recovery_tradeoff) and renders, on twin y-axes: LEFT =
%   analytic recovery (solid, the quantity used to pick the threshold), RIGHT =
%   empirical split-half reliability (dashed, validation). The x-axis uses the
%   same inverse-log warp as Python (linear on [0, 0.9]; the [0.9, 1] tail
%   expanded so the crowded approach to 1.0 fans out). Mirrors the Python
%   _plot_recovery_tradeoff panel.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <ax> - axes handle to draw the panel into.
%
% <results> - struct of PSN results; uses results.recovery_tradeoff (the curves
%   and marker structs produced by attach_recovery_tradeoff). If that field is
%   missing or empty, a "no recovery data" placeholder is drawn.
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% (none) - draws into <ax> as a side effect.

    title(ax, {'Recovery vs. signal retained', ...
               '(solid = analytic recovery, dashed = split-half r)'});

    if ~isfield(results, 'recovery_tradeoff') || isempty(results.recovery_tradeoff)
        text(ax, 0.5, 0.5, 'no recovery data', 'Units', 'normalized', ...
             'HorizontalAlignment', 'center');
        return;
    end
    rec = results.recovery_tradeoff;

    % Max-tradeoff carries the upper-right inset. local_lim then sets the top to 4x
    % the peak-to-bottom range so the data peak lands in the lower quarter and the top
    % three-quarters is guaranteed clear for the inset, whatever the data. The legend
    % sits top-left, so no bottom whitespace strip is needed here.
    is_mt = isfield(rec, 'criterion') && strcmp(rec.criterion, 'max-tradeoff');
    bot_margin = 0;

    % Inverse-log x transform parameters (split=0.9, expand=0.4, K=99).
    a = 0.9; W = 0.4; K = 99; lk = log10(1 + K); aW = a + W;
    slope = W * K / ((1 - a) * log(10) * lk);
    fx = @(x) invlog_fwd(x, a, W, K, lk, slope, aW);

    % {key, analytic color (solid), split-half color (dashed), name}. Analytic keeps
    % the basis identity color (blue = signal, green = difference) so it matches the
    % fig6 objective trace. The signal split-half is GOLD, matching the gold "TAvg vs
    % Denoised" dots in the split-half reliability panel (fig14); the difference
    % split-half is purple so the two dashed curves stay distinguishable in compare mode.
    GOLD = [1 0.84 0];
    bases = {'signal_basis',     [0.122 0.467 0.706], GOLD,                 'signal'; ...
             'difference_basis', [0.173 0.627 0.173], [0.580 0.404 0.741], 'difference'};

    yL = [];                       % left-axis (analytic recovery) values
    yR = [];                       % right-axis (split-half r) values
    legH = gobjects(0);
    legL = {};
    analytic_endpoint = [];        % analytic recovery at full retention (do-nothing)
    analytic_endpoint_x = [];
    prim_ar = []; prim_asf = [];   % primary basis analytic curve (for the chord/shade)

    present = [];
    for i = 1:size(bases, 1)
        if isfield(rec, bases{i,1}) && ~isempty(rec.(bases{i,1})), present(end+1) = i; end %#ok<AGROW>
    end
    multi = numel(present) > 1;
    % Marker/primary color follows the basis the operating point was CHOSEN on, so
    % the analytic markers (peak/chosen/trial-avg) match that trace and the fig6
    % objective (in compare mode this is the chosen, not merely the first, basis).
    prim_row = [];
    if isfield(rec, 'chosen') && isstruct(rec.chosen) && isfield(rec.chosen, 'basis') ...
            && ischar(rec.chosen.basis) && ~isempty(rec.chosen.basis)
        for i = 1:size(bases, 1)
            if strcmp(bases{i,1}, [rec.chosen.basis '_basis']) && any(present == i)
                prim_row = i; break;
            end
        end
    end
    if isempty(prim_row)
        if isempty(present), prim_row = 1; else, prim_row = present(1); end
    end
    mc_analytic = bases{prim_row, 2}; mc_splithalf = bases{prim_row, 3};

    % ---- LEFT axis: analytic recovery curves (solid) ----
    yyaxis(ax, 'left');
    hold(ax, 'on');
    set(ax, 'YColor', 'k');
    for i = 1:size(bases, 1)
        key = bases{i, 1};
        if isfield(rec, key) && ~isempty(rec.(key)) && isfield(rec.(key), 'analytic_recovery')
            b = rec.(key);
            h = plot(ax, fx(b.analytic_sv_frac), b.analytic_recovery, '-', ...
                     'Color', bases{i, 2}, 'LineWidth', 2);
            if multi, lbl = sprintf('%s: analytic recovery', bases{i, 4}); else, lbl = 'analytic recovery'; end
            legH(end+1) = h; legL{end+1} = lbl; %#ok<AGROW>
            yL = [yL; b.analytic_recovery(:)]; %#ok<AGROW>
            if isempty(analytic_endpoint)
                ar = b.analytic_recovery(:); asf = b.analytic_sv_frac(:);
                analytic_endpoint = ar(end); analytic_endpoint_x = asf(end);
                prim_ar = ar; prim_asf = asf;
            end
        end
    end

    % Max-tradeoff geometry: chord from the prediction peak to the do-nothing
    % (trial-average) point, and the shaded gap between it and the analytic
    % recovery curve on the descending limb (where max-tradeoff picks the farthest point).
    if numel(prim_ar) >= 3
        [~, kpk] = max(prim_ar);            % 1-based index of the peak
        % Peak of the analytic recovery curve (prediction peak): triangle.
        hpk = plot(ax, fx(prim_asf(kpk)), prim_ar(kpk), '^', 'MarkerSize', 11, ...
                   'MarkerFaceColor', mc_analytic, 'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
        legH(end+1) = hpk; legL{end+1} = 'prediction peak (analytic)'; %#ok<AGROW>
        yL = [yL; prim_ar(kpk)]; %#ok<AGROW>
        if kpk < numel(prim_ar) && prim_asf(end) ~= prim_asf(kpk)
            xs = prim_asf(kpk:end);
            yc = prim_ar(kpk:end);
            ychord = prim_ar(kpk) + (prim_ar(end) - prim_ar(kpk)) .* ...
                     (xs - prim_asf(kpk)) ./ (prim_asf(end) - prim_asf(kpk));
            hgap = patch(ax, [fx(xs); flipud(fx(xs))], [ychord; flipud(yc)], [0.5 0.5 0.5], ...
                         'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            uistack(hgap, 'bottom');
            plot(ax, fx(xs), ychord, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.0);
        end
    end

    % ---- RIGHT axis: split-half reliability curves (dashed) + reference points ----
    yyaxis(ax, 'right');
    hold(ax, 'on');
    set(ax, 'YColor', 'k');

    % ---- faint per-unit reliability traces (hybrid mode), drawn underneath ----
    for i = 1:size(bases, 1)
        key = bases{i, 1};
        if isfield(rec, key) && isstruct(rec.(key)) && isfield(rec.(key), 'units') && ~isempty(rec.(key).units)
            uu = rec.(key).units;
            xg = fx(uu.sv_frac);
            for j = 1:size(uu.split_half_r, 1)
                yv = uu.split_half_r(j, :)';
                m = isfinite(yv);
                if any(m)
                    plot(ax, xg(m), yv(m), '-', 'Color', [0.5 0.5 0.5 0.22], ...
                         'LineWidth', 0.4, 'HandleVisibility', 'off');
                end
            end
            if isfield(uu, 'markers') && ~isempty(uu.markers)
                mk = uu.markers;
                scatter(ax, fx(mk.sv_frac), mk.split_half_r, 9, [0.5 0.5 0.5], 'filled', ...
                        'MarkerFaceAlpha', 0.35, 'HandleVisibility', 'off');
                yR = [yR; mk.split_half_r(:)]; %#ok<AGROW>
            end
        end
    end

    for i = 1:size(bases, 1)
        key = bases{i, 1};
        if isfield(rec, key) && ~isempty(rec.(key)) && isfield(rec.(key), 'split_half_r')
            b = rec.(key);
            h = plot(ax, fx(b.sv_frac), b.split_half_r, '--', ...
                     'Color', bases{i, 3}, 'LineWidth', 1.6);
            if multi, lbl = sprintf('%s: split-half r', bases{i, 4}); else, lbl = 'split-half r'; end
            legH(end+1) = h; legL{end+1} = lbl; %#ok<AGROW>
            yR = [yR; b.split_half_r(:)]; %#ok<AGROW>
        end
    end

    if isfield(rec, 'trial_average') && ~isempty(rec.trial_average)
        ta = rec.trial_average;
        h = plot(ax, fx(ta.sv_frac), ta.split_half_r, 's', 'MarkerSize', 9, ...
                 'MarkerFaceColor', mc_splithalf, 'MarkerEdgeColor', 'k');
        legH(end+1) = h; legL{end+1} = 'trial-avg (split-half)'; %#ok<AGROW>
        yR = [yR; ta.split_half_r]; %#ok<AGROW>
    end
    if isfield(rec, 'wiener') && ~isempty(rec.wiener)
        w = rec.wiener;
        h = plot(ax, fx(w.sv_frac), w.split_half_r, 'd', 'MarkerSize', 9, ...
                 'MarkerFaceColor', [0.196 0.804 0.196], 'MarkerEdgeColor', 'k');
        legH(end+1) = h; legL{end+1} = 'Wiener'; %#ok<AGROW>
        yR = [yR; w.split_half_r]; %#ok<AGROW>
    end
    if isfield(rec, 'chosen') && ~isempty(rec.chosen) && ~isnan(rec.chosen.split_half_r)
        h = plot(ax, fx(rec.chosen.sv_frac), rec.chosen.split_half_r, 'p', 'MarkerSize', 18, ...
             'MarkerFaceColor', mc_splithalf, 'MarkerEdgeColor', 'k', 'LineWidth', 0.9);
        legH(end+1) = h; legL{end+1} = 'PSN chosen (split-half)'; %#ok<AGROW>
        yR = [yR; rec.chosen.split_half_r]; %#ok<AGROW>
    end
    ylabel(ax, 'split-half r  (TAvg vs Denoised)');
    rlim = local_lim(yR, is_mt, bot_margin);
    if ~isempty(rlim); ylim(ax, rlim); end

    % ---- GOLD markers on the LEFT (analytic recovery) trajectory:
    %      trial-avg (box) + PSN chosen (star) ----
    yyaxis(ax, 'left');
    if isfield(rec, 'trial_average') && ~isempty(rec.trial_average) && ~isempty(analytic_endpoint)
        h = plot(ax, fx(analytic_endpoint_x), analytic_endpoint, 's', 'MarkerSize', 9, ...
                 'MarkerFaceColor', mc_analytic, 'MarkerEdgeColor', 'k');
        legH(end+1) = h; legL{end+1} = 'trial-avg (analytic)'; %#ok<AGROW>
        yL = [yL; analytic_endpoint]; %#ok<AGROW>
    end
    if isfield(rec, 'chosen') && ~isempty(rec.chosen) && isfield(rec.chosen, 'recovery') ...
            && ~isnan(rec.chosen.recovery)
        h = plot(ax, fx(rec.chosen.sv_frac), rec.chosen.recovery, 'p', 'MarkerSize', 18, ...
                 'MarkerFaceColor', mc_analytic, 'MarkerEdgeColor', 'k', 'LineWidth', 0.9);
        legH(end+1) = h; legL{end+1} = 'PSN chosen (analytic)'; %#ok<AGROW>
        yL = [yL; rec.chosen.recovery]; %#ok<AGROW>
    end
    ylabel(ax, 'analytic recovery  (cumsum signal - noise/t)');
    llim = local_lim(yL, is_mt, bot_margin);
    if ~isempty(llim); ylim(ax, llim); end

    % ---- inverse-log x-axis: ticks at the warped positions, labelled by value ----
    xlim(ax, [-0.01, aW + 0.06]);
    tickvals = [0 0.25 0.5 0.75 0.9 0.95 0.99 1.0];
    set(ax, 'XTick', fx(tickvals), ...
            'XTickLabel', {'0', '0.25', '0.5', '0.75', '0.9', '0.95', '0.99', '1'});
    xtickangle(ax, 45);

    xlabel(ax, 'frac. signal var. retained');
    grid(ax, 'on');
    if ~isempty(legH)
        % Order so the analytic (gold) entries form one block and the split-half
        % (gray) entries another, with gold box/star adjacent, gray box/star adjacent.
        rank = zeros(1, numel(legL));
        for i = 1:numel(legL)
            rank(i) = local_legrank(legL{i});
        end
        [~, perm] = sort(rank);
        % Top-left. Two columns when there is no inset (so the legend runs along the
        % top and blocks less data); single column in max-tradeoff mode, where the
        % taller top whitespace and the upper-right inset leave room for one column.
        if is_mt, ncol = 1; else, ncol = 2; end
        legend_behind(ax, legH(perm), legL(perm), 'FontSize', 7, ...
                      'Location', 'northwest', 'NumColumns', ncol);
    end
    hold(ax, 'off');

    % Inset (lower-left): linear-axis zoom of the max-tradeoff selection geometry.
    draw_max_tradeoff_inset(ax, rec);
end


function draw_max_tradeoff_inset(ax, rec)
% Lower-left inset that zooms, on LINEAR axes, into the descending limb where the
% max-tradeoff criterion picks the operating point. Coordinates are normalized so
% the recovery peak sits at (0,1) and the trial-average (do-nothing) at (1,0); the
% peak->trial-average chord is then the anti-diagonal u+v=1, and max-tradeoff
% selects the curve point of greatest perpendicular distance from it (argmax of the
% shaded gap). argmax perpendicular distance is affine-invariant, so this clean
% square view marks the SAME operating point psn() chose while drawing a true right
% angle. Only rendered in max-tradeoff mode.

    if ~isfield(rec, 'criterion') || ~strcmp(rec.criterion, 'max-tradeoff'); return; end

    % Basis the operating point was chosen on (fallback: first present).
    key = '';
    if isfield(rec, 'chosen') && isstruct(rec.chosen) && isfield(rec.chosen, 'basis') ...
            && ischar(rec.chosen.basis) && ~isempty(rec.chosen.basis)
        key = [rec.chosen.basis '_basis'];
    end
    if isempty(key) || ~isfield(rec, key) || isempty(rec.(key))
        if isfield(rec, 'signal_basis') && ~isempty(rec.signal_basis)
            key = 'signal_basis';
        elseif isfield(rec, 'difference_basis') && ~isempty(rec.difference_basis)
            key = 'difference_basis';
        else
            return;
        end
    end
    b = rec.(key);
    if ~isfield(b, 'analytic_recovery') || ~isfield(b, 'analytic_sv_frac'); return; end
    asf = b.analytic_sv_frac(:); ar = b.analytic_recovery(:);
    if numel(ar) < 3; return; end
    [rpk, kpk] = max(ar);
    xpk = asf(kpk); xend = asf(end); rend = ar(end);
    if kpk >= numel(ar) || xend == xpk || (rpk - rend) <= 1e-12; return; end
    if ~isfield(rec, 'chosen') || isempty(rec.chosen) || ~isfield(rec.chosen, 'recovery') ...
            || ~isfinite(rec.chosen.recovery) || ~isfield(rec.chosen, 'sv_frac'); return; end

    % Normalized descending limb: u in [0,1] (0=peak, 1=trial-avg), v in [0,1] (1=peak).
    idx = kpk:numel(ar);
    u = (asf(idx) - xpk) ./ (xend - xpk);
    v = (ar(idx)  - rend) ./ (rpk - rend);
    u_ch = (rec.chosen.sv_frac  - xpk) / (xend - xpk);
    v_ch = (rec.chosen.recovery - rend) / (rpk - rend);
    tt   = (u_ch + v_ch - 1) / 2;                 % foot of perpendicular on u+v=1
    foot = [u_ch - tt, v_ch - tt];

    mc = [0.122 0.467 0.706];                     % analytic color, matches the panel trace
    if strcmp(key, 'difference_basis'); mc = [0.173 0.627 0.173]; end
    perpc = [0.85 0.10 0.10];

    fig = ancestor(ax, 'figure');
    p = ax.Position;                              % normalized figure units
    % Upper-right corner; the +20% top headroom above pushes the curves/markers
    % down so this sits over clear space.
    ipos = [p(1) + 0.635*p(3), p(2) + 0.52*p(4), 0.36*p(3), 0.36*p(4)];
    axi = axes('Parent', fig, 'Units', 'normalized', 'Position', ipos);
    hold(axi, 'on'); box(axi, 'on');
    set(axi, 'Color', 'w', 'FontSize', 7, 'Layer', 'top');

    patch(axi, [u; flipud(u)], [v; flipud(1 - u)], [0.5 0.5 0.5], ...
          'FaceAlpha', 0.12, 'EdgeColor', 'none');                     % gap curve<->chord
    plot(axi, [0 1], [1 0], '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0);   % chord
    plot(axi, u, v, '-', 'Color', mc, 'LineWidth', 1.6);                          % recovery curve
    plot(axi, 0, 1, '^', 'MarkerSize', 7, 'MarkerFaceColor', mc, 'MarkerEdgeColor', 'k');  % peak
    plot(axi, 1, 0, 's', 'MarkerSize', 7, 'MarkerFaceColor', mc, 'MarkerEdgeColor', 'k');  % trial-avg
    plot(axi, [u_ch foot(1)], [v_ch foot(2)], '-', 'Color', perpc, 'LineWidth', 1.4);      % perpendicular
    plot(axi, foot(1), foot(2), 'o', 'MarkerSize', 4, 'MarkerFaceColor', 'w', ...
         'MarkerEdgeColor', perpc, 'LineWidth', 1.0);                                       % foot
    plot(axi, u_ch, v_ch, 'p', 'MarkerSize', 13, 'MarkerFaceColor', mc, ...
         'MarkerEdgeColor', 'k', 'LineWidth', 0.7);                                         % chosen

    set(axi, 'DataAspectRatio', [1 1 1]);         % equal units -> the right angle reads true
    axis(axi, [-0.06 1.10 -0.06 1.10]);
    % Tick labels in the ACTUAL units: both axes are linear (affine) maps of them,
    % u -> frac. signal var. retained, v -> analytic recovery, so labelling the
    % fixed tick positions with the real values is exact.
    xt = [0 0.5 1];
    set(axi, 'XTick', xt, 'XTickLabel', fmt_ticks(xpk  + xt * (xend - xpk)));
    set(axi, 'YTick', xt, 'YTickLabel', fmt_ticks(rend + xt * (rpk  - rend)));
    title(axi, 'max-tradeoff (zoom)', 'FontSize', 7.5, 'FontWeight', 'normal');
    xlabel(axi, 'frac. signal retained', 'FontSize', 7);
    ylabel(axi, 'analytic recovery', 'FontSize', 7);
    hold(axi, 'off');
end


function s = fmt_ticks(vals)
% Format tick values with the fewest decimals that keeps them all distinct (the
% descending limb can crowd near frac=1, so a fixed precision would collapse
% adjacent labels).
    vals = vals(:)';
    for d = 1:8
        s = arrayfun(@(x) sprintf('%.*f', d, x), vals, 'UniformOutput', false);
        if numel(unique(s)) == numel(s); return; end
    end
end


function T = invlog_fwd(x, a, W, K, lk, slope, aW)
% Forward inverse-log map: linear on [0, a]; [a, 1] expanded; linear past 1.
    x = x(:);
    d = min(max((1 - x) / (1 - a), 0), 1);          % 1 at split, 0 at x=1
    tail = a + W * (1 - log10(1 + K * d) / lk);
    over = aW + slope * (x - 1);
    T = x;
    mid = (x > a) & (x <= 1);
    T(mid) = tail(mid);
    T(x > 1) = over(x > 1);
end


function r = local_legrank(lbl)
% Order legend so analytic (gold-axis) entries form one block and split-half
% entries another, with each block's box+star adjacent. Content-based so it is
% robust to the single- vs multi-basis label variants.
    L = lower(lbl);
    if contains(L, 'prediction peak')
        if contains(L, 'analytic'), r = 0.5; else, r = 9; end
    elseif contains(L, 'chosen')
        if contains(L, 'analytic'), r = 2; else, r = 14; end
    elseif contains(L, 'trial-avg')
        if contains(L, 'analytic'), r = 1; else, r = 13; end
    elseif contains(L, 'tradeoff')
        r = 3;
    elseif contains(L, 'analytic recovery')
        r = 0;
    elseif contains(L, 'split-half r')
        r = 10;
    elseif contains(L, 'wiener')
        r = 12;
    else
        r = 99;
    end
end

function lim = local_lim(yvals, is_mt, bot_margin)
% Set the top to top_mult x the peak-to-bottom range so the data peak lands at
% 1/top_mult of the axis height, keeping the top clear for the top-left legend
% (and, in max-tradeoff mode, the upper-right inset too). 4x in max-tradeoff mode
% (needs room for the inset), 2x otherwise (legend only). When values go negative
% (do-nothing tail on noisy data) the lower limit follows the data.
    if nargin < 2 || isempty(is_mt); is_mt = false; end
    if nargin < 3 || isempty(bot_margin); bot_margin = 0; end
    lim = [];
    if isempty(yvals); return; end
    v = yvals(isfinite(yvals));
    if isempty(v); return; end
    vmax = max(max(v), 1e-3);
    vmin = min(v);
    if is_mt, top_mult = 4.0; else, top_mult = 2.0; end
    if vmin >= 0
        bottom = -bot_margin * vmax;
    else
        span = vmax - vmin;
        bottom = vmin - (0.05 + bot_margin) * span;
    end
    lim = [bottom, bottom + top_mult * (vmax - bottom)];
end
