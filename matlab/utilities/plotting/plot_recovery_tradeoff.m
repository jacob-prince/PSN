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

    % Inverse-log x transform parameters (split=0.9, expand=0.4, K=99).
    a = 0.9; W = 0.4; K = 99; lk = log10(1 + K); aW = a + W;
    slope = W * K / ((1 - a) * log(10) * lk);
    fx = @(x) invlog_fwd(x, a, W, K, lk, slope, aW);

    % {key, analytic color (solid), split-half color (dashed), name}. Color encodes
    % analytic vs split-half; the chosen/trial-avg markers reuse the primary basis's
    % colors so each box/star matches its trace.
    bases = {'signal_basis',     [0.122 0.467 0.706], [1.000 0.498 0.055], 'signal'; ...
             'difference_basis', [0.173 0.627 0.173], [0.580 0.404 0.741], 'difference'};

    yL = [];                       % left-axis (analytic recovery) values
    yR = [];                       % right-axis (split-half r) values
    legH = gobjects(0);
    legL = {};
    analytic_endpoint = [];        % analytic recovery at full retention (do-nothing)
    analytic_endpoint_x = [];
    prim_ar = []; prim_asf = [];   % primary basis analytic curve (for the chord/shade)

    % primary basis (first present) colors -> used for the chosen/trial-avg markers;
    % multi = both bases present (compare), which switches labels to include the name.
    present = [];
    for i = 1:size(bases, 1)
        if isfield(rec, bases{i,1}) && ~isempty(rec.(bases{i,1})), present(end+1) = i; end %#ok<AGROW>
    end
    multi = numel(present) > 1;
    if isempty(present)
        mc_analytic = [0.122 0.467 0.706]; mc_splithalf = [1.000 0.498 0.055];
    else
        mc_analytic = bases{present(1), 2}; mc_splithalf = bases{present(1), 3};
    end

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
                         'FaceAlpha', 0.15, 'EdgeColor', 'none');
            uistack(hgap, 'bottom');
            plot(ax, fx(xs), ychord, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.0);
            legH(end+1) = hgap; legL{end+1} = 'max-tradeoff gap'; %#ok<AGROW>
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
    rlim = local_lim(yR);
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
    llim = local_lim(yL);
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
        legend(ax, legH(perm), legL(perm), 'FontSize', 7, 'Location', 'southwest');
    end
    hold(ax, 'off');
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

function lim = local_lim(yvals)
% Top = max + headroom (so the legend has room). When values go negative (e.g.
% the do-nothing tail of the analytic-recovery curve on noisy data) the lower
% limit follows the data instead of clipping at 0, so the full curve and the
% trial-average anchor stay visible. Matches Python _lim.
    lim = [];
    if isempty(yvals); return; end
    v = yvals(isfinite(yvals));
    if isempty(v); return; end
    vmax = max(max(v), 1e-3);
    vmin = min(v);
    if vmin >= 0
        lim = [0, vmax * 1.45];
    else
        span = vmax - vmin;
        lim = [vmin - 0.05*span, vmax + 0.45*span];
    end
end
