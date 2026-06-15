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

    bases = {'signal_basis', [0.122 0.467 0.706], 'signal'; ...
             'difference_basis', [0.173 0.627 0.173], 'difference'};

    yL = [];                       % left-axis (analytic recovery) values
    yR = [];                       % right-axis (split-half r) values
    legH = gobjects(0);
    legL = {};

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
            legH(end+1) = h; legL{end+1} = sprintf('%s (recovery)', bases{i, 3}); %#ok<AGROW>
            yL = [yL; b.analytic_recovery(:)]; %#ok<AGROW>
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
                     'Color', bases{i, 2}, 'LineWidth', 1.6);
            legH(end+1) = h; legL{end+1} = sprintf('%s (split-half)', bases{i, 3}); %#ok<AGROW>
            yR = [yR; b.split_half_r(:)]; %#ok<AGROW>
        end
    end

    if isfield(rec, 'trial_average') && ~isempty(rec.trial_average)
        ta = rec.trial_average;
        h = plot(ax, fx(ta.sv_frac), ta.split_half_r, 's', 'MarkerSize', 9, ...
                 'MarkerFaceColor', [0.4 0.4 0.4], 'MarkerEdgeColor', 'k');
        legH(end+1) = h; legL{end+1} = 'trial-avg'; %#ok<AGROW>
        yR = [yR; ta.split_half_r]; %#ok<AGROW>
    end
    if isfield(rec, 'wiener') && ~isempty(rec.wiener)
        w = rec.wiener;
        h = plot(ax, fx(w.sv_frac), w.split_half_r, 'd', 'MarkerSize', 9, ...
                 'MarkerFaceColor', [0.839 0.153 0.157], 'MarkerEdgeColor', 'k');
        legH(end+1) = h; legL{end+1} = 'Wiener'; %#ok<AGROW>
        yR = [yR; w.split_half_r]; %#ok<AGROW>
    end
    if isfield(rec, 'chosen') && ~isempty(rec.chosen) && ~isnan(rec.chosen.split_half_r)
        plot(ax, fx(rec.chosen.sv_frac), rec.chosen.split_half_r, 'p', 'MarkerSize', 18, ...
             'MarkerFaceColor', [1 0.84 0], 'MarkerEdgeColor', 'k', 'LineWidth', 0.9);
        yR = [yR; rec.chosen.split_half_r]; %#ok<AGROW>
    end
    ylabel(ax, 'split-half r  (TAvg vs Denoised)');
    rlim = local_lim(yR);
    if ~isempty(rlim); ylim(ax, rlim); end

    % ---- chosen gold star on the LEFT (analytic recovery) trajectory ----
    yyaxis(ax, 'left');
    if isfield(rec, 'chosen') && ~isempty(rec.chosen) && isfield(rec.chosen, 'recovery') ...
            && ~isnan(rec.chosen.recovery)
        h = plot(ax, fx(rec.chosen.sv_frac), rec.chosen.recovery, 'p', 'MarkerSize', 18, ...
                 'MarkerFaceColor', [1 0.84 0], 'MarkerEdgeColor', 'k', 'LineWidth', 0.9);
        legH(end+1) = h; legL{end+1} = sprintf('PSN chosen (%s)', rec.chosen.label); %#ok<AGROW>
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
        legend(ax, legH, legL, 'FontSize', 7, 'Location', 'northwest');
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


function lim = local_lim(yvals)
% Floor at 0, top = max + headroom (so the legend has room), matching Python.
    lim = [];
    if isempty(yvals); return; end
    v = yvals(isfinite(yvals));
    if isempty(v); return; end
    lim = [0, max(max(v), 1e-3) * 1.45];
end
