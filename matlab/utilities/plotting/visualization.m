function fig = visualization(data, results, varargin)
% VISUALIZATION  Generate the diagnostic figure for PSN denoising results.
%
%   fig = visualization(data, results) builds the multi-panel PSN diagnostic
%   figure (input/denoised/residual maps, covariances, eigenspectra, recovery
%   tradeoff, etc.) for the given training data and results struct.
%
%   fig = visualization(data, results, 'Visible', 'off', 'cmap', C) controls
%   figure visibility and the colormap used for the data panels.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <data> - [nunits x nconds x ntrials] training data used for denoising.
%
% <results> - struct, the results structure returned by psn().
%
% <varargin> (optional) - name-value pairs:
%   'Visible' - 'on' (default) or 'off' to control figure visibility.
%   'cmap'    - [m x 3] colormap for the input/denoised/residual panels.
%               Default: opt_used.cmap if present, else cmapsign4(256).
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <fig> - handle to the created figure (e.g. for saving).

    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'Visible', 'on');
    addParameter(p, 'cmap', []);  % colormap for data plots (empty = use default or from opt)
    parse(p, varargin{:});

    % Get colormap for data plots (input, denoised, residuals)
    % Priority: 1) explicit argument, 2) opt.cmap from results, 3) default cmapsign4
    if ~isempty(p.Results.cmap)
        data_cmap = p.Results.cmap;
    elseif isfield(results, 'opt_used') && isfield(results.opt_used, 'cmap') && ~isempty(results.opt_used.cmap)
        data_cmap = results.opt_used.cmap;
    else
        data_cmap = cmapsign4(256);
    end

    % Create a large figure (visible or invisible based on parameter)
    % Match Python: figsize=(24, 15) -> approximately 2400x1500 pixels
    fig = figure('Position', [50, 50, 2400, 1500], 'Visible', p.Results.Visible);

    % Set default font sizes to match Python rcParams
    set(fig, 'DefaultAxesFontSize', 11);
    set(fig, 'DefaultTextFontSize', 12);
    set(fig, 'DefaultAxesTitleFontSizeMultiplier', 1.18);  % 13/11
    set(fig, 'DefaultAxesLabelFontSizeMultiplier', 1.0);   % 11/11

    % Extract data dimensions
    [nunits, nconds, ntrials] = size(data);

    % Subsampling for large datasets (>500 units or >500 conditions)
    % Randomly select 100 units/conditions for certain plots, but keep full population for means
    n_subsample = 100;

    % Unit subsampling
    subsample_units = nunits > 500;
    if subsample_units
        su = RandStream('mt19937ar', 'Seed', 42);  % local; leave global rng untouched
        subsample_idx = sort(randperm(su, nunits, n_subsample));
    else
        subsample_idx = 1:nunits;
    end

    % Condition subsampling (for trace plots)
    subsample_conds = nconds > 500;
    if subsample_conds
        sc = RandStream('mt19937ar', 'Seed', 43);  % local; leave global rng untouched
        subsample_cond_idx = sort(randperm(sc, nconds, n_subsample));
    else
        subsample_cond_idx = 1:nconds;
    end

    % Build suffix strings
    if subsample_units && subsample_conds
        subsample_suffix = sprintf('\n(randomly subsampling %d units, %d conditions)', n_subsample, n_subsample);
        subsample_suffix_units = sprintf('\n(randomly subsampling %d units)', n_subsample);
        subsample_suffix_conds = sprintf('\n(randomly subsampling %d conditions)', n_subsample);
        subsample_suffix_traces = sprintf('\n(randomly subsampling %d units, %d conditions)', n_subsample, n_subsample);
    elseif subsample_units
        subsample_suffix = sprintf('\n(randomly subsampling %d units)', n_subsample);
        subsample_suffix_units = subsample_suffix;
        subsample_suffix_conds = '';
        subsample_suffix_traces = subsample_suffix;
    elseif subsample_conds
        subsample_suffix = '';
        subsample_suffix_units = '';
        subsample_suffix_conds = sprintf('\n(randomly subsampling %d conditions)', n_subsample);
        subsample_suffix_traces = subsample_suffix_conds;
    else
        subsample_suffix = '';
        subsample_suffix_units = '';
        subsample_suffix_conds = '';
        subsample_suffix_traces = '';
    end

    % Get options if stored
    if isfield(results, 'opt_used')
        opt = results.opt_used;
    else
        opt = struct();
    end

    % Extract basis type description
    if isfield(opt, 'basis')
        if ischar(opt.basis)
            basis_desc = opt.basis;
        else
            basis_desc = sprintf('custom [%dx%d]', size(opt.basis, 1), size(opt.basis, 2));
        end
    else
        basis_desc = 'unknown';
    end

    % Flag for full-rank Wiener (changes how threshold lines are labeled),
    % matching Python's is_fullrank_wiener = (basis_desc == 'wiener').
    is_fullrank_wiener = strcmp(basis_desc, 'wiener');

    % Extract threshold method
    if isfield(opt, 'threshold_method')
        threshold_method = opt.threshold_method;
    else
        threshold_method = 'unknown';
    end

    % Extract criterion
    if isfield(opt, 'criterion')
        criterion = opt.criterion;
    else
        criterion = 'unknown';
    end

    % Extract basis_ordering
    if isfield(opt, 'basis_ordering')
        basis_ordering = opt.basis_ordering;
    else
        basis_ordering = 'eigenvalues';  % default
    end

    % Check for NaNs and compute average number of trials
    has_nans = any(isnan(data(:)));
    if has_nans
        validcnt = sum(~any(isnan(data), 1), 3);
        ntrials_avg = sum(validcnt(validcnt > 1)) / nconds;
        if ntrials_avg < 1
            ntrials_avg = 1;
        end
    else
        ntrials_avg = ntrials;
    end

    % Create title (order: Basis, Criterion, Method to match API)
    if has_nans
        title_text = sprintf('Data: %d units × %d conditions × %d max trials (avg %.1f)  |  Basis: %s  |  Criterion: %s  |  Method: %s', ...
                            nunits, nconds, ntrials, ntrials_avg, basis_desc, criterion, threshold_method);
    else
        title_text = sprintf('Data: %d units × %d conditions × %d trials  |  Basis: %s  |  Criterion: %s  |  Method: %s', ...
                            nunits, nconds, ntrials, basis_desc, criterion, threshold_method);
    end

    % Add threshold info if conservative mode or variance criterion is used
    threshold_info = {};
    if isfield(opt, 'allowable_thresholds') && ~isempty(opt.allowable_thresholds)
        allowable = opt.allowable_thresholds;
        if numel(allowable) == 1
            threshold_info{end+1} = sprintf('Forced threshold: %d', allowable(1));
        else
            threshold_info{end+1} = sprintf('Allowable thresholds: [%s]', num2str(allowable(:)'));
        end
    end
    if strcmp(criterion, 'variance') || strcmp(criterion, 'variance_eigenvalues')
        if isfield(opt, 'variance_threshold')
            vt = opt.variance_threshold;
        else
            vt = 0.99;
        end
        threshold_info{end+1} = sprintf('Variance threshold: %.2f', vt);
    end

    if ~isempty(threshold_info)
        title_text = [title_text '  |  ' strjoin(threshold_info, ', ')];
    end

    % Get trial-averaged and denoised data (use nanmean for NaN data)
    if has_nans
        trial_avg = nanmean(data, 3);
    else
        trial_avg = mean(data, 3);
    end
    denoised = results.denoiseddata;
    noise = trial_avg - denoised;

    % =====================================================================
    % Create tiledlayout to match the Python figure.
    % Row 1 (6 panels x 2 cols): cSb, cNb, Top5 dims, Eigenvalues, Sig/Noise var, Objective
    % Rows 2-4 (4 panels x 3 cols): Row 2 leads with the Recovery-tradeoff panel.
    % 12 columns = lcm(6,4) so both panel counts tile exactly.
    % =====================================================================
    t = tiledlayout(4, 12, 'TileSpacing', 'compact', 'Padding', 'compact');

    % Add supertitle (must be after tiledlayout creation)
    sgtitle(title_text, 'FontSize', 14, 'FontWeight', 'bold');

    % =====================================================================
    % Plot 1: Basis source matrix (signal covariance or basis-specific)
    % =====================================================================
    ax1 = nexttile(t, [1 2]);  % Row 1, columns 1-2
    if isfield(results, 'gsn_result') && isfield(results.gsn_result, 'cSb')
        cSb = results.gsn_result.cSb;
        cNb = results.gsn_result.cNb;

        % Determine which matrix based on basis type
        if isfield(opt, 'basis')
            basis_type = opt.basis;
            if ischar(basis_type) || isstring(basis_type)
                basis_type = char(basis_type);
                switch basis_type
                    case 'difference'
                        plot_matrix_1 = cSb - cNb / ntrials_avg;
                        plot_title = sprintf('cSb - cNb/%.1f (difference)', ntrials_avg);
                    case 'noise'
                        plot_matrix_1 = cNb;
                        plot_title = 'Noise Covariance (cNb)';
                    case 'pca'
                        trial_avg_demeaned = trial_avg - results.unit_means;
                        plot_matrix_1 = cov(trial_avg_demeaned');
                        plot_title = 'Trial-Avg Data Covariance';
                    otherwise
                        plot_matrix_1 = cSb;
                        plot_title = 'Signal Covariance (cSb)';
                end
            else
                plot_matrix_1 = cSb;
                plot_title = 'Signal Covariance (cSb)';
            end
        else
            plot_matrix_1 = cSb;
            plot_title = 'Signal Covariance (cSb)';
        end

        % Compute symmetric colorbar limits around 0 (use 99th percentile for better contrast)
        data_absmax = prctile(abs(plot_matrix_1(:)), 99);
        if data_absmax > 0
            clim_1 = [-data_absmax, data_absmax];
        else
            clim_1 = [-1, 1];
        end

        imagesc(ax1, plot_matrix_1, clim_1);
        colormap(ax1, redblue);
        colorbar(ax1);
        title(ax1, plot_title);
        xlabel(ax1, 'Units');
        ylabel(ax1, 'Units');
        axis(ax1, 'equal', 'tight');
    else
        text(ax1, 0.5, 0.5, {'Covariance', 'Not Available'}, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Units', 'normalized');
    end

    % =====================================================================
    % Plot 2: Noise Covariance (cNb) - NEW
    % =====================================================================
    ax2 = nexttile(t, [1 2]);  % Row 1, columns 3-4
    if isfield(results, 'gsn_result') && isfield(results.gsn_result, 'cNb')
        cNb = results.gsn_result.cNb;

        % Compute symmetric colorbar limits around 0 (use 99th percentile for better contrast)
        data_absmax_cNb = prctile(abs(cNb(:)), 99);
        if data_absmax_cNb > 0
            clim_cNb = [-data_absmax_cNb, data_absmax_cNb];
        else
            clim_cNb = [-1, 1];
        end

        imagesc(ax2, cNb, clim_cNb);
        colormap(ax2, redblue);
        colorbar(ax2);
        title(ax2, 'Noise Covariance (cNb)');
        xlabel(ax2, 'Units');
        ylabel(ax2, 'Units');
        axis(ax2, 'equal', 'tight');
    else
        text(ax2, 0.5, 0.5, {'Noise Covariance', 'Not Available'}, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Units', 'normalized');
    end

    % =====================================================================
    % Plot 3: Top 5 PCs as vertical line plots (half width)
    % =====================================================================
    ax3 = nexttile(t, [1 2]);  % Row 1, panel 3
    if isfield(results, 'fullbasis')
        num_pcs = min(5, size(results.fullbasis, 2));

        % Normalize each PC for visualization
        hold(ax3, 'on');
        y_units = 1:nunits;
        colors = lines(num_pcs);

        % Find max absolute loading across top 5 PCs for scaling
        max_loading = max(abs(results.fullbasis(:, 1:num_pcs)), [], 'all');
        if max_loading > 0
            scale_factor = 0.4 / max_loading;  % Scale to fit within 0.4 x-units
        else
            scale_factor = 1;
        end

        for pc = 1:num_pcs
            % Center each PC at position pc, with loadings as horizontal deviations
            x_vals = pc + results.fullbasis(:, pc) * scale_factor;
            plot(ax3, x_vals, y_units, 'LineWidth', 1.5, 'Color', colors(pc, :));

            % Add vertical reference line at center
            plot(ax3, [pc, pc], [1, nunits], 'k--', 'LineWidth', 0.5);
        end
        hold(ax3, 'off');

        ylabel(ax3, 'Units');
        title(ax3, 'Top 5 Basis Dims');
        xlim(ax3, [0.5, num_pcs + 0.5]);
        ylim(ax3, [1, nunits]);
        set(ax3, 'YDir', 'reverse');  % Flip y-axis to match heatmaps
        set(ax3, 'XTick', 1:num_pcs);
        % Add eigenvalue labels under PC indices (MATLAB is 1-indexed)
        if isfield(results, 'basis_eigenvalues') && ~isempty(results.basis_eigenvalues) && length(results.basis_eigenvalues) >= num_pcs
            evals = results.basis_eigenvalues;
            % Use simple tick labels for PC index, add lambda as text below
            set(ax3, 'XTick', 1:num_pcs, 'XTickLabel', 1:num_pcs);
            % Get axis position for placing text
            ylims = ylim(ax3);
            y_text = ylims(2) + 0.08 * (ylims(2) - ylims(1));  % Just below x-axis (y is reversed)
            for pc = 1:num_pcs
                text(ax3, pc, y_text, sprintf('\\lambda=%.2f', evals(pc)), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
                    'FontSize', 7);
            end
        end
        % Add xlabel with extra padding to avoid overlap with lambda labels
        xh = xlabel(ax3, 'Principal Component');
        xh.Position(2) = xh.Position(2) + 0.07 * (ylims(2) - ylims(1));
        grid(ax3, 'on');
    else
        text(ax3, 0.5, 0.5, {'Basis', 'Not Available'}, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Units', 'normalized');
    end

    % =====================================================================
    % Plot 4: Global dimension ranking (eigenvalues or signal variance) (half width)
    % =====================================================================
    ax4 = nexttile(t, [1 2]);  % Row 1, panel 4

    % Determine if we should use log scale for x-axis (for large datasets)
    use_logscale = nunits > 50;

    % Check if eigenvalues were used for ranking (and are available)
    use_eigenvalues = strcmp(basis_ordering, 'eigenvalues') && ...
                      isfield(results, 'basis_eigenvalues') && ...
                      ~isempty(results.basis_eigenvalues);

    % Check if prediction ordering was used
    use_prediction_ordering = strcmp(basis_ordering, 'prediction');

    if use_eigenvalues
        % Show eigenvalues (SORTED - what was actually used for ranking)
        evals = results.basis_eigenvalues;  % Already sorted in descending order
        nd4 = length(evals);
        % Head-tail log x-axis (matches Python _set_headtail_log_xscale): warp the
        % data onto a plain linear axis, since MATLAB has no 'function' x-scale.
        [fwd4, axmode4] = dim_xaxis_transform(use_logscale, nd4, 10);
        if use_logscale, xr4 = 1:nd4; else, xr4 = 0:(nd4-1); end  % 1-indexed on log, 0-indexed on linear
        plot(ax4, fwd4(xr4), evals, 'LineWidth', 1.5, 'Color', [0.5, 0, 0.5], 'DisplayName', '\lambda_k(\Sigma_S)');
        hold(ax4, 'on');

        % Threshold -> legend entry (Python labels the axvline; no rotated text here).
        if isfield(results, 'best_threshold')
            if isscalar(results.best_threshold) && results.best_threshold > 0
                xline(ax4, fwd4(results.best_threshold), 'r--', 'LineWidth', 2, ...
                      'DisplayName', thresh_legend_label(results.best_threshold, is_fullrank_wiener, false));
            elseif ~isscalar(results.best_threshold) && mean(results.best_threshold) > 0
                mean_thresh = mean(results.best_threshold);
                xline(ax4, fwd4(mean_thresh), 'r--', 'LineWidth', 2, ...
                      'DisplayName', sprintf('Mean threshold = %.1f', mean_thresh));
            end
        end
        hold(ax4, 'off');

        xlabel(ax4, 'Dimension k (signal eigenbasis)');
        ylabel(ax4, 'Eigenvalue');
        if is_fullrank_wiener
            title(ax4, '\Sigma_S Eigenvalues (signal basis)');
        else
            title(ax4, 'Basis Eigenvalues');
        end
        legend_behind(ax4, 'Location', 'best', 'FontSize', 7);
        grid(ax4, 'on');
        finalize_dim_xaxis(ax4, fwd4, axmode4, nd4, 10, [-0.5, nd4 - 0.5]);

    elseif use_prediction_ordering && isfield(results, 'signalvar') && isfield(results, 'noisevar')
        % Show prediction ordering criterion (signal - noise/ntrials) and signal variance
        signal_vars = results.signalvar;
        noise_vars = results.noisevar;
        prediction_obj = signal_vars - noise_vars / ntrials_avg;
        nd4 = length(signal_vars);
        [fwd4, axmode4] = dim_xaxis_transform(use_logscale, nd4, 10);
        if use_logscale, xr4 = 1:nd4; else, xr4 = 0:(nd4-1); end
        plot(ax4, fwd4(xr4), signal_vars, 'LineWidth', 1.5, 'Color', 'blue', 'DisplayName', 'Signal Var');
        hold(ax4, 'on');
        plot(ax4, fwd4(xr4), prediction_obj, 'LineWidth', 1.5, 'Color', [0.5, 0, 0.5], 'DisplayName', 'SigVar - NoiseVar/ntrials');

        % Threshold -> red rotated text (Python uses 'Threshold = int(best_t)' here).
        if isfield(results, 'best_threshold')
            if isscalar(results.best_threshold) && results.best_threshold > 0
                bt = results.best_threshold;
                hxl = xline(ax4, fwd4(bt), 'r--', 'LineWidth', 2);
                hxl.HandleVisibility = 'off';  % Exclude from legend
                ylims = ylim(ax4);
                y_pos = ylims(1) + 0.7 * (ylims(2) - ylims(1));
                if use_logscale, tx = bt * 1.05; else, tx = bt + 0.5; end
                text(ax4, fwd4(tx), y_pos, sprintf('Threshold = %d', floor(bt)), ...
                    'FontSize', 9, 'Color', 'r', 'Rotation', 90, ...
                    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
            elseif ~isscalar(results.best_threshold) && mean(results.best_threshold) > 0
                mean_thresh = mean(results.best_threshold);
                hxl = xline(ax4, fwd4(mean_thresh), 'r--', 'LineWidth', 2);
                hxl.HandleVisibility = 'off';  % Exclude from legend
                ylims = ylim(ax4);
                y_pos = ylims(1) + 0.7 * (ylims(2) - ylims(1));
                if use_logscale, tx = mean_thresh * 1.05; else, tx = mean_thresh + 0.5; end
                text(ax4, fwd4(tx), y_pos, sprintf('Mean Threshold = %.1f', mean_thresh), ...
                    'FontSize', 9, 'Color', 'r', 'Rotation', 90, ...
                    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
            end
        end
        hold(ax4, 'off');

        xlabel(ax4, 'Dimension');
        ylabel(ax4, 'Variance');
        title(ax4, 'Ordering Criterion');
        legend_behind(ax4, 'Location', 'best', 'FontSize', 7);
        grid(ax4, 'on');
        finalize_dim_xaxis(ax4, fwd4, axmode4, nd4, 10, [-0.5, nd4 - 0.5]);

    elseif isfield(results, 'signalvar')
        % Show signal variance (SORTED - what was actually used for ranking)
        signal_vars = results.signalvar;  % Already sorted in descending order
        nd4 = length(signal_vars);
        [fwd4, axmode4] = dim_xaxis_transform(use_logscale, nd4, 10);
        if use_logscale, xr4 = 1:nd4; else, xr4 = 0:(nd4-1); end
        plot(ax4, fwd4(xr4), signal_vars, 'LineWidth', 1.5, 'Color', 'blue');
        hold(ax4, 'on');

        % Threshold -> red rotated text (Python uses 'Threshold = int(best_t)' here).
        if isfield(results, 'best_threshold')
            if isscalar(results.best_threshold) && results.best_threshold > 0
                bt = results.best_threshold;
                xline(ax4, fwd4(bt), 'r--', 'LineWidth', 2);
                ylims = ylim(ax4);
                y_pos = ylims(1) + 0.7 * (ylims(2) - ylims(1));
                if use_logscale, tx = bt * 1.05; else, tx = bt + 0.5; end
                text(ax4, fwd4(tx), y_pos, sprintf('Threshold = %d', floor(bt)), ...
                    'FontSize', 9, 'Color', 'r', 'Rotation', 90, ...
                    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
            elseif ~isscalar(results.best_threshold) && mean(results.best_threshold) > 0
                mean_thresh = mean(results.best_threshold);
                xline(ax4, fwd4(mean_thresh), 'r--', 'LineWidth', 2);
                ylims = ylim(ax4);
                y_pos = ylims(1) + 0.7 * (ylims(2) - ylims(1));
                if use_logscale, tx = mean_thresh * 1.05; else, tx = mean_thresh + 0.5; end
                text(ax4, fwd4(tx), y_pos, sprintf('Mean Threshold = %.1f', mean_thresh), ...
                    'FontSize', 9, 'Color', 'r', 'Rotation', 90, ...
                    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
            end
        end
        hold(ax4, 'off');

        xlabel(ax4, 'Dimension');
        ylabel(ax4, 'Signal Var');
        title(ax4, 'Signal Variance');
        grid(ax4, 'on');
        finalize_dim_xaxis(ax4, fwd4, axmode4, nd4, 10, [-0.5, nd4 - 0.5]);
    else
        text(ax4, 0.5, 0.5, {'Ranking Info', 'Not Available'}, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Units', 'normalized');
    end

    % =====================================================================
    % Plot 5: Signal vs Noise variance
    % =====================================================================
    ax5 = nexttile(t, [1 2]);  % Row 1, columns 7-8
    if isfield(results, 'signalvar') && isfield(results, 'noisevar')
        if ~iscell(results.signalvar)
            % Global or averaged
            sv = results.signalvar;
            nv = results.noisevar;
            nd5 = length(sv);
            % Same head-tail log x-axis as ax4 (warp data onto a linear axis).
            [fwd5, axmode5] = dim_xaxis_transform(use_logscale, nd5, 10);
            if use_logscale, xr5 = 1:nd5; else, xr5 = 0:(nd5-1); end

            % Left y-axis for variance
            yyaxis(ax5, 'left');
            plot(ax5, fwd5(xr5), sv, '-', 'LineWidth', 1.5, 'Color', 'blue', 'DisplayName', 'Signal var');
            hold(ax5, 'on');
            plot(ax5, fwd5(xr5), nv, '-', 'LineWidth', 1.5, 'Color', [1 0.5 0], 'DisplayName', 'Noise var');
            plot(ax5, fwd5(xr5), nv / ntrials_avg, '-', 'LineWidth', 1.5, 'Color', [1 0.85 0.6], 'DisplayName', sprintf('Noise var / %.1f trials', ntrials_avg));
            ylabel(ax5, 'Variance');

            % Right y-axis for NCSNR
            yyaxis(ax5, 'right');
            % Rectify negative signal variance to 0 (cSb need not be strictly PSD);
            % avoids a complex sqrt and matches the standard NCSNR convention.
            % NCSNR is undefined where the projected noise variance underflows to
            % ~0: a signal-basis direction can land in a null direction of cNb, so
            % noisevar = u'*cNb*u ~ 0 while signalvar is appreciable. Dividing by the
            % bare +eps floor (sqrt(eps) ~ 1.5e-8) then explodes the trace to ~1e7.
            % Floor relative to the data and mask those dims so the line shows a gap
            % instead of a spurious spike.
            noise_floor = max(eps, 1e-8 * max(nv(:)));
            ncsnr_trace = sqrt(max(sv, 0)) ./ sqrt(max(nv, noise_floor));
            ncsnr_trace(nv < noise_floor) = NaN;
            plot(ax5, fwd5(xr5), ncsnr_trace, '-', 'LineWidth', 1.5, 'Color', 'magenta', 'DisplayName', 'NCSNR');
            ylabel(ax5, 'NCSNR');

            % Threshold (on left axis) -> legend entry only (matches Python; no rotated text).
            yyaxis(ax5, 'left');
            if isfield(results, 'best_threshold')
                if isscalar(results.best_threshold) && results.best_threshold > 0
                    xline(ax5, fwd5(results.best_threshold), 'r--', 'LineWidth', 2, ...
                          'DisplayName', thresh_legend_label(results.best_threshold, is_fullrank_wiener, true));
                elseif ~isscalar(results.best_threshold) && mean(results.best_threshold) > 0
                    mean_thresh = mean(results.best_threshold);
                    xline(ax5, fwd5(mean_thresh), 'r--', 'LineWidth', 2, ...
                          'DisplayName', sprintf('Mean threshold = %.1f', mean_thresh));
                end
            end
            hold(ax5, 'off');

            if is_fullrank_wiener
                xlabel(ax5, 'Dimension k (signal eigenbasis)');
            else
                xlabel(ax5, 'Dimension');
            end
            title(ax5, 'Signal and Noise Variance');
            legend_behind(ax5, 'Location', 'best', 'FontSize', 7);
            grid(ax5, 'on');
            finalize_dim_xaxis(ax5, fwd5, axmode5, nd5, 10, [-0.5, nd5 * 1.02]);
        else
            text(ax5, 0.5, 0.5, {'Per-Unit Variance', '(Averaged across units)'}, ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'Units', 'normalized');
        end
    else
        text(ax5, 0.5, 0.5, {'Variance Info', 'Not Available'}, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Units', 'normalized');
    end

    % =====================================================================
    % Plot 6: Objective function (row 2, col 1-2)
    % =====================================================================
    ax6 = nexttile(t, [1 2]);  % Row 2, columns 1-2
    % Match Python: only draw the objective curve when it actually exists and is
    % non-empty. Full-rank Wiener (criterion='wiener') leaves objective empty, so
    % we fall through to "Objective Not Available" instead of crashing on a
    % length-0 vector.
    if isfield(results, 'objective') && ~isempty(results.objective)
        % For log scale, we need to handle x=0 specially
        % Use x=0.5 for the zero-dimension case, then shift rest by 1
        % This allows log scale while preserving the "0 dims" data point
        zero_placeholder = 0.5;  % Position for "0 dimensions" in log scale

        % Objective trace/marker color, matched to the recovery panel (fig7) and the
        % basis the operating point was chosen on: blue = signal, green = difference
        % (default green otherwise).
        obj_basis = '';
        if isfield(results, 'threshold_selection') && isfield(results.threshold_selection, 'basis') ...
                && (ischar(results.threshold_selection.basis) || isstring(results.threshold_selection.basis))
            obj_basis = char(results.threshold_selection.basis);
        end
        if isempty(obj_basis) && isfield(opt, 'basis') && (ischar(opt.basis) || isstring(opt.basis))
            obj_basis = char(opt.basis);
        end
        switch obj_basis
            case 'signal';     obj_color = [0.122 0.467 0.706];
            case 'difference'; obj_color = [0.173 0.627 0.173];
            otherwise;         obj_color = [0.3 0.7 0.3];
        end

        % Check if unit-specific objectives are available
        if isfield(results, 'unit_objectives') && ~isempty(results.unit_objectives)
            % Unit-specific mode: use dual y-axes
            % Left axis: unit curves (gray) - use subsampling if needed
            yyaxis(ax6, 'left');
            hold(ax6, 'on');
            h_units = [];
            units_to_plot = subsample_idx;  % Use subsampled units
            for ii = 1:length(units_to_plot)
                u = units_to_plot(ii);
                curve_u = results.unit_objectives{u};
                if use_logscale
                    % x=0 -> zero_placeholder, x=1 -> 1, x=2 -> 2, etc.
                    x_unit = [zero_placeholder, 1:length(curve_u)-1];
                else
                    x_unit = 0:length(curve_u)-1;
                end
                h_units(end+1) = plot(ax6, x_unit, curve_u, '-', 'LineWidth', 0.5, 'Color', [0.5 0.5 0.5 0.3], 'Marker', 'none');
            end

            % Mark each unit's chosen threshold (on left axis) - use subsampling
            % Check if unit_groups are available for coloring
            unit_groups = [];
            if isfield(opt, 'unit_groups') && ~isempty(opt.unit_groups)
                unit_groups = opt.unit_groups;
                unique_groups = unique(unit_groups);
                n_groups = length(unique_groups);
                % Use hsv colormap which handles arbitrary numbers of groups
                % Scale from 0 to 0.9 to avoid wrapping back to red
                hsv_vals = hsv(ceil(n_groups / 0.9));
                group_colors = hsv_vals(1:n_groups, :);
                % Create mapping from group to color index
                group_to_idx = containers.Map(unique_groups, 1:n_groups);
            end

            if isfield(results, 'best_threshold')
                x_thresh = [];
                y_thresh = [];
                c_thresh = [];  % Colors for each threshold point
                for ii = 1:length(units_to_plot)
                    u = units_to_plot(ii);
                    k_u = results.best_threshold(u);
                    curve_u = results.unit_objectives{u};
                    if k_u >= 0 && k_u < length(curve_u)  % k_u=0 is valid (keep 0 dims)
                        if use_logscale
                            if k_u == 0
                                x_thresh(end+1) = zero_placeholder;
                            else
                                x_thresh(end+1) = k_u;
                            end
                        else
                            x_thresh(end+1) = k_u;
                        end
                        y_thresh(end+1) = curve_u(k_u+1);  % +1 for MATLAB 1-indexing

                        % Determine color for this point
                        if ~isempty(unit_groups) && u <= length(unit_groups)
                            c_thresh(end+1, :) = group_colors(group_to_idx(unit_groups(u)), :);
                        else
                            c_thresh(end+1, :) = [1 0.3 0.3];  % Default red
                        end
                    end
                end
                if ~isempty(x_thresh)
                    scatter(ax6, x_thresh, y_thresh, 20, c_thresh, 'filled', 'MarkerFaceAlpha', 0.6);
                end
            end
            ylabel(ax6, {'unit analytic recovery', '(cumsum signal - noise/t)'});
            set(ax6, 'YColor', [0.4 0.4 0.4]);

            % Right axis: population sum (green) - FULL population
            yyaxis(ax6, 'right');
            if use_logscale
                x_obj = [zero_placeholder, 1:length(results.objective)-1];
            else
                x_obj = 0:length(results.objective)-1;
            end
            h_sum = plot(ax6, x_obj, results.objective, 'LineWidth', 2, 'Color', obj_color);
            ylabel(ax6, 'analytic recovery (population)');
            set(ax6, 'YColor', obj_color);

            % Peak of population analytic recovery (triangle) + do-nothing box at
            % full retention. Per-unit chosen thresholds are the scatter dots above.
            objc = obj_color;
            obj = results.objective(:);
            n_dim = length(obj) - 1;
            [~, max_idx] = max(obj);
            kp = max_idx - 1;
            x_peak = kp; if use_logscale && kp == 0, x_peak = zero_placeholder; end
            hold(ax6, 'on');
            % Alpha-specific: shade the interpolation range (prediction peak -> chosen).
            if isfield(results, 'alpha_info') && ~isempty(results.alpha_info)
                x_lo = results.alpha_info.k_pred; if use_logscale && x_lo == 0, x_lo = zero_placeholder; end
                x_hi = results.alpha_info.k_var;  if use_logscale && x_hi == 0, x_hi = zero_placeholder; end
                if x_lo ~= x_hi
                    yl = [min(obj) max(obj)];
                    pad = 0.05 * (yl(2) - yl(1) + eps);
                    patch(ax6, [x_lo x_hi x_hi x_lo], ...
                          [yl(1)-pad yl(1)-pad yl(2)+pad yl(2)+pad], [0 0 1], ...
                          'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
                end
            end
            h_peak = plot(ax6, x_peak, obj(max_idx), '^', 'MarkerSize', 10, ...
                          'MarkerFaceColor', objc, 'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
            h_box = plot(ax6, n_dim, obj(end), 's', 'MarkerSize', 11, ...
                         'MarkerFaceColor', objc, 'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
            hold(ax6, 'off');

            % Add legend
            legend_behind(ax6, [h_units(1), h_sum, h_peak, h_box], ...
                   {'units', 'analytic recovery (population)', ...
                    sprintf('prediction peak (K=%d)', kp), ...
                    sprintf('trial-avg / do-nothing (K=%d)', n_dim)}, ...
                   'Location', 'best', 'FontSize', 7);

            title(ax6, ['Analytic recovery vs. dimensions (unit-specific)' subsample_suffix_units]);
        else
            % Global mode: single curve
            if use_logscale
                x_obj = [zero_placeholder, 1:length(results.objective)-1];
            else
                x_obj = 0:length(results.objective)-1;
            end
            plot(ax6, x_obj, results.objective, 'LineWidth', 1.5, 'Color', obj_color);

            % Markers (parity with Python): triangle = peak (prediction peak),
            % star = chosen threshold, box = trial-average / do-nothing (keep
            % every dimension). Curve-colored, distinguished by shape.
            objc = obj_color;
            obj = results.objective(:);
            n_dim = length(obj) - 1;
            hold(ax6, 'on');

            % Alpha-specific: shade the interpolation range (prediction peak -> chosen).
            if isfield(results, 'alpha_info') && ~isempty(results.alpha_info)
                x_lo = results.alpha_info.k_pred; if use_logscale && x_lo == 0, x_lo = zero_placeholder; end
                x_hi = results.alpha_info.k_var;  if use_logscale && x_hi == 0, x_hi = zero_placeholder; end
                if x_lo ~= x_hi
                    yl = [min(obj) max(obj)];
                    pad = 0.05 * (yl(2) - yl(1) + eps);
                    patch(ax6, [x_lo x_hi x_hi x_lo], ...
                          [yl(1)-pad yl(1)-pad yl(2)+pad yl(2)+pad], [0 0 1], ...
                          'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
                end
            end

            % Peak of the analytic recovery curve (prediction peak): triangle.
            [~, max_idx] = max(obj);
            kp = max_idx - 1;
            x_peak = kp; if use_logscale && kp == 0, x_peak = zero_placeholder; end
            h_peak = plot(ax6, x_peak, obj(max_idx), '^', 'MarkerSize', 10, ...
                          'MarkerFaceColor', objc, 'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
            leg_h = h_peak;
            leg_l = {sprintf('prediction peak (K=%d)', kp)};

            % Chosen threshold (may be constrained, so not necessarily the peak): star.
            if isfield(results, 'best_threshold') && isscalar(results.best_threshold)
                k = results.best_threshold;
                if k >= 0 && k < length(obj)
                    x_ch = k; if use_logscale && k == 0, x_ch = zero_placeholder; end
                    h_chosen = plot(ax6, x_ch, obj(k+1), 'p', 'MarkerSize', 16, ...
                                    'MarkerFaceColor', objc, 'MarkerEdgeColor', 'k', 'LineWidth', 0.9);
                    leg_h(end+1) = h_chosen;
                    leg_l{end+1} = sprintf('PSN chosen (K=%d)', k);
                end
            end

            % Trial-average / do-nothing (keep every dimension): box at full retention.
            h_box = plot(ax6, n_dim, obj(end), 's', 'MarkerSize', 11, ...
                         'MarkerFaceColor', objc, 'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
            leg_h(end+1) = h_box;
            leg_l{end+1} = sprintf('trial-avg / do-nothing (K=%d)', n_dim);

            hold(ax6, 'off');
            legend_behind(ax6, leg_h, leg_l, 'Location', 'best', 'FontSize', 7);

            if isfield(results, 'alpha_info') && ~isempty(results.alpha_info)
                title(ax6, sprintf('Analytic recovery vs. dimensions (alpha=%g)', results.alpha_info.alpha));
            else
                title(ax6, 'Analytic recovery vs. dimensions');
            end

            % Set ylabel based on criterion (only for global mode)
            if strcmp(criterion, 'variance')
                ylabel(ax6, 'cumulative signal variance');
            else
                ylabel(ax6, 'analytic recovery  (cumsum signal - noise/t)');
            end
        end

        xlabel(ax6, 'Number of Dimensions');
        grid(ax6, 'on');

        % Apply log scale and fix tick labels if needed
        if use_logscale
            set(ax6, 'XScale', 'log');
            % Set xlim to start at zero_placeholder (so "0" point is visible)
            n_dims = length(results.objective) - 1;  % max dimension
            xlim(ax6, [zero_placeholder * 0.8, n_dims * 1.1]);
            % Set custom ticks to show "0" at the zero_placeholder position
            % Choose nice tick values for log scale
            tick_vals = [zero_placeholder];
            tick_labels = {'0'};
            % Add powers of 10 and intermediate values
            log_ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
            for lt = log_ticks
                if lt <= n_dims
                    tick_vals(end+1) = lt;
                    tick_labels{end+1} = num2str(lt);
                end
            end
            set(ax6, 'XTick', tick_vals, 'XTickLabel', tick_labels);
        end
    else
        text(ax6, 0.5, 0.5, {'Objective', 'Not Available'}, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Units', 'normalized');
    end

    % =====================================================================
    % Recovery-tradeoff panel (Row 2, first slot): twin-axis analytic recovery
    % (solid, left) + split-half reliability (dashed, right). Matches Python.
    % =====================================================================
    ax_rec = nexttile(t, [1 3]);
    plot_recovery_tradeoff(ax_rec, results);

    % =====================================================================
    % Plot 7-9: Raw, Denoised, Noise
    % =====================================================================

    % Compute shared colorbar limits across all three plots
    all_data_789 = [trial_avg(:); denoised(:); noise(:)];
    if has_nans
        shared_mean = nanmean(all_data_789);
        shared_std = nanstd(all_data_789);
    else
        shared_mean = mean(all_data_789);
        shared_std = std(all_data_789);
    end
    if shared_std > 0
        clim_shared = [shared_mean - 2*shared_std, shared_mean + 2*shared_std];
    else
        clim_shared = [shared_mean - 1, shared_mean + 1];
    end

    % Plot 7: Raw trial-averaged data
    ax7 = nexttile(t, [1 3]);  % Row 2, input data
    imagesc(ax7, trial_avg, clim_shared);
    colormap(ax7, data_cmap);
    colorbar(ax7);
    if has_nans
        title(ax7, 'Input Data (trial-averaged, with NaNs)');
    else
        title(ax7, 'Input Data (trial-averaged)');
    end
    xlabel(ax7, 'Conditions');
    ylabel(ax7, 'Units');

    % Plot 8: Denoised data
    ax8 = nexttile(t, [1 3]);  % Row 2, denoised data
    imagesc(ax8, denoised, clim_shared);
    colormap(ax8, data_cmap);
    colorbar(ax8);
    title(ax8, 'PSN Denoised Data');
    xlabel(ax8, 'Conditions');
    ylabel(ax8, 'Units');

    % Plot 9: Noise (residual)
    ax9 = nexttile(t, [1 3]);  % Row 2, residual
    imagesc(ax9, noise, clim_shared);
    colormap(ax9, data_cmap);
    colorbar(ax9);
    if has_nans
        title(ax9, 'Residual (Noise, with NaNs)');
    else
        title(ax9, 'Residual (Noise)');
    end
    xlabel(ax9, 'Conditions');
    ylabel(ax9, 'Units');

    % =====================================================================
    % Plot 10: Denoiser matrix
    % =====================================================================
    ax10 = nexttile(t, [1 3]);  % Row 3, denoiser matrix
    % Compute symmetric colorbar limits around 0 (use 99th percentile for better contrast)
    data_absmax = prctile(abs(results.denoiser(:)), 99);
    if data_absmax > 0
        clim_10 = [-data_absmax, data_absmax];
    else
        clim_10 = [-1, 1];
    end
    imagesc(ax10, results.denoiser, clim_10);
    colormap(ax10, redblue);
    colorbar(ax10);
    title(ax10, 'Denoiser Matrix');
    xlabel(ax10, 'Units');
    ylabel(ax10, 'Units');
    axis(ax10, 'square');

    % =====================================================================
    % Plot 11-12: Traces (use subsampled units and/or conditions if needed)
    % =====================================================================
    % Color conditions by mean response (handle NaNs)
    % Use subsampled conditions for coloring
    n_conds_to_plot = length(subsample_cond_idx);
    cond_means = nanmean(trial_avg, 1);
    cond_means_sub = cond_means(subsample_cond_idx);
    [~, sorted_indices] = sort(cond_means_sub);
    colors = jet(n_conds_to_plot);
    trace_colors = zeros(n_conds_to_plot, 3);
    for rank = 1:n_conds_to_plot
        cond_idx = sorted_indices(rank);
        trace_colors(cond_idx, :) = colors(rank, :);
    end

    % Get subsampled data for traces (both units and conditions)
    trial_avg_sub = trial_avg(subsample_idx, subsample_cond_idx);
    denoised_sub = denoised(subsample_idx, subsample_cond_idx);

    % Trial-averaged traces
    ax11 = nexttile(t, [1 3]);  % Row 3, trial-averaged traces
    hold(ax11, 'on');
    x_units = 1:length(subsample_idx);  % 1-indexed for MATLAB
    for c = 1:n_conds_to_plot
        plot(ax11, x_units, trial_avg_sub(:, c), 'Color', trace_colors(c, :), 'LineWidth', 0.5);
    end
    hold(ax11, 'off');
    xlabel(ax11, 'Units');
    ylabel(ax11, 'Activity');
    title(ax11, ['Trial-Averaged Traces' subsample_suffix_traces]);
    grid(ax11, 'on');
    xlim(ax11, [min(x_units), max(x_units)]);

    % Denoised traces
    ax12 = nexttile(t, [1 3]);  % Row 3, denoised traces
    hold(ax12, 'on');
    for c = 1:n_conds_to_plot
        plot(ax12, x_units, denoised_sub(:, c), 'Color', trace_colors(c, :), 'LineWidth', 0.5);
    end
    hold(ax12, 'off');
    xlabel(ax12, 'Units');
    ylabel(ax12, 'Activity');
    title(ax12, ['PSN Denoised Traces' subsample_suffix_traces]);
    grid(ax12, 'on');
    xlim(ax12, [min(x_units), max(x_units)]);

    % Match y-limits (handle NaNs) - use subsampled data for ylim
    all_trace_data = [trial_avg_sub(:); denoised_sub(:)];
    if has_nans
        y_min = nanmin(all_trace_data);
        y_max = nanmax(all_trace_data);
    else
        y_min = min(all_trace_data);
        y_max = max(all_trace_data);
    end
    y_range = y_max - y_min;
    y_margin = y_range * 0.05;

    ylim(ax11, [y_min - y_margin, y_max + y_margin]);
    ylim(ax12, [y_min - y_margin, y_max + y_margin]);

    % =====================================================================
    % Plot 13: Split-half reliability (use subsampling for scatter, full pop for means)
    % =====================================================================
    ax13 = nexttile(t, [1 3]);  % Row 3, split-half reliability

    % Split trials by odd/even indices (interleaved) to handle NaN patterns
    % where later trials may have more NaNs due to variable repetition counts
    odd_idx = 1:2:ntrials;   % 1, 3, 5, ... (MATLAB is 1-indexed)
    even_idx = 2:2:ntrials;  % 2, 4, 6, ...
    data_A = data(:, :, odd_idx);
    data_B = data(:, :, even_idx);

    % Trial averages (use nanmean to handle NaNs)
    if has_nans
        tavg_A = nanmean(data_A, 3);
        tavg_B = nanmean(data_B, 3);
    else
        tavg_A = mean(data_A, 3);
        tavg_B = mean(data_B, 3);
    end

    % Denoise both splits
    denoiser = results.denoiser;
    unit_means = results.unit_means;

    % Apply the denoiser. ALWAYS use the transpose: it is correct for symmetric
    % global denoisers (denoiser' == denoiser) AND for the non-symmetric hybrid
    % and full-rank Wiener denoisers, which require the transpose. (The full-rank
    % Wiener denoiser is non-symmetric even though threshold_method is 'global',
    % so the old global-vs-other branch applied the wrong operator for Wiener.)
    % Matches psn.m and the Python figure.
    dn_A = denoiser' * (tavg_A - unit_means) + unit_means;
    dn_B = denoiser' * (tavg_B - unit_means) + unit_means;

    % Per-unit split-half metric: Pearson r (default) or MSE (split_half_metric).
    use_mse = isfield(opt, 'split_half_metric') && strcmp(opt.split_half_metric, 'mse');

    corr_tavg = nan(nunits, 1);
    corr_cross = nan(nunits, 1);
    corr_dn = nan(nunits, 1);

    for u = 1:nunits
        % Get valid (non-NaN) indices for this unit
        valid_tavg = ~isnan(tavg_A(u,:)) & ~isnan(tavg_B(u,:));
        valid_cross_AB = ~isnan(tavg_A(u,:)) & ~isnan(dn_B(u,:));
        valid_cross_BA = ~isnan(dn_A(u,:)) & ~isnan(tavg_B(u,:));
        valid_dn = ~isnan(dn_A(u,:)) & ~isnan(dn_B(u,:));

        if use_mse
            % Mean squared error per unit (matches Python use_mse branch).
            if sum(valid_tavg) > 0
                corr_tavg(u) = mean((tavg_A(u, valid_tavg) - tavg_B(u, valid_tavg)).^2);
            end
            if sum(valid_cross_AB) > 0 && sum(valid_cross_BA) > 0
                mse_AB = mean((tavg_A(u, valid_cross_AB) - dn_B(u, valid_cross_AB)).^2);
                mse_BA = mean((dn_A(u, valid_cross_BA) - tavg_B(u, valid_cross_BA)).^2);
                corr_cross(u) = (mse_AB + mse_BA) / 2;
            end
            if sum(valid_dn) > 0
                corr_dn(u) = mean((dn_A(u, valid_dn) - dn_B(u, valid_dn)).^2);
            end
        else
            % TAvg vs TAvg correlation
            if sum(valid_tavg) > 1 && std(tavg_A(u, valid_tavg)) > 0 && std(tavg_B(u, valid_tavg)) > 0
                corr_tavg(u) = corr(tavg_A(u, valid_tavg)', tavg_B(u, valid_tavg)');
            end

            % Cross-method (average both directions)
            if sum(valid_cross_AB) > 1 && sum(valid_cross_BA) > 1 && ...
               std(tavg_A(u, valid_cross_AB)) > 0 && std(dn_B(u, valid_cross_AB)) > 0 && ...
               std(dn_A(u, valid_cross_BA)) > 0 && std(tavg_B(u, valid_cross_BA)) > 0
                corr_AB = corr(tavg_A(u, valid_cross_AB)', dn_B(u, valid_cross_AB)');
                corr_BA = corr(dn_A(u, valid_cross_BA)', tavg_B(u, valid_cross_BA)');
                corr_cross(u) = (corr_AB + corr_BA) / 2;
            end

            % Denoised vs Denoised correlation
            if sum(valid_dn) > 1 && std(dn_A(u, valid_dn)) > 0 && std(dn_B(u, valid_dn)) > 0
                corr_dn(u) = corr(dn_A(u, valid_dn)', dn_B(u, valid_dn)');
            end
        end
    end

    % Subsampled correlations for plotting
    corr_tavg_sub = corr_tavg(subsample_idx);
    corr_cross_sub = corr_cross(subsample_idx);
    corr_dn_sub = corr_dn(subsample_idx);
    n_sub = length(subsample_idx);

    % Plot
    x_positions = [1, 2, 3];
    labels = {'TAvg vs TAvg', 'TAvg vs Denoised', 'Denoised vs Denoised'};

    % Add jitter for subsampled units (local RNG; leave global rng untouched)
    sj = RandStream('mt19937ar', 'Seed', 42);
    x_jitter_sub = (rand(sj, n_sub, 1) - 0.5) * 0.16;

    hold(ax13, 'on');

    % Connecting lines (subsampled)
    for ii = 1:n_sub
        values = [corr_tavg_sub(ii), corr_cross_sub(ii), corr_dn_sub(ii)];
        if ~any(isnan(values))
            plot(ax13, x_positions + x_jitter_sub(ii), values, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.3);
        end
    end

    % Scatter points (subsampled)
    scatter(ax13, x_positions(1) + x_jitter_sub, corr_tavg_sub, 15, 'blue', 'filled', 'MarkerFaceAlpha', 0.4);
    scatter(ax13, x_positions(2) + x_jitter_sub, corr_cross_sub, 15, [1 0.84 0], 'filled', 'MarkerFaceAlpha', 0.4);
    scatter(ax13, x_positions(3) + x_jitter_sub, corr_dn_sub, 15, [0.5 0.8 0.3], 'filled', 'MarkerFaceAlpha', 0.4);

    % Medians (FULL population). Use the median, not the mean, so these central
    % markers match the recovery-tradeoff panel's median per-unit split-half
    % reliability (and the Python figure). Variable names kept for brevity.
    mean_tavg = median(corr_tavg, 'omitnan');
    mean_cross = median(corr_cross, 'omitnan');
    mean_dn = median(corr_dn, 'omitnan');

    scatter(ax13, x_positions(1), mean_tavg, 100, 'blue', 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
    scatter(ax13, x_positions(2), mean_cross, 100, [1 0.84 0], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
    scatter(ax13, x_positions(3), mean_dn, 100, [0.2 0.6 0.2], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);

    % Labels (FULL population means)
    y_offset = 0.08;
    text(ax13, x_positions(1), mean_tavg + y_offset, sprintf('%.3f', mean_tavg), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
    text(ax13, x_positions(2), mean_cross + y_offset, sprintf('%.3f', mean_cross), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
    text(ax13, x_positions(3), mean_dn + y_offset, sprintf('%.3f', mean_dn), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

    hold(ax13, 'off');

    set(ax13, 'XTick', x_positions, 'XTickLabel', labels, 'XTickLabelRotation', 0);
    ax13.XAxis.FontSize = 7;  % Only set x-tick label font size smaller
    if use_mse
        ylabel(ax13, 'MSE');
    else
        ylabel(ax13, 'Pearson r');
    end
    if has_nans
        % Count actual valid trials per split
        valid_A = sum(~any(isnan(data_A), 1), 3);
        valid_B = sum(~any(isnan(data_B), 1), 3);
        avg_valid_A = mean(valid_A(valid_A > 0));
        avg_valid_B = mean(valid_B(valid_B > 0));
        title(ax13, sprintf('Split-Half Reliability\n(%.1f vs %.1f avg trials)%s', avg_valid_A, avg_valid_B, subsample_suffix_units));
    else
        title(ax13, sprintf('Split-Half Reliability\n(%d vs %d trials)%s', size(data_A,3), size(data_B,3), subsample_suffix_units));
    end
    grid(ax13, 'on');
    xlim(ax13, [0.5, 3.5]);
    yline(ax13, 0, 'k-', 'LineWidth', 1);

    % Set y-limits (use subsampled data for visualization range)
    all_corr = [corr_tavg_sub; corr_cross_sub; corr_dn_sub];
    valid_corr = all_corr(~isnan(all_corr));
    if ~isempty(valid_corr)
        y_min_c = min(valid_corr);
        y_max_c = max(valid_corr);
        y_range_c = y_max_c - y_min_c;
        y_pad = max(0.1, y_range_c * 0.15);
        ylim(ax13, [y_min_c - y_pad, y_max_c + y_pad]);
    else
        ylim(ax13, [-1, 1]);
    end

    % =====================================================================
    % Plot 14-17: Signal/Noise Diagnostics (use subsampling for scatter, full pop for means)
    % =====================================================================

    % Extract signal/noise variance data
    if isfield(results, 'svnv_before') && isfield(results, 'svnv_after')
        sv_before = results.svnv_before(:, 1);
        nv_before = results.svnv_before(:, 2);
        sv_after = results.svnv_after(:, 1);
        nv_after = results.svnv_after(:, 2);

        % Compute noise-ceiling SNR (NCSNR). Rectify negative signal variance
        % to 0 (standard NCSNR convention; avoids a complex sqrt).
        % Floor the denominator relative to the data (mirrors the per-dimension
        % NCSNR guard) so a unit whose noise variance underflows to ~0 cannot blow
        % the ratio up to ~1e7. Kept finite (no NaN) because the means/limits below
        % consume these.
        nfloor_before = max(eps, 1e-8 * max(nv_before(:)));
        nfloor_after = max(eps, 1e-8 * max(nv_after(:)));
        ncsnr_before = sqrt(max(sv_before, 0)) ./ sqrt(max(nv_before, nfloor_before));
        ncsnr_after = sqrt(max(sv_after, 0)) ./ sqrt(max(nv_after, nfloor_after));

        % Compute noise ceiling percentage (use ntrials_avg for NaN data)
        noiseceiling_before = 100 * (ncsnr_before.^2 ./ (ncsnr_before.^2 + 1/ntrials_avg));
        noiseceiling_after = 100 * (ncsnr_after.^2 ./ (ncsnr_after.^2 + 1/ntrials_avg));

        % Subsampled versions for plotting
        sv_before_sub = sv_before(subsample_idx);
        sv_after_sub = sv_after(subsample_idx);
        nv_before_sub = nv_before(subsample_idx);
        nv_after_sub = nv_after(subsample_idx);
        ncsnr_before_sub = ncsnr_before(subsample_idx);
        ncsnr_after_sub = ncsnr_after(subsample_idx);
        noiseceiling_before_sub = noiseceiling_before(subsample_idx);
        noiseceiling_after_sub = noiseceiling_after(subsample_idx);
        n_sub = length(subsample_idx);

        % Define x positions
        x_before = 1;
        x_after = 2;
        sjd = RandStream('mt19937ar', 'Seed', 43);  % local; leave global rng untouched
        x_jitter_diag = (rand(sjd, n_sub, 1) - 0.5) * 0.1;

        % Plot 14: Signal Variance
        ax14 = nexttile(t, [1 3]);  % Row 4, signal variance
        hold(ax14, 'on');
        for ii = 1:n_sub
            plot(ax14, [x_before, x_after] + x_jitter_diag(ii), [sv_before_sub(ii), sv_after_sub(ii)], ...
                'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
        end
        scatter(ax14, x_before + x_jitter_diag, sv_before_sub, 40, [0.3 0.5 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
        scatter(ax14, x_after + x_jitter_diag, sv_after_sub, 40, [0.8 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.6);

        % Means from FULL population
        mean_sv_before = mean(sv_before);
        mean_sv_after = mean(sv_after);
        scatter(ax14, x_before, mean_sv_before, 120, [0.1 0.3 0.6], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
        scatter(ax14, x_after, mean_sv_after, 120, [0.6 0.1 0.1], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);

        % Calculate y_offset dynamically (use subsampled for range)
        y_range_sv = max([sv_before_sub; sv_after_sub]) - min([sv_before_sub; sv_after_sub]);
        y_offset_sv = y_range_sv * 0.08;

        text(ax14, x_before, mean_sv_before + y_offset_sv, sprintf('%.3f', mean_sv_before), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
        text(ax14, x_after, mean_sv_after + y_offset_sv, sprintf('%.3f', mean_sv_after), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

        hold(ax14, 'off');
        xlim(ax14, [0.5, 2.5]);
        set(ax14, 'XTick', [1, 2], 'XTickLabel', {'Before', 'After'});
        ylabel(ax14, 'Signal Variance');
        title(ax14, ['Signal Variance' subsample_suffix_units]);
        grid(ax14, 'on');

        % Plot 15: Noise Variance
        ax15 = nexttile(t, [1 3]);  % Row 4, noise variance
        hold(ax15, 'on');
        for ii = 1:n_sub
            plot(ax15, [x_before, x_after] + x_jitter_diag(ii), [nv_before_sub(ii), nv_after_sub(ii)], ...
                'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
        end
        scatter(ax15, x_before + x_jitter_diag, nv_before_sub, 40, [0.3 0.5 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
        scatter(ax15, x_after + x_jitter_diag, nv_after_sub, 40, [0.8 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.6);

        % Means from FULL population
        mean_nv_before = mean(nv_before);
        mean_nv_after = mean(nv_after);
        scatter(ax15, x_before, mean_nv_before, 120, [0.1 0.3 0.6], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
        scatter(ax15, x_after, mean_nv_after, 120, [0.6 0.1 0.1], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);

        % Calculate y_offset dynamically
        y_range_nv = max([nv_before_sub; nv_after_sub]) - min([nv_before_sub; nv_after_sub]);
        y_offset_nv = y_range_nv * 0.08;

        text(ax15, x_before, mean_nv_before + y_offset_nv, sprintf('%.3f', mean_nv_before), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
        text(ax15, x_after, mean_nv_after + y_offset_nv, sprintf('%.3f', mean_nv_after), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

        hold(ax15, 'off');
        xlim(ax15, [0.5, 2.5]);
        set(ax15, 'XTick', [1, 2], 'XTickLabel', {'Before', 'After'});
        ylabel(ax15, 'Noise Variance / ntrials');
        title(ax15, ['Trial-Averaged Noise Variance' subsample_suffix_units]);
        grid(ax15, 'on');

        % Set unified ylims for both signal and noise variance plots (use subsampled for range)
        all_sv_vals_sub = [sv_before_sub; sv_after_sub];
        all_nv_vals_sub = [nv_before_sub; nv_after_sub];
        all_variance_vals_sub = [all_sv_vals_sub; all_nv_vals_sub];
        y_max_unified = max(all_variance_vals_sub);
        y_pad_unified = y_max_unified * 0.05;  % 5% padding at bottom
        unified_ylim = [-y_pad_unified, y_max_unified + y_max_unified * 0.15];

        % Apply unified limits to both subplots
        ylim(ax14, unified_ylim);
        yline(ax14, 0, 'k--', 'LineWidth', 0.5);

        ylim(ax15, unified_ylim);
        yline(ax15, 0, 'k--', 'LineWidth', 0.5);

        % Plot 16: NCSNR
        ax16 = nexttile(t, [1 3]);  % Row 4, NCSNR
        hold(ax16, 'on');
        for ii = 1:n_sub
            plot(ax16, [x_before, x_after] + x_jitter_diag(ii), [ncsnr_before_sub(ii), ncsnr_after_sub(ii)], ...
                'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
        end
        scatter(ax16, x_before + x_jitter_diag, ncsnr_before_sub, 40, [0.3 0.5 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
        scatter(ax16, x_after + x_jitter_diag, ncsnr_after_sub, 40, [0.8 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.6);

        % Means from FULL population
        mean_ncsnr_before = mean(ncsnr_before);
        mean_ncsnr_after = mean(ncsnr_after);
        scatter(ax16, x_before, mean_ncsnr_before, 120, [0.1 0.3 0.6], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
        scatter(ax16, x_after, mean_ncsnr_after, 120, [0.6 0.1 0.1], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);

        % Calculate y_offset dynamically
        y_range_ncsnr = max([ncsnr_before_sub; ncsnr_after_sub]) - min([ncsnr_before_sub; ncsnr_after_sub]);
        y_offset_ncsnr = y_range_ncsnr * 0.08;

        text(ax16, x_before, mean_ncsnr_before + y_offset_ncsnr, sprintf('%.3f', mean_ncsnr_before), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
        text(ax16, x_after, mean_ncsnr_after + y_offset_ncsnr, sprintf('%.3f', mean_ncsnr_after), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

        % Set ylims with padding (use subsampled for range)
        all_ncsnr_vals_sub = [ncsnr_before_sub; ncsnr_after_sub];
        y_max_ncsnr = max(all_ncsnr_vals_sub);
        y_pad_ncsnr_bottom = y_max_ncsnr * 0.05;  % 5% padding at bottom
        y_pad_ncsnr_top = y_max_ncsnr * 0.15;  % 15% padding at top
        ylim(ax16, [-y_pad_ncsnr_bottom, y_max_ncsnr + y_pad_ncsnr_top]);
        yline(ax16, 0, 'k--', 'LineWidth', 0.5);

        hold(ax16, 'off');
        xlim(ax16, [0.5, 2.5]);
        set(ax16, 'XTick', [1, 2], 'XTickLabel', {'Before', 'After'});
        ylabel(ax16, 'NCSNR');
        title(ax16, ['Noise Ceiling SNR (NCSNR)' subsample_suffix_units]);
        grid(ax16, 'on');

        % Plot 17: Noise Ceiling %
        ax17 = nexttile(t, [1 3]);  % Row 4, noise ceiling percentage
        hold(ax17, 'on');
        for ii = 1:n_sub
            plot(ax17, [x_before, x_after] + x_jitter_diag(ii), [noiseceiling_before_sub(ii), noiseceiling_after_sub(ii)], ...
                'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
        end
        scatter(ax17, x_before + x_jitter_diag, noiseceiling_before_sub, 40, [0.3 0.5 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
        scatter(ax17, x_after + x_jitter_diag, noiseceiling_after_sub, 40, [0.8 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.6);

        % Means from FULL population
        mean_nc_before = mean(noiseceiling_before);
        mean_nc_after = mean(noiseceiling_after);
        scatter(ax17, x_before, mean_nc_before, 120, [0.1 0.3 0.6], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
        scatter(ax17, x_after, mean_nc_after, 120, [0.6 0.1 0.1], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);

        % Fixed y_offset for noise ceiling (percentage scale 0-100)
        y_offset_nc = 100 * 0.08;

        text(ax17, x_before, mean_nc_before + y_offset_nc, sprintf('%.3f', mean_nc_before), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
        text(ax17, x_after, mean_nc_after + y_offset_nc, sprintf('%.3f', mean_nc_after), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

        hold(ax17, 'off');
        xlim(ax17, [0.5, 2.5]);
        ylim(ax17, [-5, 100]);  % Add negative padding to make yline at 0 visible
        yline(ax17, 0, 'k--', 'LineWidth', 0.5);

        set(ax17, 'XTick', [1, 2], 'XTickLabel', {'Before', 'After'});
        ylabel(ax17, 'Noise Ceiling (%)');
        if has_nans
            title(ax17, sprintf('Noise Ceiling Percentage (%.1f avg trials)%s', ntrials_avg, subsample_suffix_units));
        else
            title(ax17, sprintf('Noise Ceiling Percentage (%d trials)%s', ntrials, subsample_suffix_units));
        end
        grid(ax17, 'on');
    end

end


function s = thresh_legend_label(best_t, is_fullrank_wiener, with_formula)
% Legend label for the threshold line on ax4 (eigenvalues) and ax5, mirroring
% Python: full-rank Wiener shows tr(D); otherwise the integer dimension count K.
    if is_fullrank_wiener
        if with_formula
            s = sprintf('tr(D) = %.1f  where D = \\Sigma_S(\\Sigma_S + \\Sigma_N/t)^{-1}', best_t);
        else
            s = sprintf('tr(D) = %.1f', best_t);
        end
    else
        s = sprintf('Threshold K = %d', floor(best_t));
    end
end


function [fwd, axmode] = dim_xaxis_transform(use_logscale, ndims, tail)
% MATLAB has no matplotlib-style custom ('function') x-scale, so to reproduce
% Python's _set_headtail_log_xscale we WARP the data coordinates and plot on a
% plain linear axis. Returns:
%   fwd    - function handle: raw dimension index -> plotted x position
%   axmode - 'linear'    : no log; identity warp on a native linear axis
%            'nativelog' : few dims; identity warp on a native log axis
%            'warp'      : many dims; head-tail mirror-log warp on a linear axis
% Head-tail (tail=10): ordinary log for head dims 1..N-tail, then the final
% `tail` dims fan back out as the mirror image (inverse log) of the head so the
% leading AND trailing dims stay legible.
    if nargin < 3, tail = 10; end
    if ~use_logscale
        fwd = @(x) double(x);
        axmode = 'linear';
    elseif ndims <= 2*tail + 1
        fwd = @(x) double(x);
        axmode = 'nativelog';
    else
        N = double(ndims); j = N - tail;
        logj = log10(j); logt = log10(tail + 1.0);
        fwd = @(x) headtail_warp(double(x), N, j, logj, logt);
        axmode = 'warp';
    end
end


function y = headtail_warp(x, N, j, logj, logt)
% Forward warp matching Python _set_headtail_log_xscale.fwd: plain log on the
% head (x<=j); mirror-log on the tail (x>j) so the last `tail` dims fan out.
    head  = log10(max(x, 1e-9));
    tailv = logj + logt - log10(max(N - x + 1.0, 1e-3));
    y = head;
    m = x > j;
    y(m) = tailv(m);
end


function finalize_dim_xaxis(ax, fwd, axmode, ndims, tail, lin_xlim, zero_at)
% Apply axis scale, x-limits, ticks and (for 'warp') the axis-break mark, after
% the data have been plotted in warped coordinates.
%   lin_xlim - [lo hi] used only for axmode 'linear'
%   zero_at  - [] or a placeholder x (e.g. 0.5) that gets a '0' tick (ax6)
    if nargin < 7, zero_at = []; end
    switch axmode
        case 'linear'
            xlim(ax, lin_xlim);
        case 'nativelog'
            set(ax, 'XScale', 'log');
            if isempty(zero_at), lo = 0.8; else, lo = zero_at * 0.8; end
            xlim(ax, [lo, ndims + 0.5]);
            ax.XAxis.TickLabelRotation = 45;
        case 'warp'
            N = double(ndims); j = N - tail; logj = log10(j);
            if isempty(zero_at), left = 0.8; else, left = zero_at * 0.8; end
            xlim(ax, [fwd(left), fwd(N + 0.5)]);
            tvals = []; tlabs = {};
            if ~isempty(zero_at)
                tvals(end+1) = zero_at; tlabs{end+1} = '0';
            end
            cand = [1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 50000];
            for tk = cand
                if tk >= 1 && tk < j && (logj - log10(tk)) > 0.18
                    tvals(end+1) = tk; tlabs{end+1} = num2str(tk); %#ok<AGROW>
                end
            end
            jj = round(j); tvals(end+1) = jj; tlabs{end+1} = num2str(jj);
            if N > jj, tvals(end+1) = round(N); tlabs{end+1} = num2str(round(N)); end
            [uv, ia] = unique(tvals, 'stable');
            set(ax, 'XTick', fwd(uv), 'XTickLabel', tlabs(ia));
            ax.XAxis.TickLabelRotation = 45;
            draw_dim_axis_break(ax, fwd(j));
    end
end


function draw_dim_axis_break(ax, xw)
% Draw a small '//' break mark on the bottom spine at warped-x ``xw`` to signal
% the log -> mirror-log junction (mirrors Python _draw_axis_break).
    xl = get(ax, 'XLim');
    xa = (xw - xl(1)) / (xl(2) - xl(1));
    if ~(xa > 0 && xa < 1), return; end
    yl = get(ax, 'YLim');
    % Pin the limits to manual BEFORE drawing: the break straddles the bottom
    % spine, so its lower point sits just below yl(1). With auto limits MATLAB
    % would expand the y-axis to swallow it, dropping the spine and leaving the
    % marks floating above the axis. Pinning keeps them on the spine.
    ylim(ax, yl); xlim(ax, xl);
    d = 0.014;
    xspan = xl(2) - xl(1); yspan = yl(2) - yl(1);
    washold = ishold(ax); hold(ax, 'on');
    for off = [-0.6*d, 0.6*d]
        xx = xl(1) + (xa + [-d, d] + off) * xspan;
        yy = yl(1) + [-d, d] * yspan;
        plot(ax, xx, yy, 'k-', 'LineWidth', 1.0, 'Clipping', 'off', 'HandleVisibility', 'off');
    end
    if ~washold, hold(ax, 'off'); end
end
