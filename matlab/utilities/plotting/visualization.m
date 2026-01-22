function fig = visualization(data, results, varargin)
% VISUALIZATION Generate diagnostic figures for PSN denoising results (NEW API)
%
% This visualization works with the new PSN API and results structure.
%
% Parameters:
% -----------
% data : array [nunits x nconds x ntrials]
%     Training data used for denoising
% results : struct
%     Results structure from psn function
% varargin : optional name-value pairs
%     'Visible' - 'on' (default) or 'off' to control figure visibility
%
% Returns:
% --------
% fig : figure handle
%     Handle to the created figure (for saving)

    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'Visible', 'on');
    parse(p, varargin{:});

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

    sgtitle(title_text, 'FontSize', 14, 'FontWeight', 'bold');

    % Get trial-averaged and denoised data (use nanmean for NaN data)
    if has_nans
        trial_avg = nanmean(data, 3);
    else
        trial_avg = mean(data, 3);
    end
    denoised = results.denoiseddata;
    noise = trial_avg - denoised;

    % =====================================================================
    % Create tiledlayout to match Python GridSpec(4, 8)
    % Row 1: cSb(2), cNb(2), Top5 PCs(1), Eigenvalues(1), Sig/Noise var(2)
    % Row 2-4: standard 4x2 pattern (each subplot spans 2 columns)
    % =====================================================================
    t = tiledlayout(4, 8, 'TileSpacing', 'compact', 'Padding', 'compact');

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
    ax3 = nexttile(t, [1 1]);  % Row 1, column 5 (single column)
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

        xlabel(ax3, 'PC');
        ylabel(ax3, 'Units');
        title(ax3, 'Top 5 Basis Dims');
        xlim(ax3, [0.5, num_pcs + 0.5]);
        ylim(ax3, [1, nunits]);
        set(ax3, 'YDir', 'reverse');  % Flip y-axis to match heatmaps
        set(ax3, 'XTick', 1:num_pcs);
        grid(ax3, 'on');
    else
        text(ax3, 0.5, 0.5, {'Basis', 'Not Available'}, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Units', 'normalized');
    end

    % =====================================================================
    % Plot 4: Global dimension ranking (eigenvalues or signal variance) (half width)
    % =====================================================================
    ax4 = nexttile(t, [1 1]);  % Row 1, column 6 (single column)

    % Check if eigenvalues were used for ranking (and are available)
    use_eigenvalues = strcmp(basis_ordering, 'eigenvalues') && ...
                      isfield(results, 'basis_eigenvalues') && ...
                      ~isempty(results.basis_eigenvalues);

    if use_eigenvalues
        % Show eigenvalues (SORTED - what was actually used for ranking)
        evals = results.basis_eigenvalues;  % Already sorted in descending order
        plot(ax4, 0:length(evals)-1, evals, 'LineWidth', 1.5, 'Color', [0.5, 0, 0.5]);
        hold(ax4, 'on');

        % Add threshold indicators (only if threshold > 0)
        if isfield(results, 'best_threshold')
            if isscalar(results.best_threshold) && results.best_threshold > 0
                xline(ax4, results.best_threshold, 'r--', 'LineWidth', 1.5);
            elseif ~isscalar(results.best_threshold) && mean(results.best_threshold) > 0
                mean_thresh = mean(results.best_threshold);
                xline(ax4, mean_thresh, 'r--', 'LineWidth', 1.5);
                % Add rotated text annotation for mean threshold
                ylims = ylim(ax4);
                y_pos = ylims(1) + 0.7 * (ylims(2) - ylims(1));
                text(ax4, mean_thresh + 0.5, y_pos, sprintf('Mean: %.1f', mean_thresh), ...
                    'FontSize', 9, 'Color', 'r', 'Rotation', 90, ...
                    'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
            end
        end
        hold(ax4, 'off');

        xlabel(ax4, 'Dim');
        ylabel(ax4, 'Eigenvalue');
        title(ax4, 'Basis Eigenvalues');
        grid(ax4, 'on');

    elseif isfield(results, 'signalvar')
        % Show signal variance (SORTED - what was actually used for ranking)
        signal_vars = results.signalvar;  % Already sorted in descending order
        plot(ax4, 0:length(signal_vars)-1, signal_vars, 'LineWidth', 1.5, 'Color', 'blue');
        hold(ax4, 'on');

        % Add threshold indicators (only if threshold > 0)
        if isfield(results, 'best_threshold')
            if isscalar(results.best_threshold) && results.best_threshold > 0
                xline(ax4, results.best_threshold, 'r--', 'LineWidth', 1.5);
            elseif ~isscalar(results.best_threshold) && mean(results.best_threshold) > 0
                mean_thresh = mean(results.best_threshold);
                xline(ax4, mean_thresh, 'r--', 'LineWidth', 1.5);
                % Add rotated text annotation for mean threshold
                ylims = ylim(ax4);
                y_pos = ylims(1) + 0.7 * (ylims(2) - ylims(1));
                text(ax4, mean_thresh + 0.5, y_pos, sprintf('Mean: %.1f', mean_thresh), ...
                    'FontSize', 9, 'Color', 'r', 'Rotation', 90, ...
                    'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
            end
        end
        hold(ax4, 'off');

        xlabel(ax4, 'Dim');
        ylabel(ax4, 'Signal Var');
        title(ax4, 'Signal Variance');
        grid(ax4, 'on');
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
            % Left y-axis for variance
            yyaxis(ax5, 'left');
            x_dims = 0:length(results.signalvar)-1;
            plot(ax5, x_dims, results.signalvar, '-', 'LineWidth', 1.5, 'Color', 'blue', 'DisplayName', 'Signal var');
            hold(ax5, 'on');
            plot(ax5, x_dims, results.noisevar, '-', 'LineWidth', 1.5, 'Color', [1 0.5 0], 'DisplayName', 'Noise var');
            plot(ax5, x_dims, results.noisevar / ntrials_avg, '-', 'LineWidth', 1.5, 'Color', [1 0.85 0.6], 'DisplayName', sprintf('Noise var / %.1f trials', ntrials_avg));
            ylabel(ax5, 'Variance');

            % Right y-axis for NCSNR
            yyaxis(ax5, 'right');
            ncsnr_trace = sqrt(results.signalvar) ./ sqrt(results.noisevar + eps);
            plot(ax5, x_dims, ncsnr_trace, '-', 'LineWidth', 1.5, 'Color', 'magenta', 'DisplayName', 'NCSNR');
            ylabel(ax5, 'NCSNR');

            % Add threshold (on left axis, only if > 0)
            yyaxis(ax5, 'left');
            if isfield(results, 'best_threshold')
                if isscalar(results.best_threshold) && results.best_threshold > 0
                    xline(ax5, results.best_threshold, 'r--', 'LineWidth', 1, 'DisplayName', 'Threshold');
                elseif ~isscalar(results.best_threshold) && mean(results.best_threshold) > 0
                    xline(ax5, mean(results.best_threshold), 'r--', 'LineWidth', 1, 'DisplayName', 'Mean Threshold');
                end
            end
            hold(ax5, 'off');

            xlabel(ax5, 'Dimension');
            title(ax5, 'Signal and Noise Variance');
            legend(ax5, 'Location', 'best', 'FontSize', 7);
            grid(ax5, 'on');
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
    if isfield(results, 'objective')
        % Check if unit-specific objectives are available
        if isfield(results, 'unit_objectives') && ~isempty(results.unit_objectives)
            % Unit-specific mode: use dual y-axes
            % Left axis: unit curves (gray)
            yyaxis(ax6, 'left');
            hold(ax6, 'on');
            h_units = [];
            for u = 1:length(results.unit_objectives)
                curve_u = results.unit_objectives{u};
                x_unit = 0:length(curve_u)-1;
                h_units(end+1) = plot(ax6, x_unit, curve_u, '-', 'LineWidth', 0.5, 'Color', [0.5 0.5 0.5 0.3], 'Marker', 'none');
            end

            % Mark each unit's chosen threshold (on left axis)
            if isfield(results, 'best_threshold')
                x_thresh = [];
                y_thresh = [];
                for u = 1:length(results.unit_objectives)
                    k_u = results.best_threshold(u);
                    curve_u = results.unit_objectives{u};
                    if k_u >= 0 && k_u < length(curve_u)  % k_u=0 is valid (keep 0 dims)
                        x_thresh(end+1) = k_u;
                        y_thresh(end+1) = curve_u(k_u+1);  % +1 for MATLAB 1-indexing
                    end
                end
                if ~isempty(x_thresh)
                    scatter(ax6, x_thresh, y_thresh, 20, [1 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.6);
                end
            end
            ylabel(ax6, {'Unit-Specific Objective', '(SignalVar - NoiseVar/ntrials)'});
            set(ax6, 'YColor', [0.4 0.4 0.4]);

            % Right axis: population sum (green)
            yyaxis(ax6, 'right');
            x_obj = 0:length(results.objective)-1;
            h_sum = plot(ax6, x_obj, results.objective, 'LineWidth', 2, 'Color', [0.3 0.7 0.3]);
            ylabel(ax6, 'Population Objective');
            set(ax6, 'YColor', [0.3 0.7 0.3]);
            hold(ax6, 'off');

            % Add legend
            legend(ax6, [h_units(1), h_sum], {'Units', 'Population (=Global)'}, 'Location', 'best');

            title(ax6, 'Objective Function (unit-specific)');
        else
            % Global mode: single curve
            x_obj = 0:length(results.objective)-1;
            plot(ax6, x_obj, results.objective, 'LineWidth', 1.5, 'Color', [0.3 0.7 0.3]);

            % Mark chosen threshold (not maximum - threshold may be constrained)
            if isfield(results, 'best_threshold') && isscalar(results.best_threshold)
                k = results.best_threshold;
                if k >= 0 && k < length(results.objective)
                    hold(ax6, 'on');
                    plot(ax6, k, results.objective(k+1), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
                    hold(ax6, 'off');
                end
            else
                % Fallback to maximum if no threshold stored
                [~, max_idx] = max(results.objective);
                hold(ax6, 'on');
                plot(ax6, max_idx-1, results.objective(max_idx), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
                hold(ax6, 'off');
            end

            title(ax6, 'Objective Function');

            % Set ylabel based on criterion (only for global mode)
            if strcmp(criterion, 'variance')
                ylabel(ax6, 'Cumulative SignalVar');
            else
                ylabel(ax6, 'Cumulative SignalVar - NoiseVar/ntrials');
            end
        end

        xlabel(ax6, 'Number of Dimensions');
        grid(ax6, 'on');
    else
        text(ax6, 0.5, 0.5, {'Objective', 'Not Available'}, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Units', 'normalized');
    end

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
        clim_shared = [shared_mean - 3*shared_std, shared_mean + 3*shared_std];
    else
        clim_shared = [shared_mean - 1, shared_mean + 1];
    end

    % Plot 7: Raw trial-averaged data
    ax7 = nexttile(t, [1 2]);  % Row 2, columns 3-4
    imagesc(ax7, trial_avg, clim_shared);
    colormap(ax7, redblue);
    colorbar(ax7);
    if has_nans
        title(ax7, 'Input Data (trial-averaged, with NaNs)');
    else
        title(ax7, 'Input Data (trial-averaged)');
    end
    xlabel(ax7, 'Conditions');
    ylabel(ax7, 'Units');

    % Plot 8: Denoised data
    ax8 = nexttile(t, [1 2]);  % Row 2, columns 5-6
    imagesc(ax8, denoised, clim_shared);
    colormap(ax8, redblue);
    colorbar(ax8);
    title(ax8, 'PSN Denoised Data');
    xlabel(ax8, 'Conditions');
    ylabel(ax8, 'Units');

    % Plot 9: Noise (residual)
    ax9 = nexttile(t, [1 2]);  % Row 2, columns 7-8
    imagesc(ax9, noise, clim_shared);
    colormap(ax9, redblue);
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
    ax10 = nexttile(t, [1 2]);  % Row 3, columns 1-2
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
    % Plot 11-12: Traces
    % =====================================================================
    % Color conditions by mean response (handle NaNs)
    cond_means = nanmean(trial_avg, 1);
    [~, sorted_indices] = sort(cond_means);
    colors = jet(nconds);
    trace_colors = zeros(nconds, 3);
    for rank = 1:nconds
        cond_idx = sorted_indices(rank);
        trace_colors(cond_idx, :) = colors(rank, :);
    end

    % Trial-averaged traces
    ax11 = nexttile(t, [1 2]);  % Row 3, columns 3-4
    hold(ax11, 'on');
    x_units = 1:nunits;  % 1-indexed for MATLAB
    for c = 1:nconds
        plot(ax11, x_units, trial_avg(:, c), 'Color', trace_colors(c, :), 'LineWidth', 0.5);
    end
    hold(ax11, 'off');
    xlabel(ax11, 'Units');
    ylabel(ax11, 'Activity');
    title(ax11, 'Trial-Averaged Traces');
    grid(ax11, 'on');
    xlim(ax11, [min(x_units), max(x_units)]);

    % Denoised traces
    ax12 = nexttile(t, [1 2]);  % Row 3, columns 5-6
    hold(ax12, 'on');
    for c = 1:nconds
        plot(ax12, x_units, denoised(:, c), 'Color', trace_colors(c, :), 'LineWidth', 0.5);
    end
    hold(ax12, 'off');
    xlabel(ax12, 'Units');
    ylabel(ax12, 'Activity');
    title(ax12, 'PSN Denoised Traces');
    grid(ax12, 'on');
    xlim(ax12, [min(x_units), max(x_units)]);

    % Match y-limits (handle NaNs)
    all_trace_data = [trial_avg(:); denoised(:)];
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
    % Plot 13: Split-half reliability
    % =====================================================================
    ax13 = nexttile(t, [1 2]);  % Row 3, columns 7-8

    % Split trials
    half_idx = floor(ntrials / 2);
    data_A = data(:, :, 1:half_idx);
    data_B = data(:, :, half_idx+1:end);

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

    % Handle symmetric vs non-symmetric denoiser
    if strcmp(threshold_method, 'global')
        % Symmetric: standard multiplication
        dn_A = denoiser * (tavg_A - unit_means) + unit_means;
        dn_B = denoiser * (tavg_B - unit_means) + unit_means;
    else
        % Non-symmetric: transpose multiplication
        dn_A = denoiser' * (tavg_A - unit_means) + unit_means;
        dn_B = denoiser' * (tavg_B - unit_means) + unit_means;
    end

    % Compute correlations
    corr_tavg = zeros(nunits, 1);
    corr_cross = zeros(nunits, 1);
    corr_dn = zeros(nunits, 1);

    for u = 1:nunits
        if std(tavg_A(u,:)) > 0 && std(tavg_B(u,:)) > 0
            corr_tavg(u) = corr(tavg_A(u,:)', tavg_B(u,:)');
        else
            corr_tavg(u) = NaN;
        end

        % Cross-method (average both directions)
        if std(tavg_A(u,:)) > 0 && std(dn_B(u,:)) > 0 && std(dn_A(u,:)) > 0 && std(tavg_B(u,:)) > 0
            corr_cross(u) = (corr(tavg_A(u,:)', dn_B(u,:)') + corr(dn_A(u,:)', tavg_B(u,:)')) / 2;
        else
            corr_cross(u) = NaN;
        end

        if std(dn_A(u,:)) > 0 && std(dn_B(u,:)) > 0
            corr_dn(u) = corr(dn_A(u,:)', dn_B(u,:)');
        else
            corr_dn(u) = NaN;
        end
    end

    % Plot
    x_positions = [1, 2, 3];
    labels = {'TAvg vs TAvg', 'TAvg vs Denoised', 'Denoised vs Denoised'};

    % Add jitter
    rng(42);
    x_jitter = (rand(nunits, 1) - 0.5) * 0.16;

    hold(ax13, 'on');

    % Connecting lines
    for u = 1:nunits
        values = [corr_tavg(u), corr_cross(u), corr_dn(u)];
        if ~any(isnan(values))
            plot(ax13, x_positions + x_jitter(u), values, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.3);
        end
    end

    % Scatter points
    scatter(ax13, x_positions(1) + x_jitter, corr_tavg, 15, 'blue', 'filled', 'MarkerFaceAlpha', 0.4);
    scatter(ax13, x_positions(2) + x_jitter, corr_cross, 15, [1 0.84 0], 'filled', 'MarkerFaceAlpha', 0.4);
    scatter(ax13, x_positions(3) + x_jitter, corr_dn, 15, [0.5 0.8 0.3], 'filled', 'MarkerFaceAlpha', 0.4);

    % Means
    mean_tavg = nanmean(corr_tavg);
    mean_cross = nanmean(corr_cross);
    mean_dn = nanmean(corr_dn);

    scatter(ax13, x_positions(1), mean_tavg, 100, 'blue', 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
    scatter(ax13, x_positions(2), mean_cross, 100, [1 0.84 0], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
    scatter(ax13, x_positions(3), mean_dn, 100, [0.2 0.6 0.2], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);

    % Labels
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
    ylabel(ax13, 'Pearson r');
    if has_nans
        % Count actual valid trials per split
        valid_A = sum(~any(isnan(data_A), 1), 3);
        valid_B = sum(~any(isnan(data_B), 1), 3);
        avg_valid_A = mean(valid_A(valid_A > 0));
        avg_valid_B = mean(valid_B(valid_B > 0));
        title(ax13, sprintf('Split-Half Reliability\n(%.1f vs %.1f avg trials)', avg_valid_A, avg_valid_B));
    else
        title(ax13, sprintf('Split-Half Reliability\n(%d vs %d trials)', size(data_A,3), size(data_B,3)));
    end
    grid(ax13, 'on');
    xlim(ax13, [0.5, 3.5]);
    yline(ax13, 0, 'k-', 'LineWidth', 1);

    % Set y-limits
    all_corr = [corr_tavg; corr_cross; corr_dn];
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
    % Plot 14-17: Signal/Noise Diagnostics
    % =====================================================================

    % Extract signal/noise variance data
    if isfield(results, 'svnv_before') && isfield(results, 'svnv_after')
        sv_before = results.svnv_before(:, 1);
        nv_before = results.svnv_before(:, 2);
        sv_after = results.svnv_after(:, 1);
        nv_after = results.svnv_after(:, 2);

        % Compute noise-corrected SNR (ncsnr)
        ncsnr_before = sqrt(sv_before) ./ sqrt(nv_before + eps);
        ncsnr_after = sqrt(sv_after) ./ sqrt(nv_after + eps);

        % Compute noise ceiling percentage (use ntrials_avg for NaN data)
        noiseceiling_before = 100 * (ncsnr_before.^2 ./ (ncsnr_before.^2 + 1/ntrials_avg));
        noiseceiling_after = 100 * (ncsnr_after.^2 ./ (ncsnr_after.^2 + 1/ntrials_avg));

        % Define x positions
        x_before = 1;
        x_after = 2;
        x_jitter_diag = (rand(nunits, 1) - 0.5) * 0.1;

        % Plot 14: Signal Variance
        ax14 = nexttile(t, [1 2]);  % Row 4, columns 1-2
        hold(ax14, 'on');
        for u = 1:nunits
            plot(ax14, [x_before, x_after] + x_jitter_diag(u), [sv_before(u), sv_after(u)], ...
                'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
        end
        scatter(ax14, x_before + x_jitter_diag, sv_before, 40, [0.3 0.5 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
        scatter(ax14, x_after + x_jitter_diag, sv_after, 40, [0.8 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.6);

        mean_sv_before = mean(sv_before);
        mean_sv_after = mean(sv_after);
        scatter(ax14, x_before, mean_sv_before, 120, [0.1 0.3 0.6], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
        scatter(ax14, x_after, mean_sv_after, 120, [0.6 0.1 0.1], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);

        % Calculate y_offset dynamically
        y_range_sv = max([sv_before; sv_after]) - min([sv_before; sv_after]);
        y_offset_sv = y_range_sv * 0.08;

        text(ax14, x_before, mean_sv_before + y_offset_sv, sprintf('%.3f', mean_sv_before), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
        text(ax14, x_after, mean_sv_after + y_offset_sv, sprintf('%.3f', mean_sv_after), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

        hold(ax14, 'off');
        xlim(ax14, [0.5, 2.5]);
        set(ax14, 'XTick', [1, 2], 'XTickLabel', {'Before', 'After'});
        ylabel(ax14, 'Signal Variance');
        title(ax14, 'Signal Variance');
        grid(ax14, 'on');

        % Plot 15: Noise Variance
        ax15 = nexttile(t, [1 2]);  % Row 4, columns 3-4
        hold(ax15, 'on');
        for u = 1:nunits
            plot(ax15, [x_before, x_after] + x_jitter_diag(u), [nv_before(u), nv_after(u)], ...
                'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
        end
        scatter(ax15, x_before + x_jitter_diag, nv_before, 40, [0.3 0.5 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
        scatter(ax15, x_after + x_jitter_diag, nv_after, 40, [0.8 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.6);

        mean_nv_before = mean(nv_before);
        mean_nv_after = mean(nv_after);
        scatter(ax15, x_before, mean_nv_before, 120, [0.1 0.3 0.6], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
        scatter(ax15, x_after, mean_nv_after, 120, [0.6 0.1 0.1], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);

        % Calculate y_offset dynamically
        y_range_nv = max([nv_before; nv_after]) - min([nv_before; nv_after]);
        y_offset_nv = y_range_nv * 0.08;

        text(ax15, x_before, mean_nv_before + y_offset_nv, sprintf('%.3f', mean_nv_before), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
        text(ax15, x_after, mean_nv_after + y_offset_nv, sprintf('%.3f', mean_nv_after), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

        hold(ax15, 'off');
        xlim(ax15, [0.5, 2.5]);
        set(ax15, 'XTick', [1, 2], 'XTickLabel', {'Before', 'After'});
        ylabel(ax15, 'Noise Variance / ntrials');
        title(ax15, 'Trial-Averaged Noise Variance');
        grid(ax15, 'on');

        % Set unified ylims for both signal and noise variance plots
        all_sv_vals = [sv_before; sv_after];
        all_nv_vals = [nv_before; nv_after];
        all_variance_vals = [all_sv_vals; all_nv_vals];
        y_max_unified = max(all_variance_vals);
        y_pad_unified = y_max_unified * 0.05;  % 5% padding at bottom
        unified_ylim = [-y_pad_unified, y_max_unified + y_max_unified * 0.15];

        % Apply unified limits to both subplots
        ylim(ax14, unified_ylim);
        yline(ax14, 0, 'k--', 'LineWidth', 0.5);

        ylim(ax15, unified_ylim);
        yline(ax15, 0, 'k--', 'LineWidth', 0.5);

        % Plot 16: NCSNR
        ax16 = nexttile(t, [1 2]);  % Row 4, columns 5-6
        hold(ax16, 'on');
        for u = 1:nunits
            plot(ax16, [x_before, x_after] + x_jitter_diag(u), [ncsnr_before(u), ncsnr_after(u)], ...
                'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
        end
        scatter(ax16, x_before + x_jitter_diag, ncsnr_before, 40, [0.3 0.5 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
        scatter(ax16, x_after + x_jitter_diag, ncsnr_after, 40, [0.8 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.6);

        mean_ncsnr_before = mean(ncsnr_before);
        mean_ncsnr_after = mean(ncsnr_after);
        scatter(ax16, x_before, mean_ncsnr_before, 120, [0.1 0.3 0.6], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
        scatter(ax16, x_after, mean_ncsnr_after, 120, [0.6 0.1 0.1], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);

        % Calculate y_offset dynamically
        y_range_ncsnr = max([ncsnr_before; ncsnr_after]) - min([ncsnr_before; ncsnr_after]);
        y_offset_ncsnr = y_range_ncsnr * 0.08;

        text(ax16, x_before, mean_ncsnr_before + y_offset_ncsnr, sprintf('%.3f', mean_ncsnr_before), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
        text(ax16, x_after, mean_ncsnr_after + y_offset_ncsnr, sprintf('%.3f', mean_ncsnr_after), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

        % Set ylims with padding
        all_ncsnr_vals = [ncsnr_before; ncsnr_after];
        y_max_ncsnr = max(all_ncsnr_vals);
        y_pad_ncsnr_bottom = y_max_ncsnr * 0.05;  % 5% padding at bottom
        y_pad_ncsnr_top = y_max_ncsnr * 0.15;  % 15% padding at top
        ylim(ax16, [-y_pad_ncsnr_bottom, y_max_ncsnr + y_pad_ncsnr_top]);
        yline(ax16, 0, 'k--', 'LineWidth', 0.5);

        hold(ax16, 'off');
        xlim(ax16, [0.5, 2.5]);
        set(ax16, 'XTick', [1, 2], 'XTickLabel', {'Before', 'After'});
        ylabel(ax16, 'NCSNR');
        title(ax16, 'Noise Ceiling SNR (NCSNR)');
        grid(ax16, 'on');

        % Plot 17: Noise Ceiling %
        ax17 = nexttile(t, [1 2]);  % Row 4, columns 7-8
        hold(ax17, 'on');
        for u = 1:nunits
            plot(ax17, [x_before, x_after] + x_jitter_diag(u), [noiseceiling_before(u), noiseceiling_after(u)], ...
                'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
        end
        scatter(ax17, x_before + x_jitter_diag, noiseceiling_before, 40, [0.3 0.5 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
        scatter(ax17, x_after + x_jitter_diag, noiseceiling_after, 40, [0.8 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.6);

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
            title(ax17, sprintf('Noise Ceiling Percentage (%.1f avg trials)', ntrials_avg));
        else
            title(ax17, sprintf('Noise Ceiling Percentage (%d trials)', ntrials));
        end
        grid(ax17, 'on');
    end

end
