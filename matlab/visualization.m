function visualization(data, results)
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

    % Create a large figure
    figure('Position', [100, 100, 1800, 1200]);

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

    % Create title
    if has_nans
        title_text = sprintf('Data: %d units × %d conditions × %d max trials (avg %.1f)  |  Basis: %s  |  Method: %s  |  Criterion: %s', ...
                            nunits, nconds, ntrials, ntrials_avg, basis_desc, threshold_method, criterion);
    else
        title_text = sprintf('Data: %d units × %d conditions × %d trials  |  Basis: %s  |  Method: %s  |  Criterion: %s', ...
                            nunits, nconds, ntrials, basis_desc, threshold_method, criterion);
    end
    sgtitle(title_text, 'FontSize', 12, 'FontWeight', 'bold');

    % Get trial-averaged and denoised data (use nanmean for NaN data)
    if has_nans
        trial_avg = nanmean(data, 3);
    else
        trial_avg = mean(data, 3);
    end
    denoised = results.denoiseddata;
    noise = trial_avg - denoised;

    % =====================================================================
    % Plot 1: Basis source matrix (covariance)
    % =====================================================================
    subplot(4, 4, 1);
    if isfield(results, 'gsn_result') && isfield(results.gsn_result, 'cSb')
        cSb = results.gsn_result.cSb;
        cNb = results.gsn_result.cNb;

        % Determine which matrix to show based on basis type
        if isfield(opt, 'basis')
            basis_type = opt.basis;
            if ischar(basis_type) || isstring(basis_type)
                basis_type = char(basis_type);

                switch basis_type
                    case 'difference'
                        % Show signal - noise/ntrials_avg
                        plot_matrix = cSb - cNb / ntrials_avg;
                        plot_title = sprintf('cSb - cNb/%.1f (difference)', ntrials_avg);

                    case 'noise'
                        % Show noise covariance
                        plot_matrix = cNb;
                        plot_title = 'Noise Covariance (cNb)';

                    case 'pca'
                        % Show covariance of trial-averaged data
                        trial_avg_demeaned = trial_avg - results.unit_means;
                        plot_matrix = cov(trial_avg_demeaned');
                        plot_title = 'Trial-Avg Data Covariance';

                    otherwise
                        % Default: signal covariance (for 'signal' and others)
                        plot_matrix = cSb;
                        plot_title = 'Signal Covariance (cSb)';
                end
            else
                % Custom basis matrix
                plot_matrix = cSb;
                plot_title = 'Signal Covariance (cSb)';
            end
        else
            % No basis specified, use signal
            plot_matrix = cSb;
            plot_title = 'Signal Covariance (cSb)';
        end

        if has_nans
            matrix_max = nanmax(abs(plot_matrix(:)));
        else
            matrix_max = max(abs(plot_matrix(:)));
        end
        if matrix_max > 0
            imagesc(plot_matrix, [-matrix_max, matrix_max]);
        else
            imagesc(plot_matrix);
        end
        colormap(gca, redblue);
        colorbar;
        title(plot_title);
        xlabel('Units');
        ylabel('Units');
        axis equal tight;
    else
        text(0.5, 0.5, 'Covariance\nNot Available', ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Units', 'normalized');
    end

    % =====================================================================
    % Plot 2: Full basis matrix
    % =====================================================================
    subplot(4, 4, 2);
    if isfield(results, 'fullbasis')
        if has_nans
            basis_max = nanmax(abs(results.fullbasis(:)));
        else
            basis_max = max(abs(results.fullbasis(:)));
        end
        if basis_max > 0
            imagesc(results.fullbasis, [-basis_max, basis_max]);
        else
            imagesc(results.fullbasis);
        end
        colormap(gca, redblue);
        colorbar;
        title('Full Basis Matrix');
        xlabel('Dimension');
        ylabel('Units');
        axis square;
    else
        text(0.5, 0.5, 'Basis\nNot Available', ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Units', 'normalized');
    end

    % =====================================================================
    % Plot 3: Signal variance spectrum (or PCA eigenvalues for PCA basis)
    % =====================================================================
    subplot(4, 4, 3);

    % Check if we're using PCA basis and have eigenvalues
    use_pca_eigenvalues = false;
    if isfield(opt, 'basis') && ischar(opt.basis) && strcmp(opt.basis, 'pca')
        if isfield(results, 'basis_eigenvalues') && ~isempty(results.basis_eigenvalues)
            use_pca_eigenvalues = true;
        end
    end

    % Determine if we should use log spacing (more than 50 units)
    use_log_x = nunits > 50;

    if use_pca_eigenvalues
        % For PCA basis: show PCA eigenvalues
        pca_evals = results.basis_eigenvalues;
        if use_log_x
            semilogx(1:length(pca_evals), pca_evals, 'LineWidth', 1.5, 'Color', [0.5, 0, 0.5]);
        else
            plot(1:length(pca_evals), pca_evals, 'LineWidth', 1.5, 'Color', [0.5, 0, 0.5]);
        end
        hold on;

        % Add threshold indicators
        if isfield(results, 'best_threshold')
            if isscalar(results.best_threshold) && results.best_threshold > 0
                xline(results.best_threshold, 'r--', 'LineWidth', 1.5, 'Label', sprintf('Thresh: %d', results.best_threshold));
            elseif ~isscalar(results.best_threshold)
                mean_thresh = mean(results.best_threshold);
                xline(mean_thresh, 'r--', 'LineWidth', 1.5, 'Label', sprintf('Mean: %.1f', mean_thresh));
            end
        end

        xlabel('Dimension');
        ylabel('PCA Eigenvalue');
        title('PCA Eigenspectrum');
        grid on;

    elseif isfield(results, 'signal_proj_viz')
        % Use original order (before ranking)
        signal_vars = results.signal_proj_viz;
        if use_log_x
            semilogx(1:length(signal_vars), signal_vars, 'LineWidth', 1.5, 'Color', 'blue');
        else
            plot(1:length(signal_vars), signal_vars, 'LineWidth', 1.5, 'Color', 'blue');
        end
        hold on;

        % Add threshold indicators
        if isfield(results, 'best_threshold')
            if isscalar(results.best_threshold) && results.best_threshold > 0
                xline(results.best_threshold, 'r--', 'LineWidth', 1.5, 'Label', sprintf('Thresh: %d', results.best_threshold));
            elseif ~isscalar(results.best_threshold)
                mean_thresh = mean(results.best_threshold);
                xline(mean_thresh, 'r--', 'LineWidth', 1.5, 'Label', sprintf('Mean: %.1f', mean_thresh));
            end
        end

        xlabel('Dimension');
        ylabel('Signal Variance');
        title('Signal Variance Spectrum');
        grid on;
    else
        text(0.5, 0.5, 'Signal Variance\nNot Available', ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Units', 'normalized');
    end

    % =====================================================================
    % Plot 4: Signal vs Noise variance
    % =====================================================================
    subplot(4, 4, 4);
    if isfield(results, 'signalvar') && isfield(results, 'noisevar')
        if ~iscell(results.signalvar)
            % Global or averaged
            % Left y-axis for variance
            yyaxis left;
            if use_log_x
                semilogx(1:length(results.signalvar), results.signalvar, '-', 'LineWidth', 1.5, 'Color', 'blue', 'DisplayName', 'Signal var');
                hold on;
                semilogx(1:length(results.noisevar), results.noisevar, '-', 'LineWidth', 1.5, 'Color', [1 0.5 0], 'DisplayName', 'Noise var');
            else
                plot(results.signalvar, '-', 'LineWidth', 1.5, 'Color', 'blue', 'DisplayName', 'Signal var');
                hold on;
                plot(results.noisevar, '-', 'LineWidth', 1.5, 'Color', [1 0.5 0], 'DisplayName', 'Noise var');
            end
            ylabel('Variance');

            % Right y-axis for NCSNR
            yyaxis right;
            ncsnr_trace = sqrt(results.signalvar) ./ sqrt(results.noisevar + eps);
            if use_log_x
                semilogx(1:length(ncsnr_trace), ncsnr_trace, '-', 'LineWidth', 1.5, 'Color', 'magenta', 'DisplayName', 'NCSNR');
            else
                plot(ncsnr_trace, '-', 'LineWidth', 1.5, 'Color', 'magenta', 'DisplayName', 'NCSNR');
            end
            ylabel('NCSNR');

            % Add threshold (on left axis)
            yyaxis left;
            if isfield(results, 'best_threshold')
                if isscalar(results.best_threshold)
                    xline(results.best_threshold, 'r--', 'LineWidth', 1, 'DisplayName', 'Threshold');
                else
                    xline(mean(results.best_threshold), 'r--', 'LineWidth', 1, 'DisplayName', 'Mean Threshold');
                end
            end

            xlabel('Dimension');
            title('Signal and Noise Variance');
            legend('Location', 'best');
            grid on;
        else
            text(0.5, 0.5, 'Per-Unit Variance\n(Averaged across units)', ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'Units', 'normalized');
        end
    else
        text(0.5, 0.5, 'Variance Info\nNot Available', ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Units', 'normalized');
    end

    % =====================================================================
    % Plot 5: Objective function
    % =====================================================================
    subplot(4, 4, 5);
    if isfield(results, 'objective')
        hold on;

        % Check if unit-specific objectives are available
        if isfield(results, 'unit_objectives') && ~isempty(results.unit_objectives)
            % Unit-specific mode: plot all unit curves
            for u = 1:length(results.unit_objectives)
                curve_u = results.unit_objectives{u};
                if use_log_x
                    semilogx(1:length(curve_u), curve_u, 'LineWidth', 0.5, 'Color', [0.5 0.5 0.5 0.3]);
                else
                    plot(0:length(curve_u)-1, curve_u, 'LineWidth', 0.5, 'Color', [0.5 0.5 0.5 0.3]);
                end
            end

            % Plot population average as thick line
            if use_log_x
                semilogx(1:length(results.objective), results.objective, 'LineWidth', 2, 'Color', [0.3 0.7 0.3]);
            else
                plot(0:length(results.objective)-1, results.objective, 'LineWidth', 2, 'Color', [0.3 0.7 0.3]);
            end

            % Mark each unit's chosen threshold
            if isfield(results, 'best_threshold')
                x_thresh = [];
                y_thresh = [];
                for u = 1:length(results.unit_objectives)
                    k_u = results.best_threshold(u);
                    curve_u = results.unit_objectives{u};
                    if k_u > 0 && k_u <= length(curve_u)
                        if use_log_x
                            x_thresh(end+1) = k_u + 1;  % 1-indexed for log plot
                        else
                            x_thresh(end+1) = k_u;
                        end
                        y_thresh(end+1) = curve_u(k_u+1);
                    end
                end
                if ~isempty(x_thresh)
                    scatter(x_thresh, y_thresh, 20, [1 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.6);
                end
            end

            title('Objective Function (unit-specific)');
        else
            % Global mode: single curve
            if use_log_x
                semilogx(1:length(results.objective), results.objective, 'LineWidth', 1.5, 'Color', [0.3 0.7 0.3]);
            else
                plot(0:length(results.objective)-1, results.objective, 'LineWidth', 1.5, 'Color', [0.3 0.7 0.3]);
            end

            % Mark maximum
            [~, max_idx] = max(results.objective);
            if use_log_x
                plot(max_idx, results.objective(max_idx), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
            else
                plot(max_idx-1, results.objective(max_idx), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
            end

            title('Objective Function');
        end

        xlabel('Number of Dimensions');

        % Set ylabel based on criterion
        if strcmp(criterion, 'variance')
            ylabel('Cumulative Variance');
        else
            ylabel('Cumulative Signal - Noise/ntrials');
        end

        grid on;
    else
        text(0.5, 0.5, 'Objective\nNot Available', ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'Units', 'normalized');
    end

    % =====================================================================
    % Plot 6-8: Raw, Denoised, Noise
    % =====================================================================
    all_data = [trial_avg(:); denoised(:); noise(:)];
    if has_nans
        max_abs_val = nanmax(abs(all_data));
    else
        max_abs_val = max(abs(all_data));
    end
    if max_abs_val > 0
        data_clim = [-max_abs_val, max_abs_val];
    else
        data_clim = [-1, 1];  % Default if all zeros
    end

    subplot(4, 4, 6);
    imagesc(trial_avg, data_clim);
    colormap(gca, redblue);
    colorbar;
    if has_nans
        title('Input Data (trial-averaged, with NaNs)');
    else
        title('Input Data (trial-averaged)');
    end
    xlabel('Conditions');
    ylabel('Units');

    subplot(4, 4, 7);
    imagesc(denoised, data_clim);
    colormap(gca, redblue);
    colorbar;
    title('PSN Denoised Data');
    xlabel('Conditions');
    ylabel('Units');

    subplot(4, 4, 8);
    imagesc(noise, data_clim);
    colormap(gca, redblue);
    colorbar;
    if has_nans
        title('Residual (Noise, with NaNs)');
    else
        title('Residual (Noise)');
    end
    xlabel('Conditions');
    ylabel('Units');

    % =====================================================================
    % Plot 9: Denoiser matrix
    % =====================================================================
    subplot(4, 4, 9);
    if has_nans
        denoiser_max = nanmax(abs(results.denoiser(:)));
    else
        denoiser_max = max(abs(results.denoiser(:)));
    end
    if denoiser_max > 0
        imagesc(results.denoiser, [-denoiser_max, denoiser_max]);
    else
        imagesc(results.denoiser);
    end
    colormap(gca, redblue);
    colorbar;
    title('Denoiser Matrix');
    xlabel('Units');
    ylabel('Units');
    axis square;

    % =====================================================================
    % Plot 10-11: Traces
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
    subplot(4, 4, 10);
    hold on;
    x_units = 0:nunits-1;
    for c = 1:nconds
        plot(x_units, trial_avg(:, c), 'Color', trace_colors(c, :), 'LineWidth', 0.5);
    end
    xlabel('Units');
    ylabel('Activity');
    title('Trial-Averaged Traces');
    grid on;
    xlim([min(x_units), max(x_units)]);

    % Denoised traces
    subplot(4, 4, 11);
    hold on;
    for c = 1:nconds
        plot(x_units, denoised(:, c), 'Color', trace_colors(c, :), 'LineWidth', 0.5);
    end
    xlabel('Units');
    ylabel('Activity');
    title('PSN Denoised Traces');
    grid on;
    xlim([min(x_units), max(x_units)]);

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

    subplot(4, 4, 10);
    ylim([y_min - y_margin, y_max + y_margin]);
    subplot(4, 4, 11);
    ylim([y_min - y_margin, y_max + y_margin]);

    % =====================================================================
    % Plot 12: Split-half reliability
    % =====================================================================
    subplot(4, 4, 12);

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

    hold on;

    % Connecting lines
    for u = 1:nunits
        values = [corr_tavg(u), corr_cross(u), corr_dn(u)];
        if ~any(isnan(values))
            plot(x_positions + x_jitter(u), values, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.3);
        end
    end

    % Scatter points
    scatter(x_positions(1) + x_jitter, corr_tavg, 15, 'blue', 'filled', 'MarkerFaceAlpha', 0.4);
    scatter(x_positions(2) + x_jitter, corr_cross, 15, [1 0.84 0], 'filled', 'MarkerFaceAlpha', 0.4);
    scatter(x_positions(3) + x_jitter, corr_dn, 15, [0.5 0.8 0.3], 'filled', 'MarkerFaceAlpha', 0.4);

    % Means
    mean_tavg = nanmean(corr_tavg);
    mean_cross = nanmean(corr_cross);
    mean_dn = nanmean(corr_dn);

    scatter(x_positions(1), mean_tavg, 100, 'blue', 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
    scatter(x_positions(2), mean_cross, 100, [1 0.84 0], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
    scatter(x_positions(3), mean_dn, 100, [0.2 0.6 0.2], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);

    % Labels
    y_offset = 0.08;
    text(x_positions(1), mean_tavg + y_offset, sprintf('%.3f', mean_tavg), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
    text(x_positions(2), mean_cross + y_offset, sprintf('%.3f', mean_cross), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
    text(x_positions(3), mean_dn + y_offset, sprintf('%.3f', mean_dn), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

    set(gca, 'XTick', x_positions, 'XTickLabel', labels, 'XTickLabelRotation', 0, 'FontSize', 7);
    ylabel('Pearson r');
    if has_nans
        % Count actual valid trials per split
        valid_A = sum(~any(isnan(data_A), 1), 3);
        valid_B = sum(~any(isnan(data_B), 1), 3);
        avg_valid_A = mean(valid_A(valid_A > 0));
        avg_valid_B = mean(valid_B(valid_B > 0));
        title(sprintf('Split-Half Reliability\n(%.1f vs %.1f avg trials)', avg_valid_A, avg_valid_B));
    else
        title(sprintf('Split-Half Reliability\n(%d vs %d trials)', size(data_A,3), size(data_B,3)));
    end
    grid on;
    xlim([0.5, 3.5]);
    yline(0, 'k-', 'LineWidth', 1);

    % Set y-limits
    all_corr = [corr_tavg; corr_cross; corr_dn];
    valid_corr = all_corr(~isnan(all_corr));
    if ~isempty(valid_corr)
        y_min_c = min(valid_corr);
        y_max_c = max(valid_corr);
        y_range_c = y_max_c - y_min_c;
        y_pad = max(0.1, y_range_c * 0.15);
        ylim([y_min_c - y_pad, y_max_c + y_pad]);
    else
        ylim([-1, 1]);
    end

    % =====================================================================
    % Plot 13-16: Signal/Noise Diagnostics
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

        % Plot 13: Signal Variance
        subplot(4, 4, 13);
        hold on;
        for u = 1:nunits
            plot([x_before, x_after] + x_jitter_diag(u), [sv_before(u), sv_after(u)], ...
                'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
        end
        scatter(x_before + x_jitter_diag, sv_before, 40, [0.3 0.5 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
        scatter(x_after + x_jitter_diag, sv_after, 40, [0.8 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.6);

        mean_sv_before = mean(sv_before);
        mean_sv_after = mean(sv_after);
        scatter(x_before, mean_sv_before, 120, [0.1 0.3 0.6], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
        scatter(x_after, mean_sv_after, 120, [0.6 0.1 0.1], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);

        % Calculate y_offset dynamically
        y_range_sv = max([sv_before; sv_after]) - min([sv_before; sv_after]);
        y_offset_sv = y_range_sv * 0.08;

        text(x_before, mean_sv_before + y_offset_sv, sprintf('%.3f', mean_sv_before), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
        text(x_after, mean_sv_after + y_offset_sv, sprintf('%.3f', mean_sv_after), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

        xlim([0.5, 2.5]);
        set(gca, 'XTick', [1, 2], 'XTickLabel', {'Before', 'After'});
        ylabel('Signal Variance');
        title('Signal Variance');
        grid on;

        % Plot 14: Noise Variance
        subplot(4, 4, 14);
        hold on;
        for u = 1:nunits
            plot([x_before, x_after] + x_jitter_diag(u), [nv_before(u), nv_after(u)], ...
                'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
        end
        scatter(x_before + x_jitter_diag, nv_before, 40, [0.3 0.5 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
        scatter(x_after + x_jitter_diag, nv_after, 40, [0.8 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.6);

        mean_nv_before = mean(nv_before);
        mean_nv_after = mean(nv_after);
        scatter(x_before, mean_nv_before, 120, [0.1 0.3 0.6], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
        scatter(x_after, mean_nv_after, 120, [0.6 0.1 0.1], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);

        % Calculate y_offset dynamically
        y_range_nv = max([nv_before; nv_after]) - min([nv_before; nv_after]);
        y_offset_nv = y_range_nv * 0.08;

        text(x_before, mean_nv_before + y_offset_nv, sprintf('%.3f', mean_nv_before), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
        text(x_after, mean_nv_after + y_offset_nv, sprintf('%.3f', mean_nv_after), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

        xlim([0.5, 2.5]);
        set(gca, 'XTick', [1, 2], 'XTickLabel', {'Before', 'After'});
        ylabel('Noise Variance');
        title('Noise Variance');
        grid on;

        % Set unified ylims for both signal and noise variance plots
        % Based on whichever has the bigger range
        % Start at 0 (or slightly below) since variance is non-negative
        all_sv_vals = [sv_before; sv_after];
        all_nv_vals = [nv_before; nv_after];
        all_variance_vals = [all_sv_vals; all_nv_vals];
        y_max_unified = max(all_variance_vals);
        y_pad_unified = y_max_unified * 0.05;  % 5% padding at bottom
        unified_ylim = [-y_pad_unified, y_max_unified + y_max_unified * 0.15];

        % Apply unified limits to both subplots
        subplot(4, 4, 13);
        ylim(unified_ylim);
        yline(0, 'k--', 'LineWidth', 0.5);

        subplot(4, 4, 14);
        ylim(unified_ylim);
        yline(0, 'k--', 'LineWidth', 0.5);

        % Plot 15: NCSNR
        subplot(4, 4, 15);
        hold on;
        for u = 1:nunits
            plot([x_before, x_after] + x_jitter_diag(u), [ncsnr_before(u), ncsnr_after(u)], ...
                'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
        end
        scatter(x_before + x_jitter_diag, ncsnr_before, 40, [0.3 0.5 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
        scatter(x_after + x_jitter_diag, ncsnr_after, 40, [0.8 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.6);

        mean_ncsnr_before = mean(ncsnr_before);
        mean_ncsnr_after = mean(ncsnr_after);
        scatter(x_before, mean_ncsnr_before, 120, [0.1 0.3 0.6], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
        scatter(x_after, mean_ncsnr_after, 120, [0.6 0.1 0.1], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);

        % Calculate y_offset dynamically
        y_range_ncsnr = max([ncsnr_before; ncsnr_after]) - min([ncsnr_before; ncsnr_after]);
        y_offset_ncsnr = y_range_ncsnr * 0.08;

        text(x_before, mean_ncsnr_before + y_offset_ncsnr, sprintf('%.3f', mean_ncsnr_before), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
        text(x_after, mean_ncsnr_after + y_offset_ncsnr, sprintf('%.3f', mean_ncsnr_after), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

        % Set ylims with padding
        % Start at 0 (or slightly below) since NCSNR is non-negative
        all_ncsnr_vals = [ncsnr_before; ncsnr_after];
        y_max_ncsnr = max(all_ncsnr_vals);
        y_pad_ncsnr_bottom = y_max_ncsnr * 0.05;  % 5% padding at bottom
        y_pad_ncsnr_top = y_max_ncsnr * 0.15;  % 15% padding at top
        ylim([-y_pad_ncsnr_bottom, y_max_ncsnr + y_pad_ncsnr_top]);
        yline(0, 'k--', 'LineWidth', 0.5);

        xlim([0.5, 2.5]);
        set(gca, 'XTick', [1, 2], 'XTickLabel', {'Before', 'After'});
        ylabel('NCSNR');
        title('Noise Ceiling SNR (NCSNR)');
        grid on;

        % Plot 16: Noise Ceiling %
        subplot(4, 4, 16);
        hold on;
        for u = 1:nunits
            plot([x_before, x_after] + x_jitter_diag(u), [noiseceiling_before(u), noiseceiling_after(u)], ...
                'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
        end
        scatter(x_before + x_jitter_diag, noiseceiling_before, 40, [0.3 0.5 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
        scatter(x_after + x_jitter_diag, noiseceiling_after, 40, [0.8 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.6);

        mean_nc_before = mean(noiseceiling_before);
        mean_nc_after = mean(noiseceiling_after);
        scatter(x_before, mean_nc_before, 120, [0.1 0.3 0.6], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
        scatter(x_after, mean_nc_after, 120, [0.6 0.1 0.1], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);

        % Fixed y_offset for noise ceiling (percentage scale 0-100)
        y_offset_nc = 100 * 0.08;

        text(x_before, mean_nc_before + y_offset_nc, sprintf('%.3f', mean_nc_before), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
        text(x_after, mean_nc_after + y_offset_nc, sprintf('%.3f', mean_nc_after), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

        xlim([0.5, 2.5]);
        ylim([0, 100]);
        yline(0, 'k--', 'LineWidth', 0.5);

        set(gca, 'XTick', [1, 2], 'XTickLabel', {'Before', 'After'});
        ylabel('Noise Ceiling (%)');
        if has_nans
            title(sprintf('Noise Ceiling Percentage (%.1f avg trials)', ntrials_avg));
        else
            title(sprintf('Noise Ceiling Percentage (%d trials)', ntrials));
        end
        grid on;
    end

end

% Red-Blue colormap
function cmap = redblue
    n = 256;
    cmap = zeros(n, 3);
    mid = ceil(n/2);

    % Blue to white
    cmap(1:mid, 1) = linspace(0, 1, mid);
    cmap(1:mid, 2) = linspace(0, 1, mid);
    cmap(1:mid, 3) = ones(mid, 1);

    % White to red
    cmap(mid+1:n, 1) = ones(n-mid, 1);
    cmap(mid+1:n, 2) = linspace(1, 0, n-mid);
    cmap(mid+1:n, 3) = linspace(1, 0, n-mid);
end
