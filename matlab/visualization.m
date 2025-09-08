function plot_diagnostic_figures(data, results, test_data)
% PLOT_DIAGNOSTIC_FIGURES Generate diagnostic figures for PSN denoising results.
%
% Generates comprehensive diagnostic plots showing the denoising process,
% including basis matrices, eigenspectrum, cross-validation results, and
% performance evaluation.
%
% Parameters:
% -----------
% data : array
%     Training data used for denoising, shape [nunits x nconds x ntrials]
% results : struct
%     Results structure from psn function
% test_data : array, optional
%     Data to use for testing in the bottom row plots, shape [nunits x nconds x ntrials].
%     If not provided, will use leave-one-out cross-validation on the training data.

    % Set random seed for reproducibility
    rng('default');
    rng(42, 'twister');
    
    % Create a single large figure with proper spacing
    figure('Position', [100, 100, 1600, 960]);
    
    % Extract data dimensions
    [nunits, nconds, ntrials] = size(data);
    
    % Add text at the top of the figure
    V_type = results.V;  % Get V directly from results
    if isnumeric(V_type) && ~isscalar(V_type)
        V_desc = sprintf('user-supplied [%dx%d]', size(V_type, 1), size(V_type, 2));
    else
        V_desc = num2str(V_type);
    end
        
    % Create title text with data shape and PSN application info
    title_text = sprintf('Data shape: %d units × %d conditions × %d trials    |    V = %s\n', ...
                        nunits, nconds, ntrials, V_desc);
    
    % Add cv_mode and magnitude thresholding info to title
    if isfield(results, 'opt') && isfield(results.opt, 'cv_mode')
        cv_mode = results.opt.cv_mode;
        if cv_mode == -1
            mag_type = results.opt.mag_type;
            mag_frac = results.opt.mag_frac;
            mag_frac_str = sprintf('%.3f', mag_frac);
            mag_frac_str = regexprep(mag_frac_str, '0*$', '');
            mag_frac_str = regexprep(mag_frac_str, '\.$', '');
            title_text = sprintf(['Data shape: %d units × %d conditions × %d trials    |    ' ...
                         'V = %s    |    cv_mode = %d    |    ' ...
                         'mag_type = %d, mag_frac = %s\n'], ...
                        nunits, nconds, ntrials, V_desc, cv_mode, mag_type, mag_frac_str);
        else
            threshold_per = results.opt.cv_threshold_per;
            title_text = sprintf(['Data shape: %d units × %d conditions × %d trials    |    ' ...
                         'V = %s    |    cv_mode = %d    |    thresh = %s\n'], ...
                        nunits, nconds, ntrials, V_desc, cv_mode, threshold_per);
        end
    end
    
    if nargin < 3 || isempty(test_data)
        title_text = [title_text sprintf('psn applied to all %d trials', ntrials)];
    else
        title_text = [title_text sprintf('psn applied to %d trials, tested on 1 heldout trial', ntrials)];
    end
    
    sgtitle(title_text, 'FontSize', 14);

    % Get raw and denoised data
    if isfield(results, 'opt') && isfield(results.opt, 'denoisingtype') && results.opt.denoisingtype == 0
        raw_data = mean(data, 3);  % Average across trials for trial-averaged denoising
        denoised_data = results.denoiseddata;
    else
        % For single-trial denoising, we'll plot the first trial
        if ndims(data) == 3
            raw_data = data(:, :, 1);
        else
            raw_data = data;
        end
        if ndims(results.denoiseddata) == 3
            denoised_data = results.denoiseddata(:, :, 1);
        else
            denoised_data = results.denoiseddata;
        end
    end

    % Compute noise as difference
    noise = raw_data - denoised_data;

    % Initialize lists for basis dimension analysis
    ncsnrs = [];
    sigvars = [];
    noisevars = [];

    if isfield(results, 'fullbasis') && isfield(results, 'mags')
        % Project data into basis
        data_reshaped = permute(data, [2, 3, 1]);  % [nconds x ntrials x nunits]
        eigvecs = results.fullbasis;
        for i = 1:size(eigvecs, 2)
            this_eigv = eigvecs(:, i);
            proj_data = zeros(nconds, ntrials);
            for j = 1:nconds
                for k = 1:ntrials
                    proj_data(j, k) = dot(squeeze(data_reshaped(j, k, :)), this_eigv);
                end
            end
            
            [noiseceiling, ncsnr, sigvar, noisevar] = compute_noise_ceiling(reshape(proj_data, [1, nconds, ntrials]));
            ncsnrs(end+1) = ncsnr;
            sigvars(end+1) = sigvar;
            noisevars(end+1) = noisevar;
        end

        % Convert to column vectors
        sigvars = sigvars(:);
        ncsnrs = ncsnrs(:);
        noisevars = noisevars(:);
        S = results.mags(:);
        if isfield(results, 'opt')
            opt = results.opt;
        else
            opt = struct();
        end
        if isfield(results, 'best_threshold')
            best_threshold = results.best_threshold;
        else
            if isfield(results, 'dimsretained')
                best_threshold = results.dimsretained;
            else
                best_threshold = [];
            end
        end

        % Plot 1: basis source matrix (top left)
        subplot(3, 4, 1);
        V = results.V;
        
        if isnumeric(V) && isscalar(V)
            if ismember(V, [0, 1, 2, 3])
                % Show the basis source matrix
                if isfield(results, 'basis_source') && ~isempty(results.basis_source)
                    matrix_to_show = results.basis_source;
                    if V == 0
                        title_str = 'GSN Signal Covariance (cSb)';
                    elseif V == 1
                        title_str = sprintf('GSN Transformed Signal Cov\n(inv(cNb)*cSb)');
                    elseif V == 2
                        title_str = 'GSN Noise Covariance (cNb)';
                    else  % V == 3
                        title_str = sprintf('Naive Trial-avg Data\nCovariance');
                    end
                    
                    matrix_max = max(abs(matrix_to_show(:)));
                    
                    imagesc(matrix_to_show, [-matrix_max, matrix_max]);
                    colormap(gca, redblue);
                    colorbar;
                    title(title_str);
                    xlabel('Units');
                    ylabel('Units');
                    axis equal tight;
                else
                    text(0.5, 0.5, sprintf('Covariance Matrix\nNot Available for V=%d', V), ...
                         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                         'Units', 'normalized');
                    title('');
                end
            else  % V == 4
                text(0.5, 0.5, sprintf('Random Basis\n(No Matrix to Show)'), ...
                     'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                     'Units', 'normalized');
                title('');
            end
        elseif isnumeric(V)
            % Handle case where V is a matrix
            text(0.5, 0.5, sprintf('User-Supplied Basis\nShape: [%dx%d]', size(V, 1), size(V, 2)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'Units', 'normalized');
            title('User-Supplied Basis');
        else
            % Handle any other case
            text(0.5, 0.5, 'No Basis Information Available', ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'Units', 'normalized');
            title('');
        end

        % Plot 2: Full basis matrix (top middle-left)
        subplot(3, 4, 2);
        basis_max = max(abs(results.fullbasis(:)));
        imagesc(results.fullbasis, [-basis_max, basis_max]);
        colormap(gca, redblue);
        colorbar;
        title('Full Basis Matrix');
        xlabel('Dimension');
        ylabel('Units');
        
        % Plot 3: Eigenspectrum (top middle)
        subplot(3, 4, 3);
        plot(S, 'LineWidth', 1, 'Color', 'blue');
        hold on;
        
        % Calculate and plot threshold indicators based on mode
        if isfield(opt, 'cv_mode')
            cv_mode = opt.cv_mode;
        else
            cv_mode = 0;
        end
        
        if isfield(opt, 'cv_threshold_per')
            cv_threshold_per = opt.cv_threshold_per;
        else
            cv_threshold_per = 'unit';
        end
        
        if isfield(opt, 'mag_type')
            mag_type = opt.mag_type;
        else
            mag_type = 0;
        end
        
        if cv_mode >= 0  % Cross-validation mode
            if strcmp(cv_threshold_per, 'population')
                % Single line for population threshold
                if isnumeric(best_threshold) && length(best_threshold) > 1
                    best_threshold = best_threshold(1);  % Take first value if array
                end
                xline(best_threshold, 'r--', 'LineWidth', 1, ...
                      'DisplayName', sprintf('Population threshold: %d dims', best_threshold));
            else  % Unit mode
                % Mean line and asterisks for unit-specific thresholds
                if isnumeric(best_threshold) && length(best_threshold) > 1
                    mean_threshold = mean(best_threshold);
                    xline(mean_threshold, 'r--', 'LineWidth', 1, ...
                          'DisplayName', sprintf('Mean threshold: %.1f dims', mean_threshold));
                    % Add asterisks at the top for each unit's threshold
                    unique_thresholds = unique(best_threshold);
                    ylims = ylim;
                    for i = 1:length(unique_thresholds)
                        thresh = unique_thresholds(i);
                        plot(thresh, ylims(2), 'r*', 'MarkerSize', 5);
                    end
                end
            end
        else  % Magnitude thresholding mode - show included dimensions
            if isnumeric(best_threshold) && length(best_threshold) > 0
                % Add circles for included dimensions 
                plot(best_threshold, S(best_threshold), 'ro', 'MarkerSize', 4, ...
                     'DisplayName', 'Included dimensions');
                % Show vertical line for number of dimensions retained
                threshold_len = length(best_threshold);
                xline(threshold_len, 'r--', 'LineWidth', 1, ...
                      'DisplayName', sprintf('Dims retained: %d', threshold_len));
            end
        end
        
        xlabel('Dimension');
        ylabel('Eigenvalue');
        title(sprintf('Denoising Basis\nEigenspectrum'));
        grid on;
        legend;

        % Plot 4: Signal and noise variances with NCSNR (top right)
        subplot(3, 4, 4);
        yyaxis left;
        plot(sigvars, 'LineWidth', 1, 'DisplayName', 'Sig. var');
        hold on;
        plot(noisevars, 'LineWidth', 1, 'DisplayName', 'Noise var');
        
        % Handle thresholds based on mode
        if cv_mode >= 0  % Cross-validation mode
            if isnumeric(best_threshold) && length(best_threshold) > 0
                if length(best_threshold) > 1
                    threshold_val = mean(best_threshold);
                    xline(threshold_val, 'r--', 'LineWidth', 1, ...
                          'DisplayName', sprintf('Mean thresh: %.1f dims', threshold_val));
                else
                    xline(best_threshold, 'r--', 'LineWidth', 1, ...
                          'DisplayName', sprintf('Thresh: %d dims', best_threshold));
                end
            end
        else  % Magnitude thresholding mode
            if isnumeric(best_threshold) && length(best_threshold) > 0
                threshold_len = length(best_threshold);
                xline(threshold_len, 'r--', 'LineWidth', 1, ...
                      'DisplayName', sprintf('Dims retained: %d', threshold_len));
                % Add circles for included dimensions
                if mag_type == 0
                    plot(best_threshold, sigvars(best_threshold), 'ro', 'MarkerSize', 4, ...
                         'DisplayName', 'Included dimensions');
                end
            end
        end
        
        ylabel('Variance');
        
        % Add NCSNR on secondary y-axis
        yyaxis right;
        plot(ncsnrs, 'LineWidth', 1, 'Color', 'magenta', 'DisplayName', 'NCSNR');
        ylabel('NCSNR');
        
        xlabel('Dimension');
        title(sprintf('Signal and Noise Variance for \nData Projected into Basis'));
        grid on;
        legend;

        % Plot 5: Cross-validation results (first subplot in middle row)
        subplot(3, 4, 5);
        if isfield(results, 'cv_scores') && isfield(opt, 'cv_mode') && opt.cv_mode > -1
            cv_data = mean(results.cv_scores, 2);  % Average over trials
            
            % Get thresholds, handling both list and array types
            if isfield(opt, 'cv_thresholds')
                cv_thresholds = opt.cv_thresholds;
            else
                cv_thresholds = 1:size(results.cv_scores, 1);
            end
            
            % Truncate thresholds that exceed data dimensionality
            max_dim = size(results.cv_scores, 1);
            valid_mask = cv_thresholds <= max_dim;
            thresholds = cv_thresholds(valid_mask);
            cv_data = cv_data(valid_mask, :);
            
            % Z-score the data
            cv_data = zscore(cv_data, 0, 1);  % Z-score along first dimension
            
            imagesc(cv_data');
            colorbar;
            xlabel('PC exclusion threshold');
            ylabel('Units');
            title('Cross-validation scores (z)');
            
            % Set x-ticks to show actual threshold values
            step = max(floor(length(thresholds) / 10), 1);  % Show ~10 ticks or less
            tick_positions = 1:step:length(thresholds);
            set(gca, 'XTick', tick_positions, 'XTickLabel', thresholds(tick_positions));
            
            if strcmp(opt.cv_threshold_per, 'unit')
                if isnumeric(best_threshold) && length(best_threshold) == nunits
                    % For each unit, find the threshold index that gives maximum CV score
                    unit_indices = 1:nunits;
                    threshold_positions = [];
                    
                    % Check if unit_groups are being used
                    if isfield(opt, 'unit_groups')
                        unit_groups = opt.unit_groups;
                    else
                        unit_groups = (0:nunits-1)';
                    end
                    
                    if isfield(opt, 'unit_groups') && ~isequal(unit_groups, (0:nunits-1)')
                        % Unit groups are being used - show group-based thresholds
                        unique_groups = unique(unit_groups);
                        
                        for unit_idx = 1:nunits
                            % Find which group this unit belongs to
                            unit_group = unit_groups(unit_idx);
                            
                            % Get all units in this group
                            group_mask = unit_groups == unit_group;
                            
                            % Average CV scores across units in this group
                            group_cv_scores = mean(cv_data(:, group_mask), 2);  % Average across group units
                            
                            % Find threshold index with maximum group score
                            [~, max_thresh_idx] = max(group_cv_scores);
                            
                            % Position at center of that threshold's cell
                            threshold_positions(end+1) = max_thresh_idx;
                        end
                    else
                        % No unit grouping - use individual unit's maximum CV score
                        for unit_idx = 1:nunits
                            % Get CV scores for this unit across all thresholds
                            unit_cv_scores = cv_data(:, unit_idx);  % cv_data shape: (n_thresholds, n_units)
                            
                            % Find threshold index with maximum score
                            [~, max_thresh_idx] = max(unit_cv_scores);
                            
                            % Position at center of that threshold's cell
                            threshold_positions(end+1) = max_thresh_idx;
                        end
                    end
                    
                    hold on;
                    plot(threshold_positions, unit_indices, 'r.', 'MarkerSize', 4);
                end
            end
        else
            text(0.5, 0.5, sprintf('No Cross-validation\nScores Available'), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'Units', 'normalized');
            title('Cross-validation scores');
        end

        % Plot 6-8: Raw data, denoised data, noise (rest of middle row)
        all_data = [raw_data(:); denoised_data(:); noise(:)];
        max_abs_val = max(abs(all_data));
        data_clim = [-max_abs_val, max_abs_val];
        
        % Raw data
        subplot(3, 4, 6);
        imagesc(raw_data, data_clim);
        colormap(gca, redblue);
        colorbar;
        title('Input Data (trial-averaged)');
        xlabel('Conditions');
        ylabel('Units');

        % Denoised data
        subplot(3, 4, 7);
        imagesc(denoised_data, data_clim);
        colormap(gca, redblue);
        colorbar;
        title('Data projected into basis');
        xlabel('Conditions');
        ylabel('Units');

        % Noise
        subplot(3, 4, 8);
        imagesc(noise, data_clim);
        colormap(gca, redblue);
        colorbar;
        title('Residual');
        xlabel('Conditions');
        ylabel('Units');

        % Plot denoising matrix (first subplot in bottom row)
        subplot(3, 4, 9);
        denoiser_max = max(abs(results.denoiser(:)));
        denoiser_clim = [-denoiser_max, denoiser_max];
        imagesc(results.denoiser, denoiser_clim);
        colormap(gca, redblue);
        colorbar;
        title('Optimal Basis Matrix');
        xlabel('Units');
        ylabel('Units');

        % Compute R2 and correlations for bottom row
        if nargin < 3 || isempty(test_data)
            % Use leave-one-out cross-validation on training data
            raw_r2_per_unit = zeros(ntrials, nunits);
            denoised_r2_per_unit = zeros(ntrials, nunits);
            raw_corr_per_unit = zeros(ntrials, nunits);
            denoised_corr_per_unit = zeros(ntrials, nunits);

            for tr = 1:ntrials
                train_trials = setdiff(1:ntrials, tr);
                train_avg = mean(data(:, :, train_trials), 3);
                test_trial = data(:, :, tr);
                
                for v = 1:nunits
                    raw_r2_per_unit(tr, v) = compute_r2(test_trial(v, :), train_avg(v, :));
                    raw_corr_per_unit(tr, v) = corr(test_trial(v, :)', train_avg(v, :)');
                end
                
                % Demean before denoising for consistent handling
                if isfield(results, 'unit_means')
                    train_avg_demeaned = train_avg - results.unit_means;
                    test_trial_demeaned = test_trial - results.unit_means;
                else
                    train_avg_demeaned = train_avg;
                    test_trial_demeaned = test_trial;
                end
                train_avg_denoised = (train_avg_demeaned' * results.denoiser)';
                test_trial_denoised = (test_trial_demeaned' * results.denoiser)';
                if isfield(results, 'unit_means')
                    train_avg_denoised = train_avg_denoised + results.unit_means;
                    test_trial_denoised = test_trial_denoised + results.unit_means;
                end
                    
                for v = 1:nunits
                    denoised_r2_per_unit(tr, v) = compute_r2(test_trial(v, :), train_avg_denoised(v, :));
                    denoised_corr_per_unit(tr, v) = corr(test_trial(v, :)', train_avg_denoised(v, :)');
                end
            end
        else
            % Use provided test data
            if ndims(test_data) > 2
                test_avg = mean(test_data, 3);
            else
                test_avg = test_data;
            end
            train_avg = mean(data, 3);
            
            raw_r2_per_unit = zeros(1, nunits);
            denoised_r2_per_unit = zeros(1, nunits);
            raw_corr_per_unit = zeros(1, nunits);
            denoised_corr_per_unit = zeros(1, nunits);
            
            for v = 1:nunits
                raw_r2_per_unit(1, v) = compute_r2(test_avg(v, :), train_avg(v, :));
                raw_corr_per_unit(1, v) = corr(test_avg(v, :)', train_avg(v, :)');
            end
            
            % Demean before denoising for consistent handling
            if isfield(results, 'unit_means')
                train_avg_demeaned = train_avg - results.unit_means;
                test_avg_demeaned = test_avg - results.unit_means;
            else
                train_avg_demeaned = train_avg;
                test_avg_demeaned = test_avg;
            end
            train_avg_denoised = (train_avg_demeaned' * results.denoiser)';
            test_avg_denoised = (test_avg_demeaned' * results.denoiser)';
            if isfield(results, 'unit_means')
                train_avg_denoised = train_avg_denoised + results.unit_means;
                test_avg_denoised = test_avg_denoised + results.unit_means;
            end
                
            for v = 1:nunits
                denoised_r2_per_unit(1, v) = compute_r2(test_avg(v, :), train_avg_denoised(v, :));
                denoised_corr_per_unit(1, v) = corr(test_avg(v, :)', train_avg_denoised(v, :)');
            end
        end

        % Compute mean and SEM
        raw_r2_mean = mean(raw_r2_per_unit, 1);
        raw_r2_sem = std(raw_r2_per_unit, 0, 1) / sqrt(size(raw_r2_per_unit, 1));
        denoised_r2_mean = mean(denoised_r2_per_unit, 1);
        denoised_r2_sem = std(denoised_r2_per_unit, 0, 1) / sqrt(size(denoised_r2_per_unit, 1));

        raw_corr_mean = mean(raw_corr_per_unit, 1);
        raw_corr_sem = std(raw_corr_per_unit, 0, 1) / sqrt(size(raw_corr_per_unit, 1));
        denoised_corr_mean = mean(denoised_corr_per_unit, 1);
        denoised_corr_sem = std(denoised_corr_per_unit, 0, 1) / sqrt(size(denoised_corr_per_unit, 1));

        % Plot bottom row histograms and R² progression
        train_trials = ntrials - 1;
        if nargin >= 3 && ~isempty(test_data)
            train_trials = size(data, 3);
            if ndims(test_data) > 2
                test_trials = size(test_data, 3);
            else
                test_trials = 1;
            end
        else
            test_trials = 1;
        end
        
        % Plot baseline generalization
        subplot(3, 4, 10);
        plot_bottom_histogram(raw_r2_mean, raw_corr_mean, 'blue', 'lightblue', ...
                              sprintf('Baseline Generalization\nTrial-avg Train (%d trials) vs\nTrial-avg Test (%d trials)', ...
                                      train_trials, test_trials));

        % Plot denoised generalization
        subplot(3, 4, 11);
        plot_bottom_histogram(denoised_r2_mean, denoised_corr_mean, 'green', 'lightgreen', ...
                              sprintf('Denoised Generalization\nTrial-avg Train + denoised (%d trials) vs\nTrial-avg Test (%d trials)', ...
                                      train_trials, test_trials));

        % Add R² progression plot
        subplot(3, 4, 12);
        x_positions = [1, 2];  % Two positions for the two conditions
        
        % Plot lines for each unit
        for v = 1:nunits
            values = [raw_r2_mean(v), denoised_r2_mean(v)];
            plot(x_positions, values, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
            hold on;
            scatter(x_positions(1), values(1), 20, 'blue', 'filled', 'MarkerFaceAlpha', 0.5);
            scatter(x_positions(2), values(2), 20, 'green', 'filled', 'MarkerFaceAlpha', 0.5);
        end
        
        % Plot mean performance
        mean_values = [mean(raw_r2_mean), mean(denoised_r2_mean)];
        plot(x_positions, mean_values, 'Color', 'magenta', 'LineWidth', 2, 'DisplayName', 'Mean');
        scatter(x_positions(1), mean_values(1), 100, 'blue', 'filled', 'MarkerEdgeColor', 'magenta', 'LineWidth', 2);
        scatter(x_positions(2), mean_values(2), 100, 'green', 'filled', 'MarkerEdgeColor', 'magenta', 'LineWidth', 2);
        
        set(gca, 'XTick', x_positions, 'XTickLabel', {'Trial Averaged', 'With Denoising'});
        ylabel('R²');
        title(sprintf('Impact of denoising on R² (%d units)', nunits));
        grid on;
        legend;
        xlim([0.5, 2.5]);
        ylim([-1, 1]);
        yline(0, 'k-', 'LineWidth', 2);
    end
end

% Helper function to plot bottom row with rotated histograms
function plot_bottom_histogram(r2_mean, corr_mean, r2_color, corr_color, title_str)
    hold on;
    xline(0, 'k-', 'LineWidth', 2);
    
    % Calculate histogram bins
    bins = linspace(-1, 1, 50);
    bin_width = bins(2) - bins(1);
    
    % Plot R2 histogram
    [r2_hist, ~] = histcounts(r2_mean, bins);
    bar(bins(1:end-1) + bin_width/2, r2_hist, bin_width, ...
        'FaceColor', r2_color, 'FaceAlpha', 0.6, 'DisplayName', sprintf('Mean R² = %.3f', mean(r2_mean)));
    
    % Plot correlation histogram
    [corr_hist, ~] = histcounts(corr_mean, bins);
    bar(bins(1:end-1) + bin_width/2, corr_hist, bin_width, ...
        'FaceColor', corr_color, 'FaceAlpha', 0.6, 'DisplayName', sprintf('Mean r = %.3f', mean(corr_mean)));
    
    ylabel('# Units');
    xlabel('R² / Pearson r');
    title(title_str);
    grid on;
    legend;
    xlim([-1, 1]);
end

% Helper function to compute noise ceiling
function [noiseceiling, ncsnr, signalvar, noisevar] = compute_noise_ceiling(data_in)
    % noisevar: mean variance across trials for each unit
    noisevar = mean(std(data_in, 0, 3) .^ 2, 2);

    % datavar: variance of the trial means across conditions for each unit
    datavar = std(mean(data_in, 3), 0, 2) .^ 2;

    % signalvar: signal variance, obtained by subtracting noise variance from data variance
    signalvar = max(datavar - noisevar / size(data_in, 3), 0);  % Ensure non-negative variance

    % ncsnr: signal-to-noise ratio (SNR) for each unit
    ncsnr = sqrt(signalvar) ./ sqrt(noisevar);

    % noiseceiling: percentage noise ceiling based on SNR
    noiseceiling = 100 * (ncsnr .^ 2 ./ (ncsnr .^ 2 + 1 / size(data_in, 3)));
end

% Helper function to compute R2 score
function r2 = compute_r2(y_true, y_pred)
    residual_ss = sum((y_true - y_pred) .^ 2);
    total_ss = sum((y_true - mean(y_true)) .^ 2);
    r2 = 1 - (residual_ss / total_ss);
end

% Red-Blue colormap function
function cmap = redblue
    % Create a red-blue colormap similar to Python's RdBu_r
    n = 256;
    cmap = zeros(n, 3);
    
    % Blue to white to red
    mid = ceil(n/2);
    
    % Blue to white (first half)
    cmap(1:mid, 1) = linspace(0, 1, mid);      % Red component
    cmap(1:mid, 2) = linspace(0, 1, mid);      % Green component  
    cmap(1:mid, 3) = ones(mid, 1);             % Blue component (full)
    
    % White to red (second half)
    cmap(mid+1:n, 1) = ones(n-mid, 1);         % Red component (full)
    cmap(mid+1:n, 2) = linspace(1, 0, n-mid);  % Green component
    cmap(mid+1:n, 3) = linspace(1, 0, n-mid);  % Blue component
end
