function visualization(data, results, test_data)
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
                         'V = %s    |    cv\\_mode = %d    |    ' ...
                         'mag\\_type = %d, mag\\_frac = %s\n'], ...
                        nunits, nconds, ntrials, V_desc, cv_mode, mag_type, mag_frac_str);
        else
            threshold_per = results.opt.cv_threshold_per;
            title_text = sprintf(['Data shape: %d units × %d conditions × %d trials    |    ' ...
                         'V = %s    |    cv\\_mode = %d    |    thresh = %s\n'], ...
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
            elseif V == 4
                text(0.5, 0.5, sprintf('Random Basis\n(No Matrix to Show)'), ...
                     'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                     'Units', 'normalized');
                title('');
            elseif V == 5  % ICA basis
                % Show the ICA components (mixing matrix)
                if isfield(results, 'ica_mixing') && ~isempty(results.ica_mixing)
                    matrix_to_show = results.ica_mixing;
                    matrix_max = max(abs(matrix_to_show(:)));

                    imagesc(matrix_to_show, [-matrix_max, matrix_max]);
                    colormap(gca, redblue);
                    colorbar;
                    title(sprintf('ICA Mixing Matrix\n(Components)'));
                    xlabel('Component');
                    ylabel('Units');
                else
                    text(0.5, 0.5, sprintf('ICA Components\nNot Available'), ...
                         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                         'Units', 'normalized');
                    title('');
                end
            else
                text(0.5, 0.5, sprintf('V=%d\n(No Matrix to Show)', V), ...
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
        
        % Plot 3: Magnitude spectrum (top middle)
        subplot(3, 4, 3);

        % Determine labels based on ranking method
        if isfield(results, 'opt') && isfield(results.opt, 'ranking')
            ranking = results.opt.ranking;
        else
            ranking = 'signal_variance';  % default
        end

        % Define labels for different ranking methods
        if strcmp(ranking, 'eigenvalue')
            legend_label = 'Eigenvalues';
            ylabel_str = 'Eigenvalue';
            title_str = 'Eigenspectrum (decreasing)';
        elseif strcmp(ranking, 'eigenvalue_asc')
            legend_label = 'Eigenvalues';
            ylabel_str = 'Eigenvalue';
            title_str = 'Eigenspectrum (increasing)';
        elseif strcmp(ranking, 'signal_variance')
            legend_label = 'Signal Variance';
            ylabel_str = 'Signal Variance';
            title_str = 'Signal Variance Spectrum';
        elseif strcmp(ranking, 'snr')
            legend_label = 'Noise-Ceiling SNR';
            ylabel_str = 'NCSNR';
            title_str = 'NCSNR Spectrum';
        elseif strcmp(ranking, 'signal_specificity')
            legend_label = 'Signal% - Noise%';
            ylabel_str = 'Signal% - Noise%';
            title_str = 'Signal Specificity Spectrum';
        else
            legend_label = 'Magnitude';
            ylabel_str = 'Magnitude';
            title_str = 'Magnitude Spectrum';
        end

        % Get eigenvalues from results - try multiple sources
        S_local = [];
        if isfield(results, 'mags') && ~isempty(results.mags)
            S_local = results.mags(:);  % Primary source: magnitude thresholding mode
        elseif exist('S', 'var') && ~isempty(S)
            S_local = S(:);  % Secondary source: from the computation above
        else
            % Try to compute eigenvalues from available data
            % This might happen in cross-validation mode where mags is not set
            if isfield(results, 'fullbasis') && ~isempty(results.fullbasis)
                % We have the basis but not the eigenvalues
                % Show a placeholder message
                text(0.5, 0.5, sprintf('Magnitude spectrum not available\nin cross-validation mode'), ...
                     'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                     'Units', 'normalized');
                S_local = [];  % Mark as unavailable
            else
                text(0.5, 0.5, sprintf('Magnitude spectrum not available\n(no basis data)'), ...
                     'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                     'Units', 'normalized');
                S_local = [];  % Mark as unavailable
            end
        end

        % Plot magnitude spectrum if data is available
        if ~isempty(S_local)
            % Clear any previous plots and plot the spectrum
            cla;
            h_eigen = plot(1:length(S_local), S_local, 'LineWidth', 1, 'Color', 'blue');
            set(h_eigen, 'DisplayName', legend_label);

            % Set axis limits immediately after main plot
            if length(S_local) > 1
                xlim([0.5, length(S_local) + 0.5]);
            else
                xlim([0.5, 1.5]);
            end

            if max(S_local) > min(S_local)
                y_range = max(S_local) - min(S_local);
                ylim([min(S_local) - 0.1*y_range, max(S_local) + 0.1*y_range]);
            end
        end
        
        hold on;
        
        % Add threshold indicators based on mode
        cv_mode = 0;  % default
        if isfield(opt, 'cv_mode')
            cv_mode = opt.cv_mode;
        end
        
        cv_threshold_per = 'unit';  % default
        if isfield(opt, 'cv_threshold_per')
            cv_threshold_per = opt.cv_threshold_per;
        end
            
        % Add threshold indicators based on mode
        if cv_mode >= 0  % Cross-validation mode
            if strcmp(cv_threshold_per, 'population')
                % Single line for population threshold
                if isnumeric(best_threshold) && length(best_threshold) > 1
                    best_threshold = best_threshold(1);  % Take first value if array
                end
                if isnumeric(best_threshold) && best_threshold > 0
                    xline(best_threshold, 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
                end
            else  % Unit mode
                % Mean line for unit-specific thresholds
                if isnumeric(best_threshold) && length(best_threshold) > 1
                    mean_threshold = mean(best_threshold);
                    xline(mean_threshold, 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
                    % Add asterisks at the top for each unit's threshold
                    unique_thresholds = unique(best_threshold);
                    y_max = max(S);
                    for i = 1:length(unique_thresholds)
                        thresh = unique_thresholds(i);
                        if thresh <= length(S)
                            plot(thresh, y_max, 'r*', 'MarkerSize', 5, 'HandleVisibility', 'off');
                        end
                    end
                end
            end
        else  % Magnitude thresholding mode - show included dimensions
            mag_type = 0;  % default
            if isfield(opt, 'mag_type')
                mag_type = opt.mag_type;
            end
            
            if isnumeric(best_threshold) && length(best_threshold) > 0
                % Add circles for included dimensions 
                valid_indices = best_threshold(best_threshold <= length(S));
                if ~isempty(valid_indices)
                    plot(valid_indices, S(valid_indices), 'ro', 'MarkerSize', 4, 'HandleVisibility', 'off');
                end
                % Show vertical line for number of dimensions retained
                threshold_len = length(best_threshold);
                if threshold_len <= length(S)
                    xline(threshold_len, 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
                end
            end
        end
        
        % Show truncated dimensions if truncate > 0
        if isfield(opt, 'truncate') && opt.truncate > 0
            truncate_val = opt.truncate;
            % Highlight truncated dimensions in red
            truncate_range = min(truncate_val, length(S_local));
            if truncate_range > 0
                plot(1:truncate_range, S_local(1:truncate_range), 'rx', 'MarkerSize', 6, ...
                    'DisplayName', sprintf('Truncated dims (first %d)', truncate_range));
            end
        end

        % Set labels
        xlabel('Dimension');
        ylabel(ylabel_str);
        title(sprintf('Denoising Basis\n%s', title_str));
        grid on;
        legend;

        % Plot 4: Signal and noise variances with NCSNR (top right)
        subplot(3, 4, 4);
        yyaxis left;
        plot(sigvars, '-', 'LineWidth', 1, 'Color', 'blue', 'DisplayName', 'Sig. var');
        hold on;
        plot(noisevars, '-', 'LineWidth', 1, 'Color', [1 0.5 0], 'DisplayName', 'Noise var');  % Solid orange

        % Show truncated dimensions if truncate > 0
        if isfield(opt, 'truncate') && opt.truncate > 0
            truncate_val = opt.truncate;
            % Highlight truncated dimensions in red
            truncate_range = min(truncate_val, length(sigvars));
            if truncate_range > 0
                plot(1:truncate_range, sigvars(1:truncate_range), 'rx', 'MarkerSize', 6, ...
                    'DisplayName', sprintf('Truncated dims (first %d)', truncate_range));
            end
        end
        
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
            
            % Z-score the data (MATCH PYTHON: axis=0 means along columns in MATLAB)
            cv_data = zscore(cv_data, 1, 1);  % Z-score along columns (axis=0 in Python)

            % Plot without extent - use default pixel coordinates with 'nearest' interpolation
            imagesc(cv_data');
            colorbar;

            % Update xlabel based on whether truncation is used
            if isfield(opt, 'truncate') && opt.truncate > 0
                xlabel(sprintf('PC threshold (starting from PC %d)', opt.truncate));
            else
                xlabel('PC exclusion threshold');
            end
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
                    % In MATLAB, imagesc displays matrix elements centered at integer coordinates (pixel indices)
                    plot(threshold_positions, unit_indices, 'r.', 'MarkerSize', 8);
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

        % Plot trial-averaged and denoised traces similar to EEG notebook
        % Get trial-averaged data
        trial_avg_full = mean(data, 3);  % (nunits x nconds)
        denoised_full = results.denoiseddata;

        % If denoised data is 3D, average it
        if ndims(denoised_full) == 3
            denoised_full = mean(denoised_full, 3);
        end

        % Calculate mean response across conditions for rainbow coloring
        cond_means = mean(trial_avg_full, 1);  % Mean across units for each condition
        [~, sorted_cond_indices] = sort(cond_means);
        colors_rainbow = jet(nconds);

        % Create color array where each condition gets its color based on rank
        trace_colors = zeros(nconds, 3);
        for rank = 1:nconds
            cond_idx = sorted_cond_indices(rank);
            trace_colors(cond_idx, :) = colors_rainbow(rank, :);
        end

        % Plot trial-averaged traces
        subplot(3, 4, 10);
        hold on;
        x_units = 0:nunits-1;  % 0-indexed to match Python
        for cond_idx = 1:nconds
            plot(x_units, trial_avg_full(:, cond_idx), 'Color', trace_colors(cond_idx, :), ...
                'LineWidth', 0.5);
        end
        xlabel('Units');
        ylabel('Activity');
        title(sprintf('Trial-Averaged Traces\n(rainbow: conditions by mean response)'));
        grid on;
        xlim([min(x_units), max(x_units)]);

        % Plot denoised traces
        subplot(3, 4, 11);
        hold on;
        for cond_idx = 1:nconds
            plot(x_units, denoised_full(:, cond_idx), 'Color', trace_colors(cond_idx, :), ...
                'LineWidth', 0.5);
        end
        xlabel('Units');
        ylabel('Activity');
        title(sprintf('PSN Denoised Traces\n(same condition coloring)'));
        grid on;
        xlim([min(x_units), max(x_units)]);

        % Match y-axis limits across both plots
        all_trace_data = [trial_avg_full(:); denoised_full(:)];
        y_min = min(all_trace_data);
        y_max = max(all_trace_data);
        y_range = y_max - y_min;
        y_margin = y_range * 0.05;

        subplot(3, 4, 10);
        ylim([y_min - y_margin, y_max + y_margin]);

        subplot(3, 4, 11);
        ylim([y_min - y_margin, y_max + y_margin]);

        % Add split-half correlation comparison plot
        subplot(3, 4, 12);

        % Split trials in half
        half_idx = floor(ntrials / 2);
        data_A = data(:, :, 1:half_idx);
        data_B = data(:, :, half_idx+1:end);

        % Compute trial averages for each split
        trial_avg_A = mean(data_A, 3);  % (nunits x nconds)
        trial_avg_B = mean(data_B, 3);

        % Denoise each split separately
        denoiser = results.denoiser;
        unit_means = results.unit_means;

        % Denoise split A
        trial_avg_A_demeaned = trial_avg_A - unit_means;
        denoised_A = (trial_avg_A_demeaned' * denoiser)' + unit_means;

        % Denoise split B
        trial_avg_B_demeaned = trial_avg_B - unit_means;
        denoised_B = (trial_avg_B_demeaned' * denoiser)' + unit_means;

        % Compute correlations for each unit
        corr_tavg_tavg = zeros(nunits, 1);
        corr_cross_AB = zeros(nunits, 1);
        corr_cross_BA = zeros(nunits, 1);
        corr_dn_dn = zeros(nunits, 1);

        for unit_idx = 1:nunits
            % Trial avg vs trial avg
            if std(trial_avg_A(unit_idx, :)) > 0 && std(trial_avg_B(unit_idx, :)) > 0
                corr_tavg_tavg(unit_idx) = corr(trial_avg_A(unit_idx, :)', trial_avg_B(unit_idx, :)');
            else
                corr_tavg_tavg(unit_idx) = NaN;
            end

            % Cross-method: trial avg A vs denoised B
            if std(trial_avg_A(unit_idx, :)) > 0 && std(denoised_B(unit_idx, :)) > 0
                corr_cross_AB(unit_idx) = corr(trial_avg_A(unit_idx, :)', denoised_B(unit_idx, :)');
            else
                corr_cross_AB(unit_idx) = NaN;
            end

            % Cross-method: denoised A vs trial avg B
            if std(denoised_A(unit_idx, :)) > 0 && std(trial_avg_B(unit_idx, :)) > 0
                corr_cross_BA(unit_idx) = corr(denoised_A(unit_idx, :)', trial_avg_B(unit_idx, :)');
            else
                corr_cross_BA(unit_idx) = NaN;
            end

            % Denoised vs denoised
            if std(denoised_A(unit_idx, :)) > 0 && std(denoised_B(unit_idx, :)) > 0
                corr_dn_dn(unit_idx) = corr(denoised_A(unit_idx, :)', denoised_B(unit_idx, :)');
            else
                corr_dn_dn(unit_idx) = NaN;
            end
        end

        % Average the two cross-method correlations
        corr_cross = (corr_cross_AB + corr_cross_BA) / 2;

        % Three x positions
        x_positions = [1, 2, 3];
        labels = {'TAvg vs TAvg', 'TAvg vs Denoised', 'Denoised vs Denoised'};

        % Add jitter
        jitter = 0.08;
        rng(42);  % Set seed for reproducibility
        x_jitter = (rand(nunits, 1) - 0.5) * 2 * jitter;

        hold on;

        % Plot connecting lines for each unit
        for unit_idx = 1:nunits
            values = [corr_tavg_tavg(unit_idx), corr_cross(unit_idx), corr_dn_dn(unit_idx)];
            if ~any(isnan(values))
                x_vals = x_positions + x_jitter(unit_idx);
                plot(x_vals, values, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.3);
            end
        end

        % Plot individual dots
        scatter(x_positions(1) + x_jitter, corr_tavg_tavg, 15, 'blue', 'filled', 'MarkerFaceAlpha', 0.4);
        scatter(x_positions(2) + x_jitter, corr_cross, 15, [1 0.84 0], 'filled', 'MarkerFaceAlpha', 0.4);  % Gold
        scatter(x_positions(3) + x_jitter, corr_dn_dn, 15, [0.5 0.8 0.3], 'filled', 'MarkerFaceAlpha', 0.4);  % Lime green

        % Compute means (excluding NaN)
        mean_tavg_tavg = nanmean(corr_tavg_tavg);
        mean_cross = nanmean(corr_cross);
        mean_dn_dn = nanmean(corr_dn_dn);

        % Plot mean dots (larger, with edge)
        scatter(x_positions(1), mean_tavg_tavg, 100, 'blue', 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
        scatter(x_positions(2), mean_cross, 100, [1 0.84 0], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);
        scatter(x_positions(3), mean_dn_dn, 100, [0.2 0.6 0.2], 'filled', 'MarkerEdgeColor', 'white', 'LineWidth', 2);

        % Add mean values as text above the dots
        y_text_offset = 0.08;
        text(x_positions(1), mean_tavg_tavg + y_text_offset, sprintf('%.3f', mean_tavg_tavg), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
        text(x_positions(2), mean_cross + y_text_offset, sprintf('%.3f', mean_cross), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
        text(x_positions(3), mean_dn_dn + y_text_offset, sprintf('%.3f', mean_dn_dn), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

        % Formatting
        set(gca, 'XTick', x_positions);
        set(gca, 'XTickLabel', labels);
        set(gca, 'XTickLabelRotation', 0);
        set(gca, 'FontSize', 7);
        ylabel('Pearson r');
        title(sprintf('Split-Half Reliability (%d units)\nSplit: %d vs %d trials', nunits, size(data_A, 3), size(data_B, 3)));
        grid on;
        xlim([0.5, 3.5]);

        % Set y-axis limits based on data range
        all_corr_values = [corr_tavg_tavg; corr_cross; corr_dn_dn];
        valid_corr_values = all_corr_values(~isnan(all_corr_values));
        if ~isempty(valid_corr_values)
            y_min_corr = min(valid_corr_values);
            y_max_corr = max(valid_corr_values);
            y_range_corr = y_max_corr - y_min_corr;
            y_padding = max(0.1, y_range_corr * 0.15);
            ylim([y_min_corr - y_padding, y_max_corr + y_padding]);
        else
            ylim([-1, 1]);
        end

        yline(0, 'k-', 'LineWidth', 1);
    end
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
