function plot_data_diagnostic(data, ground_truth, params)
% PLOT_DATA_DIAGNOSTIC Generate a comprehensive diagnostic figure for simulated data.
%
% Creates a 12-subplot figure showing eigenvalue spectra, signal-to-noise ratios,
% covariance matrices, ground truth signal, example trial data, eigenvectors,
% alignment visualization, and trial-averaged data.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <data> - [nvox x ncond x ntrial] the simulated data
%
% <ground_truth> - struct with ground truth information containing:
%   .signal_cov  - [nvox x nvox] signal covariance matrix
%   .noise_cov   - [nvox x nvox] noise covariance matrix
%   .signal_eigs - [nvox x 1] signal eigenvalues
%   .noise_eigs  - [nvox x 1] noise eigenvalues
%   .U_signal    - [nvox x nvox] signal eigenvectors
%   .U_noise     - [nvox x nvox] noise eigenvectors
%   .signal      - [ncond x nvox] ground truth signal
%
% <params> - struct with simulation parameters containing:
%   .nvox            - number of units
%   .ncond           - number of conditions
%   .ntrial          - number of trials
%   .signal_decay    - signal eigenvalue decay rate
%   .noise_decay     - noise eigenvalue decay rate
%   .noise_multiplier - noise scaling factor
%   .align_alpha     - alignment strength (0=orthogonal, 1=aligned)
%   .align_k         - number of aligned dimensions
%   .user_provided   - (optional) struct indicating what was user-provided
%   .clustered       - (optional) whether units were reordered by clustering

    % Extract important parameters and ground truth values
    nvox = params.nvox;
    ncond = params.ncond;
    ntrial = params.ntrial;
    signal_decay = params.signal_decay;
    noise_decay = params.noise_decay;
    noise_multiplier = params.noise_multiplier;
    align_alpha = params.align_alpha;
    align_k = params.align_k;

    if isfield(params, 'user_provided')
        user_provided = params.user_provided;
    else
        user_provided = struct('signal_cov', false, 'true_signal', false);
    end

    % Extract ground truth matrices
    signal_cov = ground_truth.signal_cov;
    noise_cov = ground_truth.noise_cov;
    signal_eigs = ground_truth.signal_eigs;
    noise_eigs = ground_truth.noise_eigs;
    U_signal = ground_truth.U_signal;
    U_noise = ground_truth.U_noise;
    true_signal = ground_truth.signal;

    % Create example trial data for visualization
    trial_avg = mean(data, 3);  % Average across trials
    example_trial = data(:, :, 1);  % First trial

    % Create figure with custom layout
    fig = figure('Position', [100, 100, 1800, 1200]);

    % Add title with parameters
    title_text = sprintf('Simulated Data: %d units × %d conditions × %d trials\n', nvox, ncond, ntrial);

    % Add appropriate source info based on what was user-provided
    if user_provided.true_signal
        title_text = [title_text 'Using user-provided ground truth signal' newline];
    elseif user_provided.signal_cov
        title_text = [title_text 'Using user-provided signal covariance matrix' newline];
    else
        title_text = [title_text sprintf('Signal decay=%.2f, Noise decay=%.2f, Noise multiplier=%.2f\n', ...
                     signal_decay, noise_decay, noise_multiplier)];
    end

    title_text = [title_text sprintf('Alignment: alpha=%.2f (0=orthogonal, 1=aligned), k=%d top PCs', ...
                 align_alpha, align_k)];

    % Add note about clustering if units were reordered
    if isfield(params, 'clustered') && params.clustered
        title_text = [title_text newline 'Units reordered by hierarchical clustering'];
    end

    sgtitle(title_text, 'FontSize', 14, 'FontWeight', 'bold');

    % Plot 1a: Eigenvalue spectra - Log scale
    subplot(3, 4, 1);
    semilogy(0:nvox-1, signal_eigs, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Signal eigenvalues');
    hold on;
    semilogy(0:nvox-1, noise_eigs, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Noise eigenvalues');
    if align_k > 0
        xline(align_k-1, 'Color', [0.5 0.5 0.5], 'LineStyle', '--', ...
             'DisplayName', sprintf('Alignment cutoff (k=%d)', align_k));
    end
    xlabel('Dimension');
    ylabel('Eigenvalue (log scale)');
    title('Eigenspectrum - Log Scale');
    legend('Location', 'best');
    grid on;

    % Plot 1b: Eigenvalue spectra - Linear scale
    subplot(3, 4, 2);
    plot(0:nvox-1, signal_eigs, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Signal eigenvalues');
    hold on;
    plot(0:nvox-1, noise_eigs, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Noise eigenvalues');
    if align_k > 0
        xline(align_k-1, 'Color', [0.5 0.5 0.5], 'LineStyle', '--', ...
             'DisplayName', sprintf('Alignment cutoff (k=%d)', align_k));
    end
    xlabel('Dimension');
    ylabel('Eigenvalue (linear scale)');
    title('Eigenspectrum - Linear Scale');
    % Show top 25% of dimensions for better visualization in linear scale
    dims_to_show = max(floor(nvox * 0.25), align_k + 5);
    xlim([0, dims_to_show]);
    legend('Location', 'best');
    grid on;

    % Plot 2: Signal-to-noise ratio per dimension
    subplot(3, 4, [3 4]);
    snr = signal_eigs ./ noise_eigs;
    plot(0:nvox-1, snr, 'g-', 'LineWidth', 1.5);
    xlabel('Dimension');
    ylabel('SNR');
    title('Signal-to-Noise Ratio per Dimension');
    grid on;

    % Plot 3: Signal covariance matrix
    subplot(3, 4, 5);
    % Calculate clim values similar to gsn_denoise.py
    cov_max = prctile(abs(signal_cov(:)), 95);
    imagesc(signal_cov, [-cov_max, cov_max]);
    colormap(gca, redblue);
    colorbar;

    % Update title to be more informative about signal_cov source
    if user_provided.true_signal
        title_suffix = sprintf('\n(Derived from user-provided GT signal)');
    else
        title_suffix = '';
    end
    title(['Signal Covariance Matrix' title_suffix]);
    xlabel('Unit');
    ylabel('Unit');
    axis square;

    % Plot 4: Noise covariance matrix
    subplot(3, 4, 6);
    cov_max = prctile(abs(noise_cov(:)), 95);
    imagesc(noise_cov, [-cov_max, cov_max]);
    colormap(gca, redblue);
    colorbar;
    title('Noise Covariance Matrix');
    xlabel('Unit');
    ylabel('Unit');
    axis square;

    % Calculate shared colorbar limits for signal and trial data
    % Combine the ground truth signal and example trial data to find common limits
    all_data = [true_signal'; example_trial];
    data_min = prctile(all_data(:), 1);  % Use 1st percentile instead of min to avoid outliers
    data_max = prctile(all_data(:), 99);  % Use 99th percentile instead of max to avoid outliers
    data_abs_max = max(abs(data_min), abs(data_max));

    % Use symmetric limits for better visualization
    signal_clim = [-data_abs_max, data_abs_max];

    % Plot 5: Example of ground truth signal - show full matrix
    subplot(3, 4, 7);
    imagesc(true_signal', signal_clim);
    colormap(gca, redblue);
    colorbar;
    title('Ground Truth Signal');
    xlabel('Condition');
    ylabel('Unit');

    % Plot 6: Example trial data - show full matrix
    subplot(3, 4, 8);
    imagesc(example_trial, signal_clim);
    colormap(gca, redblue);
    colorbar;
    title('Example Single Trial (with noise)');
    xlabel('Condition');
    ylabel('Unit');

    % Plot 7: Signal eigenvectors
    subplot(3, 4, 9);
    imagesc(U_signal, [-0.3, 0.3]);
    colormap(gca, redblue);
    colorbar;
    title('Signal Eigenvectors');
    xlabel('Dimension');
    ylabel('Unit');

    % Plot 8: Noise eigenvectors
    subplot(3, 4, 10);
    imagesc(U_noise, [-0.3, 0.3]);
    colormap(gca, redblue);
    colorbar;
    title('Noise Eigenvectors');
    xlabel('Dimension');
    ylabel('Unit');

    % Plot 9: Alignment visualization (dot products between signal and noise eigenvectors)
    subplot(3, 4, 11);
    % Calculate full alignment matrix for all dimensions
    full_alignment_matrix = abs(U_signal' * U_noise);

    % Determine how many dimensions to display in the visualization
    % If nvox is large, subsample the matrix but ensure we include the aligned dimensions
    max_dims_to_show = min(25, nvox);  % Limit to 25x25 at most for readability

    if nvox <= max_dims_to_show
        % If we have fewer dimensions than the limit, show all
        alignment_matrix = full_alignment_matrix;
        dims_shown = nvox;
    else
        % Otherwise, show a subset with emphasis on aligned dimensions
        if align_k > 0
            % Always include the aligned dimensions
            indices = 1:align_k;

            % Add additional dimensions, evenly spaced
            remaining_spots = max_dims_to_show - align_k;
            if remaining_spots > 0
                % Determine spacing for remaining dimensions
                step = (nvox - align_k) / (remaining_spots + 1);
                for i = 1:remaining_spots
                    idx = align_k + floor(i * step);
                    indices(end+1) = min(idx, nvox);  % Ensure we don't exceed bounds
                end
            end

            % Sort indices to maintain proper order
            indices = sort(indices);
        else
            % No alignment, just evenly space the indices
            indices = round(linspace(1, nvox, max_dims_to_show));
        end

        % Extract the submatrix
        alignment_matrix = full_alignment_matrix(indices, indices);
        dims_shown = length(indices);
    end

    % Create the visualization
    imagesc(alignment_matrix, [0, 1]);
    colormap(gca, viridis_colormap());
    colorbar;
    title('Eigenvector Alignment (dot products)');
    xlabel('Noise dimension');
    ylabel('Signal dimension');
    axis square;

    % Add diagonal values text to show exact alignment of corresponding eigenvectors
    if align_k > 0
        hold on;
        for i = 1:min(dims_shown, align_k)
            text(i, i, sprintf('%.2f', alignment_matrix(i, i)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'Color', 'white', 'FontWeight', 'bold');
        end
    end

    % Show the total number of dimensions in title if we're displaying a subset
    if dims_shown < nvox
        title(sprintf('Eigenvector Alignment (showing %d/%d dimensions)', dims_shown, nvox));
    end

    % Plot 10: Trial-averaged data - also use the same colorbar limits
    subplot(3, 4, 12);
    imagesc(trial_avg, signal_clim);
    colormap(gca, redblue);
    colorbar;
    title(sprintf('Trial-averaged Data (%d trials)', ntrial));
    xlabel('Condition');
    ylabel('Unit');
end
