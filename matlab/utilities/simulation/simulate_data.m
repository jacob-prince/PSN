function varargout = simulate_data(varargin)
% SIMULATE_DATA  Generate simulated neural data with controlled signal and noise properties.
%
%   [train_data, test_data, ground_truth] = simulate_data('generate_data', ...) generates
%     homogeneous populations with controlled signal/noise properties.
%
%   [train_data, test_data, ground_truth] = simulate_data('generate_heterogeneous_populations', ...)
%     generates heterogeneous subpopulations with conflicting optimal bases.
%
% This module provides tools to generate synthetic neural data with specific covariance
% structures for both signal and noise components. The data generation process allows for:
% - Control over signal and noise eigenvalue decay rates
% - Alignment between signal and noise principal components
% - Separate train and test datasets with matched properties
% - Custom signal or noise covariance matrices
% - Heterogeneous subpopulations with conflicting optimal bases
%
% See documentation for generate_data and generate_heterogeneous_populations below
% for detailed parameter descriptions.

    % Route to the requested function
    if nargin == 0
        error('simulate requires a function name as first argument');
    end

    func_name = varargin{1};
    args = varargin(2:end);

    switch func_name
        case 'generate_data'
            [varargout{1:nargout}] = generate_data(args{:});
        case 'generate_heterogeneous_populations'
            [varargout{1:nargout}] = generate_heterogeneous_populations(args{:});
        otherwise
            error('Unknown function: %s', func_name);
    end
end

function [train_data, test_data, ground_truth] = generate_data(varargin)
% GENERATE_DATA  Generate synthetic neural data with controlled signal and noise properties.
%
% Generates synthetic neural data with specified covariance structures for both
% signal and noise components. Supports both automatic generation with controlled
% decay rates and custom user-provided covariance matrices. The function can also
% align signal and noise principal components to create specific correlation patterns.
%
% -------------------------------------------------------------------------
% Inputs (all optional name-value pairs):
% -------------------------------------------------------------------------
%
% <nvox> (optional) - Number of voxels/units. If 'true_signal' is provided, this is
%   automatically inferred from size(true_signal, 2). Default: 50.
%
% <ncond> (optional) - Number of conditions. If 'true_signal' is provided, this is
%   automatically inferred from size(true_signal, 1). Default: 200.
%
% <ntrial> - Number of trials per condition. Default: 5.
%
% <signal_decay> - Rate of eigenvalue decay for signal covariance. Signal eigenvalues
%   are generated as 1/(i^signal_decay) where i is the dimension index.
%   Ignored if 'signal_cov' or 'true_signal' is provided. Default: 2.0.
%
% <noise_decay> - Rate of eigenvalue decay for noise covariance. Noise eigenvalues are
%   generated as noise_multiplier/(i^noise_decay).
%   Ignored if 'noise_cov' is provided. Default: 1.25.
%
% <noise_multiplier> - Scaling factor for noise variance. Higher values increase overall
%   noise level. Default: 3.0.
%
% <align_alpha> - Alignment between signal and noise principal components.
%   1.0 = perfectly aligned (signal and noise PCs point in same direction)
%   0.0 = orthogonal (signal and noise PCs are perpendicular)
%   Ignored if 'noise_cov' is provided. Default: 0.5.
%
% <align_k> - Number of top principal components to align between signal and noise.
%   Set to 0 to disable alignment. Ignored if 'noise_cov' is provided. Default: 10.
%
% <random_seed> (optional) - Random seed for reproducibility. Set to empty [] to use
%   current random state. Default: 42.
%
% <want_fig> - Whether to display a diagnostic figure showing the generated data
%   properties. Default: true.
%
% <signal_cov> (optional) - User-provided signal covariance matrix of shape (nvox, nvox).
%   If provided, overrides 'signal_decay' parameter. Overridden by 'true_signal' if
%   both are provided. Default: [].
%
% <true_signal> (optional) - User-provided ground truth signal of shape (ncond, nvox).
%   If provided, overrides both 'signal_cov' and 'signal_decay'. The signal covariance
%   will be calculated as the sample covariance of this signal. Default: [].
%
% <noise_cov> (optional) - User-provided noise covariance matrix of shape (nvox, nvox).
%   If provided, overrides 'noise_decay' and disables alignment. Default: [].
%
% <cluster_units> - Whether to reorder units based on hierarchical clustering of the
%   signal covariance matrix. This is purely cosmetic for visualization and does not
%   affect data properties. The original unit order is saved in ground_truth.unit_order.
%   Default: false.
%
% <verbose> - Whether to print diagnostic information during generation. Default: true.
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <train_data> - [nvox x ncond x ntrial]. Training dataset with independent noise realizations.
%
% <test_data> - [nvox x ncond x ntrial]. Test dataset with independent noise realizations.
%
% <ground_truth> - struct containing ground truth parameters with fields:
%   .signal         - [ncond x nvox] ground truth signal
%   .signal_cov     - [nvox x nvox] signal covariance matrix
%   .noise_cov      - [nvox x nvox] noise covariance matrix
%   .U_signal       - [nvox x nvox] signal eigenvectors
%   .U_noise        - [nvox x nvox] noise eigenvectors
%   .signal_eigs    - [nvox x 1] signal eigenvalues
%   .noise_eigs     - [nvox x 1] noise eigenvalues
%   .user_provided  - struct with flags for what was user-provided
%   .unit_order     - [nvox x 1] original unit indices (if cluster_units=true)

    % Default parameters
    % Use empty [] for nvox/ncond so we can detect if user provided them
    p = inputParser;
    addParameter(p, 'nvox', []);
    addParameter(p, 'ncond', []);
    addParameter(p, 'ntrial', 5);
    addParameter(p, 'signal_decay', 2.0);
    addParameter(p, 'noise_decay', 1.25);
    addParameter(p, 'noise_multiplier', 3.0);
    addParameter(p, 'align_alpha', 0.5);
    addParameter(p, 'align_k', 10);
    addParameter(p, 'random_seed', 42);
    addParameter(p, 'want_fig', true);
    addParameter(p, 'signal_cov', []);
    addParameter(p, 'true_signal', []);
    addParameter(p, 'noise_cov', []);
    addParameter(p, 'cluster_units', false);
    addParameter(p, 'verbose', true);

    parse(p, varargin{:});
    nvox = p.Results.nvox;
    ncond = p.Results.ncond;
    ntrial = p.Results.ntrial;
    signal_decay = p.Results.signal_decay;
    noise_decay = p.Results.noise_decay;
    noise_multiplier = p.Results.noise_multiplier;
    align_alpha = p.Results.align_alpha;
    align_k = p.Results.align_k;
    random_seed = p.Results.random_seed;
    want_fig = p.Results.want_fig;
    signal_cov = p.Results.signal_cov;
    true_signal = p.Results.true_signal;
    noise_cov = p.Results.noise_cov;
    cluster_units = p.Results.cluster_units;
    verbose = p.Results.verbose;

    if ~isempty(random_seed)
        rng(random_seed);
    end

    % Infer nvox and ncond from true_signal if provided
    if ~isempty(true_signal)
        if ndims(true_signal) ~= 2
            error('true_signal must be a 2D array with shape (ncond, nvox), got shape [%s]', ...
                  num2str(size(true_signal)));
        end

        [true_signal_ncond, true_signal_nvox] = size(true_signal);

        % If user provided nvox/ncond, verify they match true_signal
        % Otherwise, infer from true_signal
        if ~isempty(nvox) && nvox ~= true_signal_nvox
            error('Provided nvox=%d doesn''t match true_signal shape(2)=%d', ...
                  nvox, true_signal_nvox);
        end
        if ~isempty(ncond) && ncond ~= true_signal_ncond
            error('Provided ncond=%d doesn''t match true_signal shape(1)=%d', ...
                  ncond, true_signal_ncond);
        end

        % Use dimensions from true_signal
        nvox = true_signal_nvox;
        ncond = true_signal_ncond;
    end

    % Apply defaults if still empty
    if isempty(nvox)
        nvox = 50;  % Default
    end
    if isempty(ncond)
        ncond = 200;  % Default
    end

    % Check that required parameters are now available
    if isempty(ntrial)
        error('ntrial must be provided');
    end

    % Track what was user-provided before we potentially modify these variables
    user_provided_signal_cov = ~isempty(signal_cov);
    user_provided_true_signal = ~isempty(true_signal);
    user_provided_noise_cov = ~isempty(noise_cov);

    % Check input dimensions if provided
    if ~isempty(signal_cov)
        if ~isequal(size(signal_cov), [nvox, nvox])
            error('Provided signal_cov has shape [%s], expected [%d, %d]', ...
                  num2str(size(signal_cov)), nvox, nvox);
        end
    end

    if ~isempty(true_signal)
        if ~isequal(size(true_signal), [ncond, nvox])
            error('Provided true_signal has shape [%s], expected [%d, %d]', ...
                  num2str(size(true_signal)), ncond, nvox);
        end
    end

    if ~isempty(noise_cov)
        if ~isequal(size(noise_cov), [nvox, nvox])
            error('Provided noise_cov has shape [%s], expected [%d, %d]', ...
                  num2str(size(noise_cov)), nvox, nvox);
        end
    end

    % Generate random orthonormal matrices for signal & noise
    [U_noise, ~, ~] = svd(randn(nvox, nvox));

    % For signal, either use SVD of provided covariance or generate random
    if ~isempty(signal_cov)
        % Use provided signal covariance
        [U_signal, S_signal, ~] = svd(signal_cov);
        signal_eigs = diag(S_signal);
        signal_cov = signal_cov;  % Ensure we have a copy to avoid modifying the input
    else
        % Generate random orthonormal matrix for signal
        [U_signal, ~, ~] = svd(randn(nvox, nvox));
        % Create diagonal eigenvalues
        signal_eigs = 1.0 ./ ((1:nvox)' .^ signal_decay);
        % Build signal covariance matrix
        signal_cov = U_signal * diag(signal_eigs) * U_signal';
    end

    % For noise, either use SVD of provided covariance or generate random
    if user_provided_noise_cov
        % Use provided noise covariance
        [U_noise, S_noise, ~] = svd(noise_cov);
        noise_eigs = diag(S_noise);
        noise_cov = noise_cov;  % Ensure we have a copy to avoid modifying the input

        % Warn if alignment was requested but noise_cov is provided
        if align_k > 0 && verbose
            fprintf('Warning: align_k > 0 but noise_cov was provided. Using provided noise covariance without alignment.\n');
        end
    else
        % Generate noise covariance after potential alignment
        % Align noise PCs to signal PCs if requested
        if align_k > 0
            % Cap align_k to not exceed available dimensions
            effective_k = min(align_k, nvox);
            U_noise = adjust_alignment_gradient_descent(...
                U_signal, U_noise, align_alpha, effective_k, verbose);
        end

        % Create diagonal eigenvalues for noise
        noise_eigs = noise_multiplier ./ ((1:nvox)' .^ noise_decay);
        % Build noise covariance matrix
        noise_cov = U_noise * diag(noise_eigs) * U_noise';
    end

    % Generate the ground truth signal
    if ~isempty(true_signal)
        % Use provided ground truth signal
        true_signal = true_signal;  % Ensure we have a copy

        % Recalculate signal covariance based on the provided true signal
        % This ensures signal_cov matches the actual covariance of true_signal
        signal_cov = cov(true_signal, 0);  % 0 for normalization by N-1

        % Recompute signal eigendecomposition for consistency
        [U_signal, S_signal, ~] = svd(signal_cov);
        signal_eigs = diag(S_signal);

        % Re-align noise after recalculating U_signal (only if noise_cov was not user-provided)
        if align_k > 0 && ~user_provided_noise_cov
            U_noise = adjust_alignment_gradient_descent(...
                U_signal, U_noise, align_alpha, align_k, verbose);
            % Rebuild noise covariance matrix with the realigned eigenvectors
            noise_cov = U_noise * diag(noise_eigs) * U_noise';
        elseif align_k > 0 && user_provided_noise_cov
            if verbose
                fprintf('Warning: align_k > 0 but noise_cov was provided. Skipping noise alignment.\n');
            end
        end
    else
        % Generate from covariance
        true_signal = mvnrnd(zeros(1, nvox), signal_cov, ncond);  % shape (ncond, nvox)
    end

    % Preallocate train/test data in shape (ntrial, nvox, ncond)
    train_data = zeros(ntrial, nvox, ncond);
    test_data = zeros(ntrial, nvox, ncond);

    % Generate data
    for t = 1:ntrial
        % Independent noise for each trial
        train_noise = mvnrnd(zeros(1, nvox), noise_cov, ncond);  % shape (ncond, nvox)
        test_noise = mvnrnd(zeros(1, nvox), noise_cov, ncond);   % shape (ncond, nvox)

        % Add noise to signal
        train_data(t, :, :) = (true_signal + train_noise)';
        test_data(t, :, :)  = (true_signal + test_noise)';
    end

    % Reshape to (nvox, ncond, ntrial)
    train_data = permute(train_data, [2, 3, 1]);
    test_data  = permute(test_data, [2, 3, 1]);

    % Optionally reorder units based on hierarchical clustering
    unit_order = [];
    if cluster_units
        % Cluster based on ground truth signal patterns
        % true_signal shape is (ncond, nvox), so transpose to get (nvox, ncond)
        % Standardize each unit's activity pattern for better clustering
        signal_for_clustering = zscore(true_signal, 0, 1)';  % zscore across conditions for each unit

        % Use correlation distance and average linkage for more balanced clusters
        Z = linkage(signal_for_clustering, 'average', 'correlation');
        unit_order = optimalleaforder(Z, pdist(signal_for_clustering, 'correlation'));

        % Reorder all relevant matrices and arrays
        train_data = train_data(unit_order, :, :);
        test_data = test_data(unit_order, :, :);
        true_signal = true_signal(:, unit_order);
        signal_cov = signal_cov(unit_order, unit_order);
        noise_cov = noise_cov(unit_order, unit_order);
        U_signal = U_signal(unit_order, :);
        U_noise = U_noise(unit_order, :);
    end

    ground_truth = struct();
    ground_truth.signal = true_signal;
    ground_truth.signal_cov = signal_cov;
    ground_truth.noise_cov = noise_cov;
    ground_truth.U_signal = U_signal;
    ground_truth.U_noise = U_noise;
    ground_truth.signal_eigs = signal_eigs;
    ground_truth.noise_eigs = noise_eigs;
    ground_truth.user_provided = struct(...
        'signal_cov', user_provided_signal_cov, ...
        'true_signal', user_provided_true_signal, ...
        'noise_cov', user_provided_noise_cov);

    if ~isempty(unit_order)
        ground_truth.unit_order = unit_order;
    end

    if want_fig
        plot_data_diagnostic(train_data, ground_truth, struct(...
            'nvox', nvox, ...
            'ncond', ncond, ...
            'ntrial', ntrial, ...
            'signal_decay', signal_decay, ...
            'noise_decay', noise_decay, ...
            'noise_multiplier', noise_multiplier, ...
            'align_alpha', align_alpha, ...
            'align_k', align_k, ...
            'random_seed', random_seed, ...
            'user_provided', ground_truth.user_provided, ...
            'clustered', cluster_units));
    end
end

function [train_data, test_data, ground_truth] = generate_heterogeneous_populations(varargin)
% GENERATE_HETEROGENEOUS_POPULATIONS  Generate data with heterogeneous subpopulations.
%
% Generates neural data with heterogeneous subpopulations that have conflicting
% optimal basis orderings. This creates a challenging scenario where different
% groups of units have different signal subspaces, making global/population-based
% approaches suboptimal. Useful for testing unit-wise denoising methods.
%
% -------------------------------------------------------------------------
% Inputs (all optional name-value pairs):
% -------------------------------------------------------------------------
%
% <n_populations> - Number of distinct subpopulations to generate. Default: 3.
%
% <units_per_pop> - Number of units per subpopulation. Total units will be
%   n_populations × units_per_pop. Default: 20.
%
% <ncond> - Number of conditions. Default: 100.
%
% <ntrial> - Number of trials per condition. Default: 3.
%
% <signal_decay> - Rate of eigenvalue decay for signal covariance within each
%   population. Signal eigenvalues are generated as 1/(i^signal_decay). Default: 2.0.
%
% <noise_decay> - Rate of eigenvalue decay for noise covariance. Noise eigenvalues
%   are generated as noise_multiplier/(i^noise_decay). Default: 1.25.
%
% <noise_multiplier> - Scaling factor for noise variance. Higher values increase
%   overall noise level. Default: 3.0.
%
% <population_orthogonality> - Controls how different the population-specific signal
%   subspaces are. 0.0 = identical populations (same signal basis), 1.0 = maximally
%   orthogonal populations (completely different signal bases). Controls the angle
%   between population-specific signal subspaces via random Givens rotations.
%   Default: 0.9.
%
% <random_seed> (optional) - Random seed for reproducibility. Set to empty [] to use
%   current random state. Default: 42.
%
% <want_fig> - Whether to display diagnostic figures. Currently not implemented for
%   heterogeneous populations. Default: false.
%
% <verbose> - Whether to print diagnostic information during generation. Default: true.
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <train_data> - [nvox x ncond x ntrial]. Training dataset with independent noise realizations.
%
% <test_data> - [nvox x ncond x ntrial]. Test dataset with independent noise realizations.
%
% <ground_truth> - struct containing ground truth parameters with fields:
%   .signal                    - [ncond x nvox] ground truth signal
%   .signal_cov                - [nvox x nvox] overall signal covariance
%   .noise_cov                 - [nvox x nvox] overall noise covariance
%   .population_labels         - [nvox x 1] which population each unit belongs to
%   .population_bases          - cell array of [units_per_pop x units_per_pop] optimal basis for each population
%   .population_signals        - cell array of [ncond x units_per_pop] signal for each population
%   .n_populations             - number of populations
%   .units_per_pop             - units per population
%   .population_orthogonality  - orthogonality parameter used

    % Default parameters
    p = inputParser;
    addParameter(p, 'n_populations', 3);
    addParameter(p, 'units_per_pop', 20);
    addParameter(p, 'ncond', 100);
    addParameter(p, 'ntrial', 3);
    addParameter(p, 'signal_decay', 2.0);
    addParameter(p, 'noise_decay', 1.25);
    addParameter(p, 'noise_multiplier', 3.0);
    addParameter(p, 'population_orthogonality', 0.9);
    addParameter(p, 'random_seed', 42);
    addParameter(p, 'want_fig', false);
    addParameter(p, 'verbose', true);

    parse(p, varargin{:});
    n_populations = p.Results.n_populations;
    units_per_pop = p.Results.units_per_pop;
    ncond = p.Results.ncond;
    ntrial = p.Results.ntrial;
    signal_decay = p.Results.signal_decay;
    noise_decay = p.Results.noise_decay;
    noise_multiplier = p.Results.noise_multiplier;
    population_orthogonality = p.Results.population_orthogonality;
    random_seed = p.Results.random_seed;
    want_fig = p.Results.want_fig;
    verbose = p.Results.verbose;

    if ~isempty(random_seed)
        rng(random_seed);
    end

    nvox = n_populations * units_per_pop;

    if verbose
        fprintf('\n%s\n', repmat('=', 1, 80));
        fprintf('GENERATING HETEROGENEOUS POPULATION DATA\n');
        fprintf('%s\n', repmat('=', 1, 80));
        fprintf('  Populations: %d\n', n_populations);
        fprintf('  Units per population: %d\n', units_per_pop);
        fprintf('  Total units: %d\n', nvox);
        fprintf('  Conditions: %d\n', ncond);
        fprintf('  Trials: %d\n', ntrial);
        fprintf('  Population orthogonality: %.2f\n', population_orthogonality);
        fprintf('%s\n\n', repmat('=', 1, 80));
    end

    % Create population-specific signal subspaces
    % Each population will have a different optimal basis ordering
    population_bases = cell(n_populations, 1);
    population_signals = cell(n_populations, 1);
    population_labels = zeros(nvox, 1);

    % Generate a base random orthonormal matrix
    [U_base, ~, ~] = svd(randn(units_per_pop, units_per_pop));

    for pop_idx = 1:n_populations
        % Create population-specific basis by rotating from base
        if pop_idx == 1
            % First population uses the base
            U_pop = U_base;
        else
            % Subsequent populations are rotated versions
            % Use Givens rotations to create controlled orthogonality
            U_pop = U_base;

            % Apply rotation based on population_orthogonality
            % Higher orthogonality = more rotation = more different preferences
            n_rotations = max(1, floor(units_per_pop * population_orthogonality));

            for rot = 1:n_rotations
                % Random Givens rotation
                ij = randperm(units_per_pop, 2);
                i = ij(1);
                j = ij(2);
                theta = rand() * pi * population_orthogonality;

                % Apply rotation in plane (i, j)
                c = cos(theta);
                s = sin(theta);
                G = eye(units_per_pop);
                G(i, i) = c;
                G(i, j) = -s;
                G(j, i) = s;
                G(j, j) = c;
                U_pop = G * U_pop;
            end
        end

        population_bases{pop_idx} = U_pop;

        % Generate population-specific signal with decaying eigenvalues
        signal_eigs_pop = 1.0 ./ ((1:units_per_pop)' .^ signal_decay);
        signal_cov_pop = U_pop * diag(signal_eigs_pop) * U_pop';

        % Generate signal for this population
        signal_pop = mvnrnd(zeros(1, units_per_pop), signal_cov_pop, ncond);  % (ncond, units_per_pop)

        population_signals{pop_idx} = signal_pop;

        % Track which units belong to which population
        start_idx = (pop_idx - 1) * units_per_pop + 1;
        end_idx = start_idx + units_per_pop - 1;
        population_labels(start_idx:end_idx) = pop_idx;
    end

    % Assemble full signal matrix
    true_signal = horzcat(population_signals{:});  % (ncond, nvox)

    % Compute overall signal covariance (will be block-diagonal-ish)
    signal_cov = cov(true_signal, 0);

    % Generate noise covariance (could be global or population-specific)
    % For simplicity, use a global noise structure
    [U_noise, ~, ~] = svd(randn(nvox, nvox));
    noise_eigs = noise_multiplier ./ ((1:nvox)' .^ noise_decay);
    noise_cov = U_noise * diag(noise_eigs) * U_noise';

    % Preallocate train/test data
    train_data = zeros(ntrial, nvox, ncond);
    test_data = zeros(ntrial, nvox, ncond);

    % Generate noisy data
    for t = 1:ntrial
        train_noise = mvnrnd(zeros(1, nvox), noise_cov, ncond);  % (ncond, nvox)
        test_noise = mvnrnd(zeros(1, nvox), noise_cov, ncond);   % (ncond, nvox)

        train_data(t, :, :) = (true_signal + train_noise)';
        test_data(t, :, :) = (true_signal + test_noise)';
    end

    % Reshape to (nvox, ncond, ntrial)
    train_data = permute(train_data, [2, 3, 1]);
    test_data = permute(test_data, [2, 3, 1]);

    % Create ground truth dictionary
    ground_truth = struct();
    ground_truth.signal = true_signal;
    ground_truth.signal_cov = signal_cov;
    ground_truth.noise_cov = noise_cov;
    ground_truth.population_labels = population_labels;
    ground_truth.population_bases = population_bases;
    ground_truth.population_signals = population_signals;
    ground_truth.n_populations = n_populations;
    ground_truth.units_per_pop = units_per_pop;
    ground_truth.population_orthogonality = population_orthogonality;

    if want_fig
        warning('MATLAB plotting for simulate.generate_heterogeneous_populations not yet implemented');
    end
end
