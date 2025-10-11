function [results] = psn(data, V, opt, wantfig)
% PSN Denoise neural data using Partitioning Signal and Noise (PSN).
%
% This function requires the GSN toolbox. The GSN dependency is included as a
% git submodule. If you encounter errors about missing 'performgsn', please ensure you
% have cloned with submodules:
%   git clone --recurse-submodules <repository-url>
% Or if already cloned:
%   git submodule update --init --recursive
%
% Algorithm Details:
% -----------------
% The GSN denoising algorithm works by identifying and separating dimensions in the neural
% data that correspond to signal and noise. The algorithm works as follows:
%
% 1. Signal and Noise Estimation
%     - The parameters of signal and noise multivariate gaussians are
%     estimated (means and covariances; see performgsn.m)
%
% 2. Denoising Basis Selection (<V> parameter):
%     - V=0: Uses eigenvectors of signal covariance (cSb)
%     - V=1: Uses eigenvectors of signal covariance transformed by inverse noise covariance
%     - V=2: Uses eigenvectors of noise covariance (cNb)
%     - V=3: Uses naive PCA on trial-averaged data
%     - V=4: Uses random orthonormal basis
%     - V=5: Uses ICA (Independent Component Analysis) basis
%     - V=matrix: Uses user-supplied orthonormal basis
%
% 3. Dimension Selection:
%     The algorithm must decide how many dimensions to retain and call "signal".
%     This can be done in two ways:
%
%     a) Cross-validation (<cv_mode> = 0 or 1):
%         - Splits trials into training and testing sets
%         - For training set:
%             * Projects data onto different numbers of basis dimensions
%             * Creates denoising matrix for each dimensionality
%         - For test set:
%             * Measures how well denoised training data predicts test data
%             * Uses mean squared error (MSE) as prediction metric by
%             default
%         - Selects number of dimensions that gives best prediction
%         - Can be done per-unit or for whole population
%
%     b) Magnitude Thresholding (<cv_mode> = -1):
%         - Computes "magnitude" for each dimension:
%             * Either the eigenvalues of the denoising basis (proxy for amount of signal)
%             * Or signal variance estimated from the data projected into
%             the basis
%         - Sets threshold as fraction of maximum magnitude
%         - Keeps dimensions above threshold either:
%             * Contiguously from strongest (leftmost) dimension
%             * Or any dimension above threshold
%
% 4. Denoising:
%     - Creates denoising matrix using selected dimensions
%     - For trial-averaged denoising:
%         * Averages data across trials
%         * Projects through denoising matrix
%     - For single-trial denoising:
%         * Projects each trial through denoising matrix, separately
%     - Returns denoised data and diagnostic information
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <data> - shape [nunits x nconds x ntrials]. This indicates the measured
%   responses to different conditions on distinct trials.
%   The number of trials (<ntrials>) must be at least 2.
% <V> - shape [nunits x nunits] or scalar. Indicates the set of basis functions to use.
%   0 means perform GSN and use the eigenvectors of the
%     signal covariance estimate (cSb)
%   1 means perform GSN and use the eigenvectors of the
%     signal covariance estimate, transformed by the inverse of
%     the noise covariance estimate (inv(cNb)*cSb)
%   2 means perform GSN and use the eigenvectors of the
%     noise covariance estimate (cNb)
%   3 means naive PCA (i.e. eigenvectors of the covariance
%     of the trial-averaged data)
%   4 means use a randomly generated orthonormal basis [nunits x nunits]
%   5 means use ICA (Independent Component Analysis) basis.
%     Requires FastICA toolbox. Will fall back to PCA if not available.
%   B means use user-supplied basis B. The dimensionality of B
%     should be [nunits x D] where D >= 1. The columns of B should
%     unit-length and pairwise orthogonal.
%   Default: 0.
% <opt> - struct with the following optional fields:
%   <cv_mode> - scalar. Indicates how to determine the optimal threshold:
%     0 means cross-validation using n-1 (train) / 1 (test) splits of trials.
%     1 means cross-validation using 1 (train) / n-1 (test) splits of trials.
%    -1 means do not perform cross-validation and instead set the threshold
%       based on when the magnitudes of components drop below
%       a certain fraction (see <mag_frac>).
%     Default: 0.
%   <cv_threshold_per> - string. 'population' or 'unit', specifying
%     whether to use unit-wise thresholding (possibly different thresholds
%     for different units) or population thresholding (one threshold for
%     all units). Matters only when <cv_mode> is 0 or 1. Default: 'unit'.
%   <unit_groups> - shape [nunits x 1]. Integer array specifying which units should
%     receive the same cv threshold. This is only applicable when <cv_threshold_per>
%     is 'unit'. Units with the same integer value get the same cv threshold
%     (computed by averaging scores for those groups of units). If <cv_threshold_per>
%     is 'population', all units should have unit_group = 0. Default: (0:nunits-1)'
%     (each unit gets its own threshold).
%   <cv_thresholds> - shape [1 x n_thresholds]. Vector of thresholds to evaluate in
%     cross-validation. Matters only when <cv_mode> is 0 or 1.
%     Each threshold is a non-negative integer indicating a potential
%     number of dimensions to retain. Should be in sorted order and
%     elements should be unique. Default: 0:D where D is the
%     maximum number of dimensions.
%   <cv_scoring_fn> - function handle. For <cv_mode> 0 or 1 only.
%     It is a function handle to compute denoiser performance.
%     Default: @negative_mse_columns.
%   <mag_type> - scalar. Indicates how to obtain component magnitudes.
%     Matters only when <cv_mode> is -1.
%     0 means use signal variance computed from the data
%     1 means use eigenvalues (<V> must be 0, 1, 2, or 3)
%     Default: 0.
%   <mag_frac> - scalar. Indicates the fraction of total variance to retain.
%     Matters only when <cv_mode> is -1. The algorithm will sort dimensions
%     by magnitude and select the top dimensions that cumulatively account
%     for this fraction of the total variance.
%     Default: 0.95.
%   <denoisingtype> - scalar. Indicates denoising type:
%     0 means denoising in the trial-averaged sense
%     1 means single-trial-oriented denoising
%     Note that if <cv_mode> is 0, you probably want <denoisingtype> to be 0,
%     and if <cv_mode> is 1, you probably want <denoisingtype> to be 1, but
%     the code is deliberately flexible for users to specify what they want.
%     Default: 0.
%   <truncate> - scalar. Number of early PCs to remove from the retained dimensions.
%     If set to 1, the first PC will be excluded from denoising in addition to
%     whatever later dimensions are deemed optimal to remove via cross validation.
%     This is useful for removing unwanted global signals or artifacts.
%     Default: 0.
%   <ranking> - string. Method for ranking basis dimensions:
%     'eigs' means rank by eigenvalue magnitude (decreasing)
%     'eig-inv' means rank by eigenvalue magnitude (increasing)
%     'signal' means rank by signal variance (decreasing)
%     'ncsnr' means rank by noise-ceiling SNR (decreasing)
%     'sig-noise' means rank by difference between signal % and noise % (decreasing)
%     Default: 'ncsnr' for most cases, 'sig-noise' for V=2 or V=4 with unit-wise thresholding
% <wantfig> - bool. Whether to generate diagnostic figures showing the denoising results.
%   Default: true.
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% Return in all cases:
%   <denoiser> - shape [nunits x nunits]. This is the denoising matrix.
%   <fullbasis> - shape [nunits x dims]. This is the full set of basis functions.
%
% In the case that <denoisingtype> is 0, we return:
%   <denoiseddata> - shape [nunits x nconds]. This is the trial-averaged data
%     after applying the denoiser.
%
% In the case that <denoisingtype> is 1, we return:
%   <denoiseddata> - shape [nunits x nconds x ntrials]. This is the
%     single-trial data after applying the denoiser.
%
% In the case that <cv_mode> is 0 or 1 (cross-validation):
%   If <cv_threshold_per> is 'population', we return:
%     <best_threshold> - shape [1 x 1]. The optimal threshold (a single integer),
%       indicating how many dimensions are retained.
%     <signalsubspace> - shape [nunits x best_threshold]. This is the final set of basis
%       functions selected for denoising (i.e. the subspace into which
%       we project). The number of basis functions is equal to <best_threshold>.
%     <dimreduce> - shape [best_threshold x nconds] or [best_threshold x nconds x ntrials]. This
%       is the trial-averaged data (or single-trial data) after denoising.
%       Importantly, we do not reconstruct the original units but leave
%       the data projected into the set of reduced dimensions.
%   If <cv_threshold_per> is 'unit', we return:
%     <best_threshold> - shape [1 x nunits]. The optimal threshold for each unit.
%   In both cases ('population' or 'unit'), we return:
%     <denoised_cv_scores> - shape [n_thresholds x ntrials x nunits].
%       Cross-validation performance scores for each threshold.
%
% In the case that <cv_mode> is -1 (magnitude-based):
%   <mags> - shape [1 x dims]. Component magnitudes used for thresholding.
%   <dimsretained> - shape [1 x n_retained]. The indices of the dimensions retained.
%   <signalsubspace> - shape [nunits x n_retained]. This is the final set of basis
%     functions selected for denoising (i.e. the subspace into which
%     we project).
%   <dimreduce> - shape [n_retained x nconds] or [n_retained x nconds x ntrials]. This
%     is the trial-averaged data (or single-trial data) after denoising.
%     Importantly, we do not reconstruct the original units but leave
%     the data projected into the set of reduced dimensions.
%
% -------------------------------------------------------------------------
% Examples:
% -------------------------------------------------------------------------
%
%   % Basic usage with default options
%   data = randn(100, 200, 3);  % 100 voxels, 200 conditions, 3 trials
%   opt.cv_mode = 0;  % n-1 train / 1 test split
%   opt.cv_threshold_per = 'unit';  % Unit-wise thresholding
%   opt.cv_thresholds = 0:100;  % Test all possible dimensions (including 0)
%   opt.cv_scoring_fn = @negative_mse_columns;  % Use negative MSE as scoring function
%   opt.denoisingtype = 1;  % Single-trial denoising
%   results = psn(data, [], opt);
%
%   % Using magnitude thresholding
%   opt = struct();
%   opt.cv_mode = -1;  % Use magnitude thresholding
%   opt.mag_frac = 0.95;  % Keep components that account for 95% of variance
%   opt.mag_type = 0;  % Use signal variance
%   results = psn(data, 0, opt);
%
%   % Single-trial denoising with population threshold
%   opt = struct();
%   opt.denoisingtype = 1;  % Single-trial mode
%   opt.cv_threshold_per = 'population';  % Same dims for all units
%   results = psn(data, 0, opt);
%   denoised_trials = results.denoiseddata;  % [nunits x nconds x ntrials]
%
%   % Custom basis
%   nunits = size(data, 1);
%   [custom_basis, ~] = qr(randn(nunits));
%   results = psn(data, custom_basis);
%
% -------------------------------------------------------------------------
% History:
% -------------------------------------------------------------------------
%
%   - 2025/01/06 - Initial version.
%   - 2025/01/07 - Refactored to match Python structure.

    % Setup GSN dependency path
    gsn_matlab_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'external', 'gsn', 'matlab');
    if exist(gsn_matlab_path, 'dir')
        addpath(gsn_matlab_path);
        gsn_utilities_path = fullfile(gsn_matlab_path, 'utilities');
        if exist(gsn_utilities_path, 'dir')
            addpath(gsn_utilities_path);
        end
    else
        error(['GSN dependency not found at: %s\n' ...
               'Please ensure GSN submodule is initialized:\n' ...
               '  git submodule update --init --recursive'], gsn_matlab_path);
    end

    % Setup FastICA dependency path (for V=5 ICA basis)
    fastica_path = fullfile(fileparts(mfilename('fullpath')), 'FastICA_25');
    if exist(fastica_path, 'dir')
        addpath(fastica_path);
    end

    % 1) Validate data
    [nunits, nconds, ntrials] = validate_data(data);

    % 2) Set defaults for optional inputs
    if ~exist('V','var') || isempty(V)
        V = 0;
    end
    if ~exist('wantfig','var') || isempty(wantfig)
        wantfig = true;
    end
    if ~exist('opt','var') || isempty(opt)
        opt = struct();
    end

    % 3) Validate and set default options
    opt = validate_and_set_defaults(opt, nunits);

    % 4) Compute unit means (removed during denoising, added back later)
    trial_avg = mean(data, 3);
    unit_means = mean(trial_avg, 2);
    results.unit_means = unit_means;

    % 5) Compute or validate basis
    gsn_results = [];
    ica_mixing = [];
    if isnumeric(V) && isscalar(V)
        [basis, mags, basis_source, gsn_results] = compute_basis_from_mode(data, V, unit_means);

        % Set default ranking based on V value and threshold type if not explicitly provided
        if ~isfield(opt, 'ranking')
            % Special case: V=2 (noise) or V=4 (random) with unit thresholding -> use sig-noise
            if ismember(V, [2, 4]) && strcmp(opt.cv_threshold_per, 'unit')
                opt.ranking = 'sig-noise';
            else
                % Default for all other cases: use ncsnr (noise-ceiling SNR)
                % This has been empirically validated as the most robust across scenarios
                opt.ranking = 'ncsnr';
            end
        end

        % Rank basis dimensions according to the specified ranking method
        [basis, mags, ranked_source] = rank_basis_dimensions(basis, basis_source, data, mags, opt.ranking);

        % For ICA, store the ranked mixing matrix separately
        if V == 5
            ica_mixing = ranked_source;
            % For ICA, we don't want basis_source in visualization, replace with []
            results.basis_source = [];
            results.ica_mixing = ica_mixing;
        else
            % For other basis types, update basis_source with ranked version
            results.basis_source = ranked_source;
            results.ica_mixing = [];
        end
    else
        [basis, mags] = validate_and_normalize_custom_basis(V, data, nunits, unit_means);
        results.basis_source = [];
        results.ica_mixing = [];

        % Set default ranking for user-supplied basis
        if ~isfield(opt, 'ranking')
            % Use ncsnr as default (most robust across scenarios)
            opt.ranking = 'ncsnr';
        end

        % Rank user-supplied basis if ranking is specified
        if ~strcmp(opt.ranking, 'eigs') || ~issorted(mags, 'descend')
            [basis, mags, ~] = rank_basis_dimensions(basis, [], data, mags, opt.ranking);
        end
    end

    stored_mags = mags;

    % 6) Set default cross-validation thresholds if not provided
    if ~isfield(opt, 'cv_thresholds')
        opt.cv_thresholds = 0:size(basis, 2);
    else
        validate_cv_thresholds(opt.cv_thresholds);
    end

    % 7) Initialize return structure
    results.denoiser = [];
    results.cv_scores = [];
    results.best_threshold = [];
    results.denoiseddata = [];
    results.fullbasis = basis;
    results.signalsubspace = [];
    results.dimreduce = [];
    results.mags = [];
    results.dimsretained = [];

    % 8) Perform cross-validation or magnitude thresholding
    if opt.cv_mode >= 0
        [denoiser, cv_scores, best_threshold, denoiseddata, fullbasis_out, signalsubspace, dimreduce] = ...
            perform_cross_validation(data, basis, opt, unit_means);

        results.denoiser = denoiser;
        results.cv_scores = cv_scores;
        results.best_threshold = best_threshold;
        results.denoiseddata = denoiseddata;
        results.fullbasis = fullbasis_out;
        results.mags = stored_mags;

        if strcmp(opt.cv_threshold_per, 'population')
            results.signalsubspace = signalsubspace;
            results.dimreduce = dimreduce;
        end

    else
        [denoiser, cv_scores, best_threshold, denoiseddata, fullbasis_out, signalsubspace, dimreduce, mags_out, dimsretained] = ...
            perform_magnitude_thresholding(data, basis, gsn_results, opt, V, unit_means, stored_mags);

        results.denoiser = denoiser;
        results.cv_scores = cv_scores;
        results.best_threshold = best_threshold;
        results.denoiseddata = denoiseddata;
        results.fullbasis = fullbasis_out;
        results.mags = mags_out;
        results.dimsretained = dimsretained;
        results.signalsubspace = signalsubspace;
        results.dimreduce = dimreduce;
    end

    % Store the input data and parameters in results for later visualization
    results.input_data = data;
    results.V = V;
    results.opt = opt;

    if wantfig
        visualization(data, results);
    end
end


% =========================================================================
% Helper Functions
% =========================================================================

function [nunits, nconds, ntrials] = validate_data(data)
    % VALIDATE_DATA Check data shape and values.

    if any(~isfinite(data(:)))
        error('Data contains infinite or NaN values.');
    end

    [nunits, nconds, ntrials] = size(data);

    if ntrials < 2
        error('Data must have at least 2 trials.');
    end

    if nconds < 2
        error('Data must have at least 2 conditions to estimate covariance.');
    end
end


function [evecs_sorted, evals_sorted, matrix_sym] = compute_symmetric_eigen(matrix)
    % COMPUTE_SYMMETRIC_EIGEN Compute eigendecomposition with consistent sorting and sign.
    %
    % Inputs:
    %   <matrix> - square matrix to decompose
    %
    % Returns:
    %   <evecs_sorted> - eigenvectors sorted by magnitude, with standardized signs
    %   <evals_sorted> - eigenvalues sorted by magnitude
    %   <matrix_sym> - symmetrized version of input matrix

    % Force symmetry for numerical stability
    matrix_sym = (matrix + matrix') / 2;

    % Compute eigendecomposition
    [evecs, evals] = eig(matrix_sym, 'vector');

    % Sort by magnitude (descending)
    [~, idx] = sort(abs(evals), 'descend');
    evecs_sorted = evecs(:, idx);
    evals_sorted = abs(evals(idx));

    % Standardize eigenvector signs (make mean positive)
    evecs_sorted = standardize_eigenvector_signs(evecs_sorted);
end


function evecs = standardize_eigenvector_signs(evecs)
    % STANDARDIZE_EIGENVECTOR_SIGNS Make each eigenvector have positive mean.
    %
    % Inputs:
    %   <evecs> - matrix of eigenvectors (one per column)
    %
    % Returns:
    %   <evecs> - eigenvectors with standardized signs

    for i = 1:size(evecs, 2)
        if mean(evecs(:, i)) < 0
            evecs(:, i) = -evecs(:, i);
        end
    end
end


function [basis, mags, basis_source, gsn_results] = compute_basis_from_mode(data, V, unit_means)
    % COMPUTE_BASIS_FROM_MODE Compute denoising basis based on V mode.
    %
    % Inputs:
    %   <data> - neural response data
    %   <V> - integer mode (0-5)
    %   <unit_means> - mean response for each unit
    %
    % Returns:
    %   <basis> - orthonormal basis vectors
    %   <mags> - component magnitudes
    %   <basis_source> - source matrix used to compute basis
    %   <gsn_results> - GSN computation results (if applicable)

    if ~ismember(V, [0, 1, 2, 3, 4, 5])
        error('V must be in [0..5] (int) or a 2D numeric array.');
    end

    gsn_results = [];
    nunits = size(data, 1);

    % Modes 0-2 require GSN
    if ismember(V, [0, 1, 2])
        gsn_opt = struct();
        gsn_opt.wantverbose = 0;
        gsn_opt.wantshrinkage = 1;
        gsn_opt.random_seed = 42;
        gsn_results = performgsn(data, gsn_opt);
        cSb = gsn_results.cSb;
        cNb = gsn_results.cNb;
    end

    if V == 0
        % Signal covariance
        [basis, mags, basis_source] = compute_symmetric_eigen(cSb);

    elseif V == 1
        % Whitened signal covariance
        cNb_inv = pinv(cNb);
        matM = cNb_inv * cSb;
        [basis, mags, basis_source] = compute_symmetric_eigen(matM);

    elseif V == 2
        % Noise covariance
        [basis, mags, basis_source] = compute_symmetric_eigen(cNb);

    elseif V == 3
        % PCA on trial-averaged data
        trial_avg = mean(data, 3);
        trial_avg_demeaned = trial_avg - unit_means;
        cov_matrix = cov(trial_avg_demeaned.');
        [basis, mags, basis_source] = compute_symmetric_eigen(cov_matrix);

    elseif V == 4
        % Random orthonormal basis
        rng('default');
        rng(42, 'twister');
        rand_mat = randn(nunits);
        [basis, ~] = qr(rand_mat, 0);
        basis = basis(:, 1:nunits);
        mags = ones(nunits, 1);
        basis_source = [];

    else  % V == 5
        % ICA (Independent Component Analysis) basis
        trial_avg = mean(data, 3);
        trial_avg_demeaned = trial_avg - unit_means;
        nconds = size(data, 2);

        n_components = min(nunits, nconds);

        % Use MATLAB's rica (Robust ICA) or fastica if available
        % We'll match Python's sklearn FastICA behavior as closely as possible
        try
            % Try using fastica if available (from FastICA toolbox)
            % Note: FastICA expects each row to be an observed signal (unit)
            % trial_avg_demeaned is (nunits x nconds), which is the correct format
            % Use deterministic random initialization matching Python (seed 42)
            rng('default');
            rng(42, 'twister');
            [~, A, ~] = fastica(trial_avg_demeaned, 'numOfIC', n_components, ...
                               'verbose', 'off', 'maxNumIterations', 1000, ...
                               'epsilon', 1e-4, 'initGuess', randn(n_components, n_components));

            % A is the mixing matrix (nunits x n_components)
            mixing_directions = A;
        catch
            % Fallback to PCA if FastICA is not available
            warning('FastICA not available, falling back to PCA for V=5');
            cov_matrix = cov(trial_avg_demeaned.');
            [basis, mags, basis_source] = compute_symmetric_eigen(cov_matrix);
            return;
        end

        % Normalize mixing directions to unit length
        mixing_directions_normalized = mixing_directions ./ ...
            sqrt(sum(mixing_directions.^2, 1));

        % Apply QR decomposition to make orthonormal
        [basis, ~] = qr(mixing_directions_normalized, 0);

        % Store the original normalized mixing matrix in basis_source
        basis_source = mixing_directions_normalized;

        % Add random basis vectors if necessary to complete the basis
        if size(basis, 2) < nunits
            remaining_dims = nunits - size(basis, 2);
            rng('default');
            rng(43, 'twister');
            rand_mat = randn(nunits, remaining_dims);

            % Orthogonalize against existing basis using modified Gram-Schmidt
            for i = 1:remaining_dims
                v = rand_mat(:, i);
                % Project out existing basis vectors
                for j = 1:size(basis, 2)
                    v = v - dot(v, basis(:, j)) * basis(:, j);
                end
                % Normalize
                norm_v = norm(v);
                if norm_v > 1e-10
                    v = v / norm_v;
                    rand_mat(:, i) = v;
                else
                    % Generate a new random vector if current one is too small
                    v = randn(nunits, 1);
                    for j = 1:size(basis, 2)
                        v = v - dot(v, basis(:, j)) * basis(:, j);
                    end
                    v = v / norm(v);
                    rand_mat(:, i) = v;
                end
            end

            basis = [basis, rand_mat];
        end

        % Placeholder magnitudes (will be recomputed with noise ceiling in main function)
        projections_avg = trial_avg_demeaned' * basis;  % (nconds x nunits)
        mags = var(projections_avg, 0, 1)';
    end
end


function [basis, mags] = validate_and_normalize_custom_basis(V, data, nunits, unit_means)
    % VALIDATE_AND_NORMALIZE_CUSTOM_BASIS Validate and potentially normalize custom basis.
    %
    % Inputs:
    %   <V> - custom basis matrix
    %   <data> - neural response data
    %   <nunits> - number of units
    %   <unit_means> - mean response for each unit
    %
    % Returns:
    %   <basis> - validated/normalized basis
    %   <mags> - component magnitudes

    if ~ismatrix(V)
        error('If V is not int, it must be a numeric matrix.');
    end
    if size(V, 1) ~= nunits
        error('Basis must have %d rows, got %d.', nunits, size(V, 1));
    end
    if size(V, 2) < 1
        error('Basis must have at least 1 column.');
    end

    % Check and normalize if needed
    vector_norms = sqrt(sum(V.^2, 1));
    if any(abs(vector_norms - 1) > 1e-10)
        fprintf('Normalizing basis vectors to unit length...\n');
        V = V ./ vector_norms;
    end

    % Check and enforce orthogonality if needed
    gram = V' * V;
    if ~all(abs(gram - eye(size(gram))) < 1e-10, 'all')
        fprintf('Adjusting basis vectors to ensure orthogonality...\n');
        V = make_orthonormal(V);
    end

    basis = V;

    % Compute magnitudes based on variance in basis
    trial_avg = mean(data, 3);
    trial_avg_reshaped = trial_avg.';
    proj_data = trial_avg_reshaped * basis;
    mags = var(proj_data, 0, 1).';
end


function opt = validate_and_set_defaults(opt, nunits)
    % VALIDATE_AND_SET_DEFAULTS Validate options and set defaults.
    %
    % Inputs:
    %   <opt> - options struct
    %   <nunits> - number of units
    %
    % Returns:
    %   <opt> - validated options with defaults set

    % Validate cv_threshold_per
    if isfield(opt, 'cv_threshold_per')
        if ~any(strcmp(opt.cv_threshold_per, {'unit','population'}))
            error('cv_threshold_per must be ''unit'' or ''population''');
        end
    end

    % Set defaults
    if ~isfield(opt, 'cv_scoring_fn')
        opt.cv_scoring_fn = @negative_mse_columns;
    end
    if ~isfield(opt, 'cv_mode')
        opt.cv_mode = 0;
    end
    if ~isfield(opt, 'cv_threshold_per')
        opt.cv_threshold_per = 'unit';
    end
    if ~isfield(opt, 'mag_type')
        opt.mag_type = 0;
    end
    if ~isfield(opt, 'mag_frac')
        opt.mag_frac = 0.95;
    end
    if ~isfield(opt, 'denoisingtype')
        opt.denoisingtype = 0;
    end
    if ~isfield(opt, 'truncate')
        opt.truncate = 0;
    end

    % Set default unit_groups based on cv_threshold_per
    if ~isfield(opt, 'unit_groups')
        if strcmp(opt.cv_threshold_per, 'population')
            opt.unit_groups = zeros(nunits, 1);
        else
            opt.unit_groups = (0:nunits-1)';
        end
    end

    % Validate unit_groups
    unit_groups = opt.unit_groups(:);
    if length(unit_groups) ~= nunits
        error('unit_groups must have length %d, got %d', nunits, length(unit_groups));
    end
    if any(unit_groups < 0)
        error('unit_groups must contain only non-negative integers');
    end
    if strcmp(opt.cv_threshold_per, 'population') && any(unit_groups ~= 0)
        error('When cv_threshold_per=''population'', all unit_groups must be 0');
    end

    opt.unit_groups = unit_groups;
end


function validate_cv_thresholds(thresholds)
    % VALIDATE_CV_THRESHOLDS Check that CV thresholds are valid.

    if any(thresholds < 0)
        error('cv_thresholds must be non-negative integers.');
    end
    if any(thresholds ~= round(thresholds))
        error('cv_thresholds must be integers.');
    end
    if any(diff(thresholds) <= 0)
        error('cv_thresholds must be in sorted order with unique values.');
    end
end


function denoiser = create_denoiser_matrix(basis, retained_dims, truncate)
    % CREATE_DENOISER_MATRIX Build denoising matrix from basis and retained dimensions.
    %
    % Inputs:
    %   <basis> - orthonormal basis vectors
    %   <retained_dims> - indices of dimensions to retain (1-indexed)
    %   <truncate> - number of early PCs to exclude
    %
    % Returns:
    %   <denoiser> - denoising projection matrix

    denoising_fn = zeros(1, size(basis, 2));
    denoising_fn(retained_dims) = 1;
    D = diag(denoising_fn);
    denoiser = basis * D * basis';
end


function denoiseddata = apply_denoiser(data, denoiser, unit_means, denoisingtype)
    % APPLY_DENOISER Apply denoising matrix to data.
    %
    % Inputs:
    %   <data> - neural response data
    %   <denoiser> - denoising projection matrix
    %   <unit_means> - mean response for each unit
    %   <denoisingtype> - 0 for trial-averaged, 1 for single-trial
    %
    % Returns:
    %   <denoiseddata> - denoised data

    [nunits, nconds, ntrials] = size(data);

    if denoisingtype == 0
        % Trial-averaged denoising
        trial_avg = mean(data, 3);
        trial_avg_demeaned = trial_avg - unit_means;
        denoiseddata = (trial_avg_demeaned' * denoiser)' + unit_means;
    else
        % Single-trial denoising
        denoiseddata = zeros(size(data));
        for t = 1:ntrials
            data_demeaned = data(:, :, t) - unit_means;
            denoiseddata(:, :, t) = (data_demeaned' * denoiser)' + unit_means;
        end
    end
end


% =========================================================================
% Main Algorithm Functions
% =========================================================================

function [denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce] = perform_cross_validation(data, basis, opt, unit_means)
% PERFORM_CROSS_VALIDATION Perform cross-validation to determine optimal denoising dimensions.
%
% Uses cross-validation to determine how many dimensions to retain for denoising:
% 1. Split trials into training and testing sets
% 2. Project training data into basis
% 3. Create denoising matrix for each dimensionality
% 4. Measure prediction quality on test set
% 5. Select threshold that gives best predictions
%
% The splitting can be done in two ways:
% - Leave-one-out: Use n-1 trials for training, 1 for testing
% - Keep-one-in: Use 1 trial for training, n-1 for testing
%
% Inputs:
%   <data> - shape [nunits x nconds x ntrials]. Neural response data to denoise.
%   <basis> - shape [nunits x dims]. Orthonormal basis for denoising.
%   <opt> - struct with fields:
%     <cv_mode> - scalar.
%         0: n-1 train / 1 test split
%         1: 1 train / n-1 test split
%     <cv_threshold_per> - string.
%         'unit': different thresholds per unit
%         'population': same threshold for all units
%     <cv_thresholds> - shape [1 x n_thresholds].
%         Dimensions to test
%     <cv_scoring_fn> - function handle.
%         Function to compute prediction error
%     <denoisingtype> - scalar.
%         0: trial-averaged denoising
%         1: single-trial denoising
%     <unit_groups> - shape [nunits x 1]. Integer array specifying which units
%         should receive the same cv threshold. Only applicable when
%         cv_threshold_per='unit'. Units with the same integer value get
%         the same cv threshold.
%   <unit_means> - shape [nunits x 1]. Mean response for each unit.
%
% Returns:
%   <denoiser> - shape [nunits x nunits]. Matrix that projects data onto denoised space.
%   <cv_scores> - shape [n_thresholds x ntrials x nunits]. Cross-validation scores for each threshold.
%   <best_threshold> - shape [1 x nunits] or scalar. Selected threshold(s).
%   <denoiseddata> - shape [nunits x nconds] or [nunits x nconds x ntrials]. Denoised neural responses.
%   <fullbasis> - shape [nunits x dims]. Complete basis used for denoising.
%   <signalsubspace> - shape [nunits x best_threshold] or []. Final basis functions used for denoising.
%   <dimreduce> - shape [best_threshold x nconds] or [best_threshold x nconds x ntrials] or [].
%       Data projected onto signal subspace.

    % Set random seed for reproducibility
    rng('default');
    rng(42, 'twister');

    [nunits, nconds, ntrials] = size(data);
    cv_mode = opt.cv_mode;
    thresholds = opt.cv_thresholds;
    if ~isfield(opt,'cv_scoring_fn')
        opt.cv_scoring_fn = @negative_mse_columns;
    end
    threshold_per = opt.cv_threshold_per;
    scoring_fn = opt.cv_scoring_fn;
    denoisingtype = opt.denoisingtype;
    truncate = opt.truncate;

    % Initialize cv_scores
    cv_scores = zeros(length(thresholds), ntrials, nunits);

    for tr = 1:ntrials
        if cv_mode == 0
            % Denoise average of n-1 trials, test against held out trial
            train_trials = setdiff(1:ntrials, tr);
            train_avg = mean(data(:, :, train_trials), 3);
            test_data = data(:, :, tr);

            for tt = 1:length(thresholds)
                threshold = thresholds(tt);
                safe_thr = min(threshold, size(basis, 2));
                % Create denoising function with truncation
                denoising_fn = zeros(1, size(basis, 2));
                start_idx = truncate + 1;
                end_idx = min(start_idx + safe_thr - 1, size(basis, 2));
                if end_idx >= start_idx
                    denoising_fn(start_idx:end_idx) = 1;
                end
                D = diag(denoising_fn);
                denoiser_tmp = basis * D * basis';

                train_avg_demeaned = train_avg - unit_means;
                train_denoised = (train_avg_demeaned' * denoiser_tmp)';
                cv_scores(tt, tr, :) = scoring_fn(test_data', train_denoised');
            end

        elseif cv_mode == 1
            % Denoise single trial, test against average of n-1 trials
            dataA = data(:, :, tr)';
            dataB = mean(data(:, :, setdiff(1:ntrials, tr)), 3)';

            for tt = 1:length(thresholds)
                threshold = thresholds(tt);
                safe_thr = min(threshold, size(basis,2));
                denoising_fn = zeros(1, size(basis, 2));
                start_idx = truncate + 1;
                end_idx = min(start_idx + safe_thr - 1, size(basis, 2));
                if end_idx >= start_idx
                    denoising_fn(start_idx:end_idx) = 1;
                end
                D = diag(denoising_fn);
                denoiser_tmp = basis * D * basis';

                dataA_demeaned = dataA - unit_means';
                dataA_denoised = dataA_demeaned * denoiser_tmp;
                cv_scores(tt, tr, :) = scoring_fn(dataB, dataA_denoised);
            end
        end
    end

    % Decide best threshold
    if strcmp(threshold_per, 'population')
        % Average over trials and units for population threshold
        avg_scores = mean(mean(cv_scores, 3), 2);
        [~, best_ix] = max(avg_scores);
        best_threshold = thresholds(best_ix);
        safe_thr = min(best_threshold, size(basis,2));

        % Apply truncation
        start_idx = truncate + 1;
        end_idx = min(start_idx + safe_thr - 1, size(basis, 2));
        if end_idx >= start_idx
            denoiser = basis(:, start_idx:end_idx) * basis(:, start_idx:end_idx)';
        else
            denoiser = zeros(nunits, nunits);
        end
    else
        % Unit-wise: average over trials only, then group by unit_groups
        avg_scores = squeeze(mean(cv_scores, 2));
        if size(avg_scores, 2) == 1
            avg_scores = avg_scores(:);
        end

        unit_groups = opt.unit_groups;
        unique_groups = unique(unit_groups);

        best_thresh_unitwise = zeros(nunits, 1);

        for group_id = unique_groups'
            group_mask = unit_groups == group_id;
            group_avg_scores = mean(avg_scores(:, group_mask), 2);
            [~, best_idx] = max(group_avg_scores);
            best_thresh_for_group = thresholds(best_idx);
            best_thresh_unitwise(group_mask) = best_thresh_for_group;
        end

        best_threshold = best_thresh_unitwise';

        % Construct unit-wise denoiser
        denoiser = zeros(nunits, nunits);
        for unit_i = 1:nunits
            safe_thr = min(best_threshold(unit_i), size(basis,2));
            start_idx = truncate + 1;
            end_idx = min(start_idx + safe_thr - 1, size(basis, 2));
            if end_idx >= start_idx
                unit_denoiser = basis(:, start_idx:end_idx) * basis(:, start_idx:end_idx)';
                denoiser(:, unit_i) = unit_denoiser(:, unit_i);
            end
        end
    end

    % Apply denoiser
    denoiseddata = apply_denoiser(data, denoiser, unit_means, denoisingtype);

    fullbasis = basis;
    if strcmp(threshold_per, 'population')
        % Apply truncation to signalsubspace
        start_idx = truncate + 1;
        end_idx = min(start_idx + safe_thr - 1, size(basis, 2));
        if end_idx >= start_idx
            signalsubspace = basis(:, start_idx:end_idx);
        else
            signalsubspace = basis(:, 1:0);
        end

        % Project data onto signal subspace
        if denoisingtype == 0
            trial_avg = mean(data, 3);
            trial_avg_demeaned = trial_avg - unit_means;
            dimreduce = signalsubspace' * trial_avg_demeaned;
        else
            dimreduce = zeros(size(signalsubspace, 2), nconds, ntrials);
            for t = 1:ntrials
                data_demeaned = data(:, :, t) - unit_means;
                dimreduce(:, :, t) = signalsubspace' * data_demeaned;
            end
        end
    else
        signalsubspace = [];
        dimreduce = [];
    end
end


function [denoiser, cv_scores, best_threshold, denoiseddata, basis, signalsubspace, dimreduce, magnitudes, dimsretained] = ...
    perform_magnitude_thresholding(data, basis, gsn_results, opt, V, unit_means, ranked_mags)
% PERFORM_MAGNITUDE_THRESHOLDING Select dimensions using magnitude thresholding.
%
% Implements the magnitude thresholding procedure for PSN denoising.
% Selects dimensions based on cumulative variance explained rather than
% using cross-validation.
%
% Algorithm Details:
% 1. Get magnitudes either:
%    - From signal variance of the data projected into the basis (mag_type=0)
%    - Or precomputed basis eigenvalues (mag_type=1)
% 2. Use magnitudes as-is (already sorted by the ranking method)
% 3. Select the top dimensions that cumulatively account for mag_frac
%    of the total variance
% 4. Create denoising matrix using selected dimensions
%
% Inputs:
%   <data> - shape [nunits x nconds x ntrials]. Neural response data to denoise.
%   <basis> - shape [nunits x dims]. Orthonormal basis for denoising.
%   <gsn_results> - struct. Results from GSN computation containing:
%       <cSb> - shape [nunits x nunits]. Signal covariance matrix.
%       <cNb> - shape [nunits x nunits]. Noise covariance matrix.
%   <opt> - struct with fields:
%       <mag_type> - scalar. How to obtain component magnitudes:
%           0: use signal variance computed from data
%           1: use pre-computed eigenvalues from results
%       <mag_frac> - scalar. Fraction of total variance to retain (e.g., 0.95).
%       <denoisingtype> - scalar. Type of denoising:
%           0: trial-averaged
%           1: single-trial
%   <V> - scalar or matrix. Basis selection mode or custom basis.
%   <unit_means> - shape [nunits x 1]. Mean response for each unit.
%
% Returns:
%   <denoiser> - shape [nunits x nunits]. Matrix that projects data onto denoised space.
%   <cv_scores> - shape [0 x 0]. Empty array (not used in magnitude thresholding).
%   <best_threshold> - shape [1 x n_retained]. Selected dimension indices.
%   <denoiseddata> - shape [nunits x nconds] or [nunits x nconds x ntrials]. Denoised neural responses.
%   <basis> - shape [nunits x dims]. Complete basis used for denoising.
%   <signalsubspace> - shape [nunits x n_retained]. Final basis functions used for denoising.
%   <dimreduce> - shape [n_retained x nconds] or [n_retained x nconds x ntrials].
%       Data projected onto signal subspace.
%   <magnitudes> - shape [1 x dims]. Component magnitudes used for thresholding.
%   <dimsretained> - scalar. Number of dimensions retained.

    [nunits, nconds, ntrials] = size(data);
    mag_type = opt.mag_type;
    mag_frac = opt.mag_frac;
    denoisingtype = opt.denoisingtype;
    truncate = opt.truncate;

    cv_scores = [];

    % Use magnitudes provided from ranking (already sorted by rank_basis_dimensions)
    % Note: The magnitudes are assumed to already be sorted according to the ranking method.
    % For most ranking methods (eigs, signal, ncsnr, sig-noise), this is decreasing order.
    % For eig-inv, this is increasing order. We use them as-is without re-sorting.
    magnitudes = ranked_mags(:);
    sorted_indices = (1:length(magnitudes))';
    sorted_magnitudes = magnitudes;

    total_variance = sum(sorted_magnitudes);
    cumulative_variance = cumsum(sorted_magnitudes);
    cumulative_fraction = cumulative_variance / total_variance;

    % Find how many dimensions we need
    dims_needed = sum(cumulative_fraction < mag_frac) + 1;
    dims_needed = min(dims_needed, length(sorted_magnitudes));

    % Get selected indices and apply truncation
    selected_indices = sorted_indices(1:dims_needed);
    filtered_indices = selected_indices(selected_indices > truncate);

    % If we need more dimensions after truncation, add them
    if length(filtered_indices) < dims_needed
        remaining_indices = sorted_indices(dims_needed+1:end);
        remaining_valid = remaining_indices(remaining_indices > truncate);
        needed_additional = dims_needed - length(filtered_indices);
        additional_indices = remaining_valid(1:min(needed_additional, length(remaining_valid)));
        filtered_indices = [filtered_indices; additional_indices];
    end

    best_threshold = filtered_indices;
    dimsretained = length(best_threshold);

    if dimsretained == 0
        % No dimensions retained
        denoiser = zeros(nunits, nunits);
        if denoisingtype == 0
            denoiseddata = zeros(nunits, nconds) + unit_means;
        else
            denoiseddata = zeros(nunits, nconds, ntrials) + unit_means;
        end
        signalsubspace = basis(:, 1:0);
        if denoisingtype == 0
            dimreduce = zeros(0, nconds);
        else
            dimreduce = zeros(0, nconds, ntrials);
        end
        best_threshold = [];
        return
    end

    % Create denoiser and apply
    denoiser = create_denoiser_matrix(basis, best_threshold, 0);
    denoiseddata = apply_denoiser(data, denoiser, unit_means, denoisingtype);

    % Calculate signal subspace and reduced dimensions
    signalsubspace = basis(:, best_threshold);
    if denoisingtype == 0
        trial_avg = mean(data, 3);
        trial_avg_demeaned = trial_avg - unit_means;
        dimreduce = signalsubspace' * trial_avg_demeaned;
    else
        dimreduce = zeros(length(best_threshold), nconds, ntrials);
        for t = 1:ntrials
            data_demeaned = data(:, :, t) - unit_means;
            dimreduce(:, :, t) = signalsubspace' * data_demeaned;
        end
    end
end


function magnitudes = compute_magnitudes(data, basis, gsn_results, opt, V, unit_means)
    % COMPUTE_MAGNITUDES Calculate component magnitudes for thresholding.
    %
    % Inputs:
    %   <data> - neural response data
    %   <basis> - orthonormal basis
    %   <gsn_results> - GSN computation results
    %   <opt> - options struct
    %   <V> - basis mode or custom basis
    %   <unit_means> - mean response for each unit
    %
    % Returns:
    %   <magnitudes> - component magnitudes

    [~, nconds, ntrials] = size(data);
    mag_type = opt.mag_type;

    if mag_type == 1
        % Use eigenvalues
        if isnumeric(V) && isscalar(V)
            if V == 0
                evals = eig(gsn_results.cSb);
                [~, sort_idx] = sort(abs(evals), 'descend');
                magnitudes = abs(evals(sort_idx));
            elseif V == 1
                cNb_inv = pinv(gsn_results.cNb);
                matM = cNb_inv * gsn_results.cSb;
                evals = eig(matM);
                [~, sort_idx] = sort(abs(evals), 'descend');
                magnitudes = abs(evals(sort_idx));
            elseif V == 2
                evals = eig(gsn_results.cNb);
                [~, sort_idx] = sort(abs(evals), 'descend');
                magnitudes = abs(evals(sort_idx));
            elseif V == 3
                trial_avg = mean(data, 3);
                trial_avg_demeaned = trial_avg - unit_means;
                cov_mat = cov(trial_avg_demeaned.');
                evals = eig(cov_mat);
                [~, sort_idx] = sort(abs(evals), 'descend');
                magnitudes = abs(evals(sort_idx));
            else
                magnitudes = ones(size(basis,2),1);
            end
        else
            trial_avg = mean(data, 3);
            proj = (trial_avg.') * basis;
            magnitudes = var(proj, 0, 1).';
            [~, sort_idx] = sort(abs(magnitudes), 'descend');
            magnitudes = abs(magnitudes(sort_idx));
        end
    else
        % Variance-based threshold
        sigvars = [];
        data_reshaped = permute(data, [2, 3, 1]);

        for i = 1:size(basis, 2)
            this_eigv = basis(:, i);
            proj_data = zeros(nconds, ntrials);
            for j = 1:nconds
                for k = 1:ntrials
                    proj_data(j, k) = dot(squeeze(data_reshaped(j, k, :)), this_eigv);
                end
            end

            noisevar = mean(std(proj_data, 0, 2) .^ 2);
            datavar = std(mean(proj_data, 2), 0, 1) ^ 2;
            signalvar = max(datavar - noisevar / size(proj_data, 2), 0);
            sigvars(end+1) = signalvar;
        end

        magnitudes = sigvars(:);
    end
end


% =========================================================================
% Utility Functions
% =========================================================================

function scores = negative_mse_columns(x, y)
    % NEGATIVE_MSE_COLUMNS Calculate negative mean squared error between columns.
    scores = -mean((x - y).^2, 1);
end


function V_orthonormal = make_orthonormal(V)
    % MAKE_ORTHONORMAL Find the nearest matrix with orthonormal columns.

    [m, n] = size(V);
    if m < n
        error('Input matrix must have at least as many rows as columns');
    end

    [U, ~, Vh] = svd(V, 'econ');
    V_orthonormal = U(:,1:n) * Vh';

    gram = V_orthonormal' * V_orthonormal;
    if ~all(abs(gram - eye(n)) < 1e-10, 'all')
        warning('Result may not be perfectly orthonormal due to numerical precision');
    end
end


function [basis_ranked, magnitudes_ranked, basis_source_ranked] = rank_basis_dimensions(basis, basis_source, data, magnitudes, ranking)
    % RANK_BASIS_DIMENSIONS Rank basis dimensions by various criteria.
    %
    % This function ranks basis dimensions using different methods:
    % - 'eigs': By eigenvalue magnitude (decreasing)
    % - 'eig-inv': By eigenvalue magnitude (increasing)
    % - 'signal': By signal variance (decreasing)
    % - 'ncsnr': By noise-ceiling SNR (decreasing)
    % - 'sig-noise': By difference between signal % and noise % (decreasing)
    %
    % Inputs:
    %   <basis> - Orthonormal basis (nunits x ndims)
    %   <basis_source> - Source matrix before orthonormalization (nunits x ncomponents), can be []
    %   <data> - Raw trial data (nunits x nconds x ntrials)
    %   <magnitudes> - Initial eigenvalues/magnitudes for each dimension
    %   <ranking> - Ranking method - 'eigs', 'eig-inv', 'signal', 'ncsnr', or 'sig-noise'
    %
    % Returns:
    %   <basis_ranked> - Reranked orthonormal basis
    %   <magnitudes_ranked> - Ranking values in appropriate order
    %   <basis_source_ranked> - Reordered source matrix (or [])

    [nunits, nconds, ntrials] = size(data);
    ndims = size(basis, 2);

    if strcmp(ranking, 'eigs')
        % Rank by eigenvalue magnitude (decreasing) - magnitudes already computed
        [magnitudes_ranked, sort_idx] = sort(magnitudes, 'descend');

    elseif strcmp(ranking, 'eig-inv')
        % Rank by eigenvalue magnitude (increasing)
        [magnitudes_ranked, sort_idx] = sort(magnitudes, 'ascend');

    elseif ismember(ranking, {'signal', 'ncsnr', 'sig-noise'})
        % Need to compute signal and noise components for each dimension
        data_reshaped = permute(data, [2, 3, 1]);  % (nconds x ntrials x nunits)
        signal_variances = zeros(ndims, 1);
        noise_variances = zeros(ndims, 1);
        ncsnrs = zeros(ndims, 1);

        for i = 1:ndims
            % Get this basis vector
            this_basis_vec = basis(:, i);

            % Project data onto this direction: (nconds x ntrials x nunits) @ (nunits,) = (nconds x ntrials)
            proj_data = zeros(nconds, ntrials);
            for j = 1:nconds
                for k = 1:ntrials
                    proj_data(j, k) = dot(squeeze(data_reshaped(j, k, :)), this_basis_vec);
                end
            end

            % Use compute_noise_ceiling to get signal variance, noise variance, and ncsnr
            % proj_data needs shape (1 x nconds x ntrials) for compute_noise_ceiling
            [~, ncsnr, sigvar, noisevar] = compute_noise_ceiling(reshape(proj_data, [1, nconds, ntrials]));

            signal_variances(i) = sigvar;
            noise_variances(i) = noisevar;
            ncsnrs(i) = ncsnr;
        end

        if strcmp(ranking, 'signal')
            % Rank by signal variance (decreasing)
            [magnitudes_ranked, sort_idx] = sort(signal_variances, 'descend');

        elseif strcmp(ranking, 'ncsnr')
            % Rank by noise-ceiling SNR (decreasing)
            [magnitudes_ranked, sort_idx] = sort(ncsnrs, 'descend');

        elseif strcmp(ranking, 'sig-noise')
            % Rank by difference between signal % and noise %
            total_signal = sum(signal_variances);
            total_noise = sum(noise_variances);

            % Avoid division by zero
            if total_signal > 0
                signal_pcts = 100 * signal_variances / total_signal;
            else
                signal_pcts = zeros(ndims, 1);
            end

            if total_noise > 0
                noise_pcts = 100 * noise_variances / total_noise;
            else
                noise_pcts = zeros(ndims, 1);
            end

            sig_noise_diff = signal_pcts - noise_pcts;
            [magnitudes_ranked, sort_idx] = sort(sig_noise_diff, 'descend');
        end

    else
        error('Invalid ranking method: %s. Must be one of: ''eigs'', ''eig-inv'', ''signal'', ''ncsnr'', ''sig-noise''', ranking);
    end

    % Reorder basis and basis_source
    basis_ranked = basis(:, sort_idx);

    if ~isempty(basis_source) && size(basis_source, 2) == length(sort_idx)
        basis_source_ranked = basis_source(:, sort_idx);
    else
        basis_source_ranked = [];
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
