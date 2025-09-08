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
%     Each threshold is a positive integer indicating a potential 
%     number of dimensions to retain. Should be in sorted order and 
%     elements should be unique. Default: 1:D where D is the 
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
%   opt.cv_threshold_per = 'unit';  % Same threshold for all units
%   opt.cv_thresholds = 1:100;  % Test all possible dimensions
%   opt.cv_scoring_fn = @negative_mse_columns;  % Use negative MSE as scoring function
%   opt.denoisingtype = 1;  % Single-trial denoising
%   results = psn(data, [], opt);
%
%   % Using magnitude thresholding
%   opt = struct();
%   opt.cv_mode = -1;  % Use magnitude thresholding
%   opt.mag_frac = 0.1;  % Keep components > 10% of max
%   opt.mag_mode = 0;  % Use contiguous dimensions
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

    % Setup GSN dependency path
    gsn_matlab_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'external', 'gsn', 'matlab');
    if exist(gsn_matlab_path, 'dir')
        addpath(gsn_matlab_path);
        % Also add utilities subdirectory
        gsn_utilities_path = fullfile(gsn_matlab_path, 'utilities');
        if exist(gsn_utilities_path, 'dir')
            addpath(gsn_utilities_path);
        end
    else
        error(['GSN dependency not found at: %s\n' ...
               'Please ensure GSN submodule is initialized:\n' ...
               '  git submodule update --init --recursive'], gsn_matlab_path);
    end

    % 1) Check for infinite or NaN data => error if found
    if any(~isfinite(data(:)))
        error('Data contains infinite or NaN values.');
    end

    [nunits, nconds, ntrials] = size(data);

    % 2) If we have fewer than 2 trials, raise an error
    if ntrials < 2
        error('Data must have at least 2 trials.');
    end

    % 2b) Check for minimum number of conditions
    if nconds < 2
        error('Data must have at least 2 conditions to estimate covariance.');
    end

    % 3) If V is not provided => treat it as 0
    if ~exist('V','var') || isempty(V)
        V = 0;
    end

    % 4) If wantfig is not provided => treat it as true
    if ~exist('wantfig','var') || isempty(wantfig)
        wantfig = true;
    end

    % 5) Prepare default opts
    if ~exist('opt','var') || isempty(opt)
        opt = struct();
    end

    if isfield(opt, 'cv_threshold_per')
        if ~any(strcmp(opt.cv_threshold_per, {'unit','population'}))
            error('cv_threshold_per must be ''unit'' or ''population''');
        end
    end

    % Check if basis vectors are unit length and normalize if not
    if isnumeric(V) && ~isscalar(V)
        % First check and fix unit length
        vector_norms = sqrt(sum(V.^2, 1));
        if any(abs(vector_norms - 1) > 1e-10)
            fprintf('Normalizing basis vectors to unit length...\n');
            V = V ./ vector_norms;
        end

        % Then check orthogonality
        gram = V' * V;
        if ~all(abs(gram - eye(size(gram))) < 1e-10, 'all')
            fprintf('Adjusting basis vectors to ensure orthogonality...\n');
            V = make_orthonormal(V);
        end
    end

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
    
    % Set default unit_groups based on cv_threshold_per
    if ~isfield(opt, 'unit_groups')
        if strcmp(opt.cv_threshold_per, 'population')
            opt.unit_groups = zeros(nunits, 1);  % All units in group 0
        else  % 'unit'
            opt.unit_groups = (0:nunits-1)';  % Each unit gets its own group
        end
    end
    
    % Validate unit_groups
    unit_groups = opt.unit_groups(:);  % Ensure column vector
    if length(unit_groups) ~= nunits
        error('unit_groups must have length %d, got %d', nunits, length(unit_groups));
    end
    if any(unit_groups < 0)
        error('unit_groups must contain only non-negative integers');
    end
    if strcmp(opt.cv_threshold_per, 'population') && any(unit_groups ~= 0)
        error('When cv_threshold_per=''population'', all unit_groups must be 0');
    end
    
    % Store validated unit_groups back in opt
    opt.unit_groups = unit_groups;

    % compute the unit means since they are removed during denoising and will be added back
    trial_avg = mean(data, 3);
    unit_means = mean(trial_avg, 2);
    results.unit_means = unit_means;

    gsn_results = [];

    % 5) If V is an integer => glean basis from GSN results
    if isnumeric(V) && isscalar(V)
        if ~ismember(V, [0, 1, 2, 3, 4])
            error('V must be in [0..4] (int) or a 2D numeric array.');
        end

        % We rely on a function "perform_gsn" here (not shown), which returns:
        % gsn_results.cSb and gsn_results.cNb
        gsn_opt = struct();
        gsn_opt.wantverbose = 0;
        gsn_opt.wantshrinkage = 1;
        gsn_opt.random_seed = 42;  % Set random seed for reproducibility
        gsn_results = performgsn(data, gsn_opt);
        cSb = gsn_results.cSb;
        cNb = gsn_results.cNb;

        % Helper for pseudo-inversion
        inv_or_pinv = @(mat) pinv(mat);
        
        % Helper for sign standardization - make mean of each eigenvector positive
        standardize_signs = @(evecs) arrayfun(@(i) ...
            evecs(:,i) * sign(mean(evecs(:,i)) + eps), ...
            1:size(evecs,2), 'UniformOutput', false);
        standardize_signs = @(evecs) cell2mat(standardize_signs(evecs));

        if V == 0
            % Just eigen-decompose cSb
            % Force symmetry for consistency with Python's eigh
            cSb_sym = (cSb + cSb') / 2;
            [evecs, evals] = eig(cSb_sym, 'vector');  % Use vector output for eigenvalues
            [~, idx] = sort(abs(evals), 'descend');  % Sort by magnitude
            evecs_sorted = evecs(:, idx);
            evecs_sorted = standardize_signs(evecs_sorted);  % Standardize signs
            basis = evecs_sorted;
            mags = abs(evals(idx));
            results.basis_source = cSb_sym;
        elseif V == 1
            % inv(cNb)*cSb - ensure symmetric treatment
            cNb_inv = pinv(cNb);
            matM = cNb_inv * cSb;
            % Use eig with 'vector' for eigenvalues and ensure proper sorting
            [evecs, evals] = eig((matM + matM')/2, 'vector');  % Force symmetry
            [~, idx] = sort(abs(evals), 'descend');  % Sort by magnitude
            evecs_sorted = evecs(:, idx);
            evecs_sorted = standardize_signs(evecs_sorted);  % Standardize signs
            basis = evecs_sorted;
            mags = abs(evals(idx));
            results.basis_source = (matM + matM')/2;
        elseif V == 2
            % Force symmetry for consistency with Python's eigh
            cNb_sym = (cNb + cNb') / 2;
            [evecs, evals] = eig(cNb_sym, 'vector');  % Use vector output
            [~, idx] = sort(abs(evals), 'descend');  % Sort by magnitude
            evecs_sorted = evecs(:, idx);
            evecs_sorted = standardize_signs(evecs_sorted);  % Standardize signs
            basis = evecs_sorted;
            mags = abs(evals(idx));
            results.basis_source = cNb_sym;
        elseif V == 3
            trial_avg = mean(data, 3);  % shape [nunits x nconds]
            % de-mean each row of trial_avg
            trial_avg = trial_avg - unit_means;
            cov_matrix = cov(trial_avg.');  % shape [nunits x nunits]
            % Force symmetry for consistency with Python's eigh
            cov_matrix_sym = (cov_matrix + cov_matrix') / 2;
            [evecs, evals] = eig(cov_matrix_sym, 'vector');  % Use vector output
            [~, idx] = sort(abs(evals), 'descend');  % Sort by magnitude
            evecs_sorted = evecs(:, idx);
            evecs_sorted = standardize_signs(evecs_sorted);  % Standardize signs
            basis = evecs_sorted;
            mags = abs(evals(idx));
            results.basis_source = cov_matrix_sym;
        else
            % V == 4 => random orthonormal
            rng('default');  % Reset to default generator
            rng(42, 'twister');  % Set seed to match Python
            rand_mat = randn(nunits);
            [basis, ~] = qr(rand_mat, 0);  % Use economy QR
            basis = basis(:, 1:nunits);
            mags = ones(nunits, 1);
            results.basis_source = [];
        end

    else
        % If V not int => must be a numeric array
        if ~ismatrix(V)
            error('If V is not int, it must be a numeric matrix.');
        end
        if size(V, 1) ~= nunits
            error('Basis must have %d rows, got %d.', nunits, size(V, 1));
        end
        if size(V, 2) < 1
            error('Basis must have at least 1 column.');
        end

        % Check unit-length columns
        norms = sqrt(sum(V.^2, 1));
        if ~all(abs(norms - 1) < 1e-10)
            error('Basis columns must be unit length.');
        end
        % Check orthogonality
        gram = V' * V;
        if ~all(all(abs(gram - eye(size(V,2))) < 1e-10))
            error('Basis columns must be orthogonal.');
        end

        basis = V;
        % For user-supplied basis, compute magnitudes based on variance in basis
        trial_avg = mean(data, 3);      % shape [nunits x nconds]
        trial_avg_reshaped = trial_avg.';  % shape [nconds x nunits]
        proj_data = trial_avg_reshaped * basis; % shape [nconds x basis_dim]
        mags = var(proj_data, 0, 1).';  % variance along conditions
        results.basis_source = [];
    end

    % Store the magnitudes
    stored_mags = mags;

    % 6) Default cross-validation thresholds if not provided
    if ~isfield(opt, 'cv_thresholds')
        opt.cv_thresholds = 1:size(basis, 2);
    else
        thresholds = opt.cv_thresholds;
        % Validate
        if any(thresholds <= 0)
            error('cv_thresholds must be positive integers.');
        end
        if any(thresholds ~= round(thresholds))
            error('cv_thresholds must be integers.');
        end
        if any(diff(thresholds) <= 0)
            error('cv_thresholds must be in sorted order with unique values.');
        end
    end

    % Initialize return structure
    results.denoiser = [];
    results.cv_scores = [];
    results.best_threshold = [];
    results.denoiseddata = [];
    results.fullbasis = basis;
    results.signalsubspace = [];
    results.dimreduce = [];
    results.mags = [];
    results.dimsretained = [];

    % 7) Decide cross-validation or magnitude-threshold
    if opt.cv_mode >= 0
        [denoiser, cv_scores, best_threshold, denoiseddata, fullbasis_out, signalsubspace, dimreduce] = ...
            perform_cross_validation(data, basis, opt, unit_means);

        results.denoiser = denoiser;
        results.cv_scores = cv_scores;
        results.best_threshold = best_threshold;
        results.denoiseddata = denoiseddata;
        results.fullbasis = fullbasis_out;
        results.mags = stored_mags;  % Add eigenvalues for visualization

        if strcmp(opt.cv_threshold_per, 'population')
            results.signalsubspace = signalsubspace;
            results.dimreduce = dimreduce;
        end

    else
        [denoiser, cv_scores, best_threshold, denoiseddata, fullbasis_out, signalsubspace, dimreduce, mags_out, dimsretained] = ...
            perform_magnitude_thresholding(data, basis, gsn_results, opt, V, unit_means);

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

    % Initialize cv_scores
    cv_scores = zeros(length(thresholds), ntrials, nunits);

    for tr = 1:ntrials
        if cv_mode == 0
            % Denoise average of n-1 trials, test against held out trial
            train_trials = setdiff(1:ntrials, tr);
            train_avg = mean(data(:, :, train_trials), 3);  % [nunits x nconds]
            test_data = data(:, :, tr);                     % [nunits x nconds]

            for tt = 1:length(thresholds)
                threshold = thresholds(tt);
                safe_thr = min(threshold, size(basis, 2));
                denoising_fn = [ones(1, safe_thr), zeros(1, size(basis,2) - safe_thr)];
                D = diag(denoising_fn);
                denoiser_tmp = basis * D * basis';

                % Demean training average before denoising
                train_avg_demeaned = train_avg - unit_means;
                train_denoised = (train_avg_demeaned' * denoiser_tmp)';
                cv_scores(tt, tr, :) = scoring_fn(test_data', train_denoised');
            end

        elseif cv_mode == 1
            % Denoise single trial, test against average of n-1 trials
            dataA = data(:, :, tr)';  % [nconds x nunits]
            dataB = mean(data(:, :, setdiff(1:ntrials, tr)), 3)';  % [nconds x nunits]

            for tt = 1:length(thresholds)
                threshold = thresholds(tt);
                safe_thr = min(threshold, size(basis,2));
                denoising_fn = [ones(1, safe_thr), zeros(1, size(basis,2) - safe_thr)];
                D = diag(denoising_fn);
                denoiser_tmp = basis * D * basis';

                % Demean single trial before denoising
                dataA_demeaned = dataA - unit_means';
                dataA_denoised = dataA_demeaned * denoiser_tmp;
                cv_scores(tt, tr, :) = scoring_fn(dataB, dataA_denoised);
            end
        end
    end

    % Decide best threshold
    if strcmp(threshold_per, 'population')
        % Average over trials and units for population threshold
        avg_scores = mean(mean(cv_scores, 3), 2);  % shape: [length(thresholds), 1]
        [~, best_ix] = max(avg_scores);
        best_threshold = thresholds(best_ix);
        safe_thr = min(best_threshold, size(basis,2));
        denoiser = basis(:, 1:safe_thr) * basis(:, 1:safe_thr)';
    else
        % unit-wise: average over trials only, then group by unit_groups
        avg_scores = squeeze(mean(cv_scores, 2));  % shape: [length(thresholds), nunits]
        if size(avg_scores, 2) == 1
            avg_scores = avg_scores(:);  % Convert to column vector if only one unit
        end
        
        unit_groups = opt.unit_groups;
        unique_groups = unique(unit_groups);
        
        best_thresh_unitwise = zeros(nunits, 1);
        
        % For each group, find the best threshold by averaging CV scores within the group
        for group_id = unique_groups'
            group_mask = unit_groups == group_id;
            group_units = find(group_mask);
            
            % Average CV scores across units in this group
            group_avg_scores = mean(avg_scores(:, group_mask), 2);  % shape: [length(thresholds), 1]
            [~, best_idx] = max(group_avg_scores);
            best_thresh_for_group = thresholds(best_idx);
            
            % Assign this threshold to all units in the group
            best_thresh_unitwise(group_mask) = best_thresh_for_group;
        end
        
        best_threshold = best_thresh_unitwise';

        % Construct unit-wise denoiser
        denoiser = zeros(nunits, nunits);
        for unit_i = 1:nunits
            % For each unit, create its own denoising vector using its threshold
            safe_thr = min(best_threshold(unit_i), size(basis,2));
            unit_denoiser = basis(:, 1:safe_thr) * basis(:, 1:safe_thr)';
            % Use the column corresponding to this unit
            denoiser(:, unit_i) = unit_denoiser(:, unit_i);
        end
    end

    % Calculate denoiseddata based on denoisingtype
    if denoisingtype == 0
        % Trial-averaged denoising
        trial_avg = mean(data, 3);  % [nunits x nconds]
        % Demean trial average before denoising
        trial_avg_demeaned = trial_avg - unit_means;
        denoiseddata = (trial_avg_demeaned' * denoiser)';
    else
        % Single-trial denoising
        denoiseddata = zeros(size(data));
        for t = 1:ntrials
            % Demean each trial before denoising
            data_demeaned = data(:, :, t) - unit_means;
            denoiseddata(:, :, t) = (data_demeaned' * denoiser)';
        end
    end
    
    % Add back the means
    if ndims(denoiseddata) == 3  % Single-trial case
        denoiseddata = denoiseddata + unit_means;
    else  % Trial-averaged case
        denoiseddata = denoiseddata + unit_means;
    end

    fullbasis = basis;
    if strcmp(threshold_per, 'population')
        signalsubspace = basis(:, 1:safe_thr);
        % Project data onto signal subspace
        if denoisingtype == 0
            trial_avg = mean(data, 3);
            % Demean before projecting to signal subspace for consistency
            trial_avg_demeaned = trial_avg - unit_means;
            dimreduce = signalsubspace' * trial_avg_demeaned;  % [safe_thr x nconds]
        else
            dimreduce = zeros(safe_thr, nconds, ntrials);
            for t = 1:ntrials
                % Demean before projecting to signal subspace for consistency
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
    perform_magnitude_thresholding(data, basis, gsn_results, opt, V, unit_means)
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
% 2. Sort dimensions by magnitude in descending order
% 3. Select the top dimensions that cumulatively account for mag_frac 
%    of the total variance
% 4. Create denoising matrix using selected dimensions (in original order)
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

    cv_scores = [];  % Not used in magnitude thresholding
    
    % Get magnitudes based on mag_type
    if mag_type == 1
        % Use pre-computed magnitudes from stored_mags - we need to access them
        % This is a simplification - in practice, stored_mags should be passed in
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
        % Variance-based threshold in user basis
        % Initialize list to store signal variances
        sigvars = [];

        data_reshaped = permute(data, [2, 3, 1]);  % shape [nconds x ntrials x nunits]
        % Compute signal variance for each basis dimension
        for i = 1:size(basis, 2)
            this_eigv = basis(:, i);  % Select the i-th eigenvector
            proj_data = zeros(nconds, ntrials);
            for j = 1:nconds
                for k = 1:ntrials
                    proj_data(j, k) = dot(squeeze(data_reshaped(j, k, :)), this_eigv);
                end
            end

            % Compute signal variance (using same computation as in noise ceiling)
            noisevar = mean(std(proj_data, 0, 2) .^ 2);
            datavar = std(mean(proj_data, 2), 0, 1) ^ 2;
            signalvar = max(datavar - noisevar / size(proj_data, 2), 0);  % Ensure non-negative variance
            sigvars(end+1) = signalvar;
        end

        magnitudes = sigvars(:);
    end
    
    % Sort dimensions by magnitude in descending order to find cumulative variance
    [sorted_magnitudes, sorted_indices] = sort(magnitudes, 'descend');
    
    % Calculate cumulative variance explained
    total_variance = sum(sorted_magnitudes);
    cumulative_variance = cumsum(sorted_magnitudes);
    cumulative_fraction = cumulative_variance / total_variance;
    
    % Find how many dimensions we need to reach mag_frac of total variance
    dims_needed = sum(cumulative_fraction < mag_frac) + 1;  % +1 to include the dimension that crosses threshold
    dims_needed = min(dims_needed, length(sorted_magnitudes));  % Don't exceed total dimensions
    
    % Get the original indices of the selected dimensions (unsorted)
    best_threshold = sorted_indices(1:dims_needed);
    dimsretained = length(best_threshold);
    
    if dimsretained == 0
        % If no dimensions selected, return zero matrices
        denoiser = zeros(nunits, nunits);
        if denoisingtype == 0
            denoiseddata = zeros(nunits, nconds);
        else
            denoiseddata = zeros(nunits, nconds, ntrials);
        end
        signalsubspace = basis(:, 1:0);  % Empty but valid shape
        if denoisingtype == 0
            dimreduce = zeros(0, nconds);
        else
            dimreduce = zeros(0, nconds, ntrials);
        end
        best_threshold = [];
        return
    end

    % Create denoising matrix using retained dimensions
    denoising_fn = zeros(1, size(basis, 2));
    denoising_fn(best_threshold) = 1;
    D = diag(denoising_fn);
    denoiser = basis * D * basis';

    % Calculate denoised data
    if denoisingtype == 0
        % Trial-averaged denoising
        trial_avg = mean(data, 3);
        % Demean trial average before denoising
        trial_avg_demeaned = trial_avg - unit_means;
        denoiseddata = (trial_avg_demeaned' * denoiser)';
    else
        % Single-trial denoising
        denoiseddata = zeros(size(data));
        for t = 1:ntrials
            % Demean each trial before denoising
            data_demeaned = data(:, :, t) - unit_means;
            denoiseddata(:, :, t) = (data_demeaned' * denoiser)';
        end
    end
    
    % Add back the means
    if ndims(denoiseddata) == 3  % Single-trial case
        denoiseddata = denoiseddata + unit_means;
    else  % Trial-averaged case
        denoiseddata = denoiseddata + unit_means;
    end

    % Calculate signal subspace and reduced dimensions
    signalsubspace = basis(:, best_threshold);
    if denoisingtype == 0
        trial_avg = mean(data, 3);
        % Demean before projecting to signal subspace for consistency
        trial_avg_demeaned = trial_avg - unit_means;
        dimreduce = signalsubspace' * trial_avg_demeaned;
    else
        dimreduce = zeros(length(best_threshold), nconds, ntrials);
        for t = 1:ntrials
            % Demean before projecting to signal subspace for consistency
            data_demeaned = data(:, :, t) - unit_means;
            dimreduce(:, :, t) = signalsubspace' * data_demeaned;
        end
    end
end


function scores = negative_mse_columns(x, y)
    % NEGATIVE_MSE_COLUMNS Calculate negative mean squared error between columns.
    %
    % Inputs:
    %   <x> - nconds x nunits. First matrix (usually test data).
    %   <y> - nconds x nunits. Second matrix (usually predictions).
    %       Must have same shape as <x>.
    %
    % Returns:
    %   <scores> - 1 x nunits. Negative MSE for each column/unit.
    %           0 indicates perfect prediction
    %           More negative values indicate worse predictions
    %           Each unit gets its own score
    %
    % Example:
    %   x = [1 2; 3 4];  % 2 conditions, 2 units
    %   y = [1.1 2.1; 2.9 3.9];  % Predictions
    %   scores = negative_mse_columns(x, y);  % Close to 0
    %
    % Notes:
    %   The function handles empty inputs gracefully by returning zeros, which is useful
    %   when no data survives thresholding.

    % Calculate negative mean squared error for each column
    scores = -mean((x - y).^2, 1);
end

function V_orthonormal = make_orthonormal(V)
    % MAKE_ORTHONORMAL Find the nearest matrix with orthonormal columns.
    %
    % Uses Singular Value Decomposition (SVD) to find the nearest orthonormal matrix:
    % 1. Decompose <V> = <U>*<S>*<Vh> where <U> and <Vh> are orthogonal
    % 2. The nearest orthonormal matrix is <U>*<Vh>
    % 3. Take only the first n columns if m > n
    % 4. Verify orthonormality within numerical precision
    %
    % Inputs:
    %   <V> - m x n matrix where m >= n. Input matrix to be made orthonormal.
    %       The number of rows (m) must be at least as large as the number of
    %       columns (n).
    %
    % Returns:
    %   <V_orthonormal> - m x n matrix with orthonormal columns.
    %                   The resulting matrix will have:
    %                   1. All columns unit length
    %                   2. All columns pairwise orthogonal
    %
    % Example:
    %   V = randn(5,3);  % Random 5x3 matrix
    %   V_ortho = make_orthonormal(V);
    %   % Check orthonormality
    %   gram = V_ortho' * V_ortho;  % Should be very close to identity
    %   disp(max(abs(gram - eye(size(gram))), [], 'all'));  % Should be ~1e-15
    %
    % Notes:
    %   The SVD method guarantees orthonormality within numerical precision.
    %   A warning is issued if the result is not perfectly orthonormal.
    
    % Check input dimensions
    [m, n] = size(V);
    if m < n
        error('Input matrix must have at least as many rows as columns');
    end
    
    % Use SVD to find the nearest orthonormal matrix
    % SVD gives us V = U*S*Vh where U and Vh are orthogonal
    % The nearest orthonormal matrix is U*Vh
    [U, ~, Vh] = svd(V, 'econ');
    
    % Take only the first n columns of U if m > n
    V_orthonormal = U(:,1:n) * Vh';
    
    % Double check that the result is orthonormal within numerical precision
    % This is mainly for debugging - the SVD method should guarantee this
    gram = V_orthonormal' * V_orthonormal;
    if ~all(abs(gram - eye(n)) < 1e-10, 'all')
        warning('Result may not be perfectly orthonormal due to numerical precision');
    end
end
