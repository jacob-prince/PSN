function [results] = psn(varargin)
% PSN  Denoise neural data using PSN (Partitioning Signal and Noise).
%
%   results = psn(data) is the default version of PSN and is shorthand for psn(data,'standard')
%
%   results = psn(data,'conservative', opt) prioritizes retaining signal, and is shorthand for
%             psn(data,struct('basis','signal','criterion','variance','threshold_method','global'))
%
%   results = psn(data,'standard', opt) prioritizes out-of-sample generalization at the possible expense
%             of removing some signal in dimensions dominated by noise. It is shorthand for
%             psn(data,struct('basis','signal','criterion','prediction','threshold_method','hybrid'))
%
%   results = psn(data,'aggressive', opt) uses an aggressive denoising approach that flexibly adapts
%             to every unit. This approach may yield improved out-of-sample generalization compared to
%             'standard' but may yield unstable results in cases of limited data. It is shorthand for
%             psn(data,struct('basis','difference','criterion','prediction','threshold_method','unit'))
%
%   results = psn(data, opt) is a version of PSN where the user customizes the settings.
%
%   In all cases, <opt> can be omitted and default parameters are used.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <data> - shape [nunits x nconds x ntrials]. Measured responses for each
%   unit, condition, and trial. Requires ntrials >= 2.
%
%   NaN HANDLING (uneven trials across conditions):
%   - Not all conditions need to have the full set of trials. To indicate
%     the lack of data for certain trials, you can include NaNs ---
%     specifically, it is okay if data(:,i,j) consists of NaNs for some
%     combination(s) of i and j.
%   - IMPORTANT: Each condition must have at least one trial with valid
%     data across ALL units (i.e., data(:,i,j) must contain at least one
%     trial j where no units have NaNs).
%   - PSN computes the average number of trials across conditions (for
%     conditions with >=2 valid trials) and uses this average in formulas
%     involving noise/ntrials. This follows GSN's approach.
%   - The denoised output will NOT contain NaNs (PSN fills them in based
%     on the available data). Residuals will preserve NaNs in the same
%     positions as the input data.
%   (See performgsn.m for more details about uneven trials.)
%
% <opt> (optional) - struct with the following fields:
%
%   <basis> (optional) - basis specifier. Either string or matrix:
%     'signal'     -> use signal basis (eigenvectors of signal covariance, cSb)
%     'difference' -> use "difference" basis (eigenvectors of cSb - cNb / ntrials_avg)
%                     where ntrials_avg is the average number of trials (handles NaNs)
%     B            -> user-supplied basis vectors B, size [nunits x D], D >= 1, with
%                     orthonormal columns (B'*B = I).
%     'noise'      -> [NOT RECOMMENDED] use noise basis (eigenvectors of noise covariance, cNb)
%     'pca'        -> [NOT RECOMMENDED] use eigenvectors of covariance of trial-averaged data
%     'random'     -> [NOT RECOMMENDED] generate a random orthonormal basis
%     Default: 'signal'.
%
%   <criterion> (optional) - string. How to determine the threshold:
%     'prediction'            -> maximize out-of-sample generalization by analytically
%                                maximizing cumulative signal - noise/ntrials_avg
%     'variance'              -> retain dimensions until a target fraction of signal variance is reached
%     'variance_eigenvalues'  -> retain dimensions until a target fraction of the total sum of
%                                positive eigenvalues associated with the basis is reached
%     Default: 'prediction'.
%     (Note that <criterion> set to 'variance_eigenvalues' is not compatible with
%      the B and 'random' cases of <basis>, and is also not compatible with
%      <threshold_method> as 'hybrid' or 'unit'.)
%     (Also, note that when <basis> is 'signal' and <threshold_method> is 'global', then
%      identical results are produced by 'variance' vs. 'variance_eigenvalues'.)
%
%   <threshold_method> (optional) - string. How to select thresholds (i.e. the
%     number of dimensions to retain):
%     'global' -> single threshold for all units (symmetric denoiser)
%     'hybrid' -> global ordering of basis vectors, unit-specific thresholds
%     'unit'   -> unit-specific ordering of basis vectors, unit-specific thresholds
%     Default: 'hybrid'.
%
%   <basis_ordering> (optional) - string. How to set the initial global order of basis vectors:
%     'eigenvalues'    -> use descending order of eigenvalues (if available)
%     'signalvariance' -> measure signal variance and use descending order of signal variance
%     Default: 'eigenvalues'.
%     (Note that when <basis> is B or 'random', eigenvalues are not available, so we
%      necessarily fall back to 'signalvariance'.)
%
%   <variance_threshold> (optional) - scalar in [0,1]. Fraction used
%     when <criterion> is 'variance' or 'variance_eigenvalues'.
%     Default: 0.99.
%
%   <allowable_thresholds> (optional) is a vector of thresholds that are acceptable.
%     For example, a threshold of 7 means to retain the first 7 dimensions.
%     If an optimal threshold is found that is not listed in <allowable_thresholds>,
%     we force to the nearest acceptable threshold (rounding up in cases of a tie).
%     Note that setting <allowable_thresholds> to a single threshold will force
%     PSN to use exactly that many dimensions.
%     Default: [] (no constraint; any threshold between 0 and D is allowed).
%
%   <unit_groups> (optional) - [nunits x 1] non-negative integer vector specifying
%       which units must share the same threshold (applies when <threshold_method>
%       is 'hybrid' or 'unit'). Units with the same integer label are treated
%       as a group and receive the same threshold (determined by averaging
%       the criterion across units of that group). If [] or omitted, the
%       default behavior is equivalent to (1:nunits)', which indicates that
%       each unit forms its own group (i.e., distinct threshold for each unit).
%       For <threshold_method> = 'global', <unit_groups> is ignored (defaults to
%       all zeros internally).
%
%   <gsn_args> (optional) - struct of options passed directly to the GSN routine
%       (performgsn.m). Typical fields might include:
%           .wantshrinkage  - whether to use covariance shrinkage (default: 1)
%           .wantverbose    - whether to print diagnostic output (default: 0)
%           .random_seed    - RNG seed used inside GSN (only for Python)
%       If [] or omitted, defaults are used.
%
%   <wantfig> - 0 or 1. Whether to generate diagnostic figures.
%     Default: 1.
%
%   <wantverbose> - 0 or 1. Whether to show messages during execution.
%     Default: 1.
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% The function returns a single struct <results> with the following fields:
%
% Returned in all cases:
%
%   results.denoiseddata - [nunits x nconds]. Trial-averaged data after applying
%                          the denoiser. This is PSN's estimate of the signal.
%
%   results.residuals    - [nunits x nconds x ntrials]. The original data minus
%                          the <denoiseddata>. This is PSN's estimate of the noise.
%
%   results.unit_means  - [nunits x 1]. Mean response per unit. Note that PSN
%                         subtracts the mean response per unit before the denoising
%                         projection and then re-adds the means afterwards.
%
%   results.denoiser    - [nunits x nunits]. The denoising matrix. For global mode,
%                         this is symmetric. For hybrid/unit modes, this is generally
%                         non-symmetric. The denoising is: denoiser' * (data - means) + means.
%
%   results.svnv_before - [nunits x 2]. The total signal variance (column 1) and
%                         total noise variance (column 2) for each unit, before PSN.
%
%   results.svnv_after  - [nunits x 2]. The total signal variance (column 1) and
%                         total noise variance (column 2) for each unit, after PSN.
%
%   results.best_threshold - optimal threshold(s) chosen by the algorithm.
%       • If thresholding is global, this is a scalar [1 x 1],
%         i.e. the number of dimensions retained.
%       • If thresholding is unit-specific, this is [nunits x 1], giving the
%         threshold for each unit. (Note that units in the same <unit_groups>
%         will share the same value).
%
%   results.fullbasis   - [nunits x dims]. The full set of basis vectors after global
%                         ordering. For unit-specific methods, individual units may
%                         use different orderings (see unitreorderings).
%
%   results.basis_eigenvalues - [dims x 1]. Eigenvalues from basis construction (e.g.,
%                         from signal covariance, difference matrix, etc.), sorted to
%                         match the column order of fullbasis. Empty ([]) for custom
%                         or random bases. For visualization of original order, see
%                         basis_eigenvalues_viz.
%
%   results.unitreorderings - [nunits x dimindices]. Each row indicates the
%     chosen unit-specific ordering for the dimensions. If <threshold_method> is
%     'global' or 'hybrid', each row is simply 1:D in that order.
%
%   results.gsn_result  - struct. Full results from the GSN algorithm containing:
%                             .cSb  - signal covariance
%                             .cNb  - noise covariance
%                         plus any additional outputs from the GSN code.
%
%   results.input_data  - copy of the original input data
%
%   results.signalvar   - [dims x 1] or cell array. Signal variance per dimension.
%
%   results.noisevar    - [dims x 1] or cell array. Noise variance per dimension.
%
%   results.objective   - [(dims+1) x 1]. Cumulative objective curve that was actually
%                         used for threshold selection. For 'prediction' criterion, this is
%                         cumsum(signal - noise/ntrials). For 'variance', this is cumsum(signal).
%                         For 'variance_eigenvalues', this is cumsum(positive eigenvalues).
%
% Special outputs returned only when <threshold_method> is 'global':
%
%   results.signalsubspace - [nunits x K]. The final set of basis vectors selected
%     for denoising (i.e. the subspace into which data are projected).
%
%   results.dimreduce - [K x nconds]. This is the low-dimensional representation
%     of the denoised data in the selected subspace.
%
% Special outputs for unit-specific methods ('hybrid' or 'unit'):
%
%   results.unit_signal_vars - cell array of signal variances per unit
%   results.unit_noise_vars  - cell array of noise variances per unit
%   results.unit_objectives  - cell array of objective curves per unit
%
% Visualization outputs (original order before global ranking):
%
%   results.basis_viz       - [nunits x dims]. Basis vectors in original order
%                             (before global ranking/sorting).
%   results.signal_proj_viz - [dims x 1]. Signal variance projections in original order.
%   results.noise_proj_viz  - [dims x 1]. Noise variance projections in original order.
%   results.basis_eigenvalues_viz - [dims x 1]. Eigenvalues in original order,
%                             matching column order of basis_viz.
%
% -------------------------------------------------------------------------
% Algorithm Details:
% -------------------------------------------------------------------------
%
% PSN works by:
%   1) Estimating signal and noise covariances via GSN 
%   2) Constructing an orthonormal basis (eigenvectors of cSb or cSb - cNb/ntrials_avg)
%   3) Projecting signal and noise covariances into this basis
%   4) Ranking dimensions by signal variance (or eigenvalues for difference basis)
%   5) Selecting threshold to maximize signal - noise/ntrials_avg (or retain variance fraction)
%   6) Building a denoiser that projects data onto selected dimensions
%   7) Reconstructing denoised data 
%.  8) Using this updated signal estimate to compute a noise estimate (residuals)
%
% For unit-specific methods, each unit gets weighted projections based on
% the squared basis coefficients for that unit, and potentially different
% dimension orderings and thresholds.

% =========================================================================
% SETUP: Add GSN dependency to path
% =========================================================================
% PSN requires the GSN (Generative Modeling of Signal and Noise) toolbox to estimate
% signal and noise covariances. GSN is included as a git submodule.

gsn_matlab_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'external', 'gsn', 'matlab');
performgsn_file = fullfile(gsn_matlab_path, 'performgsn.m');

if exist(performgsn_file, 'file')
    addpath(gsn_matlab_path);
    gsn_utilities_path = fullfile(gsn_matlab_path, 'utilities');
    if exist(gsn_utilities_path, 'dir')
        addpath(gsn_utilities_path);
    end
else
    error(['GSN dependency not found (performgsn.m missing).\n' ...
           'Expected location: %s\n' ...
           'Please ensure GSN submodule is initialized:\n' ...
           '  git submodule update --init --recursive'], performgsn_file);
end

% =========================================================================
% STEP 1: Parse and validate inputs
% =========================================================================

[data, opt] = parse_inputs(varargin{:});
[nunits, nconds, ntrials, ntrials_avg, has_nans] = validate_data(data);
opt = set_default_options(opt, nunits);

if opt.wantverbose
    if has_nans
        fprintf('PSN: Starting denoising for %d units, %d conditions, %d max trials (avg %.2f trials)\n', ...
                nunits, nconds, ntrials, ntrials_avg);
    else
        fprintf('PSN: Starting denoising for %d units, %d conditions, %d trials\n', ...
                nunits, nconds, ntrials);
    end
end

% =========================================================================
% STEP 2: Compute unit means
% =========================================================================
% PSN removes the mean response of each unit before denoising, then adds
% it back afterward. This ensures we're only denoising the fluctuations
% around the mean, not the mean itself.

if has_nans
    trial_avg = nanmean(data, 3);        % [nunits x nconds] - ignore NaNs
else
    trial_avg = mean(data, 3);           % [nunits x nconds]
end
unit_means = mean(trial_avg, 2);     % [nunits x 1]

% =========================================================================
% STEP 3: Estimate signal and noise covariances using GSN
% =========================================================================
% GSN (Generative Modeling of Signal and Noise) estimates the covariance matrices
% that describe the signal and noise in the data.

if opt.wantverbose
    fprintf('PSN: Running GSN to estimate signal and noise covariances...\n');
end

gsn_opt = opt.gsn_args;
if ~isfield(gsn_opt, 'wantverbose'), gsn_opt.wantverbose = 0; end
if ~isfield(gsn_opt, 'wantshrinkage'), gsn_opt.wantshrinkage = 1; end

gsn_result = performgsn(data, gsn_opt);
cSb = gsn_result.cSb; % signal covariance (symmetric)
cNb = gsn_result.cNb; % noise covariance (symmetric)

% =========================================================================
% STEP 4: Construct the denoising basis
% =========================================================================
% The basis is a set of orthonormal vectors that span the space in which
% we'll perform denoising. Different basis choices emphasize different
% aspects of the data structure.

if opt.wantverbose
    if ischar(opt.basis) || isstring(opt.basis)
        fprintf('PSN: Constructing denoising basis (type: %s)...\n', char(opt.basis));
    else
        fprintf('PSN: Constructing denoising basis (type: custom matrix [%dx%d])...\n', ...
                size(opt.basis, 1), size(opt.basis, 2));
    end
end

[basis, basis_eigenvalues] = construct_basis(cSb, cNb, opt.basis, data, trial_avg, unit_means, ntrials_avg, has_nans);

% Validate allowable_thresholds against actual basis dimensions
ndims = size(basis, 2);
if ~isempty(opt.allowable_thresholds)
    if any(opt.allowable_thresholds > ndims)
        error('allowable_thresholds contains values exceeding number of basis dimensions (%d)', ndims);
    end
end

% =========================================================================
% STEP 5: Project covariances into basis space
% =========================================================================
% We compute how much signal and noise variance each basis dimension contains
% by projecting the GSN-derived signal and noise covariance matrices (cSb, cNb)
% into the coordinate system of the basis that will be used for PSN denoising.

if opt.wantverbose
    fprintf('PSN: Computing signal and noise variance per dimension...\n');
end

% Always use GSN-based projection for all basis types
[signal_proj, noise_proj] = project_covs(cSb, cNb, basis);

% Save original basis and projections for visualization (before reordering)
basis_viz = basis;
signal_proj_viz = signal_proj;
noise_proj_viz = noise_proj;
basis_eigenvalues_viz = basis_eigenvalues;  % Save original order for visualization

% =========================================================================
% STEP 6: Rank basis dimensions (global ordering)
% =========================================================================
% We order the basis dimensions according to their importance based on basis_ordering:
% - 'eigenvalues': rank by eigenvalues (if available)
% - 'signalvariance': rank by signal variance

if opt.wantverbose
    fprintf('PSN: Ranking basis dimensions globally...\n');
end

if strcmp(opt.basis_ordering, 'eigenvalues') && ~isempty(basis_eigenvalues)
    % Use eigenvalue-based ranking
    [~, sort_idx_global] = sort(basis_eigenvalues, 'descend');
    if opt.wantverbose
        fprintf('PSN: Using eigenvalue-based ordering\n');
    end
else
    % Use signal variance-based ranking (or fallback when eigenvalues unavailable)
    [~, sort_idx_global] = sort(signal_proj, 'descend');
    if opt.wantverbose
        if strcmp(opt.basis_ordering, 'eigenvalues')
            fprintf('PSN: Eigenvalues unavailable, falling back to signal variance ordering\n');
        else
            fprintf('PSN: Using signal variance ordering\n');
        end
    end
end

% Reorder basis and projections according to global ranking
basis = basis(:, sort_idx_global);
signal_proj = signal_proj(sort_idx_global);
noise_proj = noise_proj(sort_idx_global);
if ~isempty(basis_eigenvalues)
    basis_eigenvalues = basis_eigenvalues(sort_idx_global);
end

% =========================================================================
% STEP 7: Select thresholds and build denoiser
% =========================================================================
% Determine how many dimensions to retain. This depends on threshold_method:
%   - 'global': single threshold for all units (symmetric denoiser)
%   - 'hybrid': unit-specific thresholds with global basis ordering
%   - 'unit': unit-specific thresholds AND unit-specific orderings

if opt.wantverbose
    fprintf('PSN: Selecting thresholds (method: %s, criterion: %s)...\n', ...
            opt.threshold_method, opt.criterion);
end

if strcmp(opt.threshold_method, 'global')
    % GLOBAL (POPULATION) MODE
    [denoiser, best_threshold, objective, signalvar, noisevar, unit_cumsum_curves, ...
     unit_signal_vars, unit_noise_vars] = ...
        denoise_global(basis, signal_proj, noise_proj, basis_eigenvalues, ...
                       ntrials_avg, opt);

    unit_orderings = repmat(1:size(basis,2), [nunits, 1]);

else
    % UNIT-SPECIFIC MODES (hybrid or unit)
    unitwise_threshold_only = strcmp(opt.threshold_method, 'hybrid');

    [denoiser, best_threshold, objective, signalvar, noisevar, unit_cumsum_curves, ...
     unit_signal_vars, unit_noise_vars, unit_orderings] = ...
        denoise_unitwise(basis, signal_proj, noise_proj, basis_eigenvalues, ...
                         ntrials_avg, opt, unitwise_threshold_only);
end

% =========================================================================
% STEP 8: Apply denoising to data
% =========================================================================
% Project the trial-averaged data through the denoiser to get the
% final denoised estimates.

if opt.wantverbose
    fprintf('PSN: Applying denoiser to data...\n');
end

% Apply denoiser (transpose works for both symmetric and non-symmetric cases)
denoiseddata = denoiser' * (trial_avg - unit_means) + unit_means;

% =========================================================================
% STEP 9: Compute residuals (noise estimates)
% =========================================================================
% The residuals are what's left after subtracting the denoised data from
% the original data. These represent PSN's estimate of the noise.

residuals = data - repmat(denoiseddata, [1, 1, ntrials]);

% =========================================================================
% STEP 10: Compute signal and noise variances before/after denoising
% =========================================================================

[svnv_before, svnv_after] = compute_signal_noise_diagnostics(...
    opt.threshold_method, unit_signal_vars, unit_noise_vars, best_threshold, nunits);

% =========================================================================
% STEP 11: Package results
% =========================================================================

if opt.wantverbose
    fprintf('PSN: Packaging results...\n');
end

results = struct();

% Core outputs
results.denoiseddata = denoiseddata;
results.residuals = residuals;
results.unit_means = unit_means;
results.denoiser = denoiser;

% Diagnostic outputs
results.svnv_before = svnv_before;
results.svnv_after = svnv_after;
results.best_threshold = best_threshold;
results.fullbasis = basis;
results.unitreorderings = unit_orderings;

% GSN outputs
results.gsn_result = gsn_result;

% Variance outputs
results.signalvar = signalvar;
results.noisevar = noisevar;
results.objective = objective;

% Visualization basis (original order before global ranking)
results.basis_viz = basis_viz;
results.signal_proj_viz = signal_proj_viz;
results.noise_proj_viz = noise_proj_viz;
results.basis_eigenvalues_viz = basis_eigenvalues_viz;  % Original order for visualization

% Sorted eigenvalues (match fullbasis column order)
results.basis_eigenvalues = basis_eigenvalues;  % Sorted order, matches fullbasis

% Input data
results.input_data = data;

% Special outputs for global thresholding
if strcmp(opt.threshold_method, 'global')
    if best_threshold > 0
        results.signalsubspace = basis(:, 1:best_threshold);
        % Project data onto signal subspace (dimensionality reduction)
        results.dimreduce = results.signalsubspace' * (trial_avg - unit_means);
    else
        results.signalsubspace = [];
        results.dimreduce = [];
    end
else
    results.signalsubspace = [];
    results.dimreduce = [];
end

% Unit-specific outputs
if ~strcmp(opt.threshold_method, 'global')
    results.unit_signal_vars = unit_signal_vars;
    results.unit_noise_vars = unit_noise_vars;
    results.unit_objectives = unit_cumsum_curves;
end

% Store options for visualization
results.opt_used = opt;

% =========================================================================
% STEP 12: Visualization
% =========================================================================

if opt.wantfig
    if opt.wantverbose
        fprintf('PSN: Generating diagnostic figures...\n');
    end
    visualize_results(results, opt);
end

if opt.wantverbose
    if strcmp(opt.threshold_method, 'global')
        fprintf('PSN: Complete! Retained %d dimensions.\n', best_threshold);
    else
        fprintf('PSN: Complete! Retained %.1f dimensions on average (range: %d-%d).\n', ...
                mean(best_threshold), min(best_threshold), max(best_threshold));
    end
end

end  % End of main psn function


% =========================================================================
% =========================================================================
% HELPER FUNCTIONS
% =========================================================================
% =========================================================================

function [data, opt] = parse_inputs(varargin)
% PARSE_INPUTS Parse flexible input arguments to psn()
%
% Handles:
%   psn(data)
%   psn(data, 'standard')
%   psn(data, 'conservative')
%   psn(data, 'aggressive')
%   psn(data, 'standard', opt)
%   psn(data, opt)

    if nargin < 1
        error('PSN requires at least one input argument (data)');
    end

    data = varargin{1};
    opt = struct();

    if nargin >= 2
        second_arg = varargin{2};

        if ischar(second_arg) || isstring(second_arg)
            mode = char(second_arg);

            switch lower(mode)
                case 'conservative'
                    opt.basis = 'signal';
                    opt.criterion = 'variance';
                    opt.threshold_method = 'global';

                case 'standard'
                    opt.basis = 'signal';
                    opt.criterion = 'prediction';
                    opt.threshold_method = 'hybrid';

                case 'aggressive'
                    opt.basis = 'difference';
                    opt.criterion = 'prediction';
                    opt.threshold_method = 'hybrid';

                otherwise
                    error('Unknown mode: %s. Must be ''conservative'', ''standard'', or ''aggressive''', mode);
            end

            if nargin >= 3
                user_opt = varargin{3};
                if ~isstruct(user_opt)
                    error('Third argument must be an options struct');
                end
                opt = merge_structs(opt, user_opt);
            end

        elseif isstruct(second_arg)
            opt = second_arg;
        else
            error('Second argument must be a mode string or options struct');
        end
    else
        % Default: standard
        opt.basis = 'signal';
        opt.criterion = 'prediction';
        opt.threshold_method = 'hybrid';
    end
end


function merged = merge_structs(base, override)
% MERGE_STRUCTS Merge two structs, with override taking precedence

    merged = base;
    fields = fieldnames(override);
    for i = 1:length(fields)
        merged.(fields{i}) = override.(fields{i});
    end
end


function [nunits, nconds, ntrials, ntrials_avg, has_nans] = validate_data(data)
% VALIDATE_DATA Check that data has correct shape and valid values
%
% Returns:
%   nunits      - number of units
%   nconds      - number of conditions
%   ntrials     - maximum number of trials (from size of data)
%   ntrials_avg - average number of valid trials per condition (used in formulas)
%   has_nans    - whether data contains NaNs

    if ~isnumeric(data) || ndims(data) ~= 3
        error('Data must be a 3D numeric array [nunits x nconds x ntrials]');
    end

    [nunits, nconds, ntrials] = size(data);

    if ntrials < 2
        error('Data must have at least 2 trials (needed to estimate noise). Got %d trials.', ntrials);
    end

    if nconds < 2
        error('Data must have at least 2 conditions (needed to estimate covariance). Got %d conditions.', nconds);
    end

    % Check for NaNs and compute average number of trials
    % Following GSN's approach in rsanoiseceiling.m:195-239
    has_nans = any(isnan(data(:)));

    if has_nans
        % Count valid trials per condition (trials with no NaNs across all units)
        validcnt = sum(~any(isnan(data), 1), 3);  % 1 x nconds

        % Validate that each condition has at least 1 valid trial
        if any(validcnt < 1)
            error('All conditions must have at least 1 valid trial (no NaNs)');
        end

        % Compute average number of trials across conditions with >= 2 trials
        % This follows GSN's formula: ntrialBC = sum(validcnt(validcnt>1))/ncond
        ntrials_avg = sum(validcnt(validcnt > 1)) / nconds;

        if ntrials_avg < 1
            warning('Average number of trials is lopsided! Setting to 1');
            ntrials_avg = 1;
        end
    else
        % No NaNs: average equals actual
        ntrials_avg = ntrials;
    end
end


function opt = set_default_options(opt, nunits)
% SET_DEFAULT_OPTIONS Fill in any missing options with defaults

    % Normalize string inputs to char for consistent strcmp behavior
    % MATLAB string type ("text") vs char array ('text') behave differently with ischar()
    if isfield(opt, 'basis') && isstring(opt.basis)
        opt.basis = char(opt.basis);
    end
    if isfield(opt, 'criterion') && isstring(opt.criterion)
        opt.criterion = char(opt.criterion);
    end
    if isfield(opt, 'threshold_method') && isstring(opt.threshold_method)
        opt.threshold_method = char(opt.threshold_method);
    end
    if isfield(opt, 'basis_ordering') && isstring(opt.basis_ordering)
        opt.basis_ordering = char(opt.basis_ordering);
    end

    if ~isfield(opt, 'basis')
        opt.basis = 'signal';
    end

    if ~isfield(opt, 'criterion')
        opt.criterion = 'prediction';
    end

    if ~isfield(opt, 'threshold_method')
        opt.threshold_method = 'hybrid';
    end

    if ~isfield(opt, 'basis_ordering')
        opt.basis_ordering = 'eigenvalues';
    end

    if ~isfield(opt, 'variance_threshold')
        opt.variance_threshold = 0.99;
    end

    if ~isfield(opt, 'allowable_thresholds')
        opt.allowable_thresholds = [];
    end

    if ~isfield(opt, 'unit_groups')
        if strcmp(opt.threshold_method, 'global')
            opt.unit_groups = zeros(nunits, 1);
        else
            opt.unit_groups = (1:nunits)';
        end
    end

    if ~isfield(opt, 'gsn_args') || isempty(opt.gsn_args) || ~isstruct(opt.gsn_args)
        opt.gsn_args = struct();
    end

    if ~isfield(opt, 'wantfig')
        opt.wantfig = 1;
    end

    if ~isfield(opt, 'wantverbose')
        opt.wantverbose = 1;
    end

    validate_options(opt, nunits);
end


function validate_options(opt, nunits)
% VALIDATE_OPTIONS Check that all options have valid values

    valid_basis_strings = {'signal', 'difference', 'noise', 'pca', 'random'};
    if ischar(opt.basis) || isstring(opt.basis)
        if ~ismember(opt.basis, valid_basis_strings)
            error('basis must be one of: %s, or a matrix', strjoin(valid_basis_strings, ', '));
        end
    elseif ~isnumeric(opt.basis)
        error('basis must be a string or numeric matrix');
    end

    valid_criteria = {'prediction', 'variance', 'variance_eigenvalues'};
    if ~ismember(opt.criterion, valid_criteria)
        error('criterion must be one of: %s', strjoin(valid_criteria, ', '));
    end

    valid_methods = {'global', 'hybrid', 'unit'};
    if ~ismember(opt.threshold_method, valid_methods)
        error('threshold_method must be one of: %s', strjoin(valid_methods, ', '));
    end

    valid_orderings = {'eigenvalues', 'signalvariance'};
    if ~ismember(opt.basis_ordering, valid_orderings)
        error('basis_ordering must be one of: %s', strjoin(valid_orderings, ', '));
    end

    if opt.variance_threshold < 0 || opt.variance_threshold > 1
        error('variance_threshold must be between 0 and 1');
    end

    if ~isempty(opt.allowable_thresholds)
        if ~isnumeric(opt.allowable_thresholds) || ~isvector(opt.allowable_thresholds)
            error('allowable_thresholds must be a numeric vector');
        end
        if any(opt.allowable_thresholds < 0)
            error('allowable_thresholds must contain only non-negative values');
        end
        % Note: Upper bound checked later against actual basis dimensions (ndims), not nunits
    end

    if length(opt.unit_groups) ~= nunits
        error('unit_groups must have length equal to nunits (%d)', nunits);
    end
    if any(mod(opt.unit_groups, 1) ~= 0)
        error('unit_groups must contain integer values');
    end
    if any(opt.unit_groups < 0)
        error('unit_groups must contain non-negative integers (0 is allowed for global mode)');
    end

    if strcmp(opt.criterion, 'variance_eigenvalues')
        if isnumeric(opt.basis) || strcmp(opt.basis, 'random')
            error('criterion ''variance_eigenvalues'' not compatible with custom basis or ''random'' basis');
        end
        if ismember(opt.threshold_method, {'hybrid', 'unit'})
            error('criterion ''variance_eigenvalues'' only compatible with threshold_method ''global''');
        end
    end
end


function [basis, basis_eigenvalues] = construct_basis(cSb, cNb, basis_spec, data, trial_avg, unit_means, ntrials_avg, has_nans)
% CONSTRUCT_BASIS Create the orthonormal basis for denoising
%
% Different basis choices:
%   'signal'     - Eigenvectors of signal covariance (cSb)
%   'difference' - Eigenvectors of cSb - cNb/ntrials_avg (emphasize signal-dominated directions)
%   'noise'      - Eigenvectors of noise covariance (cNb)
%   'pca'        - Standard PCA on trial-averaged data
%   'random'     - Random orthonormal basis
%   matrix       - User-provided basis
%
% Parameters:
%   trial_avg   - pre-computed trial-averaged data (avoid redundant computation)
%   ntrials_avg - average number of trials (handles NaN case correctly)
%   has_nans    - whether data contains NaNs
%
% Returns:
%   basis             - Orthonormal basis vectors [nunits x ndims]
%   basis_eigenvalues - Eigenvalues associated with basis ([] if not applicable)

    nunits = size(data, 1);

    if ischar(basis_spec) || isstring(basis_spec)
        basis_spec = char(basis_spec);

        switch basis_spec
            case 'signal'
                % Eigenvectors of signal covariance (GSN returns symmetric)
                [basis_eigenvalues, basis] = eigh_descending_sym(cSb);

            case 'difference'
                % Eigenvectors of signal - scaled noise
                % Eigenvalues encode the net benefit per dimension
                % Use ntrials_avg to properly handle NaN case
                A = cSb - cNb / ntrials_avg;
                % Symmetrize derived matrix to handle numerical errors
                A = (A + A') / 2;
                [basis_eigenvalues, basis] = eigh_descending_sym(A);  % already symmetrized above

            case 'noise'
                % Eigenvectors of noise covariance (GSN returns symmetric)
                [basis_eigenvalues, basis] = eigh_descending_sym(cNb);

            case 'pca'
                % Standard PCA on trial-averaged data
                % Eigenvectors from empirical covariance, but treated exactly like signal basis
                % in all subsequent ranking/thresholding (uses GSN signal_proj, not PCA eigenvalues).
                % PCA eigenvalues are kept for visualization purposes only.
                % Use pre-computed trial_avg to avoid redundant computation
                trial_avg_demeaned = trial_avg - unit_means;
                cov_matrix = cov(trial_avg_demeaned');  % cov() returns symmetric matrix
                [basis_eigenvalues, basis] = eigh_descending_sym(cov_matrix);  % no symmetrization needed
                % Note: PCA eigenvalues stored but NOT used for ranking/thresholding

            case 'random'
                % Random orthonormal basis (no meaningful eigenvalues)
                % NOTE: This resets the global RNG state for reproducibility
                % To avoid affecting other random operations, consider using a separate RandStream
                rng('default');
                rng(42);
                [basis, ~] = qr(randn(nunits));
                basis_eigenvalues = [];

            otherwise
                error('Unknown basis type: %s', basis_spec);
        end

    else
        % User-provided custom basis (no eigenvalues available)
        basis = basis_spec;

        if size(basis, 1) ~= nunits
            error('Custom basis must have %d rows (matching nunits)', nunits);
        end
        if size(basis, 2) < 1 || size(basis, 2) > nunits
            error('Custom basis must have between 1 and %d columns', nunits);
        end

        basis = normalize_orthonormalize_basis(basis);
        basis_eigenvalues = [];
    end
end


function [evals_sorted, evecs_sorted] = eigh_descending_sym(matrix, do_symmetrize)
% EIGH_DESCENDING_SYM Compute eigendecomposition with consistent sorting
%
% Returns eigenvalues first, then eigenvectors (matching convention)
% Optionally symmetrizes matrix, computes eigenvectors/eigenvalues, sorts by
% descending eigenvalue, and standardizes eigenvector signs for reproducibility.
%
% Parameters:
%   matrix        - Input matrix (should be symmetric)
%   do_symmetrize - (optional) If true, force symmetry. Default: false.
%                   GSN returns symmetric cSb/cNb, so symmetrization is only
%                   needed for derived matrices (already symmetrized before call).
%
% Returns:
%   evals_sorted - Eigenvalues [n x 1], sorted descending by magnitude
%   evecs_sorted - Eigenvectors [n x n], columns sorted to match eigenvalues

    if nargin < 2
        do_symmetrize = false;
    end

    if do_symmetrize
        matrix = (matrix + matrix') / 2;
    end

    % Compute eigendecomposition
    [evecs, evals] = eig(matrix, 'vector');

    % Sort by eigenvalue magnitude (descending)
    [evals_sorted, order] = sort(evals, 'descend');
    evecs_sorted = evecs(:, order);

    % Deterministic sign: make largest-magnitude element positive
    [~, piv] = max(abs(evecs_sorted), [], 1);
    idx = sub2ind(size(evecs_sorted), piv, 1:size(evecs_sorted, 2));
    sgn = sign(evecs_sorted(idx));
    sgn(sgn == 0) = 1;
    evecs_sorted = evecs_sorted .* sgn;
end


function basis = normalize_orthonormalize_basis(basis)
% NORMALIZE_ORTHONORMALIZE_BASIS Ensure basis has orthonormal columns
%
% Normalizes columns then checks/enforces orthonormality using QR if needed.

    norms = sqrt(sum(basis.^2, 1));
    norms(norms == 0) = 1;
    basis = basis ./ norms;

    gram = basis' * basis;
    if ~all(all(abs(gram - eye(size(gram))) < 1e-10))
        [basis, ~] = qr(basis, 0);
    end
end


function [sig, noi] = project_covs(cS, cN, B)
% PROJECT_COVS Project covariances into basis
%
% Computes diagonal elements of B' * cS * B and B' * cN * B, which give
% the signal and noise variance for each dimension of the basis.
%
% Uses efficient element-wise computation: diag(B' * C * B) = sum((C * B) .* B, 1)'
% This is O(N^2 * K) instead of O(N^3) for full matrix multiplication.
%
% Note: cS and cN are assumed symmetric (GSN returns symmetric matrices).

    % Efficient diagonal extraction (avoids full matrix multiplication)
    % diag(B' * C * B)[i] = B[:,i]' * C * B[:,i] = sum(B[:,i] .* (C * B[:,i]))
    sig = sum((cS * B) .* B, 1)';
    noi = sum((cN * B) .* B, 1)';

    % Clamp tiny negatives from numerical error
    sig = max(sig, 0);
    noi = max(noi, 0);
end


function [unit_cumsum_curves, unit_signal_vars, unit_noise_vars, unit_orderings] = ...
    compute_unit_weighted_projections(basis, signal_proj, noise_proj, ntrials, do_unit_ranking)
% COMPUTE_UNIT_WEIGHTED_PROJECTIONS Compute unit-specific weighted variances and objective curves
%
% For each unit, computes weighted signal/noise projections based on how much each
% basis dimension affects that unit (w = basis(u,:)^2). Optionally ranks dimensions
% by each unit's signal variance.
%
% Parameters:
%   basis: (nunits x ndims) - the basis matrix
%   signal_proj: (ndims x 1) - signal variance per dimension
%   noise_proj: (ndims x 1) - noise variance per dimension
%   ntrials: scalar - number of trials
%   do_unit_ranking: logical - if true, rank dimensions by each unit's signal variance;
%                             if false, use global ordering
%
% Returns:
%   unit_cumsum_curves: cell array of (ndims+1 x 1) - cumulative objective for each unit
%   unit_signal_vars: cell array of (ndims x 1) - weighted signal variance for each unit
%   unit_noise_vars: cell array of (ndims x 1) - weighted noise variance for each unit
%   unit_orderings: (nunits x ndims) - dimension ordering for each unit

    [nunits, ndims] = size(basis);

    unit_cumsum_curves = cell(nunits, 1);
    unit_signal_vars = cell(nunits, 1);
    unit_noise_vars = cell(nunits, 1);
    unit_orderings = zeros(nunits, ndims);

    for u = 1:nunits
        % Compute weighted projections for this unit
        % w = squared basis coefficients (how much each dimension affects this unit)
        w = basis(u, :)' .^ 2;
        sig_u = w .* signal_proj;
        noi_u = w .* noise_proj;

        if do_unit_ranking
            % Rank by this unit's signal variance
            [~, sort_idx_u] = sort(sig_u, 'descend');
            sig_sorted = sig_u(sort_idx_u);
            noi_sorted = noi_u(sort_idx_u);
        else
            % Use global ordering
            sig_sorted = sig_u;
            noi_sorted = noi_u;
            sort_idx_u = (1:ndims)';
        end

        unit_orderings(u, :) = sort_idx_u';

        % Compute objective curve for this unit
        % Always use prediction-style objective (signal - noise/ntrials)
        % even for variance criterion (threshold selection handles the difference)
        scaled_noise = noi_sorted / ntrials;
        diff = sig_sorted - scaled_noise;
        curve_u = [0; cumsum(diff(:))];

        unit_cumsum_curves{u} = curve_u;
        unit_signal_vars{u} = sig_sorted;
        unit_noise_vars{u} = noi_sorted;
    end
end


function [denoiser, best_threshold, objective, signalvar, noisevar, unit_cumsum_curves, ...
          unit_signal_vars, unit_noise_vars] = ...
    denoise_global(basis, signal_proj, noise_proj, basis_eigenvalues, ntrials, opt)
% DENOISE_GLOBAL Population-level denoising (symmetric denoiser)
%
% Selects a single threshold for all units. Builds symmetric denoiser matrix.
%
% Fast path: If using difference basis + prediction criterion, eigenvalues
% already encode signal - noise/ntrials, so we can directly maximize their cumsum.

    nunits = size(basis, 1);
    use_diff_basis = ischar(opt.basis) && strcmp(opt.basis, 'difference');
    use_prediction = strcmp(opt.criterion, 'prediction');

    % Threshold selection
    if use_diff_basis && use_prediction && ~isempty(basis_eigenvalues)
        % FAST PATH: difference basis eigenvalues ARE the net benefit
        objective = [0; cumsum(basis_eigenvalues(:))];
        [~, k] = max(objective);
        k = k - 1;  % Convert index to number of dims
    else
        % Standard path (including variance_eigenvalues criterion)
        [k, objective] = select_threshold_analytic(signal_proj, noise_proj, basis_eigenvalues, ntrials, opt);
    end

    % Apply allowable_thresholds constraint
    if ~isempty(opt.allowable_thresholds)
        k = constrain_to_allowable(k, opt.allowable_thresholds);
    end

    best_threshold = k;

    % Build symmetric denoiser
    if k > 0
        denoiser = basis(:, 1:k) * basis(:, 1:k)';
    else
        denoiser = zeros(nunits, nunits);
    end

    % Outputs
    signalvar = signal_proj;
    noisevar = noise_proj;

    % Compute unit-specific weighted variances using same logic as unit-specific method
    % Even though we use a global threshold, we can still compute how much each
    % dimension contributes to each unit's signal and noise
    [unit_cumsum_curves, unit_signal_vars, unit_noise_vars, ~] = ...
        compute_unit_weighted_projections(basis, signal_proj, noise_proj, ntrials, false);
end


function [denoiser, best_threshold, objective, signalvar, noisevar, unit_cumsum_curves, ...
          unit_signal_vars, unit_noise_vars, unit_orderings] = ...
    denoise_unitwise(basis, signal_proj, noise_proj, basis_eigenvalues, ntrials, opt, threshold_only)
% DENOISE_UNITWISE Unit-specific denoising (non-symmetric denoiser)
%
% Each unit gets:
%   - Weighted signal/noise projections: w = basis(u,:)^2, sig_u = w .* signal_proj
%   - Optional unit-specific ranking (if threshold_only=false)
%   - Unit-specific threshold selection with optional unit_groups averaging
%   - Denoiser column: Bu * Bu(u,:)' where Bu = basis(:, dims_for_unit_u)
%
% The denoiser is generally non-symmetric. Apply as: denoiser' * data

    [nunits, ndims] = size(basis);

    denoiser = zeros(nunits, nunits);

    % First pass: compute weighted projections and objectives for each unit
    % If threshold_only=true (hybrid mode), use global ordering
    % If threshold_only=false (full unit-specific), rank by each unit's signal variance
    do_unit_ranking = ~threshold_only;
    [unit_cumsum_curves, unit_signal_vars, unit_noise_vars, unit_orderings] = ...
        compute_unit_weighted_projections(basis, signal_proj, noise_proj, ntrials, do_unit_ranking);

    % Second pass: select thresholds considering unit_groups
    unique_groups = unique(opt.unit_groups);
    best_threshold = zeros(nunits, 1);

    for g = unique_groups'
        group_mask = (opt.unit_groups == g);
        group_indices = find(group_mask);

        if strcmp(opt.criterion, 'prediction')
            % Average objective curves across units in this group
            % All curves should have the same length (ndims+1)
            avg_curve = mean(cat(2, unit_cumsum_curves{group_indices}), 2);
            [~, k_group] = max(avg_curve);
            k_group = k_group - 1;  % Convert index to number of dims
        elseif strcmp(opt.criterion, 'variance')
            % Average signal variances across units in this group
            avg_signal = mean(cat(2, unit_signal_vars{group_indices}), 2);
            vt = max(0, min(1, opt.variance_threshold));
            if vt == 0
                k_group = 0;
            else
                % Prepend 0 for consistency with global mode (index 1 = 0 dims)
                cs = [0; cumsum(avg_signal(:))];
                total = cs(end);
                if total <= 0
                    k_group = 0;
                else
                    k_group = find(cs >= vt * total, 1, 'first');
                    if isempty(k_group)
                        k_group = 0;
                    else
                        k_group = k_group - 1;  % Convert index to number of dims
                    end
                    k_group = min(k_group, ndims);
                end
            end
        else
            error('criterion ''variance_eigenvalues'' not supported for unit-specific modes');
        end

        % Apply allowable_thresholds constraint
        if ~isempty(opt.allowable_thresholds)
            k_group = constrain_to_allowable(k_group, opt.allowable_thresholds);
        end

        % Assign this threshold to all units in the group
        best_threshold(group_mask) = k_group;
    end

    % Third pass: build denoiser columns
    % Optimize by grouping units with same threshold and ordering
    unique_thresholds = unique(best_threshold(best_threshold > 0));

    for k = unique_thresholds'
        units_with_k = find(best_threshold == k);

        if threshold_only
            % Hybrid mode: all units share same ordering, vectorize fully
            Bu = basis(:, 1:k);
            % Vectorized: denoiser(:, units) = Bu * Bu(units, :)'
            denoiser(:, units_with_k) = Bu * Bu(units_with_k, :)';
        else
            % Full unit-specific: group by ordering within same threshold
            for u = units_with_k'
                sort_idx_u = unit_orderings(u, :);
                Bu = basis(:, sort_idx_u(1:k));
                denoiser(:, u) = Bu * Bu(u, :)';
            end
        end
    end

    % Population-level averages for visualization
    if ~isempty(unit_signal_vars)
        signalvar = mean(cat(2, unit_signal_vars{:}), 2);
        noisevar = mean(cat(2, unit_noise_vars{:}), 2);
        objective = [0; cumsum(signalvar - noisevar / ntrials)];
    else
        signalvar = [];
        noisevar = [];
        objective = zeros(1, 1);
    end
end


function [k, objective] = select_threshold_analytic(signal, noise, basis_eigenvalues, ntrials, opt)
% SELECT_THRESHOLD_ANALYTIC Choose threshold using analytic or variance criterion
%
% 'prediction': maximize cumsum(signal - noise/ntrials)
% 'variance': retain until cumsum(signal) >= variance_threshold * total_signal
% 'variance_eigenvalues': retain until cumsum(positive eigenvalues) >= variance_threshold * total
%
% Inputs:
%   signal            - Signal variance per dimension
%   noise             - Noise variance per dimension
%   basis_eigenvalues - Eigenvalues from basis construction ([] if not available)
%   ntrials           - Number of trials
%   opt               - Options struct
%
% Returns:
%   k         - Selected threshold (number of dimensions to retain)
%   objective - Curve that was ACTUALLY used for threshold selection

    ndims = length(signal);
    scaled_noise = noise / ntrials;
    diff = signal - scaled_noise;

    switch opt.criterion
        case 'prediction'
            % Maximize expected out-of-sample prediction quality
            objective = [0; cumsum(diff(:))];
            [~, k] = max(objective);
            k = k - 1;  % Index to number of dims

        case 'variance'
            % Retain fraction of signal variance
            % Return cumulative signal variance as objective
            objective = [0; cumsum(signal(:))];

            vt = max(0, min(1, opt.variance_threshold));
            if vt == 0
                k = 0;
            else
                total = objective(end);
                if total <= 0
                    k = 0;
                else
                    k = find(objective >= vt * total, 1, 'first');
                    if isempty(k)
                        k = 0;
                    else
                        k = k - 1;  % Convert index to number of dims
                    end
                    k = min(k, ndims);
                end
            end

        case 'variance_eigenvalues'
            % Retain fraction of total positive eigenvalue sum
            % Only valid when eigenvalues are available (signal, difference, noise bases)
            % For PCA: use signal variance instead (PCA eigenvalues are for visualization only)

            use_pca_basis = ischar(opt.basis) && strcmp(opt.basis, 'pca');

            if use_pca_basis
                % PCA special case: use signal variance instead of PCA eigenvalues
                objective = [0; cumsum(signal(:))];
            else
                if isempty(basis_eigenvalues)
                    error(['variance_eigenvalues criterion requires eigenvalues.\n' ...
                           'Not compatible with custom basis or random basis.']);
                end
                % Return cumulative positive eigenvalues as objective
                pos_evals = max(basis_eigenvalues(:), 0);
                objective = [0; cumsum(pos_evals)];
            end

            vt = max(0, min(1, opt.variance_threshold));
            if vt == 0
                k = 0;
            else
                total = objective(end);
                if total <= 0
                    k = 0;
                else
                    k = find(objective >= vt * total, 1, 'first');
                    if isempty(k)
                        k = 0;
                    else
                        k = k - 1;  % Convert index to number of dims
                    end
                    k = min(k, ndims);
                end
            end

        otherwise
            error('Unknown criterion: %s', opt.criterion);
    end
end


function k_constrained = constrain_to_allowable(k, allowable)
% CONSTRAIN_TO_ALLOWABLE Force threshold to nearest allowable value

    if isscalar(k)
        if ~ismember(k, allowable)
            diffs = abs(allowable - k);
            min_diff = min(diffs);
            tied_values = allowable(diffs == min_diff);
            k_constrained = max(tied_values);  % Round up on tie
        else
            k_constrained = k;
        end
    else
        % Vector case (unit-specific)
        k_constrained = k;
        for i = 1:length(k)
            if ~ismember(k(i), allowable)
                diffs = abs(allowable - k(i));
                min_diff = min(diffs);
                tied_values = allowable(diffs == min_diff);
                k_constrained(i) = max(tied_values);
            end
        end
    end
end


function [svnv_before, svnv_after] = compute_signal_noise_diagnostics(...
    threshold_method, unit_signal_vars, unit_noise_vars, best_threshold, nunits)
% COMPUTE_SIGNAL_NOISE_DIAGNOSTICS Sum signal/noise variance before/after thresholding
%
% Uses unit-specific variance projections. Sums across all dims (before) and retained dims (after).

    svnv_before = zeros(nunits, 2);
    svnv_after = zeros(nunits, 2);

    if strcmp(threshold_method, 'global')
        % Global: all units share same threshold, but each unit gets different
        % amounts of signal/noise variance based on weighted projections
        for u = 1:nunits
            sig_u = unit_signal_vars{u};
            noi_u = unit_noise_vars{u};
            k = best_threshold;  % Same threshold for all units
            svnv_before(u, :) = [sum(sig_u), sum(noi_u)];
            svnv_after(u, :) = [(k > 0) * sum(sig_u(1:k)), (k > 0) * sum(noi_u(1:k))];
        end
    else
        % Unit-specific: each unit has weighted projections and individual threshold
        for u = 1:nunits
            sig_u = unit_signal_vars{u};
            noi_u = unit_noise_vars{u};
            k_u = best_threshold(u);
            svnv_before(u, :) = [sum(sig_u), sum(noi_u)];
            svnv_after(u, :) = [(k_u > 0) * sum(sig_u(1:k_u)), (k_u > 0) * sum(noi_u(1:k_u))];
        end
    end
end


function visualize_results(results, opt)
% VISUALIZE_RESULTS Create diagnostic figures
%
% Calls the visualization.m script if available

    if exist('visualization', 'file')
        visualization(results.input_data, results);
    else
        % Basic placeholder visualization
        if opt.wantverbose
            fprintf('  (visualization.m not found - skipping figures)\n');
        end
    end
end
