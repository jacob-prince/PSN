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
%   <figurepath> (optional) - string. Path to save the diagnostic figure.
%     Can be a full path (e.g., '/path/to/figure.png') or just a filename
%     (e.g., 'psn_diagnostics.png') to save in the current directory.
%     If specified, the figure is saved to this path and closed after saving.
%     If '' or omitted, figures are displayed but not automatically saved.
%     Default: '' (empty string).
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
% SETUP: Add dependencies to path
% =========================================================================

% Add PSN utilities to path
psn_utils_path = fullfile(fileparts(mfilename('fullpath')), 'utilities');
if exist(psn_utils_path, 'dir')
    addpath(psn_utils_path);
else
    error(['PSN utilities folder not found.\n' ...
           'Expected location: %s\n' ...
           'The utilities folder should be in the same directory as psn.m'], psn_utils_path);
end

% Add GSN dependency to path
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
% - 'prediction': rank by signal variance - noise variance / ntrials

if opt.wantverbose
    fprintf('PSN: Ranking basis dimensions globally...\n');
end

if strcmp(opt.basis_ordering, 'eigenvalues') && ~isempty(basis_eigenvalues)
    % Use eigenvalue-based ranking
    [~, sort_idx_global] = sort(basis_eigenvalues, 'descend');
    if opt.wantverbose
        fprintf('PSN: Using eigenvalue-based ordering\n');
    end
elseif strcmp(opt.basis_ordering, 'prediction')
    % Use prediction objective (signal - noise/ntrials) for ranking
    prediction_obj = signal_proj - noise_proj / ntrials_avg;
    [~, sort_idx_global] = sort(prediction_obj, 'descend');
    if opt.wantverbose
        fprintf('PSN: Using prediction-based ordering (signalvar - noisevar/ntrials)\n');
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
% STEP 7: Select thresholds and build denoising matrix
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
    opt.threshold_method, unit_signal_vars, unit_noise_vars, best_threshold, nunits, ntrials_avg);

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
