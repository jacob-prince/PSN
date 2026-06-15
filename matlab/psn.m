function [results] = psn(varargin)
% PSN  Denoise neural data using PSN (Partitioning Signal and Noise).
%
%   results = psn(data) is the default version of PSN and is identical to psn(data,'standard').
%             It uses the signal basis at the "max-tradeoff" threshold: a near-unbiased,
%             deterministic choice that still captures the bulk of the achievable recovery,
%             with a single population (global) threshold. It is shorthand for
%             psn(data,struct('basis','signal','criterion','max-tradeoff','threshold_method','global'))
%
%   results = psn(data,'conservative', opt) prioritizes retaining signal, and is shorthand for
%             psn(data,struct('basis','signal','criterion','variance','threshold_method','global','variance_threshold',0.99))
%
%   results = psn(data,'standard', opt) is identical to the default psn(data): the signal basis at the
%             max-tradeoff threshold with a single population (global) threshold. It is shorthand for
%             psn(data,struct('basis','signal','criterion','max-tradeoff','threshold_method','global'))
%
%   results = psn(data,'aggressive', opt) uses a more aggressive denoising approach (difference basis at
%             the prediction peak). This may yield improved out-of-sample generalization compared to
%             'standard' but may yield unstable results in cases of limited data. It is shorthand for
%             psn(data,struct('basis','difference','criterion','prediction','threshold_method','global'))
%
%   results = psn(data,'wiener', opt) applies the full-rank matrix Wiener filter (Bayes-optimal linear
%             estimator). It is shorthand for psn(data,struct('criterion','wiener'))
%
%   results = psn(data,'compare', opt) builds both the signal and difference bases at their
%             max-tradeoff thresholds and keeps whichever has the higher analytic recovery
%             (ties keep the more near-unbiased signal basis). It is shorthand for
%             psn(data,struct('basis','compare','criterion','max-tradeoff','threshold_method','hybrid'))
%
%   results = psn(data, opt) is a version of PSN where the user customizes the settings.
%
%   In all cases, <opt> can be omitted and default parameters are used.
%
%   When a named mode is combined with an <opt> struct, the struct OVERRIDES the
%   mode's defaults (the user's options take priority). For example,
%   psn(data,'aggressive',struct('threshold_method','hybrid')) keeps the 'aggressive'
%   basis and criterion but switches to hybrid thresholds. The 'wiener' mode is the
%   exception: it is basis-free and untruncated, so combining it with a conflicting
%   <basis>, <criterion>, <threshold_method>, <basis_ordering>, <allowable_thresholds>,
%   <variance_threshold>, <alpha>, or <unit_groups> raises an error instead.
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
%     'wiener'     -> [DEPRECATED] alias for criterion='wiener' (the full-rank Wiener
%                     filter). Wiener is not a basis; see <criterion>.
%     'compare'    -> build both the 'signal' and 'difference' bases, each at its
%                     own max-tradeoff threshold, and keep whichever has the higher
%                     analytic recovery against the GSN covariances. A negligible
%                     margin keeps 'signal' (the more near-unbiased basis). The
%                     chosen basis and the per-candidate metrics -- analytic recovery
%                     (which drives the choice) and the empirical split-half
%                     reliability (reported for validation only) -- are returned in
%                     results.threshold_selection and results.diagnostics.
%     Default: 'signal'.
%
%   <criterion> (optional) - string. How to determine the threshold:
%     'prediction'            -> maximize out-of-sample generalization by analytically
%                                maximizing cumulative signal - noise/ntrials_avg
%     'max-tradeoff'          -> the most-unbiased operating point that still captures the
%                                bulk of the achievable recovery: farthest from the chord on the
%                                descending (prediction-peak -> trial-average) limb of the
%                                recovery curve. Retains more signal variance (less bias)
%                                than 'prediction' while keeping recovery high. Fully
%                                analytic. Most meaningful with basis='signal'.
%     'variance'              -> retain dimensions until a target fraction of signal variance is reached
%     'variance_eigenvalues'  -> retain dimensions until a target fraction of the total sum of
%                                positive eigenvalues associated with the basis is reached
%     'wiener'                -> full-rank matrix Wiener filter: D = cSb * (cSb + cNb/t)^{-1},
%                                the Bayes-optimal linear estimator. Unlike the other criteria
%                                this applies NO truncation -- all dimensions are kept and
%                                continuously weighted -- and it is basis-free. Because it
%                                bypasses the basis/criterion/threshold pipeline, supplying a
%                                conflicting <basis>, <threshold_method>, <basis_ordering>,
%                                <allowable_thresholds>, <variance_threshold>, <alpha>, or
%                                <unit_groups> raises an error rather than being ignored.
%     Default: 'max-tradeoff'.
%     (Note that <criterion> set to 'variance_eigenvalues' is not compatible with
%      the B and 'random' cases of <basis>, and is also not compatible with
%      <threshold_method> as 'hybrid'.)
%     (Also, note that when <basis> is 'signal' and <threshold_method> is 'global', then
%      identical results are produced by 'variance' vs. 'variance_eigenvalues'.)
%
%   <threshold_method> (optional) - string. How to select thresholds (i.e. the
%     number of dimensions to retain):
%     'global' -> single threshold for all units (symmetric denoiser)
%     'hybrid' -> global ordering of basis vectors, unit-specific thresholds
%     Default: 'global'.
%
%   <basis_ordering> (optional) - string. How to set the initial global order of basis vectors:
%     'eigenvalues'    -> use descending order of eigenvalues (if available)
%     'signalvariance' -> measure signal variance and use descending order of signal variance
%     Default: 'eigenvalues'.
%     (Note that when <basis> is B or 'random', eigenvalues are not available, so we
%      necessarily fall back to 'signalvariance'.)
%
%   <alpha> (optional) - scalar in [0,1] or []. Interpolation parameter
%     between the prediction peak and a variance retention target:
%       alpha=0   -> prediction peak (same as criterion='prediction')
%       alpha=1   -> variance retention (same as criterion='variance')
%       alpha=0.3 -> retain an additional 30% of the signal variance gap
%                    between the prediction peak and the variance target
%     When set, overrides the criterion setting. Uses variance_threshold
%     to define the variance target. Does not apply to criterion='wiener'.
%     Default: [] (disabled; existing criterion logic used).
%
%   <variance_threshold> (optional) - scalar in [0,1]. Fraction used
%     when <criterion> is 'variance' or 'variance_eigenvalues', or as
%     the variance target when <alpha> is set.
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
%       is 'hybrid'). Units with the same integer label are treated
%       as a group and receive the same threshold (determined by averaging
%       the criterion across units of that group). If [] or omitted, the
%       default behavior is equivalent to (1:nunits)', which indicates that
%       each unit forms its own group (i.e., distinct threshold for each unit).
%       For <threshold_method> = 'global', <unit_groups> is ignored (defaults to
%       all zeros internally).
%
%   <gsn_result> (optional) - struct with fields cSb and cNb from a previous PSN
%       call (i.e. results.gsn_result), OR a path to a '.mat' file holding them.
%       When provided, PSN skips the expensive GSN estimation and uses these
%       covariances directly -- useful for sweeping hyperparameters (alpha,
%       basis, criterion, ...) on the same data. If the struct also carries
%       cached eigvecs/eigvals for the requested signal/difference basis, PSN
%       additionally skips its own eigendecomposition.
%       Default: [] (run GSN normally).
%
%   <gsn_args> (optional) - struct of options passed directly to the GSN routine
%       (performgsn.m). Typical fields might include:
%           .wantshrinkage  - whether to use covariance shrinkage (default: 1)
%           .wantverbose    - whether to print diagnostic output (default: 0)
%           .random_seed    - RNG seed used inside GSN (only for Python)
%       If [] or omitted, defaults are used.
%
%   <split_half_metric> (optional) - 'correlation' or 'mse'. Metric used for the
%       split-half reliability panel of the diagnostic figure ('correlation' =
%       Pearson r per unit, 'mse' = mean squared error per unit).
%       Default: 'correlation'.
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
%   Language-specific differences (Python-only, intentionally not in MATLAB):
%     - <device>: the Python package offers GPU acceleration ('cuda'/'mps') via
%       torch. MATLAB PSN is CPU-only; there is no <device> option.
%     - scikit-learn estimator: Python ships a PSN BaseEstimator/TransformerMixin
%       class (fit/transform). MATLAB exposes only the functional psn() entry.
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
%                         this is symmetric. For hybrid mode, this is generally
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
%                         ordering. All units share this ordering.
%
%   results.basis_eigenvalues - [dims x 1]. Eigenvalues from basis construction (e.g.,
%                         from signal covariance, difference matrix, etc.), sorted to
%                         match the column order of fullbasis. Empty ([]) for custom
%                         or random bases. For visualization of original order, see
%                         basis_eigenvalues_viz.
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
% Special outputs for unit-specific thresholds (threshold_method 'hybrid'):
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
% Metadata:
%
%   results.opt_used   - struct of the resolved options actually used.
%   results.runtime    - scalar. Wall-clock seconds for the psn() call.
%   results.threshold_selection - struct, set for 'compare' and 'wiener'. Records
%                        the chosen mode/basis/criterion/best_threshold (+ recovery
%                        and sv_frac for 'compare').
%   results.diagnostics - struct, set for 'compare'. Per-candidate selection
%                        metrics (best_threshold/recovery/sv_frac for signal and
%                        difference) plus the cached candidate eigenbases.
%   results.alpha_info  - struct, set when <alpha> is active: the prediction-peak
%                        (k_pred) and variance-target (k_var) dimension counts the
%                        alpha threshold interpolates between, and alpha.
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

psn_t_start = tic;   % wall-clock timer -> results.runtime (mirrors Python)

[data, opt] = parse_inputs(varargin{:});
[nunits, nconds, ntrials, ntrials_avg, has_nans] = validate_data(data);
opt = set_default_options(opt, nunits);

% Remember the originally-requested basis (e.g. 'compare') before it is resolved,
% so the recovery-tradeoff figure can show the right curve(s).
orig_basis = opt.basis;

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

if isfield(opt, 'gsn_result') && ~isempty(opt.gsn_result)
    % Reuse a provided GSN result (struct or path to a .mat file), skipping the
    % expensive GSN estimation -- useful for sweeping hyperparameters (alpha,
    % basis, criterion, ...) on the same data. Mirrors the Python opt['gsn_result'].
    gsn_result = load_gsn_result(opt.gsn_result);
    if ~isfield(gsn_result, 'cSb') || ~isfield(gsn_result, 'cNb')
        error('opt.gsn_result must contain cSb and cNb fields');
    end
    if size(gsn_result.cSb, 1) ~= nunits
        error('gsn_result covariance size (%d) does not match data (%d units)', ...
              size(gsn_result.cSb, 1), nunits);
    end
    % If the loaded result carries cached eigvecs for the requested signal/
    % difference basis, swap basis -> matrix + eigenvalues so construct_basis
    % skips its own O(N^3) eigh (bit-equivalent to the string-basis path).
    opt = use_cached_eigvecs(opt, gsn_result);
    if opt.wantverbose
        fprintf('PSN: Using provided GSN result (skipping GSN estimation)...\n');
    end
else
    if opt.wantverbose
        fprintf('PSN: Running GSN to estimate signal and noise covariances...\n');
    end
    gsn_opt = opt.gsn_args;
    if ~isfield(gsn_opt, 'wantverbose'), gsn_opt.wantverbose = 0; end
    if ~isfield(gsn_opt, 'wantshrinkage'), gsn_opt.wantshrinkage = 1; end
    gsn_result = performgsn(data, gsn_opt);
end
cSb = gsn_result.cSb; % signal covariance (symmetric)
cNb = gsn_result.cNb; % noise covariance (symmetric)

% =========================================================================
% HIGH-LEVEL SELECTION: full-rank Wiener (criterion/basis 'wiener'), or
% 'compare' basis resolution. Wiener bypasses basis construction and
% truncation; 'compare' resolves to 'signal'/'difference' before the pipeline.
% =========================================================================
is_wiener = strcmp(opt.criterion, 'wiener') || ...
            ((ischar(opt.basis) || isstring(opt.basis)) && strcmp(char(opt.basis), 'wiener'));
if is_wiener
    if opt.wantverbose
        fprintf('PSN: Applying full-rank matrix Wiener filter...\n');
    end
    results = denoise_fullrank_wiener(cSb, cNb, data, trial_avg, unit_means, ...
                                      ntrials_avg, nunits, gsn_result, opt);
    results.threshold_selection = struct('mode', 'wiener', 'basis', [], ...
        'criterion', 'wiener', 'best_threshold', results.best_threshold);
    % Recovery-tradeoff data for the figure (Wiener point + trial-avg + chosen).
    results = attach_recovery_tradeoff(results, cSb, cNb, ntrials_avg, data, ...
                                       unit_means, has_nans, 'wiener', nunits);
    if opt.wantfig
        if opt.wantverbose
            fprintf('PSN: Generating diagnostic figures...\n');
        end
        visualize_results(results, opt);
    end
    results.runtime = toc(psn_t_start);
    if opt.wantverbose
        fprintf('PSN: Complete! Full-rank Wiener filter (%.1f effective dimensions). Runtime: %.3fs\n', ...
                results.best_threshold, results.runtime);
    end
    return;
end

cmp_info = [];   % populated only when basis='compare' resolves; lets STEP 4/5 and
                 % the recovery figure reuse the eigvecs/projections already built.
if (ischar(opt.basis) || isstring(opt.basis)) && strcmp(char(opt.basis), 'compare')
    [opt.basis, cmp_info] = select_compare_basis(cSb, cNb, ntrials_avg, opt);
    if opt.wantverbose
        fprintf('PSN: basis=''compare'' resolved to ''%s''\n', opt.basis);
    end
end

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

if ~isempty(cmp_info)
    % basis='compare' already eigendecomposed the chosen basis in
    % select_compare_basis (with the same eigh_descending_sym convention as
    % construct_basis), so reuse it instead of repeating the O(N^3) eigh.
    basis = cmp_info.chosen_V;
    basis_eigenvalues = cmp_info.chosen_evals;
else
    % Pass cached eigenvalues (set by use_cached_eigvecs) so a matrix basis can
    % skip the eigh while keeping eigenvalue-based ordering.
    if isfield(opt, 'basis_eigenvalues'), cbe = opt.basis_eigenvalues; else, cbe = []; end
    [basis, basis_eigenvalues] = construct_basis(cSb, cNb, opt.basis, data, trial_avg, unit_means, ntrials_avg, has_nans, cbe);
end

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
if ~isempty(cmp_info)
    % Reuse the projections select_compare_basis already computed for the chosen
    % basis (identical formula + clamp to project_covs).
    signal_proj = cmp_info.chosen_signal_proj;
    noise_proj  = cmp_info.chosen_noise_proj;
else
    [signal_proj, noise_proj] = project_covs(cSb, cNb, basis);
end

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
else
    % HYBRID MODE: unit-specific thresholds with a shared global basis ordering
    [denoiser, best_threshold, objective, signalvar, noisevar, unit_cumsum_curves, ...
     unit_signal_vars, unit_noise_vars] = ...
        denoise_unitwise(basis, signal_proj, noise_proj, basis_eigenvalues, ...
                         ntrials_avg, opt);
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

% Selection metadata (parity with Python). Set for 'compare' here; the wiener
% path sets its own threshold_selection above. Plain signal/difference runs
% leave both unset (matches Python).
if ~isempty(cmp_info)
    results.threshold_selection = struct( ...
        'mode', char(orig_basis), 'basis', cmp_info.chosen_basis, ...
        'criterion', opt.criterion, 'best_threshold', results.best_threshold, ...
        'recovery', cmp_info.chosen_recovery, 'sv_frac', cmp_info.chosen_sv_frac);
    results.diagnostics = struct( ...
        'mode', 'compare', 'picked_by', 'analytic_recovery', 'tie_eps', cmp_info.tie_eps, ...
        'candidates', cmp_info.candidates, ...
        'eigvecs_by_basis', struct('signal', cmp_info.V_signal, 'difference', cmp_info.V_difference));
end

% Alpha interpolation metadata (parity with Python): the prediction-peak and
% variance-target dimension counts the alpha threshold interpolates between.
if isfield(opt, 'alpha') && ~isempty(opt.alpha) && isnumeric(signalvar)
    diff_obj = signalvar(:) - noisevar(:) / ntrials_avg;
    pred_obj = [0; cumsum(diff_obj)];
    [~, kp] = max(pred_obj);  k_pred = kp - 1;               % 0-indexed argmax
    sig_cs = [0; cumsum(signalvar(:))];
    total_signal = sig_cs(end);
    vt = min(max(opt.variance_threshold, 0), 1);
    S_var = vt * total_signal;
    nd = size(basis, 2);
    if total_signal <= 0
        k_var = 0;
    else
        kv = find(sig_cs >= S_var, 1);
        if isempty(kv), k_var = nd; else, k_var = kv - 1; end
        k_var = min(k_var, nd);
    end
    results.alpha_info = struct('k_pred', k_pred, 'k_var', k_var, 'alpha', opt.alpha);
end

% =========================================================================
% STEP 11b: Recovery-tradeoff diagnostic data (analytic + split-half curves).
% Always computed so it is available regardless of wantfig; the figure reads it.
% =========================================================================
% For 'compare', hand both candidate eigvec matrices to the figure so it reuses
% them instead of re-eighing the basis the pipeline did not keep.
if ~isempty(cmp_info)
    rt_extra_bases = struct('signal', cmp_info.V_signal, 'difference', cmp_info.V_difference);
else
    rt_extra_bases = [];
end
results = attach_recovery_tradeoff(results, cSb, cNb, ntrials_avg, data, ...
                                   unit_means, has_nans, orig_basis, nunits, rt_extra_bases);

% =========================================================================
% STEP 12: Visualization
% =========================================================================

if opt.wantfig
    if opt.wantverbose
        fprintf('PSN: Generating diagnostic figures...\n');
    end
    visualize_results(results, opt);
end

results.runtime = toc(psn_t_start);
if opt.wantverbose
    fprintf('PSN: Complete! Retained %s dimensions. Runtime: %.3fs\n', ...
            format_threshold(best_threshold), results.runtime);
end

end  % End of main psn function
