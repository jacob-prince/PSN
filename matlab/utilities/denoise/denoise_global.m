function [denoiser, best_threshold, objective, signalvar, noisevar, unit_cumsum_curves, ...
          unit_signal_vars, unit_noise_vars] = ...
    denoise_global(basis, signal_proj, noise_proj, basis_eigenvalues, ntrials, opt)
% DENOISE_GLOBAL  Population-level denoising (symmetric denoiser)
%
%   [denoiser, best_threshold, objective, ...] = denoise_global(basis, signal_proj,
%   noise_proj, basis_eigenvalues, ntrials, opt) builds a symmetric denoising
%   matrix using a single threshold applied to all units.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <basis> - [nunits x ndims] orthonormal basis matrix
%
% <signal_proj> - [ndims x 1] signal variance per dimension
%
% <noise_proj> - [ndims x 1] noise variance per dimension
%
% <basis_eigenvalues> - [ndims x 1] eigenvalues from basis construction, or []
%
% <ntrials> - scalar, number of trials (or average if NaNs present)
%
% <opt> - struct with PSN options (criterion, allowable_thresholds, etc.)
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <denoiser> - [nunits x nunits] symmetric denoising matrix. If k dimensions
%   are retained, denoiser = basis(:,1:k) * basis(:,1:k)'
%
% <best_threshold> - scalar, number of dimensions retained (0 to ndims)
%
% <objective> - [ndims+1 x 1] cumulative objective curve used for threshold
%   selection. Depends on criterion: cumsum(signal - noise/ntrials) for
%   'prediction', cumsum(signal) for 'variance', or cumsum(eigenvalues)
%   for 'variance_eigenvalues'
%
% <signalvar> - [ndims x 1] signal variance per dimension (copy of signal_proj)
%
% <noisevar> - [ndims x 1] noise variance per dimension (copy of noise_proj)
%
% <unit_cumsum_curves> - {nunits x 1} cell array of unit-specific objective curves
%
% <unit_signal_vars> - {nunits x 1} cell array of unit-specific signal variances
%
% <unit_noise_vars> - {nunits x 1} cell array of unit-specific noise variances
%
% -------------------------------------------------------------------------
% Implementation notes:
% -------------------------------------------------------------------------
%
% Fast path: If using difference basis + prediction criterion, eigenvalues
% already encode signal - noise/ntrials, so we directly maximize cumsum(eigenvalues)

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

