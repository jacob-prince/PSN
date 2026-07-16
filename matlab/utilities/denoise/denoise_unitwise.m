function [denoiser, best_threshold, objective, signalvar, noisevar, unit_cumsum_curves, ...
          unit_signal_vars, unit_noise_vars] = ...
    denoise_unitwise(basis, signal_proj, noise_proj, basis_eigenvalues, ntrials, opt)
% DENOISE_UNITWISE  Unit-specific denoising (non-symmetric denoiser)
%
%   [denoiser, best_threshold, objective, ...] = denoise_unitwise(basis, signal_proj,
%   noise_proj, basis_eigenvalues, ntrials, opt) builds a generally non-symmetric
%   denoising matrix with unit-specific thresholds applied on a shared global
%   dimension ordering (threshold_method='hybrid').
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
% <opt> - struct with PSN options (criterion, allowable_thresholds, unit_groups, etc.)
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <denoiser> - [nunits x nunits] generally non-symmetric denoising matrix.
%   Column u is Bu * Bu(u,:)' where Bu are the selected dimensions for unit u
%
% <best_threshold> - [nunits x 1] number of dimensions retained per unit.
%   Units in the same unit_groups share the same threshold
%
% <objective> - [ndims+1 x 1] population-averaged cumulative objective curve
%
% <signalvar> - [ndims x 1] population-averaged signal variance per dimension
%
% <noisevar> - [ndims x 1] population-averaged noise variance per dimension
%
% <unit_cumsum_curves> - {nunits x 1} cell array of unit-specific objective curves
%
% <unit_signal_vars> - {nunits x 1} cell array of unit-specific weighted signal variances
%
% <unit_noise_vars> - {nunits x 1} cell array of unit-specific weighted noise variances
%
% -------------------------------------------------------------------------
% Algorithm:
% -------------------------------------------------------------------------
%
% Each unit receives:
%   - Weighted signal/noise projections: w = basis(u,:)^2, sig_u = w .* signal_proj
%   - Unit-specific threshold selection with optional unit_groups averaging
%   - Denoiser column: Bu * Bu(u,:)' where Bu = basis(:, 1:k_u)
%
% The denoiser is generally non-symmetric. Apply as: denoiser' * data

    [nunits, ndims] = size(basis);

    denoiser = zeros(nunits, nunits);

    % First pass: compute weighted projections and objectives for each unit.
    % Hybrid mode uses the shared global ordering (do_unit_ranking=false).
    [unit_cumsum_curves, unit_signal_vars, unit_noise_vars, ~] = ...
        compute_unit_weighted_projections(basis, signal_proj, noise_proj, ntrials, false);

    % Second pass: select thresholds considering unit_groups. When
    % allowable_thresholds restricts the choice, each group picks the BEST
    % threshold among the allowable values (best-among-allowable); a single
    % allowable value forces exactly that many dimensions.
    allow = opt.allowable_thresholds;
    unique_groups = unique(opt.unit_groups);
    best_threshold = zeros(nunits, 1);

    for g = unique_groups'
        group_mask = (opt.unit_groups == g);
        group_indices = find(group_mask);

        if isfield(opt, 'alpha') && ~isempty(opt.alpha)
            % Alpha interpolation: blend the prediction peak and the trial-average
            % (do-nothing) point on this group's averaged curves. alpha's right
            % endpoint is the full signal variance; alpha does NOT use
            % variance_threshold.
            alpha_val = opt.alpha;
            avg_signal = mean(cat(2, unit_signal_vars{group_indices}), 2);
            avg_curve = mean(cat(2, unit_cumsum_curves{group_indices}), 2);
            [~, k_pred] = max(avg_curve);
            k_pred = k_pred - 1;                      % number of dims at prediction peak
            sig_cs = [0; cumsum(avg_signal(:))];
            S_pred = sig_cs(k_pred + 1);
            total = sig_cs(end);
            S_var = total;
            target = S_pred + alpha_val * max(0, S_var - S_pred);
            if total <= 0
                k_group = 0;
            else
                idx = find(sig_cs >= target, 1, 'first');
                if isempty(idx)
                    k_group = ndims;
                else
                    k_group = idx - 1;
                end
                k_group = max(k_group, k_pred);
                k_group = min(k_group, ndims);
                if ~isempty(allow)
                    % Best-among-allowable: smallest allowable threshold that
                    % reaches the target without going below the prediction peak;
                    % else snap the unconstrained pick to the nearest allowable.
                    C = allowable_candidates(allow, ndims);
                    elig = C(C >= k_pred & sig_cs(C + 1) >= target);
                    if ~isempty(elig)
                        k_group = min(elig);
                    else
                        k_group = constrain_to_allowable(k_group, C);
                    end
                end
            end
        elseif strcmp(opt.criterion, 'prediction')
            % Average objective curves across units in this group
            % All curves should have the same length (ndims+1)
            avg_curve = mean(cat(2, unit_cumsum_curves{group_indices}), 2);
            if ~isempty(allow)
                k_group = argmax_allowable(avg_curve, allow);
            else
                [~, k_group] = max(avg_curve);
                k_group = k_group - 1;  % Convert index to number of dims
            end
        elseif strcmp(opt.criterion, 'max-tradeoff')
            % Max-tradeoff on this group's averaged recovery curve.
            avg_signal = mean(cat(2, unit_signal_vars{group_indices}), 2);
            avg_noise = mean(cat(2, unit_noise_vars{group_indices}), 2);
            k_group = max_tradeoff_threshold(avg_signal, avg_noise, ntrials, allow);
        elseif strcmp(opt.criterion, 'variance')
            % Average signal variances across units in this group
            avg_signal = mean(cat(2, unit_signal_vars{group_indices}), 2);
            vt = max(0, min(1, opt.variance_threshold));
            % Prepend 0 for consistency with global mode (index 1 = 0 dims)
            cs = [0; cumsum(avg_signal(:))];
            if ~isempty(allow)
                % Best-among-allowable: smallest allowable threshold whose
                % cumulative variance reaches the target; else the largest allowable.
                total = cs(end);
                k_group = first_reach_allowable(cs, vt * total, allow);
            elseif vt == 0
                k_group = 0;
            else
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

        % Assign this threshold to all units in the group
        best_threshold(group_mask) = k_group;
    end

    % Third pass: build denoiser columns. All units share the global ordering,
    % so group units by threshold and build each group with one matmul.
    unique_thresholds = unique(best_threshold(best_threshold > 0));

    for k = unique_thresholds'
        units_with_k = find(best_threshold == k);
        % Vectorized: denoiser(:, units) = Bu * Bu(units, :)'
        Bu = basis(:, 1:k);
        denoiser(:, units_with_k) = Bu * Bu(units_with_k, :)';
    end

    % Population-level totals for visualization
    % Sum across units to get total variance (since unit weights sum to 1,
    % mean * nunits = sum). This makes signalvar/noisevar consistent with
    % eigenvalues and global mode.
    if ~isempty(unit_signal_vars)
        signalvar = mean(cat(2, unit_signal_vars{:}), 2) * nunits;
        noisevar = mean(cat(2, unit_noise_vars{:}), 2) * nunits;
        objective = [0; cumsum(signalvar - noisevar / ntrials)];

        % Note: unit_cumsum_curves remain unscaled (per-unit contributions)
        % Their sum equals the objective (green line)
    else
        signalvar = [];
        noisevar = [];
        objective = zeros(1, 1);
    end
end
