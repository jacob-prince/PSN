function [denoiser, best_threshold, objective, signalvar, noisevar, unit_cumsum_curves, ...
          unit_signal_vars, unit_noise_vars, unit_orderings] = ...
    denoise_unitwise(basis, signal_proj, noise_proj, basis_eigenvalues, ntrials, opt, threshold_only)
% DENOISE_UNITWISE  Unit-specific denoising (non-symmetric denoiser)
%
%   [denoiser, best_threshold, objective, ...] = denoise_unitwise(basis, signal_proj,
%   noise_proj, basis_eigenvalues, ntrials, opt, threshold_only) builds a generally
%   non-symmetric denoising matrix with unit-specific thresholds and optionally
%   unit-specific dimension orderings.
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
% <threshold_only> - logical. If true, use global dimension ordering with
%   unit-specific thresholds (hybrid mode). If false, use unit-specific
%   dimension ordering and thresholds (full unit-specific mode)
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
% <unit_orderings> - [nunits x ndims] dimension ordering for each unit
%
% -------------------------------------------------------------------------
% Algorithm:
% -------------------------------------------------------------------------
%
% Each unit receives:
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
