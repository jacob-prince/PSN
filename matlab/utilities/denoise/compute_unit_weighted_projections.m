function [unit_cumsum_curves, unit_signal_vars, unit_noise_vars, unit_orderings] = ...
    compute_unit_weighted_projections(basis, signal_proj, noise_proj, ntrials, do_unit_ranking)
% COMPUTE_UNIT_WEIGHTED_PROJECTIONS  Compute unit-specific weighted variances and objective curves
%
%   [unit_cumsum_curves, unit_signal_vars, unit_noise_vars, unit_orderings] =
%   compute_unit_weighted_projections(basis, signal_proj, noise_proj, ntrials, do_unit_ranking)
%   computes how much signal and noise variance each basis dimension contributes
%   to each individual unit, and builds objective curves for unit-specific thresholding.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <basis> - [nunits x ndims] orthonormal basis matrix
%
% <signal_proj> - [ndims x 1] signal variance per dimension (from project_covs)
%
% <noise_proj> - [ndims x 1] noise variance per dimension (from project_covs)
%
% <ntrials> - scalar, number of trials (or average number of trials if NaNs present)
%
% <do_unit_ranking> - logical. If true, rank dimensions by each unit's weighted
%   signal variance (full unit-specific mode). If false, use global ordering
%   (hybrid mode with unit-specific thresholds only)
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <unit_cumsum_curves> - {nunits x 1} cell array. Each cell contains
%   [ndims+1 x 1] cumulative objective curve for that unit, computed as
%   cumsum(weighted_signal - weighted_noise/ntrials)
%
% <unit_signal_vars> - {nunits x 1} cell array. Each cell contains [ndims x 1]
%   weighted signal variance for that unit
%
% <unit_noise_vars> - {nunits x 1} cell array. Each cell contains [ndims x 1]
%   weighted noise variance for that unit
%
% <unit_orderings> - [nunits x ndims] dimension ordering indices. Row u gives
%   the dimension ordering for unit u. If do_unit_ranking=false, all rows
%   are 1:ndims
%
% -------------------------------------------------------------------------
% Algorithm:
% -------------------------------------------------------------------------
%
% For each unit u, computes weights w(d) = basis(u,d)^2, which measure how
% much dimension d affects unit u. Then computes weighted variances:
%   sig_u(d) = w(d) * signal_proj(d)
%   noi_u(d) = w(d) * noise_proj(d)

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
