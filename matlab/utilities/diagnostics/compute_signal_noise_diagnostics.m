function [svnv_before, svnv_after] = compute_signal_noise_diagnostics(...
    threshold_method, unit_signal_vars, unit_noise_vars, best_threshold, nunits, ntrials)
% COMPUTE_SIGNAL_NOISE_DIAGNOSTICS  Sum signal/noise variance before/after thresholding
%
%   [svnv_before, svnv_after] = compute_signal_noise_diagnostics(threshold_method,
%   unit_signal_vars, unit_noise_vars, best_threshold, nunits, ntrials) computes the total
%   signal and noise variance for each unit before and after applying PSN thresholding.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <threshold_method> - string, 'global' or 'hybrid'
%
% <unit_signal_vars> - {nunits x 1} cell array, each cell contains [ndims x 1]
%   weighted signal variances for that unit
%
% <unit_noise_vars> - {nunits x 1} cell array, each cell contains [ndims x 1]
%   weighted noise variances for that unit
%
% <best_threshold> - scalar (for global) or [nunits x 1] (for hybrid/unit),
%   number of dimensions retained per unit
%
% <nunits> - scalar, number of units
%
% <ntrials> - scalar, number of trials (or average if NaNs present). Used to
%   convert noise variance to trial-averaged noise variance.
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <svnv_before> - [nunits x 2] matrix. Column 1 is total signal variance,
%   column 2 is trial-averaged noise variance (noise_var / ntrials), summed
%   across all dimensions (before thresholding)
%
% <svnv_after> - [nunits x 2] matrix. Column 1 is total signal variance,
%   column 2 is trial-averaged noise variance (noise_var / ntrials), summed
%   only across retained dimensions (after thresholding)

    svnv_before = zeros(nunits, 2);
    svnv_after = zeros(nunits, 2);

    if strcmp(threshold_method, 'global')
        % Global: all units share same threshold, but each unit gets different
        % amounts of signal/noise variance based on weighted projections
        for u = 1:nunits
            sig_u = unit_signal_vars{u};
            noi_u = unit_noise_vars{u};
            k = best_threshold;  % Same threshold for all units
            svnv_before(u, :) = [sum(sig_u), sum(noi_u) / ntrials];
            svnv_after(u, :) = [(k > 0) * sum(sig_u(1:k)), (k > 0) * sum(noi_u(1:k)) / ntrials];
        end
    else
        % Unit-specific: each unit has weighted projections and individual threshold
        for u = 1:nunits
            sig_u = unit_signal_vars{u};
            noi_u = unit_noise_vars{u};
            k_u = best_threshold(u);
            svnv_before(u, :) = [sum(sig_u), sum(noi_u) / ntrials];
            svnv_after(u, :) = [(k_u > 0) * sum(sig_u(1:k_u)), (k_u > 0) * sum(noi_u(1:k_u)) / ntrials];
        end
    end
end

