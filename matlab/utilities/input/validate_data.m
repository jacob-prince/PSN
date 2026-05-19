function [nunits, nconds, ntrials, ntrials_avg, has_nans] = validate_data(data)
% VALIDATE_DATA  Check that data has correct shape and valid values
%
%   [nunits, nconds, ntrials, ntrials_avg, has_nans] = validate_data(data)
%   validates the input data array and extracts dimension information.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <data> - [nunits x nconds x ntrials] numeric array. Must have at least
%   2 trials and 2 conditions. May contain NaNs to indicate missing trials.
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <nunits> - number of units (first dimension of data)
%
% <nconds> - number of conditions (second dimension of data)
%
% <ntrials> - maximum number of trials (third dimension of data)
%
% <ntrials_avg> - average number of valid trials per condition (used in
%   PSN formulas). For data without NaNs, equals <ntrials>. For data with
%   NaNs, computed as sum(validcnt(validcnt>1))/nconds, following GSN's
%   approach in rsanoiseceiling.m
%
% <has_nans> - logical. True if data contains any NaN values

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
