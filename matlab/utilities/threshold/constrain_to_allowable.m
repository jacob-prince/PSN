
function k_constrained = constrain_to_allowable(k, allowable)
% CONSTRAIN_TO_ALLOWABLE  Force threshold to nearest allowable value
%
%   k_constrained = constrain_to_allowable(k, allowable) snaps the threshold(s)
%   to the nearest value in the allowable set, rounding up in case of ties.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <k> - scalar or [nunits x 1] vector of threshold values to constrain
%
% <allowable> - vector of allowed threshold values (e.g., [0, 5, 10, 15])
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <k_constrained> - same size as k, with each value replaced by the nearest
%   value from <allowable>. In case of a tie (equal distance to two allowable
%   values), rounds up to the larger value

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
