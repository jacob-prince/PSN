function k = argmax_allowable(objective, allowable)
% ARGMAX_ALLOWABLE  Number of dims in <allowable> maximizing objective[k].
%
%   k = argmax_allowable(objective, allowable) returns the allowable threshold
%   (number of dimensions) with the largest cumulative objective value; ties
%   resolve to the fewest dimensions (matching the unconstrained argmax). Falls
%   back to the unconstrained argmax if no candidate is in range. Mirrors the
%   Python psn.utilities.threshold.select_allowable.argmax_allowable.
%
% <objective> - [ndims+1 x 1] cumulative curve, index (1-based) = #dims + 1
% <allowable> - vector of candidate threshold values
% <k>         - selected number of dimensions to retain

    objective = objective(:);
    ndims = numel(objective) - 1;
    C = allowable_candidates(allowable, ndims);
    if isempty(C)
        [~, im] = max(objective);
        k = im - 1;                    % 1-based index -> number of dims
        return;
    end
    vals = objective(C + 1);           % objective at each candidate (#dims -> 1-based)
    [~, im] = max(vals);              % first max -> smallest c (fewest dims)
    k = C(im);
end
