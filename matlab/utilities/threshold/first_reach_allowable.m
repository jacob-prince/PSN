function k = first_reach_allowable(cum_curve, target, allowable, floor_k)
% FIRST_REACH_ALLOWABLE  Smallest allowable threshold reaching a cumulative target.
%
%   k = first_reach_allowable(cum_curve, target, allowable, floor_k) returns the
%   smallest value in <allowable> with cum_curve[k] >= target and k >= floor_k.
%   If none qualify, returns the largest allowable candidate (the closest we can
%   get from below). Mirrors the Python
%   psn.utilities.threshold.select_allowable.first_reach_allowable.
%
% <cum_curve> - [ndims+1 x 1] cumulative curve, index (1-based) = #dims + 1
% <target>    - scalar target value
% <allowable> - vector of candidate threshold values
% <floor_k>   - (optional) minimum allowable threshold (default 0)
% <k>         - selected number of dimensions to retain

    if nargin < 4 || isempty(floor_k), floor_k = 0; end
    cum_curve = cum_curve(:);
    ndims = numel(cum_curve) - 1;
    C = allowable_candidates(allowable, ndims);
    if isempty(C)
        k = 0;
        return;
    end
    elig = C(C >= floor_k & cum_curve(C + 1) >= target);
    if ~isempty(elig)
        k = min(elig);
    else
        k = max(C);
    end
end
