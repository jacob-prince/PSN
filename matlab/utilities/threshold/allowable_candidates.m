function C = allowable_candidates(allowable, ndims)
% ALLOWABLE_CANDIDATES  Sorted unique integer candidates within [0, ndims].
%
%   C = allowable_candidates(allowable, ndims) returns the unique, ascending,
%   integer-valued entries of <allowable> that fall in [0, ndims]. Mirrors the
%   Python psn.utilities.threshold.select_allowable.allowable_candidates.
%
% <allowable> - vector of candidate threshold values
% <ndims>     - scalar, number of basis dimensions (upper bound)
% <C>         - [m x 1] sorted unique candidates in [0, ndims]

    C = unique(fix(allowable(:)));     % truncate toward zero (matches numpy astype(int))
    C = C(C >= 0 & C <= ndims);
end
