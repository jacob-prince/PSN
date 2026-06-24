function s = format_threshold(best_threshold)
% FORMAT_THRESHOLD  Compact verbose string for the retained-dimension threshold(s).
%
%   s = format_threshold(best_threshold) mirrors the Python format_threshold:
%   the integer threshold for global mode, or 'mean +/- std across units' for
%   per-unit (hybrid) mode, never the full per-unit list.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <best_threshold> - scalar or [nunits] vector. Number of dimensions retained:
%   a single value (global mode) or one threshold per unit (hybrid mode).
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <s> - char. The integer threshold for global mode, or the mean +/- std summary
%   across units for hybrid mode.

    bt = best_threshold(:);
    if numel(bt) == 1
        s = sprintf('%d', floor(bt));
    else
        % std(.,1) = population std (ddof=0), matching numpy's np.std default.
        s = sprintf('%.1f %s %.1f (mean %s std across units)', ...
                    mean(bt), char(177), std(bt, 1), char(177));
    end
end
