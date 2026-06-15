function k = max_tradeoff_threshold(signal, noise, ntrials)
% MAX_TRADEOFF_THRESHOLD  Select the "max-tradeoff" threshold on the recovery curve.
%
%   k = max_tradeoff_threshold(signal, noise, ntrials) returns the most-unbiased
%   operating point that still captures the bulk of the achievable recovery: the
%   point on the descending (peak -> trial-average) limb of the recovery curve
%   that lies farthest from the chord joining those two anchors, measured in
%   (fraction-of-signal-variance-retained, normalized-recovery) coordinates.
%   Fully analytic; no cross-validation. Mirrors the Python max_tradeoff_threshold.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <signal>  - [ndims] per-dimension signal variance (basis order)
% <noise>   - [ndims] per-dimension noise variance (basis order)
% <ntrials> - scalar, number of trials (or average if NaNs present)
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <k> - selected number of dimensions to retain, in [0, ndims]

    signal = signal(:);
    noise = noise(:);
    ndims = numel(signal);
    if ndims == 0
        k = 0;
        return;
    end

    diff = signal - noise / ntrials;
    rec = [0; cumsum(diff)];            % recovery curve, length ndims+1
    sig_cum = [0; cumsum(signal)];      % cumulative signal variance
    total_S = sig_cum(end);

    [~, kp1] = max(rec);                % 1-based index of the recovery peak
    k_peak = kp1 - 1;                   % number of dims retained at the peak

    % Degenerate cases: no signal, or no usable descending limb -> the peak is
    % already the most we can do, so fall back to it.
    if total_S <= 0 || (ndims - k_peak) < 2
        k = k_peak;
        return;
    end
    rec_peak = rec(k_peak + 1);
    rec_ta = rec(ndims + 1);            % trial-average (retain all dims)
    if (rec_peak - rec_ta) <= 1e-12
        k = k_peak;
        return;
    end

    seg = (k_peak:ndims)';                              % dim counts on descending limb
    x = sig_cum(seg + 1) / total_S;                     % fraction signal var retained
    y = (rec(seg + 1) - rec_ta) / (rec_peak - rec_ta);  % recovery rescaled to [0, 1]
    x0 = x(1); y0 = y(1); x1 = x(end); y1 = y(end);
    denom = hypot(x1 - x0, y1 - y0);
    if denom <= 1e-12
        k = k_peak;
        return;
    end
    % perpendicular distance from the peak -> trial-average chord
    dist = abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom;
    [~, idx] = max(dist);
    k = seg(idx);
end
