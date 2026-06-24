function r = split_half_r(data, D, unit_means, has_nans)
% SPLIT_HALF_R  Median per-unit split-half reliability of denoiser D.
%
%   r = split_half_r(data, D, unit_means, has_nans) splits trials into even/odd
%   halves; for each half it takes the trial-average, denoises the OTHER half's
%   average with D, and correlates them per unit across conditions (symmetrized).
%   Returns the median across units. NaN entries are handled pairwise. Matches
%   the diagnostic figure's split-half value and the Python split_half_r. Used by
%   basis='compare' to choose between bases at their max-tradeoff thresholds.
%
% <data>       - [nunits x nconds x ntrials]
% <D>          - [nunits x nunits] denoiser (applied as D * (avg - means) + means)
% <unit_means> - [nunits x 1] per-unit means used for centering
% <has_nans>   - (optional) whether <data> contains NaNs (default false)

    if nargin < 4, has_nans = false; end
    ntrials = size(data, 3);
    A = data(:, :, 1:2:ntrials);
    B = data(:, :, 2:2:ntrials);
    if has_nans
        tavg_A = mean(A, 3, 'omitnan');
        tavg_B = mean(B, 3, 'omitnan');
    else
        tavg_A = mean(A, 3);
        tavg_B = mean(B, 3);
    end
    um = unit_means(:);
    dn_A = D * (tavg_A - um) + um;
    dn_B = D * (tavg_B - um) + um;
    both = 0.5 * (local_row_corr(tavg_A, dn_B, has_nans) + local_row_corr(dn_A, tavg_B, has_nans));
    m = ~isnan(both);
    if any(m), r = median(both(m)); else, r = NaN; end
end

function out = local_row_corr(X, Y, has_nans)
% Per-row (per-unit) Pearson correlation across conditions; NaN-aware.
    n = size(X, 1);
    out = nan(n, 1);
    if has_nans
        for u = 1:n
            mask = ~(isnan(X(u, :)) | isnan(Y(u, :)));
            if sum(mask) > 1 && std(X(u, mask)) > 0 && std(Y(u, mask)) > 0
                c = corrcoef(X(u, mask), Y(u, mask));
                out(u) = c(1, 2);
            end
        end
    else
        Xc = X - mean(X, 2);
        Yc = Y - mean(Y, 2);
        den = sqrt(sum(Xc.^2, 2) .* sum(Yc.^2, 2));
        nz = den > 0;
        out(nz) = sum(Xc(nz, :) .* Yc(nz, :), 2) ./ den(nz);
    end
end
