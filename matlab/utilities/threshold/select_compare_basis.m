function [chosen, info] = select_compare_basis(cSb, cNb, ntrials, opt, data, unit_means, has_nans)
% SELECT_COMPARE_BASIS  Resolve basis='compare' to 'signal' or 'difference'.
%
%   [chosen, info] = select_compare_basis(cSb, cNb, ntrials, opt, data, ...
%   unit_means, has_nans) builds the signal and difference bases, each truncated
%   at its max-tradeoff threshold, and returns whichever has the higher empirical
%   split-half r evaluated AT that threshold. Mirrors the Python select_threshold
%   'compare' branch.
%
%   The eigendecompositions use eigh_descending_sym (the same routine, with the
%   same sign convention, that construct_basis uses), so the eigenvectors
%   returned in <info> are bit-identical to what the main pipeline would rebuild.
%   The recovery comparison itself is sign-invariant (it uses sum((C*V).*V)), so
%   the choice is unchanged by the sign standardization.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <cSb> - [N x N] symmetric signal covariance matrix from GSN.
%
% <cNb> - [N x N] symmetric noise covariance matrix from GSN.
%
% <ntrials> - scalar, number of trials (or average if NaNs present).
%
% <opt> - struct with PSN options. Uses opt.criterion ('max-tradeoff' uses
%   max_tradeoff_threshold; otherwise select_threshold_analytic).
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <chosen> - char, 'signal' or 'difference' (the winning basis).
%
% <info> - struct of artifacts already computed here, so the caller can reuse
%   them instead of recomputing the (O(N^3)) eigendecomposition and projections:
%     .chosen_basis        - same as <chosen>
%     .chosen_V            - [N x N] eigenvectors of the chosen basis
%     .chosen_evals        - [N x 1] eigenvalues of the chosen basis
%     .chosen_signal_proj  - [N x 1] per-dim signal variance, clamped >= 0
%     .chosen_noise_proj   - [N x 1] per-dim noise variance, clamped >= 0
%                            (both match project_covs, which also clamps)
%     .V_signal,.V_difference - both candidate eigenvector matrices, so the
%                            recovery-tradeoff figure can reuse them (no re-eigh)

    if nargin < 7, has_nans = false; end

    bases = {'signal', 'difference'};
    recovery = zeros(1, 2);
    shr = zeros(1, 2);            % split-half r at each basis's max-tradeoff threshold
    ks = zeros(1, 2);
    svf = zeros(1, 2);
    Vs = cell(1, 2);
    evals_all = cell(1, 2);
    sig_all = cell(1, 2);
    noi_all = cell(1, 2);

    for i = 1:2
        b = bases{i};
        if strcmp(b, 'signal')
            % Match construct_basis('signal'): eigh of cSb.
            [evals, V] = eigh_descending_sym(cSb, false);
        else
            % Match construct_basis('difference'): eigh of symmetrized cSb - cNb/t.
            A = cSb - cNb / ntrials;
            A = (A + A') / 2;
            [evals, V] = eigh_descending_sym(A, false);
        end

        % Sign-invariant per-dim projections (drive the choice; kept unclamped
        % here so the recovery comparison is identical to the historical result).
        sig = sum((cSb * V) .* V, 1)';
        noi = sum((cNb * V) .* V, 1)';
        total_S = sum(sig);

        % The compare basis comparison itself is unconstrained; allowable_thresholds
        % restricts only the final denoise threshold in the main pipeline.
        if strcmp(opt.criterion, 'max-tradeoff')
            k = max_tradeoff_threshold(sig, noi, ntrials);
        else
            sub_opt = opt;
            sub_opt.basis = b;
            sub_opt.allowable_thresholds = [];
            [k, ~] = select_threshold_analytic(sig, noi, evals, ntrials, sub_opt);
        end
        k = max(0, min(k, numel(sig)));

        if total_S <= 0
            recovery(i) = 0;
            svf(i) = 0;
        else
            rec_curve = [0; cumsum(sig - noi / ntrials)] / total_S;
            recovery(i) = rec_curve(k + 1);
            sv_curve = [0; cumsum(sig)] / total_S;
            svf(i) = sv_curve(k + 1);
        end

        % Split-half r of this basis truncated to its max-tradeoff threshold.
        if k > 0
            D = V(:, 1:k) * V(:, 1:k)';
        else
            D = zeros(size(V, 1));
        end
        shr(i) = split_half_r(data, D, unit_means, has_nans);

        ks(i) = k;
        Vs{i} = V;
        evals_all{i} = evals;
        sig_all{i} = sig;
        noi_all{i} = noi;
    end

    % Choose the basis with the higher split-half r at its max-tradeoff threshold.
    if shr(2) > shr(1)
        chosen = 'difference';
        ci = 2;
    else
        chosen = 'signal';
        ci = 1;
    end

    info = struct();
    info.chosen_basis = chosen;
    info.chosen_V = Vs{ci};
    info.chosen_evals = evals_all{ci};
    % Clamp to match project_covs (covariance projections can dip slightly
    % negative from numerical error).
    info.chosen_signal_proj = max(sig_all{ci}, 0);
    info.chosen_noise_proj  = max(noi_all{ci}, 0);
    info.V_signal = Vs{1};
    info.V_difference = Vs{2};
    % Selection metrics for results.threshold_selection / results.diagnostics.
    info.chosen_best_threshold = ks(ci);
    info.chosen_recovery = recovery(ci);
    info.chosen_sv_frac = svf(ci);
    info.chosen_split_half_r = shr(ci);
    info.candidates = struct( ...
        'signal',     struct('best_threshold', ks(1), 'recovery', recovery(1), 'sv_frac', svf(1), 'split_half_r', shr(1)), ...
        'difference', struct('best_threshold', ks(2), 'recovery', recovery(2), 'sv_frac', svf(2), 'split_half_r', shr(2)));
end
