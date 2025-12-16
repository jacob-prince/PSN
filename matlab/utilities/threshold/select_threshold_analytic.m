function [k, objective] = select_threshold_analytic(signal, noise, basis_eigenvalues, ntrials, opt)
% SELECT_THRESHOLD_ANALYTIC  Choose threshold using analytic or variance criterion
%
%   [k, objective] = select_threshold_analytic(signal, noise, basis_eigenvalues,
%   ntrials, opt) determines the optimal number of dimensions to retain based
%   on the specified criterion.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <signal> - [ndims x 1] signal variance per dimension
%
% <noise> - [ndims x 1] noise variance per dimension
%
% <basis_eigenvalues> - [ndims x 1] eigenvalues from basis construction,
%   or [] if not available (e.g., for custom/random bases)
%
% <ntrials> - scalar, number of trials (or average if NaNs present)
%
% <opt> - struct with PSN options. Relevant fields:
%   .criterion          - 'prediction', 'variance', or 'variance_eigenvalues'
%   .variance_threshold - target fraction for variance-based criteria (default 0.99)
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <k> - scalar, selected threshold (number of dimensions to retain, 0 to ndims)
%
% <objective> - [ndims+1 x 1] cumulative objective curve that was actually
%   used for threshold selection. Interpretation depends on criterion:
%     'prediction'           - cumsum(signal - noise/ntrials)
%     'variance'             - cumsum(signal)
%     'variance_eigenvalues' - cumsum(max(basis_eigenvalues, 0))
%
% -------------------------------------------------------------------------
% Criteria:
% -------------------------------------------------------------------------
%
% <criterion> = 'prediction': Maximize expected out-of-sample prediction
%   quality by finding k that maximizes cumsum(signal - noise/ntrials)
%
% <criterion> = 'variance': Retain dimensions until cumulative signal
%   variance reaches variance_threshold * total_signal_variance
%
% <criterion> = 'variance_eigenvalues': Retain dimensions until cumulative
%   positive eigenvalues reach variance_threshold * total_positive_eigenvalues.
%   Only compatible with named basis types (not custom/random matrices)

    ndims = length(signal);
    scaled_noise = noise / ntrials;
    diff = signal - scaled_noise;

    switch opt.criterion
        case 'prediction'
            % Maximize expected out-of-sample prediction quality
            objective = [0; cumsum(diff(:))];
            [~, k] = max(objective);
            k = k - 1;  % Index to number of dims

        case 'variance'
            % Retain fraction of signal variance
            % Return cumulative signal variance as objective
            objective = [0; cumsum(signal(:))];

            vt = max(0, min(1, opt.variance_threshold));
            if vt == 0
                k = 0;
            else
                total = objective(end);
                if total <= 0
                    k = 0;
                else
                    k = find(objective >= vt * total, 1, 'first');
                    if isempty(k)
                        k = 0;
                    else
                        k = k - 1;  % Convert index to number of dims
                    end
                    k = min(k, ndims);
                end
            end

        case 'variance_eigenvalues'
            % Retain fraction of total positive eigenvalue sum
            % Only valid when eigenvalues are available (signal, difference, noise bases)
            % For PCA: use signal variance instead (PCA eigenvalues are for visualization only)

            use_pca_basis = ischar(opt.basis) && strcmp(opt.basis, 'pca');

            if use_pca_basis
                % PCA special case: use signal variance instead of PCA eigenvalues
                objective = [0; cumsum(signal(:))];
            else
                if isempty(basis_eigenvalues)
                    error(['variance_eigenvalues criterion requires eigenvalues.\n' ...
                           'Not compatible with custom basis or random basis.']);
                end
                % Return cumulative positive eigenvalues as objective
                pos_evals = max(basis_eigenvalues(:), 0);
                objective = [0; cumsum(pos_evals)];
            end

            vt = max(0, min(1, opt.variance_threshold));
            if vt == 0
                k = 0;
            else
                total = objective(end);
                if total <= 0
                    k = 0;
                else
                    k = find(objective >= vt * total, 1, 'first');
                    if isempty(k)
                        k = 0;
                    else
                        k = k - 1;  % Convert index to number of dims
                    end
                    k = min(k, ndims);
                end
            end

        otherwise
            error('Unknown criterion: %s', opt.criterion);
    end
