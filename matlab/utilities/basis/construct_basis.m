function [basis, basis_eigenvalues] = construct_basis(cSb, cNb, basis_spec, data, trial_avg, unit_means, ntrials_avg, has_nans)
% CONSTRUCT_BASIS  Create the orthonormal basis for denoising
%
%   [basis, basis_eigenvalues] = construct_basis(cSb, cNb, basis_spec, ...)
%   constructs an orthonormal basis for PSN denoising according to the
%   specified basis type.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <cSb> - [nunits x nunits] symmetric signal covariance matrix from GSN
%
% <cNb> - [nunits x nunits] symmetric noise covariance matrix from GSN
%
% <basis_spec> - basis specifier. Either string or matrix:
%   'signal'     - Eigenvectors of signal covariance (cSb)
%   'difference' - Eigenvectors of cSb - cNb/ntrials_avg (emphasize signal-dominated directions)
%   'noise'      - Eigenvectors of noise covariance (cNb)
%   'pca'        - Standard PCA on trial-averaged data
%   'random'     - Random orthonormal basis (uses fixed seed for reproducibility)
%   B            - User-provided basis matrix [nunits x D] with orthonormal columns
%
% <data> - [nunits x nconds x ntrials] original data array
%
% <trial_avg> - [nunits x nconds] pre-computed trial-averaged data
%
% <unit_means> - [nunits x 1] mean response per unit
%
% <ntrials_avg> - scalar, average number of valid trials (handles NaN case correctly)
%
% <has_nans> - logical, whether data contains NaNs
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <basis> - [nunits x ndims] orthonormal basis vectors. Columns are sorted
%   by descending eigenvalue magnitude (for eigenvalue-based methods) or
%   left as provided (for custom/random bases)
%
% <basis_eigenvalues> - [ndims x 1] eigenvalues associated with basis, sorted
%   to match basis columns. Empty ([]) for custom or random bases. For 'pca'
%   basis, contains PCA eigenvalues (used for ordering, but not appropriate
%   for criterion='variance_eigenvalues')

    nunits = size(data, 1);

    if ischar(basis_spec) || isstring(basis_spec)
        basis_spec = char(basis_spec);

        switch basis_spec
            case 'signal'
                % Eigenvectors of signal covariance (GSN returns symmetric)
                [basis_eigenvalues, basis] = eigh_descending_sym(cSb);

            case 'difference'
                % Eigenvectors of signal - scaled noise
                % Eigenvalues encode the net benefit per dimension
                % Use ntrials_avg to properly handle NaN case
                A = cSb - cNb / ntrials_avg;
                % Symmetrize derived matrix to handle numerical errors
                A = (A + A') / 2;
                [basis_eigenvalues, basis] = eigh_descending_sym(A);  % already symmetrized above

            case 'noise'
                % Eigenvectors of noise covariance (GSN returns symmetric)
                [basis_eigenvalues, basis] = eigh_descending_sym(cNb);

            case 'pca'
                % Standard PCA on trial-averaged data
                % Eigenvectors from empirical covariance, but treated exactly like signal basis
                % in all subsequent ranking/thresholding (uses GSN signal_proj, not PCA eigenvalues).
                % PCA eigenvalues are kept for visualization purposes only.
                % Use pre-computed trial_avg to avoid redundant computation
                trial_avg_demeaned = trial_avg - unit_means;
                cov_matrix = cov(trial_avg_demeaned');  % cov() returns symmetric matrix
                [basis_eigenvalues, basis] = eigh_descending_sym(cov_matrix);  % no symmetrization needed
                % Note: PCA eigenvalues ARE used for ordering (default behavior), but should
                % NOT be used with criterion='variance_eigenvalues' as they don't represent
                % GSN-estimated signal variance.

            case 'random'
                % Random orthonormal basis (no meaningful eigenvalues)
                % NOTE: This resets the global RNG state for reproducibility
                % To avoid affecting other random operations, consider using a separate RandStream
                rng('default');
                rng(42);
                [basis, ~] = qr(randn(nunits));
                basis_eigenvalues = [];

            otherwise
                error('Unknown basis type: %s', basis_spec);
        end

    else
        % User-provided custom basis (no eigenvalues available)
        basis = basis_spec;

        if size(basis, 1) ~= nunits
            error('Custom basis must have %d rows (matching nunits)', nunits);
        end
        if size(basis, 2) < 1 || size(basis, 2) > nunits
            error('Custom basis must have between 1 and %d columns', nunits);
        end

        basis = normalize_orthonormalize_basis(basis);
        basis_eigenvalues = [];
    end
end
