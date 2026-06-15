function results = denoise_fullrank_wiener(cSb, cNb, data, trial_avg, unit_means, ntrials_avg, nunits, gsn_result, opt)
% DENOISE_FULLRANK_WIENER  Full-rank matrix Wiener filter: D = Sigma_S (Sigma_S + Sigma_N/t)^{-1}.
%
%   results = denoise_fullrank_wiener(cSb, cNb, data, trial_avg, unit_means, ...
%   ntrials_avg, nunits, gsn_result, opt) applies the Bayes-optimal linear
%   estimator. It bypasses basis construction, ordering, criterion and
%   thresholding (no truncation; all dimensions continuously weighted). Mirrors
%   the Python denoise_fullrank_wiener and returns a complete results struct.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <cSb> - [nunits x nunits] signal covariance matrix from GSN.
%
% <cNb> - [nunits x nunits] noise covariance matrix from GSN.
%
% <data> - [nunits x nconds x ntrials] measured data (used to form residuals).
%
% <trial_avg> - [nunits x nconds] trial-averaged responses (the filter input).
%
% <unit_means> - [nunits x 1] per-unit mean, subtracted before and added back
%   after filtering.
%
% <ntrials_avg> - scalar, number of trials (or average if NaNs present), the t
%   in Sigma_N/t.
%
% <nunits> - scalar, number of units (used for the diagonal jitter scaling).
%
% <gsn_result> - struct, the raw GSN output, passed through onto results.
%
% <opt> - struct of PSN options, stored on results.opt_used.
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <results> - struct with the standard PSN fields, including:
%   .denoiseddata    - [nunits x nconds] Wiener-filtered responses
%   .residuals       - [nunits x nconds x ntrials] data minus denoiseddata
%   .denoiser        - [nunits x nunits] filter stored so denoiser' = D
%   .wiener_matrix   - [nunits x nunits] the filter matrix D = cSb * inv(A)
%   .svnv_before     - [nunits x 2] per-unit signal/noise variance before
%   .svnv_after      - [nunits x 2] per-unit signal/noise variance after
%   .best_threshold  - scalar, effective dimensions = trace(D)
%   .fullbasis,.basis_viz - [nunits x nunits] cSb eigenvectors (for the figure)
%   .basis_eigenvalues,.basis_eigenvalues_viz - [nunits x 1] cSb eigenvalues
%   .signalvar,.noisevar,.signal_proj_viz,.noise_proj_viz - [nunits x 1] per-dim
%                      signal/noise projections onto the cSb eigenvectors
%   .unit_means,.input_data,.gsn_result,.opt_used - passthrough/bookkeeping
%   .objective,.signalsubspace,.dimreduce - empty (not used in this mode)

    ntrials = size(data, 3);

    % D = Sigma_S (Sigma_S + Sigma_N/t)^{-1}. Store denoiser so that
    % denoiser' * x = D * x, i.e. denoiser = inv(A) * Sigma_S and denoiser' = D.
    A = cSb + cNb / ntrials_avg;
    jitter = 1e-10 * trace(A) / nunits;
    A = A + jitter * eye(nunits);
    denoiser = A \ cSb;        % inv(A) * cSb
    D = denoiser';             % the actual filter matrix (cSb * inv(A))

    % Apply
    denoiseddata = denoiser' * (trial_avg - unit_means) + unit_means;
    residuals = data - repmat(denoiseddata, [1, 1, ntrials]);

    % Diagnostics: signal/noise variance per unit before and after
    svnv_before = [diag(cSb), diag(cNb) / ntrials_avg];
    D_cSb_Dt = D * cSb * D';
    D_cNb_Dt = D * cNb * D';
    svnv_after = [diag(D_cSb_Dt), diag(D_cNb_Dt) / ntrials_avg];

    effective_dims = trace(D);

    % cSb eigenvectors for visualization (same as the signal basis, using the
    % standardized sign convention so it matches construct_basis and Python).
    [evals_cSb, eigvecs_cSb] = eigh_descending_sym(cSb, false);
    signal_proj = sum((cSb * eigvecs_cSb) .* eigvecs_cSb, 1)';
    noise_proj = sum((cNb * eigvecs_cSb) .* eigvecs_cSb, 1)';

    results = struct();
    results.denoiseddata = denoiseddata;
    results.residuals = residuals;
    results.unit_means = unit_means;
    results.denoiser = denoiser;
    results.svnv_before = svnv_before;
    results.svnv_after = svnv_after;
    results.best_threshold = effective_dims;
    results.fullbasis = eigvecs_cSb;
    results.basis_eigenvalues = evals_cSb;
    results.gsn_result = gsn_result;
    results.signalvar = signal_proj;
    results.noisevar = noise_proj;
    results.objective = [];
    results.basis_viz = eigvecs_cSb;
    results.signal_proj_viz = signal_proj;
    results.noise_proj_viz = noise_proj;
    results.basis_eigenvalues_viz = evals_cSb;
    results.input_data = data;
    results.signalsubspace = [];
    results.dimreduce = [];
    results.wiener_matrix = D;
    results.opt_used = opt;
end
