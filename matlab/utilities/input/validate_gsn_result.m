function gsn_result = validate_gsn_result(gsn_result, nunits)
% VALIDATE_GSN_RESULT  Structurally validate a cached gsn_result before PSN trusts it.
%
%   gsn_result = validate_gsn_result(gsn_result, nunits) checks a cache
%   (already normalized by load_gsn_result) against the data's unit count and
%   errors on the first problem found. The cache's covariances feed the
%   projection/threshold math and its cached eigenvectors can replace PSN's
%   eigh, so a malformed cache (wrong shape, NaN, non-symmetric, 1-D eigvecs)
%   would otherwise error cryptically or denoise silently wrong. Mirror of the
%   Python validate_gsn_result.
%
%   These checks are STRUCTURAL only (shape, finiteness, symmetry, a cheap
%   PSD-ish diagonal check). They cannot confirm the cache describes the same
%   population of units as the data being denoised - matching unit count is
%   necessary, not sufficient - so that pairing stays the caller's responsibility.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <gsn_result> - struct of GSN outputs. Must contain cSb and cNb; may also
%   carry cached eigvecs_signal/eigvals_signal and/or
%   eigvecs_difference/eigvals_difference.
%
% <nunits> - scalar. Unit count of the data being denoised.
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <gsn_result> - the same struct, unchanged when valid.

    if ~isfield(gsn_result, 'cSb') || ~isfield(gsn_result, 'cNb')
        error('gsn_result must contain cSb and cNb fields');
    end
    check_covariance(gsn_result.cSb, 'cSb', nunits);
    check_covariance(gsn_result.cNb, 'cNb', nunits);
    check_cached_eigvecs(gsn_result, nunits);
end

% Scale-relative tolerances (GSN covariances are symmetric and PSD).
%   symmetry: max|M - M'| <= 1e-6 * max|M|
%   PSD-ish:  min(diag) >= -1e-8 * max|diag|

function check_covariance(M, name, nunits)
% Validate one covariance matrix.
    if ~isnumeric(M) || ~ismatrix(M) || size(M, 1) ~= nunits || size(M, 2) ~= nunits
        error('gsn_result.%s must be a %dx%d matrix matching the data unit count; got size %s.', ...
              name, nunits, nunits, mat2str(size(M)));
    end
    if ~all(isfinite(M(:)))
        error('gsn_result.%s contains non-finite values (NaN/Inf); the cache is corrupt.', name);
    end
    scale = max(abs(M(:)));
    if scale > 0
        asym = max(max(abs(M - M.')));
        if asym > 1e-6 * scale
            error('gsn_result.%s is not symmetric (max asymmetry %.3e vs scale %.3e).', ...
                  name, asym, scale);
        end
    end
    % Cheap PSD sanity: diagonal variances can't be negative (full check is O(N^3)).
    d = diag(M);
    dscale = max(abs(d));
    if ~isempty(d) && min(d) < -1e-8 * max(1.0, dscale)
        error('gsn_result.%s has a negative diagonal value (%.3e); a covariance has non-negative variances.', ...
              name, min(d));
    end
end

function check_cached_eigvecs(gsn_result, nunits)
% Validate cached eigvecs/eigvals before the swap; construct_basis would
% silently re-orthonormalize a NaN/1-D array into a valid-shaped wrong basis.
    kinds = {'signal', 'difference'};
    for i = 1:numel(kinds)
        vkey = ['eigvecs_' kinds{i}];
        wkey = ['eigvals_' kinds{i}];
        has_v = isfield(gsn_result, vkey);
        has_w = isfield(gsn_result, wkey);
        if ~has_v && ~has_w
            continue;
        end
        if has_v ~= has_w
            present = wkey; missing = vkey;
            if has_v, present = vkey; missing = wkey; end
            error('gsn_result has %s but not %s; cached eigvecs and eigvals must be provided together.', ...
                  present, missing);
        end
        V = gsn_result.(vkey);
        w = gsn_result.(wkey);
        w = w(:);
        if ~isnumeric(V) || ~ismatrix(V) || size(V, 1) ~= nunits
            error('gsn_result.%s must be a 2-D [%d x k] matrix; got size %s.', ...
                  vkey, nunits, mat2str(size(V)));
        end
        k = size(V, 2);
        if k < 1 || k > nunits
            error('gsn_result.%s must have between 1 and %d columns; got %d.', vkey, nunits, k);
        end
        if numel(w) ~= k
            error('gsn_result.%s length (%d) must match the number of cached eigvec columns (%d).', ...
                  wkey, numel(w), k);
        end
        if ~all(isfinite(V(:))) || ~all(isfinite(w))
            error('gsn_result.%s / %s contain non-finite values (NaN/Inf); the cache is corrupt.', ...
                  vkey, wkey);
        end
    end
end
