function opt = use_cached_eigvecs(opt, gsn_result)
% USE_CACHED_EIGVECS  Swap a string basis for cached GSN eigvecs to skip the eigh.
%
%   opt = use_cached_eigvecs(opt, gsn_result) is a quiet no-op unless opt.basis
%   is the string 'signal' or 'difference' AND gsn_result carries the matching
%   cached eigvecs + eigvals (from GSN's eigenbasis-returns feature) AND the
%   caller has not already set opt.basis_eigenvalues. In that case it replaces
%   opt.basis with the precomputed eigenvector matrix and sets
%   opt.basis_eigenvalues, so construct_basis skips PSN's own O(N^3) eigh while
%   producing bit-equivalent results to the string-basis path. Mirror of the
%   Python use_cached_eigvecs.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <opt> - PSN options struct. Only opt.basis (and the optional
%   opt.basis_eigenvalues / opt.wantverbose) are consulted.
%
% <gsn_result> - struct of GSN outputs. The cached basis is used only when it
%   contains eigvecs_signal/eigvals_signal (for basis='signal') or
%   eigvecs_difference/eigvals_difference (for basis='difference').
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <opt> - the options struct, with opt.basis replaced by the [nunits x nunits]
%   eigenvector matrix and opt.basis_eigenvalues set when a substitution was
%   made; otherwise unchanged.

    if ~(ischar(opt.basis) || isstring(opt.basis))
        return;                                        % custom matrix already
    end
    basis = char(opt.basis);
    if strcmp(basis, 'signal')
        eigvecs_key = 'eigvecs_signal';   eigvals_key = 'eigvals_signal';
    elseif strcmp(basis, 'difference')
        eigvecs_key = 'eigvecs_difference'; eigvals_key = 'eigvals_difference';
    else
        return;                                        % nothing to substitute
    end
    if ~isfield(gsn_result, eigvecs_key) || ~isfield(gsn_result, eigvals_key)
        return;                                        % no cached eigvecs present
    end
    if isfield(opt, 'basis_eigenvalues') && ~isempty(opt.basis_eigenvalues)
        return;                                        % user already set it
    end

    opt.basis = gsn_result.(eigvecs_key);
    opt.basis_eigenvalues = gsn_result.(eigvals_key);
    if isfield(opt, 'wantverbose') && opt.wantverbose
        fprintf('PSN: using cached ''%s'' eigvecs from gsn_result (skipping PSN''s own eigh).\n', basis);
    end
end
