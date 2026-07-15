function gsn_result = load_gsn_result(gsn_result, nunits)
% LOAD_GSN_RESULT  Normalize a gsn_result reference into a struct and validate it.
%
%   gsn_result = load_gsn_result(gsn_result, nunits) accepts a result that is
%   already a struct, or a path to a '.mat' file, so a GSN run persisted to disk
%   can be reused across many psn() calls without re-running GSN. For a path, the
%   standard GSN fields (cSb, cNb, plus any optional eigvecs/eigvals from GSN's
%   eigenbasis-returns feature) are loaded into a struct. The normalized struct
%   is then structurally checked by validate_gsn_result against <nunits>. Mirror
%   of the Python load_gsn_result.
%
%   Validation cannot tell whether a persisted cache describes the same
%   population of units as the data being denoised, so keep each cache paired
%   with its population.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <gsn_result> - one of:
%     struct        -> returned unchanged (in-memory result, e.g. a prior
%                      results.gsn_result)
%     char / string -> path to a '.mat' file (must end in .mat and exist). The
%                      file may store the fields directly (save -struct) or wrap
%                      them in a single 'gsn_result' variable.
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <gsn_result> - validated struct of GSN outputs (contains at least .cSb and
%   .cNb).

    if ischar(gsn_result) || isstring(gsn_result)
        path = char(gsn_result);
        [~, ~, ext] = fileparts(path);
        if ~strcmpi(ext, '.mat')
            error('opt.gsn_result path must end in .mat; got %s', path);
        end
        if exist(path, 'file') ~= 2
            error('opt.gsn_result points to a file that does not exist: %s', path);
        end
        loaded = load(path);
        if isfield(loaded, 'cSb') && isfield(loaded, 'cNb')
            gsn_result = loaded;                       % fields stored directly
        elseif isfield(loaded, 'gsn_result') && isstruct(loaded.gsn_result)
            gsn_result = loaded.gsn_result;            % wrapped in one variable
        else
            gsn_result = loaded;                       % let caller catch missing cSb/cNb
        end
    elseif ~isstruct(gsn_result)
        error('opt.gsn_result must be a struct or a path to a .mat file');
    end
    gsn_result = validate_gsn_result(gsn_result, nunits);
end
