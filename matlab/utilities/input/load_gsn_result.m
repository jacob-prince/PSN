function gsn_result = load_gsn_result(gsn_result)
% LOAD_GSN_RESULT  Normalize a gsn_result reference into a struct of GSN arrays.
%
%   gsn_result = load_gsn_result(gsn_result) accepts a result that is already a
%   struct, or a path to a '.mat' file, so a GSN run persisted to disk can be
%   reused across many psn() calls without re-running GSN. For a path, the
%   standard GSN fields (cSb, cNb, plus any optional eigvecs/eigvals from GSN's
%   eigenbasis-returns feature) are loaded into a struct. Mirror of the Python
%   load_gsn_result.
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
% <gsn_result> - struct of GSN outputs (expected to contain at least .cSb and
%   .cNb; the caller validates).

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
end
