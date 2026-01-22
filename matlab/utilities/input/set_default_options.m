function opt = set_default_options(opt, nunits)
% SET_DEFAULT_OPTIONS  Fill in any missing options with defaults
%
%   opt = set_default_options(opt, nunits) takes a partial options struct
%   and fills in any missing fields with default values. Also normalizes
%   string inputs to char arrays for consistent strcmp behavior.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <opt> - partial struct with user-specified options. May be missing fields.
%
% <nunits> - number of units in the data (used for default unit_groups)
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <opt> - complete struct with all required fields. Defaults are:
%   basis              = 'signal'
%   criterion          = 'prediction'
%   threshold_method   = 'hybrid'
%   basis_ordering     = 'eigenvalues'
%   variance_threshold = 0.99
%   allowable_thresholds = []
%   unit_groups        = (1:nunits)' for hybrid/unit modes, zeros for global
%   gsn_args           = struct()
%   wantfig            = 1
%   figurepath         = '' (empty = display only, no save)
%   wantverbose        = 1

    % Normalize string inputs to char for consistent strcmp behavior
    % MATLAB string type ("text") vs char array ('text') behave differently with ischar()
    if isfield(opt, 'basis') && isstring(opt.basis)
        opt.basis = char(opt.basis);
    end
    if isfield(opt, 'criterion') && isstring(opt.criterion)
        opt.criterion = char(opt.criterion);
    end
    if isfield(opt, 'threshold_method') && isstring(opt.threshold_method)
        opt.threshold_method = char(opt.threshold_method);
    end
    if isfield(opt, 'basis_ordering') && isstring(opt.basis_ordering)
        opt.basis_ordering = char(opt.basis_ordering);
    end

    if ~isfield(opt, 'basis')
        opt.basis = 'signal';
    end

    if ~isfield(opt, 'criterion')
        opt.criterion = 'prediction';
    end

    if ~isfield(opt, 'threshold_method')
        opt.threshold_method = 'hybrid';
    end

    if ~isfield(opt, 'basis_ordering')
        opt.basis_ordering = 'eigenvalues';
    end

    if ~isfield(opt, 'variance_threshold')
        opt.variance_threshold = 0.99;
    end

    if ~isfield(opt, 'allowable_thresholds')
        opt.allowable_thresholds = [];
    end

    if ~isfield(opt, 'unit_groups')
        if strcmp(opt.threshold_method, 'global')
            opt.unit_groups = zeros(nunits, 1);
        else
            opt.unit_groups = (1:nunits)';
        end
    end

    if ~isfield(opt, 'gsn_args') || isempty(opt.gsn_args) || ~isstruct(opt.gsn_args)
        opt.gsn_args = struct();
    end

    if ~isfield(opt, 'wantfig')
        opt.wantfig = 1;
    end

    if ~isfield(opt, 'wantverbose')
        opt.wantverbose = 1;
    end

    if ~isfield(opt, 'figurepath')
        opt.figurepath = '';
    end

    % Auto-detect: if allowable_thresholds is a single value, force threshold_method to 'global'
    if ~isempty(opt.allowable_thresholds)
        if isscalar(opt.allowable_thresholds)
            if ~strcmp(opt.threshold_method, 'global')
                if opt.wantverbose
                    fprintf('PSN: allowable_thresholds is a single value, automatically setting threshold_method to ''global''\n');
                end
                opt.threshold_method = 'global';
                % Update unit_groups to match global mode
                opt.unit_groups = zeros(nunits, 1);
            end
        end
    end

    validate_options(opt, nunits);
end
