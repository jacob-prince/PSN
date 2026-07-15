function [data, opt] = parse_inputs(varargin)
% PARSE_INPUTS  Parse flexible input arguments to psn()
%
%   [data, opt] = parse_inputs(varargin) parses the variable input arguments
%   passed to psn() and returns the data array and options struct.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <varargin> - Variable input arguments from psn(). Can be:
%   {data}                     - Use defaults (same as 'standard')
%   {data, 'mode'}             - Use predefined mode
%   {data, opt}                - Use custom options struct
%   {data, 'mode', opt}        - Use predefined mode with option overrides
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <data> - [nunits x nconds x ntrials] numeric array of neural responses
%
% <opt> - struct with PSN options. For predefined modes:
%   'conservative' - basis='signal', criterion='variance', threshold_method='global', variance_threshold=0.99
%   'standard'     - basis='signal', criterion='max-tradeoff', threshold_method='global' (same as default)
%   'aggressive'   - basis='difference', criterion='prediction', threshold_method='global'
%   'compare'      - basis='compare', criterion='max-tradeoff', threshold_method='global'
%   'wiener'       - criterion='wiener' (full-rank matrix Wiener filter)
%
% For 'conservative'/'standard'/'aggressive'/'compare', any fields in the options
% struct OVERRIDE the mode defaults. 'wiener' is basis-free and untruncated, so a
% conflicting basis/criterion/threshold_method/etc. raises an error instead.

    if nargin < 1
        error('PSN requires at least one input argument (data)');
    end

    data = varargin{1};
    opt = struct();

    if nargin >= 2
        second_arg = varargin{2};

        if ischar(second_arg) || isstring(second_arg)
            mode = char(second_arg);

            switch lower(mode)
                case 'conservative'
                    opt.basis = 'signal';
                    opt.criterion = 'variance';
                    opt.threshold_method = 'global';
                    opt.variance_threshold = 0.99;

                case 'standard'
                    % Same as the default psn(data): signal basis at the
                    % max-tradeoff threshold, single population (global) threshold.
                    opt.basis = 'signal';
                    opt.criterion = 'max-tradeoff';
                    opt.threshold_method = 'global';

                case 'aggressive'
                    opt.basis = 'difference';
                    opt.criterion = 'prediction';
                    opt.threshold_method = 'global';

                case 'compare'
                    opt.basis = 'compare';
                    opt.criterion = 'max-tradeoff';
                    opt.threshold_method = 'global';

                case 'wiener'
                    % Full-rank matrix Wiener filter. Basis-free and untruncated,
                    % so reject any contradicting options outright.
                    opt.criterion = 'wiener';
                    if nargin >= 3 && isstruct(varargin{3})
                        reject_wiener_conflicts(varargin{3}, true);
                    end

                otherwise
                    error(['Unknown mode: %s. Must be ''conservative'', ''standard'', ' ...
                           '''aggressive'', ''compare'', or ''wiener'''], mode);
            end

            if nargin >= 3
                user_opt = varargin{3};
                if ~isstruct(user_opt)
                    error('Third argument must be an options struct');
                end
                % Also reject a Wiener request made through the override struct.
                reject_wiener_conflicts(user_opt, false);
                opt = merge_structs(opt, user_opt);
            end

        elseif isstruct(second_arg)
            % Direct options struct: reject a Wiener request combined with conflicts.
            reject_wiener_conflicts(second_arg, false);
            opt = second_arg;
        else
            error('Second argument must be a mode string or options struct');
        end
    end
    % No-arg / partial struct: missing fields are filled by set_default_options
    % (defaults: basis='signal', criterion='max-tradeoff', threshold_method='global').
end


function reject_wiener_conflicts(user_opt, force_wiener)
% Error if a Wiener request is combined with contradicting options. Only the
% user-supplied fields are inspected, so filled defaults never trigger a false
% conflict. criterion='wiener' / legacy basis='wiener' are themselves requests.
    if ~isstruct(user_opt)
        return;
    end
    requests_wiener = force_wiener;
    if isfield(user_opt, 'criterion') && (ischar(user_opt.criterion) || isstring(user_opt.criterion)) ...
            && strcmp(char(user_opt.criterion), 'wiener')
        requests_wiener = true;
    end
    if isfield(user_opt, 'basis') && (ischar(user_opt.basis) || isstring(user_opt.basis)) ...
            && strcmp(char(user_opt.basis), 'wiener')
        requests_wiener = true;
    end
    if ~requests_wiener
        return;
    end
    ignored = {'basis', 'basis_eigenvalues', 'criterion', 'threshold_method', 'basis_ordering', ...
               'allowable_thresholds', 'variance_threshold', 'alpha', 'unit_groups'};
    conflicts = {};
    for i = 1:numel(ignored)
        key = ignored{i};
        if ~isfield(user_opt, key)
            continue;
        end
        v = user_opt.(key);
        if strcmp(key, 'criterion') && (ischar(v) || isstring(v)) && strcmp(char(v), 'wiener')
            continue;   % consistent with Wiener
        end
        if strcmp(key, 'basis') && (ischar(v) || isstring(v)) && strcmp(char(v), 'wiener')
            continue;   % legacy alias, consistent with Wiener
        end
        conflicts{end+1} = key; %#ok<AGROW>
    end
    if ~isempty(conflicts)
        error(['The full-rank Wiener filter (mode/criterion ''wiener'') is basis-free and ' ...
               'applies no truncation, so it ignores the basis/criterion/threshold pipeline. ' ...
               'Remove these conflicting options: %s.'], strjoin(conflicts, ', '));
    end
end
