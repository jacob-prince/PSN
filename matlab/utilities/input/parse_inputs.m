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
%   {data}                     - Use default 'standard' mode
%   {data, 'mode'}             - Use predefined mode ('conservative', 'standard', 'aggressive')
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
%   'conservative' - Sets basis='signal', criterion='variance', threshold_method='global'
%   'standard'     - Sets basis='signal', criterion='prediction', threshold_method='hybrid'
%   'aggressive'   - Sets basis='difference', criterion='prediction', threshold_method='hybrid'

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

                case 'standard'
                    opt.basis = 'signal';
                    opt.criterion = 'prediction';
                    opt.threshold_method = 'hybrid';

                case 'aggressive'
                    opt.basis = 'difference';
                    opt.criterion = 'prediction';
                    opt.threshold_method = 'hybrid';

                otherwise
                    error('Unknown mode: %s. Must be ''conservative'', ''standard'', or ''aggressive''', mode);
            end

            if nargin >= 3
                user_opt = varargin{3};
                if ~isstruct(user_opt)
                    error('Third argument must be an options struct');
                end
                opt = merge_structs(opt, user_opt);
            end

        elseif isstruct(second_arg)
            opt = second_arg;
        else
            error('Second argument must be a mode string or options struct');
        end
    else
        % Default: standard
        opt.basis = 'signal';
        opt.criterion = 'prediction';
        opt.threshold_method = 'hybrid';
    end
end


function merged = merge_structs(base, override)
% MERGE_STRUCTS  Merge two structs, with override taking precedence
%
%   merged = merge_structs(base, override) combines two structs, with fields
%   from <override> replacing any matching fields in <base>.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <base> - struct with default or base field values
%
% <override> - struct with fields that should replace those in <base>
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <merged> - struct containing all fields from <base>, with any fields
%   present in <override> replaced by their <override> values

    merged = base;
    fields = fieldnames(override);
    for i = 1:length(fields)
        merged.(fields{i}) = override.(fields{i});
    end
end
