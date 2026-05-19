function validate_options(opt, nunits)
% VALIDATE_OPTIONS  Check that all options have valid values
%
%   validate_options(opt, nunits) validates all fields in the options struct
%   and throws errors if any values are invalid or incompatible.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <opt> - struct with PSN options (assumed to be complete after set_default_options)
%
% <nunits> - number of units in the data (for validating unit_groups size)
%
% -------------------------------------------------------------------------
% Validation checks:
% -------------------------------------------------------------------------
%
% - basis: must be 'signal', 'difference', 'noise', 'pca', 'random', or numeric matrix
% - criterion: must be 'prediction', 'variance', or 'variance_eigenvalues'
% - threshold_method: must be 'global', 'hybrid', or 'unit'
% - basis_ordering: must be 'eigenvalues', 'signalvariance', or 'prediction'
% - variance_threshold: must be in [0,1]
% - allowable_thresholds: must be numeric vector with non-negative values
% - unit_groups: must have length nunits, contain non-negative integers
% - Compatibility: 'variance_eigenvalues' requires named basis (not custom/random)
%   and only works with 'global' threshold_method

    valid_basis_strings = {'signal', 'difference', 'noise', 'pca', 'random'};
    if ischar(opt.basis) || isstring(opt.basis)
        if ~ismember(opt.basis, valid_basis_strings)
            error('basis must be one of: %s, or a matrix', strjoin(valid_basis_strings, ', '));
        end
    elseif ~isnumeric(opt.basis)
        error('basis must be a string or numeric matrix');
    end

    valid_criteria = {'prediction', 'variance', 'variance_eigenvalues'};
    if ~ismember(opt.criterion, valid_criteria)
        error('criterion must be one of: %s', strjoin(valid_criteria, ', '));
    end

    valid_methods = {'global', 'hybrid', 'unit'};
    if ~ismember(opt.threshold_method, valid_methods)
        error('threshold_method must be one of: %s', strjoin(valid_methods, ', '));
    end

    valid_orderings = {'eigenvalues', 'signalvariance', 'prediction'};
    if ~ismember(opt.basis_ordering, valid_orderings)
        error('basis_ordering must be one of: %s', strjoin(valid_orderings, ', '));
    end

    if opt.variance_threshold < 0 || opt.variance_threshold > 1
        error('variance_threshold must be between 0 and 1');
    end

    if ~isempty(opt.allowable_thresholds)
        if ~isnumeric(opt.allowable_thresholds) || ~isvector(opt.allowable_thresholds)
            error('allowable_thresholds must be a numeric vector');
        end
        if any(opt.allowable_thresholds < 0)
            error('allowable_thresholds must contain only non-negative values');
        end
        % Note: Upper bound checked later against actual basis dimensions (ndims), not nunits
    end

    if length(opt.unit_groups) ~= nunits
        error('unit_groups must have length equal to nunits (%d)', nunits);
    end
    if any(mod(opt.unit_groups, 1) ~= 0)
        error('unit_groups must contain integer values');
    end
    if any(opt.unit_groups < 0)
        error('unit_groups must contain non-negative integers (0 is allowed for global mode)');
    end

    if strcmp(opt.criterion, 'variance_eigenvalues')
        if isnumeric(opt.basis) || strcmp(opt.basis, 'random')
            error('criterion ''variance_eigenvalues'' not compatible with custom basis or ''random'' basis');
        end
        if ismember(opt.threshold_method, {'hybrid', 'unit'})
            error('criterion ''variance_eigenvalues'' only compatible with threshold_method ''global''');
        end
    end
end
