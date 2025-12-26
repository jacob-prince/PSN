"""Options validation and defaults for PSN."""

import numpy as np


def set_default_options(opt, nunits):
    """SET_DEFAULT_OPTIONS  Fill in any missing options with defaults

    opt = set_default_options(opt, nunits) takes a partial options dict
    and fills in any missing fields with default values.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <opt> - partial dict with user-specified options. May be missing fields.

    <nunits> - number of units in the data (used for default unit_groups)

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <opt> - complete dict with all required fields. Defaults are:
      basis              = 'signal'
      criterion          = 'prediction'
      threshold_method   = 'hybrid'
      basis_ordering     = 'eigenvalues'
      variance_threshold = 0.99
      allowable_thresholds = None
      unit_groups        = np.arange(nunits) for hybrid/unit modes, zeros for global
      gsn_args           = {}
      wantfig            = True
      wantverbose        = True
    """

    # Create a copy to avoid modifying the original
    opt = opt.copy()

    if 'basis' not in opt:
        opt['basis'] = 'signal'

    if 'criterion' not in opt:
        opt['criterion'] = 'prediction'

    if 'threshold_method' not in opt:
        opt['threshold_method'] = 'hybrid'

    if 'basis_ordering' not in opt:
        opt['basis_ordering'] = 'eigenvalues'

    if 'variance_threshold' not in opt:
        opt['variance_threshold'] = 0.99

    if 'allowable_thresholds' not in opt:
        opt['allowable_thresholds'] = None

    if 'unit_groups' not in opt:
        if opt['threshold_method'] == 'global':
            opt['unit_groups'] = np.zeros(nunits, dtype=int)
        else:
            opt['unit_groups'] = np.arange(nunits)

    if 'gsn_args' not in opt or opt['gsn_args'] is None or not isinstance(opt['gsn_args'], dict):
        opt['gsn_args'] = {}

    if 'wantfig' not in opt:
        opt['wantfig'] = True

    if 'wantverbose' not in opt:
        opt['wantverbose'] = True

    # Auto-detect: if allowable_thresholds is a single value, force threshold_method to 'global'
    if opt['allowable_thresholds'] is not None:
        allowable_arr = np.asarray(opt['allowable_thresholds'])
        if allowable_arr.ndim == 1 and len(allowable_arr) == 1:
            if opt['threshold_method'] != 'global':
                if opt['wantverbose']:
                    print("PSN: allowable_thresholds is a single value, automatically setting threshold_method to 'global'")
                opt['threshold_method'] = 'global'
                # Update unit_groups to match global mode
                opt['unit_groups'] = np.zeros(nunits, dtype=int)

    _validate_options(opt, nunits)
    return opt


def _validate_options(opt, nunits):
    """VALIDATE_OPTIONS  Check that all options have valid values

    _validate_options(opt, nunits) validates all fields in the options dict
    and raises errors if any values are invalid or incompatible.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <opt> - dict with PSN options (assumed to be complete after set_default_options)

    <nunits> - number of units in the data (for validating unit_groups size)

    -------------------------------------------------------------------------
    Validation checks:
    -------------------------------------------------------------------------

    - basis: must be 'signal', 'difference', 'noise', 'pca', 'random', or numeric matrix
    - criterion: must be 'prediction', 'variance', or 'variance_eigenvalues'
    - threshold_method: must be 'global', 'hybrid', or 'unit'
    - basis_ordering: must be 'eigenvalues' or 'signalvariance'
    - variance_threshold: must be in [0,1]
    - allowable_thresholds: must be numeric vector with non-negative values
    - unit_groups: must have length nunits, contain non-negative integers
    - Compatibility: 'variance_eigenvalues' requires named basis (not custom/random)
      and only works with 'global' threshold_method
    """

    valid_basis_strings = ['signal', 'difference', 'noise', 'pca', 'random']
    if isinstance(opt['basis'], str):
        if opt['basis'] not in valid_basis_strings:
            raise ValueError(f"basis must be one of: {', '.join(valid_basis_strings)}, or a matrix")
    elif not isinstance(opt['basis'], np.ndarray):
        raise ValueError('basis must be a string or numeric matrix')

    valid_criteria = ['prediction', 'variance', 'variance_eigenvalues']
    if opt['criterion'] not in valid_criteria:
        raise ValueError(f"criterion must be one of: {', '.join(valid_criteria)}")

    valid_methods = ['global', 'hybrid', 'unit']
    if opt['threshold_method'] not in valid_methods:
        raise ValueError(f"threshold_method must be one of: {', '.join(valid_methods)}")

    valid_orderings = ['eigenvalues', 'signalvariance']
    if opt['basis_ordering'] not in valid_orderings:
        raise ValueError(f"basis_ordering must be one of: {', '.join(valid_orderings)}")

    if opt['variance_threshold'] < 0 or opt['variance_threshold'] > 1:
        raise ValueError('variance_threshold must be between 0 and 1')

    if opt['allowable_thresholds'] is not None:
        if not isinstance(opt['allowable_thresholds'], (list, np.ndarray)):
            raise ValueError('allowable_thresholds must be a numeric vector')
        allowable_arr = np.asarray(opt['allowable_thresholds'])
        if allowable_arr.ndim != 1:
            raise ValueError('allowable_thresholds must be a 1D vector')
        if np.any(allowable_arr < 0):
            raise ValueError('allowable_thresholds must contain only non-negative values')
        # Note: Upper bound checked later against actual basis dimensions (ndims), not nunits

    if len(opt['unit_groups']) != nunits:
        raise ValueError(f'unit_groups must have length equal to nunits ({nunits})')

    unit_groups_arr = np.asarray(opt['unit_groups'])
    if not np.all(np.equal(np.mod(unit_groups_arr, 1), 0)):
        raise ValueError('unit_groups must contain integer values')
    if np.any(unit_groups_arr < 0):
        raise ValueError('unit_groups must contain non-negative integers (0 is allowed for global mode)')

    if opt['criterion'] == 'variance_eigenvalues':
        if isinstance(opt['basis'], np.ndarray) or opt['basis'] == 'random':
            raise ValueError("criterion 'variance_eigenvalues' not compatible with custom basis or 'random' basis")
        if opt['threshold_method'] in ['hybrid', 'unit']:
            raise ValueError("criterion 'variance_eigenvalues' only compatible with threshold_method 'global'")
