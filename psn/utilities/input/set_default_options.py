"""Options validation and defaults for PSN."""

import numpy as np

from .validate_options import validate_options as _validate_options


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
      criterion          = 'max-tradeoff'
      threshold_method   = 'global'
      basis_ordering     = 'eigenvalues'
      variance_threshold = 0.99
      allowable_thresholds = None
      unit_groups        = np.arange(nunits) for hybrid mode, zeros for global
      alpha              = None (disabled; interpolates between prediction and variance)
      gsn_result         = None (pass previous results['gsn_result'] to skip GSN)
      gsn_args           = {}
      wantfig            = True
      wantverbose        = True
      figurepath         = None
      cmap               = None (uses cmapsign4 in visualization)
      skip_split_half    = False (keep analytic recovery curve but skip the
                           costly empirical split-half tradeoff computation)
    """

    # Create a copy to avoid modifying the original
    opt = opt.copy()

    if 'basis' not in opt:
        opt['basis'] = 'signal'

    if 'criterion' not in opt:
        opt['criterion'] = 'max-tradeoff'

    if 'threshold_method' not in opt:
        opt['threshold_method'] = 'global'

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

    if 'gsn_result' not in opt:
        opt['gsn_result'] = None

    if 'gsn_args' not in opt or opt['gsn_args'] is None or not isinstance(opt['gsn_args'], dict):
        opt['gsn_args'] = {}

    if 'wantfig' not in opt:
        opt['wantfig'] = True

    if 'wantverbose' not in opt:
        opt['wantverbose'] = True

    if 'figurepath' not in opt:
        opt['figurepath'] = None

    if 'cmap' not in opt:
        opt['cmap'] = None  # Will use default cmapsign4 in visualization

    if 'split_half_metric' not in opt:
        opt['split_half_metric'] = 'correlation'  # 'correlation' or 'mse'

    if 'alpha' not in opt:
        opt['alpha'] = None

    if 'skip_split_half' not in opt:
        opt['skip_split_half'] = False

    # Auto-detect: if allowable_thresholds is a single value, force threshold_method to 'global'
    if opt['allowable_thresholds'] is not None:
        allowable_arr = np.asarray(opt['allowable_thresholds'])
        if allowable_arr.size == 1:            # bare scalar or length-1 vector
            if opt['threshold_method'] != 'global':
                if opt['wantverbose']:
                    print("PSN: allowable_thresholds is a single value, automatically setting threshold_method to 'global'")
                opt['threshold_method'] = 'global'
                # Update unit_groups to match global mode
                opt['unit_groups'] = np.zeros(nunits, dtype=int)

    # Note: conflicts between a Wiener request and the basis/criterion/threshold
    # pipeline are rejected up front in parse_inputs (see _reject_wiener_conflicts),
    # where the user-supplied keys are still distinguishable from filled defaults.

    _validate_options(opt, nunits)
    return opt
