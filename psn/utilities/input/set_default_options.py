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
      basis              = 'auto'   (auto signal-vs-difference selection)
      criterion          = 'max-tradeoff'
      threshold_method   = 'hybrid'
      basis_ordering     = 'eigenvalues'
      variance_threshold = 0.99
      allowable_thresholds = None
      unit_groups        = np.arange(nunits) for hybrid/unit modes, zeros for global
      alpha              = None (disabled; interpolates between prediction and variance)
      denoiser_type      = 'truncation' ('truncation' or 'wiener')
      ntrials_eval       = None (defaults to ntrials_avg)
      gsn_result         = None (pass previous results['gsn_result'] to skip GSN)
      gsn_args           = {}
      wantfig            = True
      wantverbose        = True
      figurepath         = None
      cmap               = None (uses cmapsign4 in visualization)
    """

    # Create a copy to avoid modifying the original
    opt = opt.copy()

    if 'basis' not in opt:
        opt['basis'] = 'auto'

    if 'criterion' not in opt:
        opt['criterion'] = 'max-tradeoff'

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

    if 'denoiser_type' not in opt:
        opt['denoiser_type'] = 'truncation'  # 'truncation' (default) or 'wiener'

    if 'ntrials_eval' not in opt:
        opt['ntrials_eval'] = None  # defaults to ntrials_avg; can differ for held-out data

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

    # Auto-detect: if basis='wiener', warn about ignored options
    if isinstance(opt['basis'], str) and opt['basis'] == 'wiener':
        ignored = []
        if opt['criterion'] != 'prediction':
            ignored.append(f"criterion='{opt['criterion']}'")
        if opt['threshold_method'] != 'hybrid':
            ignored.append(f"threshold_method='{opt['threshold_method']}'")
        if opt['basis_ordering'] != 'eigenvalues':
            ignored.append(f"basis_ordering='{opt['basis_ordering']}'")
        if opt['denoiser_type'] != 'truncation':
            ignored.append(f"denoiser_type='{opt['denoiser_type']}'")
        if opt['allowable_thresholds'] is not None:
            ignored.append('allowable_thresholds')
        if ignored and opt['wantverbose']:
            print(f"PSN: basis='wiener' bypasses basis/criterion/threshold pipeline; ignoring {', '.join(ignored)}")

    # Auto-detect: if denoiser_type is 'wiener', force threshold_method to 'global'
    if opt['denoiser_type'] == 'wiener' and opt['threshold_method'] != 'global':
        if opt['wantverbose']:
            print("PSN: denoiser_type='wiener' requires threshold_method='global', setting automatically")
        opt['threshold_method'] = 'global'
        # Update unit_groups to match global mode
        opt['unit_groups'] = np.zeros(nunits, dtype=int)

    _validate_options(opt, nunits)
    return opt
