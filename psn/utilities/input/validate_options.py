"""Options validation utility for PSN."""

import numpy as np


def validate_options(opt, nunits):
    """Check that all options have valid values.

    Validates all fields in the options dict and raises errors
    if any values are invalid or incompatible.

    Parameters
    ----------
    opt : dict
        Dict with PSN options (assumed to be complete after set_default_options).
    nunits : int
        Number of units in the data (for validating unit_groups size).

    Raises
    ------
    ValueError
        If any option value is invalid or incompatible.

    Notes
    -----
    Validation checks:
    - basis: must be 'signal', 'difference', 'noise', 'pca', 'random', or numeric matrix
    - criterion: must be 'prediction', 'max-tradeoff', 'variance', or 'variance_eigenvalues'
    - threshold_method: must be 'global' or 'hybrid'
    - basis_ordering: must be 'eigenvalues', 'signalvariance', or 'prediction'
    - variance_threshold: must be in [0,1]
    - alpha: if set, must be a numeric scalar in [0,1] (else None)
    - allowable_thresholds: numeric scalar or 1D vector with non-negative values
    - unit_groups: must have length nunits, contain non-negative integers
    - Compatibility: 'variance_eigenvalues' requires named basis (not custom/random)
      and only works with 'global' threshold_method
    """
    valid_basis_strings = ['signal', 'difference', 'noise', 'pca', 'random',
                           'wiener', 'compare']
    if isinstance(opt['basis'], str):
        if opt['basis'] not in valid_basis_strings:
            raise ValueError(f"basis must be one of: {', '.join(valid_basis_strings)}, or a matrix")
    elif not isinstance(opt['basis'], np.ndarray):
        raise ValueError('basis must be a string or numeric matrix')

    valid_criteria = ['prediction', 'max-tradeoff', 'variance', 'variance_eigenvalues', 'wiener']
    if opt['criterion'] not in valid_criteria:
        raise ValueError(f"criterion must be one of: {', '.join(valid_criteria)}")

    valid_methods = ['global', 'hybrid']
    if opt['threshold_method'] not in valid_methods:
        raise ValueError(f"threshold_method must be one of: {', '.join(valid_methods)}")

    valid_orderings = ['eigenvalues', 'signalvariance', 'prediction']
    if opt['basis_ordering'] not in valid_orderings:
        raise ValueError(f"basis_ordering must be one of: {', '.join(valid_orderings)}")

    if opt['variance_threshold'] < 0 or opt['variance_threshold'] > 1:
        raise ValueError('variance_threshold must be between 0 and 1')

    if opt.get('alpha') is not None:
        alpha = opt['alpha']
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.integer, np.floating)):
            raise ValueError('alpha must be a scalar in [0, 1] or None')
        if alpha < 0 or alpha > 1:
            raise ValueError('alpha must be a scalar in [0, 1] or None')

    if opt['allowable_thresholds'] is not None:
        # Scalar or 1D vector; a scalar means "force that many dims" (parity with MATLAB).
        allowable_arr = np.asarray(opt['allowable_thresholds'])
        if not np.issubdtype(allowable_arr.dtype, np.number) or allowable_arr.ndim > 1:
            raise ValueError('allowable_thresholds must be a numeric scalar or 1D vector')
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
        if opt['threshold_method'] == 'hybrid':
            raise ValueError("criterion 'variance_eigenvalues' only compatible with threshold_method 'global'")
