"""Input parsing utility for PSN."""

import numpy as np

from .merge_dicts import merge_dicts as _merge_dicts


def parse_inputs(*args):
    """PARSE_INPUTS  Parse flexible input arguments to psn()

    [data, opt] = parse_inputs(*args) parses the variable input arguments
    passed to psn() and returns the data array and options dict.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <args> - Variable input arguments from psn(). Can be:
      (data,)                     - Use default 'auto' mode
      (data, 'mode')              - Use predefined mode ('conservative', 'standard', 'aggressive')
      (data, opt)                 - Use custom options dict
      (data, 'mode', opt)         - Use predefined mode with option overrides

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <data> - [nunits x nconds x ntrials] numeric array of neural responses

    <opt> - dict with PSN options. For predefined modes:
      'conservative' - Sets basis='signal', criterion='variance', threshold_method='global'
      'standard'     - Sets basis='signal', criterion='prediction', threshold_method='hybrid'
      'aggressive'   - Sets basis='difference', criterion='prediction', threshold_method='hybrid'
      'auto'         - Sets basis='auto', criterion='max-tradeoff', threshold_method='hybrid'
                       (auto-selects the better of the signal-basis vs difference-basis operating point)
      'wiener'       - Sets basis='wiener' (full-rank matrix Wiener filter)
    """

    if len(args) < 1:
        raise ValueError('PSN requires at least one input argument (data)')

    data = args[0]
    opt = {}

    if len(args) >= 2:
        second_arg = args[1]

        if isinstance(second_arg, str):
            mode = second_arg.lower()

            if mode == 'conservative':
                opt['basis'] = 'signal'
                opt['criterion'] = 'variance'
                opt['threshold_method'] = 'global'

            elif mode == 'standard':
                opt['basis'] = 'signal'
                opt['criterion'] = 'prediction'
                opt['threshold_method'] = 'hybrid'

            elif mode == 'aggressive':
                opt['basis'] = 'difference'
                opt['criterion'] = 'prediction'
                opt['threshold_method'] = 'hybrid'

            elif mode == 'auto':
                # Auto-select the better of the signal-basis and difference-basis
                # max-tradeoff point (by split-half reliability). Near-unbiased,
                # fully analytic. Wiener is available separately via 'wiener'.
                opt['basis'] = 'auto'
                opt['criterion'] = 'max-tradeoff'
                opt['threshold_method'] = 'hybrid'

            elif mode == 'wiener':
                # Full-rank matrix Wiener filter (Bayes-optimal linear estimator).
                opt['basis'] = 'wiener'

            else:
                raise ValueError(f"Unknown mode: {second_arg}. Must be 'conservative', 'standard', 'aggressive', 'auto', or 'wiener'")

            if len(args) >= 3:
                user_opt = args[2]
                if not isinstance(user_opt, dict):
                    raise ValueError('Third argument must be an options dict')
                opt = _merge_dicts(opt, user_opt)

        elif isinstance(second_arg, dict):
            opt = second_arg
        else:
            raise ValueError('Second argument must be a mode string or options dict')
    else:
        # Default: 'auto' mode. Leave opt empty so set_default_options supplies
        # the defaults (basis='auto', criterion='max-tradeoff', threshold_method='hybrid').
        opt = {}

    return data, opt
