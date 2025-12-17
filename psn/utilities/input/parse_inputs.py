"""Input parsing utility for PSN."""

import numpy as np


def parse_inputs(*args):
    """PARSE_INPUTS  Parse flexible input arguments to psn()

    [data, opt] = parse_inputs(*args) parses the variable input arguments
    passed to psn() and returns the data array and options dict.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <args> - Variable input arguments from psn(). Can be:
      (data,)                     - Use default 'standard' mode
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

            else:
                raise ValueError(f"Unknown mode: {second_arg}. Must be 'conservative', 'standard', or 'aggressive'")

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
        # Default: standard mode
        opt['basis'] = 'signal'
        opt['criterion'] = 'prediction'
        opt['threshold_method'] = 'hybrid'

    return data, opt


def _merge_dicts(base, override):
    """MERGE_DICTS  Merge two dicts, with override taking precedence

    merged = _merge_dicts(base, override) combines two dicts, with fields
    from <override> replacing any matching fields in <base>.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <base> - dict with default or base field values

    <override> - dict with fields that should replace those in <base>

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <merged> - dict containing all fields from <base>, with any fields
      present in <override> replaced by their <override> values
    """

    merged = base.copy()
    for key, value in override.items():
        merged[key] = value
    return merged
