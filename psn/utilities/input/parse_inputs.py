"""Input parsing utility for PSN."""

import numpy as np

from .merge_dicts import merge_dicts as _merge_dicts


# Pipeline options that the full-rank matrix Wiener filter ignores entirely (it is
# basis-free and applies no truncation). Specifying any of these alongside a Wiener
# request is contradictory and is rejected rather than silently overridden.
_WIENER_IGNORED_OPTS = ('basis', 'criterion', 'threshold_method', 'basis_ordering',
                        'allowable_thresholds', 'variance_threshold', 'alpha',
                        'unit_groups')


def _reject_wiener_conflicts(user_opt, force_wiener=False):
    """Raise ValueError if a Wiener request is combined with contradicting options.

    Only the keys the user actually supplied (in <user_opt>) are inspected, so the
    defaults filled in later never trigger a false conflict. criterion='wiener' and
    the legacy basis='wiener' are themselves Wiener requests, not conflicts.
    """
    if not isinstance(user_opt, dict):
        return
    requests_wiener = force_wiener or user_opt.get('criterion') == 'wiener' or (
        isinstance(user_opt.get('basis'), str) and user_opt['basis'] == 'wiener')
    if not requests_wiener:
        return
    conflicts = []
    for k in _WIENER_IGNORED_OPTS:
        if k not in user_opt:
            continue
        v = user_opt[k]
        if k == 'criterion' and v == 'wiener':
            continue                       # consistent with Wiener
        if k == 'basis' and isinstance(v, str) and v == 'wiener':
            continue                       # legacy alias, consistent with Wiener
        conflicts.append(f"{k}={v!r}")
    if conflicts:
        raise ValueError(
            "The full-rank Wiener filter (mode/criterion 'wiener') is basis-free and "
            "applies no truncation, so it ignores the basis/criterion/threshold "
            "pipeline. Remove these conflicting options: " + ", ".join(conflicts) + ".")


def parse_inputs(*args):
    """PARSE_INPUTS  Parse flexible input arguments to psn()

    [data, opt] = parse_inputs(*args) parses the variable input arguments
    passed to psn() and returns the data array and options dict.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <args> - Variable input arguments from psn(). Can be:
      (data,)                     - Use defaults (same as 'standard')
      (data, 'mode')              - Use predefined mode ('conservative', 'standard', 'aggressive')
      (data, opt)                 - Use custom options dict
      (data, 'mode', opt)         - Use predefined mode with option overrides

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <data> - [nunits x nconds x ntrials] numeric array of neural responses

    <opt> - dict with PSN options. For predefined modes:
      'conservative' - Sets basis='signal', criterion='variance', threshold_method='global', variance_threshold=0.99
      'standard'     - Sets basis='signal', criterion='max-tradeoff', threshold_method='global' (same as default)
      'aggressive'   - Sets basis='difference', criterion='prediction', threshold_method='global'
      'compare'      - Sets basis='compare', criterion='max-tradeoff', threshold_method='global'
                       (compares the signal-basis vs difference-basis threshold and keeps the better one)
      'wiener'       - Sets criterion='wiener' (full-rank matrix Wiener filter)

    Option-override priority:
      For 'conservative', 'standard', 'aggressive', and 'compare', any keys supplied
      in the options dict OVERRIDE the mode's defaults: the dict takes priority.
      Example: psn(data, 'aggressive', {'threshold_method': 'hybrid'}) keeps the
      'aggressive' basis/criterion but uses hybrid thresholds.

      'wiener' is the exception: the full-rank Wiener filter is basis-free and
      applies no truncation, so it ignores the whole basis/criterion/threshold
      pipeline. Supplying a contradicting option (basis, criterion, threshold_method,
      basis_ordering, allowable_thresholds, variance_threshold, alpha, or unit_groups)
      raises a ValueError instead of being silently ignored. The same check applies
      to a direct dict that requests Wiener via criterion='wiener' or basis='wiener'.
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
                opt['variance_threshold'] = 0.99

            elif mode == 'standard':
                # Same as the default psn(data): signal basis at the
                # max-tradeoff threshold, single population (global) threshold.
                opt['basis'] = 'signal'
                opt['criterion'] = 'max-tradeoff'
                opt['threshold_method'] = 'global'

            elif mode == 'aggressive':
                opt['basis'] = 'difference'
                opt['criterion'] = 'prediction'
                opt['threshold_method'] = 'global'

            elif mode == 'compare':
                # Build the signal and difference bases at their max-tradeoff
                # points and keep whichever has the higher split-half r at that threshold.
                opt['basis'] = 'compare'
                opt['criterion'] = 'max-tradeoff'
                opt['threshold_method'] = 'global'

            elif mode == 'wiener':
                # Full-rank matrix Wiener filter (optimal linear estimator).
                # Wiener is a criterion, not a basis. It ignores the basis/criterion/
                # threshold pipeline, so reject any contradicting options outright.
                opt['criterion'] = 'wiener'
                _reject_wiener_conflicts(args[2] if len(args) >= 3 else None,
                                         force_wiener=True)

            else:
                raise ValueError(f"Unknown mode: {second_arg}. Must be 'conservative', "
                                 f"'standard', 'aggressive', 'compare', or 'wiener'")

            if len(args) >= 3:
                user_opt = args[2]
                if not isinstance(user_opt, dict):
                    raise ValueError('Third argument must be an options dict')
                # Also reject a Wiener request made through the override dict itself
                # (e.g. psn(data, 'standard', {'criterion': 'wiener', ...})).
                _reject_wiener_conflicts(user_opt)
                # For the named modes, the user's options override the mode defaults.
                opt = _merge_dicts(opt, user_opt)

        elif isinstance(second_arg, dict):
            # Direct options dict: reject a Wiener request combined with conflicts.
            _reject_wiener_conflicts(second_arg)
            opt = second_arg
        else:
            raise ValueError('Second argument must be a mode string or options dict')
    else:
        # Default: leave opt empty so set_default_options supplies the defaults
        # (basis='signal', criterion='max-tradeoff', threshold_method='global').
        opt = {}

    return data, opt
