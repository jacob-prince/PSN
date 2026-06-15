"""Crash-safe wrapper around visualize_results.

Figure generation is a side-effect; a rendering failure should never
lose the user's already-computed results dict.
"""

import warnings

from .visualize_results import visualize_results


def safe_visualize_results(results, opt):
    """Run visualize_results, swallowing any rendering failure.

    Figure generation is a side-effect: a failure (interrupted slow render,
    matplotlib backend issue, broken pickle, etc.) should never lose the
    already-computed results dict. On failure a warning is emitted and the call
    returns normally, unless opt['raise_on_fig_error'] is True.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <results> - dict. The fully-built PSN results, passed through to
        visualize_results for rendering.

    <opt> - dict of PSN options (forwarded to visualize_results). Honors
        opt['raise_on_fig_error'] (optional, bool): re-raise figure errors
        instead of warning. Default: False.

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    None. Called for its figure side-effect only.
    """
    try:
        visualize_results(results, opt)
    except BaseException as err:                # incl. KeyboardInterrupt
        if opt.get('raise_on_fig_error'):
            raise
        warnings.warn(
            f"PSN: visualize_results failed with "
            f"{type(err).__name__}: {err}. Results dict was already "
            f"built and is being returned; the diagnostic figure was "
            f"skipped. Set opt['raise_on_fig_error']=True to surface "
            f"the underlying error instead.")
