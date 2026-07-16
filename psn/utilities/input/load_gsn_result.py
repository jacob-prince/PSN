"""Normalize opt['gsn_result'] into a plain dict of arrays, then validate it.

Lets callers persist a GSN run to disk once and point any number of
downstream psn() calls at the same file, without loading it themselves.

The normalized result is checked by validate_gsn_result before it is returned,
but that cannot tell whether a persisted cache describes the same population of
units as the data being denoised, so keep each cache paired with its population.
"""

import os

import numpy as np

from .validate_gsn_result import validate_gsn_result

# Standard GSN keys we pull out of an .npz / NpzFile. Includes the
# optional eigvecs / eigvals from the GSN eigenbasis-returns feature.
_GSN_FILE_KEYS = ('cSb', 'cNb', 'mnN', 'mnS', 'ncsnr',
                  'shrinklevelN', 'shrinklevelD', 'numiters',
                  'eigvecs_signal', 'eigvals_signal',
                  'eigvecs_difference', 'eigvals_difference')


def load_gsn_result(gsn_result, nunits):
    """Normalize a gsn_result reference into a plain dict and validate it.

    Accepts a result that is already a dict, or a path / open npz handle, so a
    GSN run persisted to disk can be reused across many psn() calls. For .npz
    inputs the standard GSN keys (cSb, cNb, mnN, mnS, ncsnr, shrink/iter fields,
    plus the optional eigvecs/eigvals from GSN's eigenbasis-returns feature) are
    pulled into a dict so downstream code treats every result uniformly. The
    normalized dict is then structurally checked by validate_gsn_result against
    <nunits> (shape, finiteness, symmetry, PSD-ish).

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <gsn_result> - one of:
        dict        -> used as-is (legacy in-memory result)
        str / Path  -> path to a '.npz' file (must end in .npz and exist)
        NpzFile     -> an open np.load(...) handle (not closed here)

    <nunits> - int. Unit count of the data being denoised (for validation).

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <gsn_result> - validated dict of GSN arrays (at least 'cSb' and 'cNb', both
        coerced to ndarrays).

    Raises ValueError if a string path does not end in '.npz' or validation
    fails, and FileNotFoundError if the path does not exist.
    """
    if isinstance(gsn_result, (str, bytes)) or hasattr(gsn_result, '__fspath__'):
        path = os.fspath(gsn_result)
        if not path.endswith('.npz'):
            raise ValueError(
                f"opt['gsn_result'] string path must end in .npz; got {path!r}")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"opt['gsn_result'] points to a file that doesn't exist: {path}")
        npz = np.load(path, allow_pickle=False)
        gsn_result = {k: np.asarray(npz[k]) for k in _GSN_FILE_KEYS if k in npz.files}
        npz.close()
    elif hasattr(gsn_result, 'files') and not isinstance(gsn_result, dict):
        # NpzFile (np.load result): pull what we need without closing.
        gsn_result = {k: np.asarray(gsn_result[k]) for k in _GSN_FILE_KEYS if k in gsn_result.files}
    # else: dict / dict-like, used as-is
    return validate_gsn_result(gsn_result, nunits)
