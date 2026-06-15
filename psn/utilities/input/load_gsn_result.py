"""Normalize opt['gsn_result'] into a plain dict of arrays.

Lets callers persist a GSN run to disk once and point any number of
downstream psn() calls at the same file, without loading it themselves.
"""

import os

import numpy as np

# Standard GSN keys we pull out of an .npz / NpzFile. Includes the
# optional eigvecs / eigvals from the GSN eigenbasis-returns feature.
_GSN_FILE_KEYS = ('cSb', 'cNb', 'mnN', 'mnS', 'ncsnr',
                  'shrinklevelN', 'shrinklevelD', 'numiters',
                  'eigvecs_signal', 'eigvals_signal',
                  'eigvecs_difference', 'eigvals_difference')


def load_gsn_result(gsn_result):
    """Normalize a gsn_result reference into a plain dict of GSN arrays.

    Accepts a result that is already a dict, or a path / open npz handle, so a
    GSN run persisted to disk can be reused across many psn() calls. For .npz
    inputs the standard GSN keys (cSb, cNb, mnN, mnS, ncsnr, shrink/iter fields,
    plus the optional eigvecs/eigvals from GSN's eigenbasis-returns feature) are
    pulled into a dict so downstream code treats every result uniformly.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <gsn_result> - one of:
        dict        -> returned unchanged (legacy in-memory result)
        str / Path  -> path to a '.npz' file (must end in .npz and exist)
        NpzFile     -> an open np.load(...) handle (not closed here)

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <d> - dict mapping the present GSN keys to numpy arrays (at least 'cSb' and
        'cNb'). A path/NpzFile is reduced to this dict; a dict is passed through.

    Raises ValueError if a string path does not end in '.npz', and
    FileNotFoundError if the path does not exist.
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
        d = {}
        for k in _GSN_FILE_KEYS:
            if k in npz.files:
                d[k] = np.asarray(npz[k])
        npz.close()
        return d
    # NpzFile (np.load result): pull what we need without closing.
    if hasattr(gsn_result, 'files') and not isinstance(gsn_result, dict):
        d = {}
        for k in _GSN_FILE_KEYS:
            if k in gsn_result.files:
                d[k] = np.asarray(gsn_result[k])
        return d
    return gsn_result                                   # dict / dict-like
