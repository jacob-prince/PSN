"""Swap in cached GSN eigvecs so PSN can skip its own eigh.

When GSN was run with the eigenbasis-returns feature, the gsn_result
carries the signal / difference eigenbases. If the caller asks for the
matching string basis, we substitute the precomputed matrix and skip
PSN's O(N^3) eigh entirely.
"""

import numpy as np


def use_cached_eigvecs(opt, gsn_result):
    """Swap a string basis for cached GSN eigvecs so PSN can skip its own eigh.

    When opt['basis'] is 'signal' or 'difference' AND gsn_result carries the
    matching cached eigvecs + eigvals (GSN's eigenbasis-returns feature),
    replace the string with the precomputed eigenvector matrix and inject the
    eigenvalues, so construct_basis skips PSN's O(N^3) eigh. Quiet no-op when
    the cached arrays aren't present, when basis isn't one of those two strings,
    or when the caller already supplied basis_eigenvalues.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <opt> - dict of PSN options. Only opt['basis'] (str) and the optional
        opt['basis_eigenvalues'] / opt['wantverbose'] are consulted.

    <gsn_result> - dict of GSN outputs. The cached basis is used only when it
        contains 'eigvecs_signal'/'eigvals_signal' (for basis='signal') or
        'eigvecs_difference'/'eigvals_difference' (for basis='difference').

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <opt> - dict. A copy with opt['basis'] replaced by the [nunits x nunits]
        eigenvector matrix and opt['basis_eigenvalues'] set, when a substitution
        was made; otherwise the original opt unchanged.
    """
    basis = opt.get('basis')
    if not isinstance(basis, str):
        return opt
    if basis == 'signal':
        eigvecs_key, eigvals_key = 'eigvecs_signal', 'eigvals_signal'
    elif basis == 'difference':
        eigvecs_key, eigvals_key = 'eigvecs_difference', 'eigvals_difference'
    else:
        return opt
    if eigvecs_key not in gsn_result or eigvals_key not in gsn_result:
        return opt
    if opt.get('basis_eigenvalues') is not None:
        return opt                                    # user already set it
    new_opt = dict(opt)
    new_opt['basis'] = np.asarray(gsn_result[eigvecs_key])
    new_opt['basis_eigenvalues'] = np.asarray(gsn_result[eigvals_key])
    if new_opt.get('wantverbose'):
        print(f"PSN: using cached '{basis}' eigvecs from gsn_result "
              f"(skipping PSN's own eigh).")
    return new_opt
