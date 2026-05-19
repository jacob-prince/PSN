"""Threshold selection utility for PSN."""

import numpy as np


def select_threshold_analytic(signal, noise, basis_eigenvalues, ntrials, opt):
    """SELECT_THRESHOLD_ANALYTIC  Choose threshold using analytic or variance criterion

    [k, objective] = select_threshold_analytic(signal, noise, basis_eigenvalues,
    ntrials, opt) determines the optimal number of dimensions to retain based
    on the specified criterion.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <signal> - [ndims] signal variance per dimension

    <noise> - [ndims] noise variance per dimension

    <basis_eigenvalues> - [ndims] eigenvalues from basis construction,
      or None if not available (e.g., for custom/random bases)

    <ntrials> - scalar, number of trials (or average if NaNs present)

    <opt> - dict with PSN options. Relevant fields:
      .criterion          - 'prediction', 'variance', or 'variance_eigenvalues'
      .variance_threshold - target fraction for variance-based criteria (default 0.99)

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <k> - scalar, selected threshold (number of dimensions to retain, 0 to ndims)

    <objective> - [ndims+1] cumulative objective curve that was actually
      used for threshold selection. Interpretation depends on criterion:
        'prediction'           - cumsum(signal - noise/ntrials)
        'variance'             - cumsum(signal)
        'variance_eigenvalues' - cumsum(max(basis_eigenvalues, 0))

    -------------------------------------------------------------------------
    Criteria:
    -------------------------------------------------------------------------

    <criterion> = 'prediction': Maximize expected out-of-sample prediction
      quality by finding k that maximizes cumsum(signal - noise/ntrials)

    <criterion> = 'variance': Retain dimensions until cumulative signal
      variance reaches variance_threshold * total_signal_variance

    <criterion> = 'variance_eigenvalues': Retain dimensions until cumulative
      positive eigenvalues reach variance_threshold * total_positive_eigenvalues.
      Only compatible with named basis types (not custom/random matrices)
    """

    ndims = len(signal)
    scaled_noise = noise / ntrials
    diff = signal - scaled_noise

    # Alpha interpolation: blend between prediction peak and variance target
    if opt.get('alpha') is not None:
        alpha = opt['alpha']
        # 1. Compute prediction peak
        pred_objective = np.concatenate([[0], np.cumsum(diff)])
        k_pred = np.argmax(pred_objective)
        # 2. Compute signal cumsum (prepend 0 for index alignment)
        sig_cumsum = np.concatenate([[0], np.cumsum(signal)])
        S_pred = sig_cumsum[k_pred]
        total_signal = sig_cumsum[-1]
        vt = np.clip(opt['variance_threshold'], 0, 1)
        S_var = vt * total_signal
        # 3. Interpolate
        target = S_pred + alpha * max(0, S_var - S_pred)
        # 4. Find threshold
        if total_signal <= 0:
            k = 0
        else:
            idx = np.where(sig_cumsum >= target)[0]
            k = idx[0] if len(idx) > 0 else ndims
            k = max(k, k_pred)  # never go below prediction peak
            k = min(k, ndims)
        objective = pred_objective  # return prediction curve for viz
        return k, objective

    if opt['criterion'] == 'prediction':
        # Maximize expected out-of-sample prediction quality
        objective = np.concatenate([[0], np.cumsum(diff)])
        k = np.argmax(objective)
        # k is already the number of dims (0-indexed argmax)

    elif opt['criterion'] == 'variance':
        # Retain fraction of signal variance
        # Return cumulative signal variance as objective
        objective = np.concatenate([[0], np.cumsum(signal)])

        vt = np.clip(opt['variance_threshold'], 0, 1)
        if vt == 0:
            k = 0
        else:
            total = objective[-1]
            if total <= 0:
                k = 0
            else:
                idx = np.where(objective >= vt * total)[0]
                if len(idx) == 0:
                    k = 0
                else:
                    k = idx[0]  # First index where threshold is reached
                k = min(k, ndims)

    elif opt['criterion'] == 'variance_eigenvalues':
        # Retain fraction of total positive eigenvalue sum
        # Only valid when eigenvalues are available (signal, difference, noise bases)
        # For PCA: use signal variance instead (PCA eigenvalues are for visualization only)

        use_pca_basis = isinstance(opt['basis'], str) and opt['basis'] == 'pca'

        if use_pca_basis:
            # PCA special case: use signal variance instead of PCA eigenvalues
            objective = np.concatenate([[0], np.cumsum(signal)])
        else:
            if basis_eigenvalues is None:
                raise ValueError(
                    'variance_eigenvalues criterion requires eigenvalues.\n'
                    'Not compatible with custom basis or random basis.')
            # Return cumulative positive eigenvalues as objective
            pos_evals = np.maximum(basis_eigenvalues, 0)
            objective = np.concatenate([[0], np.cumsum(pos_evals)])

        vt = np.clip(opt['variance_threshold'], 0, 1)
        if vt == 0:
            k = 0
        else:
            total = objective[-1]
            if total <= 0:
                k = 0
            else:
                idx = np.where(objective >= vt * total)[0]
                if len(idx) == 0:
                    k = 0
                else:
                    k = idx[0]  # First index where threshold is reached
                k = min(k, ndims)

    else:
        raise ValueError(f"Unknown criterion: {opt['criterion']}")

    return k, objective
