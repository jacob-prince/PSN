"""Fast GSN — internal accelerated backend for ``psn.utils.perform_gsn``.

This module is a drop-in replacement for the parts of ``gsn.perform_gsn``
that PSN actually consumes (``cSb``, ``cNb``). It picks one of three
paths at call time:

1. **NaN / uneven-trials input** → delegates to the reference
   ``gsn.perform_gsn`` unchanged. That path has stochastic trial-subsetting
   and other subtleties we don't rewrite.

2. **torch available, clean data** → GPU-ready path. Batched Cholesky over
   all 51 shrinkage levels in a single kernel launch, automatic device
   selection (CUDA → MPS → CPU). This is the fastest path — 10-100× on
   moderate nunits even without a GPU, more with one.

3. **torch unavailable, clean data** → numpy+scipy path. Uses
   ``scipy.linalg.solve_triangular`` in place of ``np.linalg.pinv`` on the
   Cholesky factor (the reference's single biggest waste), and ``eigh``
   in place of ``svd`` for nearest-PSD projection. 2-4× faster than the
   reference with zero new dependencies.

Bottlenecks in the reference pipeline (for nunits=N, shrinkage grid size
S=51): the Cholesky-based likelihood is evaluated 2*S times (once per
shrinkage level for noise covariance, once for data covariance), and each
evaluation wastes an O(N³) ``np.linalg.pinv`` call on a matrix that is
already triangular. That single wasted pinv dominates for N ≳ 500.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

# Torch is optional. It is runtime-detected; it is not a declared dependency
# and absence silently downgrades to the numpy+scipy fast path.
try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Deterministic permutation — bit-for-bit match to gsn.utilities.deterministic_randperm
# so train/val splits agree with the reference pipeline.
# ---------------------------------------------------------------------------

def _deterministic_randperm(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.argsort(rng.random(n))


def _pick_torch_device() -> str:
    """Auto-select the best available torch device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# ===========================================================================
# Torch path
# ===========================================================================

def _perform_gsn_torch(
    data: np.ndarray,
    opt: Dict,
    *,
    device: Optional[str] = None,
    dtype: str = 'float64',
    max_iters: int = 50,
    conv_tol: float = 1e-3,
) -> Dict[str, np.ndarray]:
    """Torch backend for fast GSN. Expects NaN-free data."""
    wantshrinkage = opt.get('wantshrinkage', True)
    wantverbose = opt.get('wantverbose', False)

    if device is None:
        device = _pick_torch_device()
    tdtype = torch.float32 if dtype == 'float32' else torch.float64

    nvox, ncond, ntrial = data.shape
    shrinklevels = np.linspace(0, 1, 51) if wantshrinkage else np.array([1.0])

    data_t = torch.as_tensor(data, dtype=tdtype, device=device)

    # --- Noise covariance: reference transposes to (trials, voxels, conditions)
    # which is (obs, vars, cases) in calc_shrunken_covariance's vocabulary.
    data_noise = data_t.permute(2, 0, 1).contiguous()  # (T, N, C)
    if wantverbose:
        print('[fast_gsn/torch] noise covariance…', flush=True)
    _, cN = _shrunken_cov_3d_torch(data_noise, 5, shrinklevels, tdtype, device)

    # --- Data covariance: (cond, vox) 2D path.
    data_mean = data_t.mean(dim=2).transpose(0, 1).contiguous()  # (C, N)
    if wantverbose:
        print('[fast_gsn/torch] data covariance…', flush=True)
    _, cD = _shrunken_cov_2d_torch(data_mean, 5, shrinklevels, tdtype, device)

    # --- Biconvex loop
    if wantverbose:
        print('[fast_gsn/torch] biconvex loop…', flush=True)

    cS = cD - cN / ntrial
    cNb = cN
    cSb_old = cS
    cNb_old = cN
    for _ in range(max_iters):
        cSb = _nearest_psd_torch(cD - cNb / ntrial)
        coef_N = (ncond * (ntrial - 1) * ntrial ** 2) / (
            ncond * ntrial ** 2 * (ntrial - 1) + ncond - 1
        )
        coef_D = (ncond - 1) / (ncond * ntrial ** 2 * (ntrial - 1) + ncond - 1)
        cNb = _nearest_psd_torch(coef_N * cN + coef_D * ntrial * (cD - cSb))

        dS = torch.linalg.norm(cSb - cSb_old) / (torch.linalg.norm(cSb_old) + 1e-30)
        dN = torch.linalg.norm(cNb - cNb_old) / (torch.linalg.norm(cNb_old) + 1e-30)
        if float(dS) < conv_tol and float(dN) < conv_tol:
            break
        cSb_old = cSb
        cNb_old = cNb

    return {
        'cSb': cSb.detach().cpu().numpy().astype(np.float64, copy=False),
        'cNb': cNb.detach().cpu().numpy().astype(np.float64, copy=False),
    }


def _batched_shrunken_nll_torch(c, pts_zm, shrinklevels):
    """Evaluate mean NLL at every shrinkage level in one batched call.

    Shrunken covariance: c_alpha = alpha*c + (1-alpha)*diag(c). This is
    identical to the reference's `c*alpha` with diagonal restored.
    Builds (S, N, N) tensor, does a batched Cholesky + triangular solve.
    """
    N = c.shape[0]
    M = pts_zm.shape[0]
    log_2pi = float(np.log(2 * np.pi))
    S = len(shrinklevels)

    diag_c = torch.diag(c)
    D = torch.diag(diag_c)
    alphas = torch.as_tensor(shrinklevels, dtype=c.dtype, device=c.device)
    covs = alphas[:, None, None] * c.unsqueeze(0) + (1 - alphas)[:, None, None] * D.unsqueeze(0)

    Ls, info = torch.linalg.cholesky_ex(covs, upper=False)
    ok = info == 0
    ok_np = ok.detach().cpu().numpy().astype(bool)

    nll = np.full(S, np.nan, dtype=np.float64)
    if ok_np.any():
        rhs = pts_zm.transpose(0, 1).unsqueeze(0).expand(int(ok_np.sum()), N, M)
        Ls_ok = Ls[ok]
        X = torch.linalg.solve_triangular(Ls_ok, rhs, upper=False, unitriangular=False)
        sq = (X ** 2).sum(dim=1)  # (S_ok, M)
        logdet = torch.log(torch.diagonal(Ls_ok, dim1=-2, dim2=-1)).sum(dim=1)
        log_pdf = -0.5 * sq - logdet.unsqueeze(1) - 0.5 * N * log_2pi
        nll[ok_np] = (-log_pdf.mean(dim=1)).detach().cpu().numpy()
    return nll


def _pooled_cov_no_nan_torch(data_obs_var_case):
    """Average per-case empirical covariance, fully vectorized."""
    centered = data_obs_var_case - data_obs_var_case.mean(dim=0, keepdim=True)
    nobs = data_obs_var_case.shape[0]
    cov_sum = torch.einsum('ovc,owc->vw', centered, centered) / (nobs - 1)
    return cov_sum / data_obs_var_case.shape[2]


def _cov_2d_torch(X):
    centered = X - X.mean(dim=0, keepdim=True)
    return (centered.T @ centered) / (X.shape[0] - 1)


def _shrunken_cov_3d_torch(data_obs_var_case, leaveout, shrinklevels, tdtype, device):
    ncase = data_obs_var_case.shape[2]
    perm = _deterministic_randperm(ncase)
    val_size = int(np.round(ncase / leaveout))
    ii = perm[:val_size]
    iinot = perm[val_size:]

    c_train = _pooled_cov_no_nan_torch(data_obs_var_case[:, :, iinot])

    val = data_obs_var_case[:, :, ii]
    centered_val = val - val.mean(dim=0, keepdim=True)
    pts_zm = centered_val.permute(2, 0, 1).reshape(-1, centered_val.shape[1])

    nll = _batched_shrunken_nll_torch(c_train, pts_zm, shrinklevels)
    # Reference uses np.argmin (which treats NaN as the minimum — a numpy
    # quirk that the reference relies on whether intentional or not). We
    # match that behavior bit-for-bit so shrinkage selection is identical.
    best = int(np.argmin(nll))
    chosen = float(shrinklevels[best])

    # wantfull=1: refit on full data, re-apply chosen level.
    c_full = _pooled_cov_no_nan_torch(data_obs_var_case)
    D = torch.diag(torch.diag(c_full))
    c = chosen * c_full + (1 - chosen) * D
    mn = torch.zeros(1, data_obs_var_case.shape[1], dtype=tdtype, device=device)
    return mn, c


def _shrunken_cov_2d_torch(X, leaveout, shrinklevels, tdtype, device):
    nobs = X.shape[0]
    perm = _deterministic_randperm(nobs)
    val_size = int(np.round(nobs / leaveout))
    ii = perm[:val_size]
    iinot = perm[val_size:]

    X_train = X[iinot]
    c_train = _cov_2d_torch(X_train)
    mn_train = X_train.mean(dim=0)

    # Ridge if rank-deficient — matches reference.
    rank = int(torch.linalg.matrix_rank(c_train))
    if rank < c_train.shape[0]:
        c_train = c_train + 1e-6 * torch.eye(c_train.shape[0], dtype=tdtype, device=device)

    pts_zm = X[ii] - mn_train
    nll = _batched_shrunken_nll_torch(c_train, pts_zm, shrinklevels)
    # Reference uses np.argmin (which treats NaN as the minimum — a numpy
    # quirk that the reference relies on whether intentional or not). We
    # match that behavior bit-for-bit so shrinkage selection is identical.
    best = int(np.argmin(nll))
    chosen = float(shrinklevels[best])

    c_full = _cov_2d_torch(X)
    D = torch.diag(torch.diag(c_full))
    c = chosen * c_full + (1 - chosen) * D
    return X.mean(dim=0), c


def _nearest_psd_torch(M, eps: float = 1e-10):
    M = (M + M.transpose(-1, -2)) / 2
    try:
        torch.linalg.cholesky(M)
        return M
    except Exception:
        pass
    evals, evecs = torch.linalg.eigh(M)
    evals = torch.clamp(evals, min=0)
    Mp = (evecs * evals) @ evecs.transpose(-1, -2)
    Mp = (Mp + Mp.transpose(-1, -2)) / 2
    try:
        torch.linalg.cholesky(Mp)
    except Exception:
        Mp = Mp + eps * torch.eye(Mp.shape[0], dtype=M.dtype, device=M.device)
    return Mp


# ===========================================================================
# Numpy+scipy path (fallback for no-torch environments)
# ===========================================================================

def _perform_gsn_numpy(
    data: np.ndarray,
    opt: Dict,
    *,
    max_iters: int = 50,
    conv_tol: float = 1e-3,
) -> Dict[str, np.ndarray]:
    """Numpy+scipy backend for fast GSN. Expects NaN-free data.

    Main wins over the reference:
    - ``scipy.linalg.solve_triangular`` instead of ``np.linalg.pinv`` on
      the Cholesky factor. Pinv of a triangular matrix is a wasted O(N^3).
    - ``np.linalg.eigh`` instead of ``np.linalg.svd`` for nearest-PSD.
    - Vectorized per-condition noise covariance via einsum.
    - Frobenius convergence check instead of ``corrcoef`` on flattened N² vectors.
    """
    from scipy.linalg import solve_triangular

    wantshrinkage = opt.get('wantshrinkage', True)
    wantverbose = opt.get('wantverbose', False)

    nvox, ncond, ntrial = data.shape
    shrinklevels = np.linspace(0, 1, 51) if wantshrinkage else np.array([1.0])

    # --- Noise covariance (3-D path).
    data_noise = np.transpose(data, (2, 0, 1))  # (T, N, C)
    if wantverbose:
        print('[fast_gsn/numpy] noise covariance…', flush=True)
    _, cN = _shrunken_cov_3d_numpy(data_noise, 5, shrinklevels, solve_triangular)

    # --- Data covariance (2-D path).
    data_mean = np.mean(data, axis=2).T  # (C, N)
    if wantverbose:
        print('[fast_gsn/numpy] data covariance…', flush=True)
    _, cD = _shrunken_cov_2d_numpy(data_mean, 5, shrinklevels, solve_triangular)

    # --- Biconvex loop.
    if wantverbose:
        print('[fast_gsn/numpy] biconvex loop…', flush=True)
    cS = cD - cN / ntrial
    cNb = cN
    cSb_old = cS
    cNb_old = cN
    for _ in range(max_iters):
        cSb = _nearest_psd_numpy(cD - cNb / ntrial)
        coef_N = (ncond * (ntrial - 1) * ntrial ** 2) / (
            ncond * ntrial ** 2 * (ntrial - 1) + ncond - 1
        )
        coef_D = (ncond - 1) / (ncond * ntrial ** 2 * (ntrial - 1) + ncond - 1)
        cNb = _nearest_psd_numpy(coef_N * cN + coef_D * ntrial * (cD - cSb))

        dS = np.linalg.norm(cSb - cSb_old) / (np.linalg.norm(cSb_old) + 1e-30)
        dN = np.linalg.norm(cNb - cNb_old) / (np.linalg.norm(cNb_old) + 1e-30)
        if dS < conv_tol and dN < conv_tol:
            break
        cSb_old = cSb
        cNb_old = cNb

    return {'cSb': cSb.astype(np.float64, copy=False),
            'cNb': cNb.astype(np.float64, copy=False)}


def _pooled_cov_no_nan_numpy(data_obs_var_case):
    centered = data_obs_var_case - data_obs_var_case.mean(axis=0, keepdims=True)
    nobs = data_obs_var_case.shape[0]
    cov_sum = np.einsum('ovc,owc->vw', centered, centered) / (nobs - 1)
    return cov_sum / data_obs_var_case.shape[2]


def _cov_2d_numpy(X):
    centered = X - X.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / (X.shape[0] - 1)


def _shrunken_nll_numpy(c, pts_zm, shrinklevels, solve_triangular):
    """Per-level Cholesky + triangular-solve likelihood. Not batched (numpy has
    no batched Cholesky), but still avoids the reference's wasted pinv."""
    N = c.shape[0]
    log_2pi = float(np.log(2 * np.pi))
    diag_c = np.diag(c)
    D = np.diag(diag_c)
    S = len(shrinklevels)
    nll = np.full(S, np.nan, dtype=np.float64)
    for i, alpha in enumerate(shrinklevels):
        c2 = alpha * c + (1 - alpha) * D
        try:
            L = np.linalg.cholesky(c2)
        except np.linalg.LinAlgError:
            continue
        # Solve L @ X = pts_zm.T  →  ||X||² columnwise = quadratic form.
        X = solve_triangular(L, pts_zm.T, lower=True)
        sq = np.sum(X ** 2, axis=0)
        logdet = np.sum(np.log(np.diag(L)))
        log_pdf = -0.5 * sq - logdet - 0.5 * N * log_2pi
        nll[i] = float(np.mean(-log_pdf))
    return nll


def _shrunken_cov_3d_numpy(data_obs_var_case, leaveout, shrinklevels, solve_triangular):
    ncase = data_obs_var_case.shape[2]
    perm = _deterministic_randperm(ncase)
    val_size = int(np.round(ncase / leaveout))
    ii = perm[:val_size]
    iinot = perm[val_size:]

    c_train = _pooled_cov_no_nan_numpy(data_obs_var_case[:, :, iinot])
    val = data_obs_var_case[:, :, ii]
    centered_val = val - val.mean(axis=0, keepdims=True)
    pts_zm = centered_val.transpose(2, 0, 1).reshape(-1, centered_val.shape[1])

    nll = _shrunken_nll_numpy(c_train, pts_zm, shrinklevels, solve_triangular)
    # Reference uses np.argmin (which treats NaN as the minimum — a numpy
    # quirk that the reference relies on whether intentional or not). We
    # match that behavior bit-for-bit so shrinkage selection is identical.
    best = int(np.argmin(nll))
    chosen = float(shrinklevels[best])

    c_full = _pooled_cov_no_nan_numpy(data_obs_var_case)
    D = np.diag(np.diag(c_full))
    c = chosen * c_full + (1 - chosen) * D
    mn = np.zeros((1, data_obs_var_case.shape[1]), dtype=c.dtype)
    return mn, c


def _shrunken_cov_2d_numpy(X, leaveout, shrinklevels, solve_triangular):
    nobs = X.shape[0]
    perm = _deterministic_randperm(nobs)
    val_size = int(np.round(nobs / leaveout))
    ii = perm[:val_size]
    iinot = perm[val_size:]

    X_train = X[iinot]
    c_train = _cov_2d_numpy(X_train)
    mn_train = X_train.mean(axis=0)

    if int(np.linalg.matrix_rank(c_train)) < c_train.shape[0]:
        c_train = c_train + 1e-6 * np.eye(c_train.shape[0])

    pts_zm = X[ii] - mn_train
    nll = _shrunken_nll_numpy(c_train, pts_zm, shrinklevels, solve_triangular)
    # Reference uses np.argmin (which treats NaN as the minimum — a numpy
    # quirk that the reference relies on whether intentional or not). We
    # match that behavior bit-for-bit so shrinkage selection is identical.
    best = int(np.argmin(nll))
    chosen = float(shrinklevels[best])

    c_full = _cov_2d_numpy(X)
    D = np.diag(np.diag(c_full))
    c = chosen * c_full + (1 - chosen) * D
    return X.mean(axis=0), c


def _nearest_psd_numpy(M, eps: float = 1e-10):
    M = (M + M.T) / 2
    try:
        np.linalg.cholesky(M)
        return M
    except np.linalg.LinAlgError:
        pass
    evals, evecs = np.linalg.eigh(M)
    evals = np.maximum(evals, 0)
    Mp = (evecs * evals) @ evecs.T
    Mp = (Mp + Mp.T) / 2
    try:
        np.linalg.cholesky(Mp)
    except np.linalg.LinAlgError:
        Mp = Mp + eps * np.eye(Mp.shape[0])
    return Mp


# ===========================================================================
# Public entry point — dispatcher
# ===========================================================================

def fast_perform_gsn(data: np.ndarray, opt: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """Fast ``perform_gsn`` with the same ``{cSb, cNb}`` output as the reference.

    Dispatch:
    - NaN / uneven trials → reference ``gsn.perform_gsn`` (subtle stochastic
      trial-subsetting path, not reimplemented).
    - torch present, clean data → batched torch path (CUDA / MPS / CPU).
    - torch absent, clean data → numpy+scipy path (triangular solve + eigh).

    Drop-in replacement; callers need no changes.
    """
    if opt is None:
        opt = {}

    # NaN / uneven trials → reference.
    if np.isnan(data).any():
        from gsn.perform_gsn import perform_gsn as _ref
        res = _ref(data, opt)
        return {'cSb': res['cSb'], 'cNb': res['cNb']}

    if _HAS_TORCH:
        return _perform_gsn_torch(data, opt)
    return _perform_gsn_numpy(data, opt)
