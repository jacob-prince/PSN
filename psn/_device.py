"""Device-selection helpers for PSN's heavy linear-algebra utilities.

PSN's bottlenecks at large nunits (≥10000) are matmul-heavy:
  - project_covs:   diag(B.T @ cS @ B) — 2 × O(N^2·K) GEMM-equivalent
  - denoise_unitwise / denoise_global: build the (N, N) denoiser via
                    Bu @ Bu[u, :].T for each threshold group
  - denoise_wiener: full-rank Σ_S @ (Σ_S + Σ_N/t)^{-1}

All of these are pure GEMM / solve work that torch on a CUDA / MPS
device can do 10-50× faster than numpy on CPU at N=24640. This module
provides the minimal infrastructure to opt into that without
restructuring the call sites:

  device = resolve_device(opt.get('device', 'cpu'))
  cS_t = to_device(cS, device)
  ...
  result = from_device(t)

`device == 'cpu'` is a no-op: tensors are pure numpy (when the input
is numpy) and the matmul falls back to numpy. So callers can blindly
sprinkle `to_device(...)` without paying the torch import cost when
they don't actually need it.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _torch_available():
    return _HAS_TORCH


def resolve_device(device):
    """Translate an opt['device'] string into a concrete device handle.

    Accepts 'cpu', 'cuda', 'mps', 'auto', or None (treated as 'cpu').
    Returns 'cpu' (a sentinel string, NOT a torch device — so callers
    can stay numpy-only) when the host is CPU, or a torch.device(...)
    otherwise. Raises RuntimeError if a non-cpu device is requested
    but unavailable.
    """
    if device is None:
        return 'cpu'
    if device == 'cpu':
        return 'cpu'
    if not _HAS_TORCH:
        raise RuntimeError(
            f"opt['device']={device!r} requires torch. "
            f"Install with `pip install torch` or pass device='cpu'.")
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            return 'cpu'
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "opt['device']='cuda' requested but CUDA is unavailable. "
                "Use device='auto' to fall back, or 'cpu'.")
        return torch.device('cuda')
    if device == 'mps':
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError(
                "opt['device']='mps' requested but MPS is unavailable.")
        return torch.device('mps')
    # Already a torch.device or similar object — return as-is.
    return device


def is_cpu(device):
    """True when the resolved device is the numpy CPU path."""
    return device == 'cpu' or device is None


def to_device(arr, device, dtype=None):
    """Move a numpy array (or torch tensor) onto the device.

    If `is_cpu(device)`, returns a numpy array (no torch dependency).
    Otherwise returns a torch tensor on the target device.

    Default dtype is float32 (matches PSN's downstream expectations);
    pass dtype to override.
    """
    if is_cpu(device):
        a = np.asarray(arr)
        if dtype is not None and a.dtype != dtype:
            a = a.astype(dtype, copy=False)
        return a
    t = torch.as_tensor(arr)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t.to(device)


def from_device(t):
    """Move a torch tensor back to a numpy array. Pass-through for
    numpy inputs (so callers can call this unconditionally)."""
    if _HAS_TORCH and isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def empty_cache_if_cuda(device):
    """Free unreferenced CUDA allocations when running on a CUDA
    device. No-op elsewhere."""
    if _HAS_TORCH and isinstance(device, torch.device) and device.type == 'cuda':
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
