# PSN scikit-learn API (Python)

PSN ships a scikit-learn-compatible estimator, `PSN`, that wraps the functional
`psn()` API behind the standard `fit` / `transform` / `fit_transform` interface. It
subclasses `BaseEstimator` and `TransformerMixin`, so it supports
`get_params` / `set_params` / `clone` and drops into `Pipeline` and `GridSearchCV`.

```python
from psn import PSN
```

---

## Quick start

```python
from psn import PSN, generate_data

train, test, _ = generate_data(nvox=50, ncond=200, ntrial=5, random_seed=42)

# Fit on [nunits x nconds x ntrials]; transform returns [nunits x nconds]
model = PSN()                                   # no mode -> 'standard' default
denoised = model.fit_transform(train)          # [nunits x nconds]

# Apply the learned denoiser to held-out data
denoised_test = model.transform(test)          # [nunits x nconds x ntrials] or [nunits x nconds]
```

---

## Parameters

The constructor mirrors the functional `psn()` options — see the main
[README](README.md) for their semantics. Every parameter defaults to `None` (use the
preset / library default); any value you set **overrides** the preset.

```python
PSN(mode=None, *, basis=None, criterion=None, threshold_method=None,
    basis_ordering=None, variance_threshold=None, allowable_thresholds=None,
    unit_groups=None, alpha=None, gsn_result=None, gsn_args=None, device=None,
    split_half_metric=None, cmap=None, wantverbose=True, wantfig=False)
```

- **mode** — `None` (default; same as `'standard'`), `'conservative'`, `'standard'`,
  `'aggressive'`, `'compare'`, or `'wiener'`. A preset that any non-`None` parameter
  below overrides (except `'wiener'`, which rejects conflicting options).
- **basis / criterion / threshold_method / basis_ordering / variance_threshold /
  allowable_thresholds / unit_groups / alpha** — the core PSN options (see README).
- **gsn_result / gsn_args** — reuse precomputed GSN covariances / forward GSN options.
- **device** — `'cpu'` (default), `'cuda'`, `'mps'` (GPU only when explicitly set).
- **split_half_metric / cmap** — diagnostic-figure options.
- **wantverbose** — print progress (default `True`).
- **wantfig** — draw the diagnostic figure during `fit` (default `False`; override
  per call with `fit(..., visualize=True)`).

---

## Methods

- **`fit(X, y=None, visualize=None)`** — learn the denoiser from `X` shaped
  `[nunits x nconds x ntrials]` (`y` is ignored; present for the sklearn API).
- **`transform(X)`** — apply the learned denoiser; `X` is `[nunits x nconds x ntrials]`
  (trial-averaged internally) or already `[nunits x nconds]`. Returns `[nunits x nconds]`.
- **`fit_transform(X, y=None, visualize=None)`** — fit, then return the denoised
  trial-averaged training data.
- **`plot_diagnostics(figurepath=None, cmap=None, **kwargs)`** — diagnostic figure for
  the fitted data.
- **`get_params()` / `set_params(**params)`** — standard sklearn introspection.

### Fitted attributes (trailing underscore)

`denoiseddata_`, `residuals_`, `denoiser_`, `unit_means_`, `best_threshold_`,
`fullbasis_`, `signalvar_`, `noisevar_`, `objective_`, `svnv_before_`, `svnv_after_`,
`gsn_result_`, `recovery_tradeoff_`, `opt_used_`, and (for `'compare'` / `'wiener'`)
`threshold_selection_` / `diagnostics_`.

---

## Examples

```python
# Override options directly (no preset)
model = PSN(basis='difference', criterion='prediction', threshold_method='hybrid')

# Combine a preset with an override
model = PSN(mode='aggressive', threshold_method='hybrid')

# GPU (no mode -> 'standard' default)
model = PSN(device='cuda')

# sklearn introspection / cloning
model.get_params()['criterion']
model.set_params(alpha=0.3)
from sklearn.base import clone
clone(model)
```

Because it implements the estimator protocol, `PSN` can be used inside a `Pipeline`
or swept with `GridSearchCV` over its parameters (e.g.
`{'basis': ['signal', 'difference'], 'criterion': ['max-tradeoff', 'prediction']}`).

See the notebooks in `examples/` for runnable demos.

---

## Relationship to the functional API

`PSN` is a thin wrapper: each parameter maps 1:1 to a `psn()` option, and `fit`
simply calls `psn(X, mode, opt)`. The functional API (`from psn import psn`) remains
the primary interface — see the main [README](README.md).
