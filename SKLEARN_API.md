# PSN scikit-learn API (Python)

PSN ships a scikit-learn-style estimator, `PSN`, that wraps the functional
`psn()` API behind the standard `fit` / `transform` / `fit_transform` interface. It
subclasses `BaseEstimator` and `TransformerMixin`, so it supports
`get_params` / `set_params` / `clone` and works as a single `Pipeline` step.

> **Not a drop-in for `GridSearchCV` / `cross_val_score` / `check_estimator`.**
> PSN data is `[nunits, nconds, ntrials]` with units on axis 0, but scikit-learn
> treats axis 0 as samples, so its default CV splitters would partition *units*
> rather than hold out data. PSN is also unsupervised (no `y`, no `score`). To
> cross-validate, split the conditions axis yourself; see
> [Cross-validation & model selection](#cross-validation--model-selection).

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

Because it implements the transformer protocol, `PSN` can be used as a single
step in a `Pipeline`. See the notebooks in `examples/` for runnable demos.

---

## Cross-validation & model selection

`PSN` is **not** compatible with scikit-learn's CV utilities (`GridSearchCV`,
`cross_val_score`, `check_estimator`) as-is, for two reasons:

1. **Axis convention.** PSN expects `[nunits, nconds, ntrials]` with units on
   axis 0. scikit-learn's splitters (`KFold`, etc.) split axis 0, so they would
   partition *units*, changing the population and the covariance dimensions,
   rather than holding out data.
2. **Unsupervised.** PSN has no target `y` and no `score` method, so there is
   nothing for `GridSearchCV` to optimise without a custom scorer.

To cross-validate a hyperparameter, split the **conditions** axis yourself and
score however suits your analysis:

```python
import numpy as np
from psn import PSN

n_folds = 5
rng = np.random.default_rng(0)
folds = np.array_split(rng.permutation(train.shape[1]), n_folds)   # over conditions

for k in range(n_folds):
    test_c  = folds[k]
    train_c = np.concatenate([folds[j] for j in range(n_folds) if j != k])
    model = PSN(criterion='prediction', wantverbose=False).fit(train[:, train_c, :])
    denoised_heldout = model.transform(train[:, test_c, :])        # [nunits, len(test_c)]
    # ... score denoised_heldout against a held-out trial average, etc.
```

---

## Relationship to the functional API

`PSN` is a thin wrapper: each parameter maps 1:1 to a `psn()` option, and `fit`
simply calls `psn(X, mode, opt)`. The functional API (`from psn import psn`) remains
the primary interface — see the main [README](README.md).
