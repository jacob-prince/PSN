# PSN: Partitioning Signal and Noise

**PSN** is a library for denoising neural data by adaptively partitioning signal and noise.

> **Note:** As of June 2026, PSN is under active development. The API and algorithm are subject to change.

## What is PSN?

PSN (Partitioning Signal and Noise) is a method for denoising multi-trial neural data by:
1. Estimating signal and noise covariances with GSN
2. Projecting data into a basis (e.g., eigenvectors of the signal covariance)
3. Selecting how many dimensions to retain (a single population threshold, or per-unit)
4. Reconstructing denoised estimates that recover the underlying signal while suppressing noise

PSN builds on [Generative Modeling of Signal and Noise (GSN)](https://github.com/cvnlab/GSN).

---

## Quick Start

### Python

#### Installation

Optionally create a conda environment:
```bash
conda create -n psn python=3.9
conda activate psn
```

Clone and install:
```bash
git clone https://github.com/jacob-prince/PSN.git
cd PSN
pip install -e .
```

#### Basic Usage

PSN exposes a single functional entry point, `psn(data, opt)`, where `data` has shape
`(n_units, n_conditions, n_trials)`. The simulation utility `generate_data` is included
for quickly producing realistic synthetic data to experiment with.

```python
import numpy as np
from psn import psn, generate_data

# Generate synthetic data with known signal and noise structure
train_data, test_data, gt = generate_data(
    nvox=50, ncond=200, ntrial=5, random_seed=42
)
# train_data shape: (50, 200, 5) -> (n_units, n_conditions, n_trials)

# Apply PSN denoising with default settings ('standard' preset)
results = psn(train_data)

# Access denoised data
print(f"Denoised shape: {results['denoiseddata'].shape}")  # (50, 200)
print(f"Retained {results['best_threshold']} dimensions")
```

#### Using Presets

By default, PSN uses the `'standard'` preset. You can optionally specify a different mode:

```python
# Standard (default): signal basis at the max-tradeoff threshold (global).
# A deterministic operating point that captures most of the achievable analytic
# recovery while retaining more signal variance than the prediction peak.
# Uses: basis='signal', criterion='max-tradeoff', threshold_method='global'
results = psn(train_data, 'standard')

# Conservative: prioritizes retaining signal (99% of signal variance).
# Uses: basis='signal', criterion='variance', threshold_method='global', variance_threshold=0.99
results = psn(train_data, 'conservative')

# Aggressive: difference basis at the prediction peak. May improve recovery but
# can be unstable with limited data.
# Uses: basis='difference', criterion='prediction', threshold_method='global'
results = psn(train_data, 'aggressive')

# Compare: build the signal and difference bases (each at its max-tradeoff
# threshold) and keep whichever has the higher split-half r at that threshold.
results = psn(train_data, 'compare')

# Wiener: full-rank matrix Wiener filter (no truncation; basis-free).
results = psn(train_data, 'wiener')
```

#### Custom Configuration

```python
opt = {
    'basis': 'signal',                # 'signal', 'difference', 'pca', 'noise', 'random',
                                      #   'compare', or a custom [n_units x D] matrix
    'criterion': 'max-tradeoff',      # 'max-tradeoff' (default), 'prediction', 'variance',
                                      #   'variance_eigenvalues', or 'wiener'
    'threshold_method': 'global',     # 'global' (default) or 'hybrid' (per-unit thresholds)
    'basis_ordering': 'eigenvalues',  # 'eigenvalues' or 'signalvariance'
    'variance_threshold': 0.95,       # Used when criterion='variance'/'variance_eigenvalues'
    'wantverbose': True,
    'wantfig': True,                  # Display diagnostic figures
}
results = psn(train_data, opt)
```

The threshold criteria all operate on the **analytic recovery** curve - the
GSN-covariance-based estimate of how well the denoised output recovers the true
underlying signal out-of-sample, `cumsum(signal - noise/ntrials)` over the retained
dimensions. `'prediction'` takes its peak; `'max-tradeoff'` (the default) takes the
point farthest from the chord between the peak and the do-nothing (trial-average)
point - closer to the peak it keeps more signal variance.

#### Interpolating Between the Prediction Peak and Do-Nothing

The `alpha` parameter smoothly interpolates, in signal-variance space, between the
prediction peak and the trial-average (do-nothing) point:

```python
opt = {
    'alpha': 0.3,   # 0 = prediction peak; 1 = retain all signal variance (do nothing)
}
results = psn(train_data, opt)
```

`alpha` overrides `criterion` and does **not** use `variance_threshold` (its right
endpoint is fixed at the full signal variance).

#### Constraining the Threshold

`allowable_thresholds` restricts the choice to a set of dimensionalities. PSN selects
the **best** threshold among the allowable values for the chosen criterion (it never
evaluates a threshold outside the set); a single value forces exactly that many dims:

```python
results = psn(train_data, {'allowable_thresholds': [5, 10, 20]})  # best of these
results = psn(train_data, {'allowable_thresholds': [50]})          # force 50 dims
```

#### Caching GSN Output Across Hyperparameter Sweeps

GSN covariance estimation is the expensive step. To sweep over PSN hyperparameters
(`alpha`, `basis`, `criterion`, ...) on the same data without re-running GSN each time,
pass the previously computed `gsn_result` back in:

```python
# First call computes GSN
results = psn(train_data, {'criterion': 'max-tradeoff'})

# Subsequent calls reuse the cached covariances
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    out = psn(train_data, {
        'alpha': alpha,
        'gsn_result': results['gsn_result'],
    })
```

#### Wiener Denoising

For the full-rank matrix Wiener filter (the optimal linear estimator - minimizes
expected mean-squared error given the signal and noise covariances; no basis
truncation):

```python
results = psn(train_data, 'wiener')
# equivalently: psn(train_data, {'criterion': 'wiener'})
```

#### scikit-learn Estimator (Python only)

PSN also ships a scikit-learn-compatible estimator for use in pipelines / grid search:

```python
from psn import PSN
model = PSN(mode='standard')
denoised = model.fit_transform(train_data)
```

#### GPU Acceleration (Python only)

Set `device='cuda'` or `device='mps'` to run the covariance / projection / eigh work on
a GPU via torch (the numpy CPU path is the default):

```python
results = psn(train_data, {'device': 'cuda'})
```

#### Applying the Denoiser to Held-Out Data

The learned denoising matrix can be applied to new trial-averaged data from the same units:

```python
results = psn(train_data)
denoiser = results['denoiser']
unit_means = results['unit_means'][:, None]

# Apply to held-out test data (n_units, n_conditions)
test_avg = np.nanmean(test_data, axis=2)
denoised_test = denoiser.T @ (test_avg - unit_means) + unit_means
```

#### Output Structure

`psn` returns a dictionary. The most commonly used fields:

```python
results['denoiseddata']      # (n_units, n_conditions) - Denoised estimates
results['residuals']         # (n_units, n_conditions, n_trials) - data - denoiseddata
results['denoiser']          # (n_units, n_units) - Denoising matrix
results['unit_means']        # (n_units,) - Per-unit means used for centering
results['best_threshold']    # Number of dimensions retained (scalar, or per-unit array for 'hybrid')
results['fullbasis']         # (n_units, n_dims) - Basis vectors after global ordering
results['signalvar']         # Signal variance per dimension
results['noisevar']          # Noise variance per dimension
results['objective']         # Cumulative objective curve used for threshold selection
results['svnv_before']       # (n_units, 2) - Signal/noise variance before denoising
results['svnv_after']        # (n_units, 2) - Signal/noise variance after denoising
results['gsn_result']        # GSN output dict (cSb, cNb, ...) for reuse / caching
results['recovery_tradeoff'] # Diagnostic data behind the recovery-vs-signal-retained figure
```

---

### MATLAB

#### Installation

Clone with submodules to include the GSN dependency:
```bash
git clone --recurse-submodules https://github.com/jacob-prince/PSN.git
```

If you've already cloned without submodules:
```bash
cd PSN
git submodule update --init --recursive
```

#### Basic Usage

Open MATLAB and navigate to the PSN directory:

```matlab
cd('path/to/PSN/matlab')

% Generate test data (50 units, 100 conditions, 5 trials)
data = randn(50, 100, 5);

% Apply PSN with default settings ('standard')
results = psn(data);
fprintf('Denoised data shape: [%d x %d]\n', size(results.denoiseddata));
```

#### Using Presets

```matlab
% Standard (default): signal basis at the max-tradeoff threshold (global)
results = psn(data, 'standard');

% Conservative: prioritizes retaining signal (99% of signal variance)
results = psn(data, 'conservative');

% Aggressive: difference basis at the prediction peak
results = psn(data, 'aggressive');

% Compare: keep whichever of the signal/difference bases has higher split-half r
results = psn(data, 'compare');

% Wiener: full-rank matrix Wiener filter
results = psn(data, 'wiener');
```

#### Custom Configuration

```matlab
opt = struct();
opt.basis = 'signal';              % 'signal','difference','pca','noise','random',
                                   %   'compare', or a custom matrix
opt.criterion = 'max-tradeoff';    % 'max-tradeoff' (default),'prediction','variance',
                                   %   'variance_eigenvalues','wiener'
opt.threshold_method = 'global';   % 'global' (default) or 'hybrid'
opt.variance_threshold = 0.95;     % Used when criterion='variance'/'variance_eigenvalues'
opt.wantverbose = true;
opt.wantfig = true;                % Display diagnostic figures

results = psn(data, opt);
```

> The MATLAB and Python implementations are feature-for-feature equivalent. The only
> Python-only additions are GPU acceleration (`device`) and the scikit-learn `PSN`
> estimator class.

#### Test Installation

Add the toolbox to the path and run the test suite:
```matlab
addpath(genpath('path/to/PSN/matlab'))
test_psn_all_combinations
```

#### Troubleshooting

- **"performgsn not found"**: Run `git submodule update --init --recursive` to fetch the GSN dependency
- **Path issues**: Ensure MATLAB's working directory is `PSN/matlab`
- See `matlab/README.md` for detailed MATLAB-specific documentation

---

## Key Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| **basis** | `'signal'`, `'difference'`, `'pca'`, `'noise'`, `'random'`, `'compare'`, custom matrix | Basis for dimensionality reduction (`'wiener'` is a deprecated alias for `criterion='wiener'`) |
| **criterion** | `'max-tradeoff'` (default), `'prediction'`, `'variance'`, `'variance_eigenvalues'`, `'wiener'` | How to determine the dimensionality threshold |
| **threshold_method** | `'global'` (default), `'hybrid'` | Single population threshold, or per-unit thresholds on a shared ordering |
| **basis_ordering** | `'eigenvalues'`, `'signalvariance'` | Initial global order of basis vectors |
| **alpha** | `0.0` to `1.0` (or `None`) | Interpolates between the prediction peak (0) and the trial-average / do-nothing point (1). Does not use `variance_threshold`. |
| **variance_threshold** | `0.0` to `1.0` (default: `0.99`) | Target signal-variance fraction for `criterion='variance'`/`'variance_eigenvalues'` |
| **allowable_thresholds** | vector (or `None`) | Restrict the threshold to these values (PSN picks the best one); a single value forces it |
| **unit_groups** | `[n_units]` integer labels | Units sharing a label share a threshold (`'hybrid'` only) |
| **gsn_result** | dict / struct | Reuse precomputed GSN covariances to skip estimation |
| **device** *(Python only)* | `'cpu'` (default), `'cuda'`, `'mps'` | Run on GPU via torch |

---

## Data Format

Both Python and MATLAB expect data in the shape:
- **Python**: `(n_units, n_conditions, n_trials)`
- **MATLAB**: `[n_units x n_conditions x n_trials]`

**NaN handling**: PSN supports uneven trials across conditions. Missing trials can be
indicated with NaNs. Each condition must have at least one trial with valid data across
all units.

---

## Documentation

- **MATLAB API**: See [matlab/README.md](matlab/README.md) for detailed MATLAB documentation
- **Examples**: Explore `examples/` for Python demos and `matlab/tests/` for MATLAB examples

---

## Citation

If you use PSN in your research, please cite:

```
[Citation information will be added upon publication]
```

---

## License

PSN is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

PSN is under active development. Feedback, bug reports, and contributions are welcome! Please open an issue or pull request on GitHub.
