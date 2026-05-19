# PSN: Partitioning Signal and Noise

**PSN** is a library for denoising neural data by adaptively partitioning signal and noise.

> **Note:** As of May 2026, PSN is under active development. The API and algorithm are subject to change.

## What is PSN?

PSN (Partitioning Signal and Noise) is a method for denoising multi-trial neural data by:
1. Projecting data into a low-dimensional basis (e.g., eigenvectors of signal covariance)
2. Adaptively selecting how many dimensions to retain per unit
3. Reconstructing denoised estimates that optimize signal recovery while minimizing noise

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
# Standard (default): balances signal retention and out-of-sample generalization
# Uses: basis='signal', criterion='prediction', threshold_method='hybrid'
results = psn(train_data, 'standard')

# Conservative: prioritizes retaining signal
# Uses: basis='signal', criterion='variance', threshold_method='global'
results = psn(train_data, 'conservative')

# Aggressive: maximizes out-of-sample generalization with unit-specific adaptation
# Uses: basis='difference', criterion='prediction', threshold_method='unit'
results = psn(train_data, 'aggressive')
```

#### Custom Configuration

```python
opt = {
    'basis': 'signal',                # 'signal', 'difference', 'pca', 'wiener', or custom matrix
    'criterion': 'prediction',        # 'prediction', 'variance', or 'variance_eigenvalues'
    'threshold_method': 'hybrid',     # 'global', 'hybrid', or 'unit'
    'basis_ordering': 'eigenvalues',  # 'eigenvalues' or 'signalvariance'
    'variance_threshold': 0.95,       # Used when criterion='variance'
    'wantverbose': True,
    'wantfig': True,                  # Display diagnostic figures
}
results = psn(train_data, opt)
```

#### Interpolating Between Prediction and Variance Targets

The `alpha` parameter smoothly interpolates between the prediction-optimal threshold
and a variance-retention target, letting you trade off generalization vs. signal
preservation without committing to either extreme:

```python
opt = {
    'alpha': 0.3,                 # 0 = prediction peak, 1 = variance retention target
    'variance_threshold': 0.95,   # Defines the variance target
}
results = psn(train_data, opt)
```

#### Caching GSN Output Across Hyperparameter Sweeps

GSN covariance estimation is the expensive step. To sweep over PSN hyperparameters
(`alpha`, `basis`, `criterion`, ...) on the same data without re-running GSN each time,
pass the previously computed `gsn_result` back in:

```python
# First call computes GSN
results = psn(train_data, {'criterion': 'prediction'})

# Subsequent calls reuse the cached covariances
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    out = psn(train_data, {
        'alpha': alpha,
        'gsn_result': results['gsn_result'],
    })
```

#### Wiener Denoising

For full-rank matrix Wiener filtering (no basis truncation):

```python
results = psn(train_data, {'basis': 'wiener'})
```

To apply Wiener shrinkage on top of a global truncation:

```python
opt = {
    'basis': 'signal',
    'threshold_method': 'global',
    'denoiser_type': 'wiener',
}
results = psn(train_data, opt)
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
results['best_threshold']    # Number of dimensions retained (scalar or per-unit array)
results['fullbasis']         # (n_units, n_dims) - Basis vectors after global ordering
results['signalvar']         # Signal variance per dimension
results['noisevar']          # Noise variance per dimension
results['svnv_before']       # (n_units, 2) - Signal/noise variance before denoising
results['svnv_after']        # (n_units, 2) - Signal/noise variance after denoising
results['gsn_result']        # GSN output dict (cSb, cNb, ...) for reuse / caching
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

% Apply PSN with default settings
results = psn(data);
fprintf('Denoised data shape: [%d x %d]\n', size(results.denoiseddata));
```

#### Using Presets

```matlab
% Conservative: prioritizes retaining signal
results = psn(data, 'conservative');

% Standard: balances signal retention and generalization (default)
results = psn(data, 'standard');

% Aggressive: maximizes out-of-sample generalization
results = psn(data, 'aggressive');
```

#### Custom Configuration

```matlab
opt = struct();
opt.basis = 'signal';              % 'signal', 'difference', 'pca', or custom matrix
opt.criterion = 'prediction';      % 'prediction', 'variance', 'variance_eigenvalues'
opt.threshold_method = 'hybrid';   % 'global', 'hybrid', 'unit'
opt.variance_threshold = 0.95;     % Used when criterion='variance'
opt.wantverbose = true;
opt.wantfig = true;                % Display diagnostic figures

results = psn(data, opt);
```

#### Test Installation

Verify GSN dependency and PSN functionality:
```matlab
cd('path/to/PSN/matlab')

% Test GSN dependency
test_gsn_dependency
```

#### Troubleshooting

- **"performgsn not found"**: Run `git submodule update --init --recursive` to fetch the GSN dependency
- **Path issues**: Ensure MATLAB's working directory is `PSN/matlab`
- See `matlab/README.md` for detailed MATLAB-specific documentation

---

## Key Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| **basis** | `'signal'`, `'difference'`, `'pca'`, `'wiener'`, custom matrix | Basis for dimensionality reduction |
| **criterion** | `'prediction'`, `'variance'`, `'variance_eigenvalues'` | How to determine dimensionality threshold |
| **threshold_method** | `'global'`, `'hybrid'`, `'unit'` | How to apply thresholds across units |
| **basis_ordering** | `'eigenvalues'`, `'signalvariance'` | Initial global order of basis vectors |
| **alpha** | `0.0` to `1.0` (or `None`) | Interpolates between prediction peak (0) and variance target (1) |
| **variance_threshold** | `0.0` to `1.0` (default: `0.99`) | Target variance fraction (for `criterion='variance'` or with `alpha`) |
| **denoiser_type** | `'truncation'`, `'wiener'` | Hard truncation or Wiener shrinkage (Wiener requires `threshold_method='global'`) |
| **gsn_result** | dict | Reuse precomputed GSN covariances to skip estimation |

---

## Data Format

Both Python and MATLAB expect data in the shape:
- **Python**: `(n_units, n_conditions, n_trials)`
- **MATLAB**: `[n_units x n_conditions x n_trials]`

**NaN handling**: PSN supports uneven trials across conditions. Missing trials can be indicated with NaNs. Each condition must have at least one trial with valid data across all units.

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
