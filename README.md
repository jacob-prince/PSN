# PSN: Partitioning Signal and Noise

**PSN** is a library for denoising neural data by adaptively partitioning signal and noise.

> **Note:** As of December 2025, PSN is under active development. The API and algorithm are subject to change.

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

#### Basic Usage (Sklearn API)

PSN provides a scikit-learn compatible API with `fit()`, `transform()`, and `fit_transform()` methods:

```python
import numpy as np
from psn import PSN

# Generate test data (10 units, 25 conditions, 3 trials)
np.random.seed(42)
data = np.random.randn(10, 25, 3)

# Create and fit the model
model = PSN()
model.fit(data)

# Access denoised data
print(f"Denoised shape: {model.denoiseddata_.shape}")  # (10, 25)
print(f"Retained {model.best_threshold_} dimensions")

# Or use fit_transform for one-step denoising
denoised = PSN().fit_transform(data)
```

#### Applying to New Data

Once fitted, the model can denoise new data with the same units:

```python
# Fit on training data
model = PSN()
model.fit(train_data)

# Apply to new test data (must have same number of units)
denoised_test = model.transform(test_data)

# Works with 3D (trial) or 2D (already averaged) data
denoised_2d = model.transform(test_data_2d)
```

#### Saving and Loading Models

Fitted models can be saved and loaded using pickle:

```python
import pickle

# Save fitted model
with open('psn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load and use
with open('psn_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

denoised = loaded_model.transform(new_data)
```

#### Diagnostic Visualization

Generate diagnostic plots after fitting:

```python
model = PSN()
model.fit(data)

# Generate diagnostic figure
fig = model.plot_diagnostics()
fig.savefig('diagnostics.png')

# Or visualize during fit
model.fit(data, visualize=True)
```

#### Functional API

For simple one-shot denoising, use the functional `psn()` interface:

```python
from psn import psn

# Apply PSN denoising with default settings
results = psn(data)

# Access denoised data
denoised_data = results['denoiseddata']  # Shape: (10, 25)
print(f"Retained {results['best_threshold']} dimensions on average")
```

#### Using Presets

By default, PSN uses the `'standard'` preset. You can optionally specify a different mode:

```python
# Standard (default): balances signal retention and out-of-sample generalization
# Uses: basis='signal', criterion='prediction', threshold_method='hybrid'
model = PSN()  # equivalent to PSN(mode='standard')

# Conservative: prioritizes retaining signal
# Uses: basis='signal', criterion='variance', threshold_method='global'
model = PSN(mode='conservative')

# Aggressive: maximizes out-of-sample generalization with unit-specific adaptation
# Uses: basis='difference', criterion='prediction', threshold_method='unit'
model = PSN(mode='aggressive')
```

#### Custom Configuration

```python
# Sklearn API: pass parameters directly to constructor
model = PSN(
    mode=None,                       # Use None for fully custom config
    basis='signal',                  # 'signal', 'difference', 'pca', or custom matrix
    criterion='prediction',          # 'prediction', 'variance', or 'variance_eigenvalues'
    threshold_method='hybrid',       # 'global', 'hybrid', or 'unit'
    variance_threshold=0.95,         # Used when criterion='variance'
    wantverbose=True                 # Print progress messages
)
model.fit(data)

# Access results as attributes (with trailing underscore)
denoised_data = model.denoiseddata_      # Denoised estimates
residuals = model.residuals_             # data - denoiseddata
thresholds = model.best_threshold_       # Number of dimensions retained
basis = model.fullbasis_                 # Basis vectors used
denoiser = model.denoiser_               # Learned denoising matrix

# Functional API: pass options as dict
opt = {
    'basis': 'signal',
    'criterion': 'prediction',
    'threshold_method': 'hybrid',
    'variance_threshold': 0.95,
    'wantverbose': True,
    'wantfig': True
}
results = psn(data, opt)
```

#### Output Structure

**Sklearn API** - Access results as attributes (with trailing underscore):

```python
model = PSN()
model.fit(data)

# Primary outputs
model.denoiseddata_       # (n_units, n_conditions) - Denoised estimates
model.residuals_          # (n_units, n_conditions, n_trials) - data - denoiseddata
model.denoiser_           # (n_units, n_units) - Learned denoising matrix
model.unit_means_         # (n_units,) - Mean per unit for centering

# Diagnostics
model.best_threshold_     # Number of dimensions retained (scalar or array)
model.fullbasis_          # (n_units, n_dims) - Basis vectors
model.signalvar_          # Signal variance per dimension
model.noisevar_           # Noise variance per dimension
model.svnv_before_        # Signal/noise variance before denoising
model.svnv_after_         # Signal/noise variance after denoising
```

**Functional API** - Returns a dictionary:

```python
results = psn(data, opt)

results['denoiseddata']    # (n_units, n_conditions) - Denoised estimates
results['residuals']       # (n_units, n_conditions, n_trials) - data - denoiseddata
results['denoiser']        # (n_units, n_units) - Denoising matrix
results['best_threshold']  # Number of dimensions retained
results['fullbasis']       # (n_units, n_dims) - Basis vectors
results['gsn_result']      # Full GSN results dict
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
% Customize PSN parameters
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
| **basis** | `'signal'`, `'difference'`, `'pca'`, custom matrix | Basis for dimensionality reduction |
| **criterion** | `'prediction'`, `'variance'`, `'variance_eigenvalues'` | How to determine dimensionality threshold |
| **threshold_method** | `'global'`, `'hybrid'`, `'unit'` | How to apply thresholds across units |
| **variance_threshold** | 0.0 to 1.0 (default: 0.99) | Target variance fraction (for `criterion='variance'`) |

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
