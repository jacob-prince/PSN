# PSN: Partitioning Signal and Noise

**PSN** denoises multi-trial neural data by estimating signal and noise covariances
([GSN](https://github.com/cvnlab/GSN)), projecting into an optimal basis, and selecting
how many dimensions to retain.

> **Note:** PSN is under active development. The API and functionality are subject to change.

| Clean (ground truth) | Single noisy trial | PSN denoised |
|:---:|:---:|:---:|
| ![clean](examples/sunset_clean.jpg) | ![noisy](examples/sunset_noisy.jpg) | ![denoised](examples/sunset_denoised.jpg) |

---

## Installation

### Python

```bash
git clone https://github.com/jacob-prince/PSN.git
cd PSN
pip install -e .
```

### MATLAB

```bash
git clone --recurse-submodules https://github.com/jacob-prince/PSN.git
```

If you've already cloned without submodules:
```bash
cd PSN && git submodule update --init --recursive
```

---

## Data Format

Both Python and MATLAB expect `(n_units, n_conditions, n_trials)` with `n_trials >= 2`.

**NaN handling**: PSN supports uneven trials across conditions. Missing trials can be
indicated with NaNs. Each condition must have at least one trial with valid data across
all units.

---

## Quick Start

### Python

```python
import numpy as np
from psn import psn, generate_data

# Synthetic train and test sets: (n_units, n_conditions, n_trials)
train_data, test_data, ground_truth = generate_data(
    nvox=50, ncond=200, ntrial=5, random_seed=42
)

results = psn(train_data)                  # recommended default: balanced denoising
results = psn(train_data, 'conservative')  # remove noise cautiously, keep more of the signal
results = psn(train_data, 'aggressive')    # remove noise more strongly (best with plenty of data)
results = psn(train_data, 'compare')       # auto-pick whichever setting best reproduces the data
results = psn(train_data, 'wiener')        # optimal linear filter, no cutoff (minimizes error)

denoised = results['denoiseddata']         # (n_units, n_conditions)

# Apply the fitted denoiser to held-out test data
denoiser   = results['denoiser']           # (n_units, n_units)
unit_means = results['unit_means'][:, None]
test_avg   = np.nanmean(test_data, axis=2)                       # (n_units, n_conditions)
denoised_test = denoiser.T @ (test_avg - unit_means) + unit_means
```

### MATLAB

```matlab
cd('path/to/PSN/matlab')
data = randn(50, 100, 5);                 % [n_units x n_conditions x n_trials]

results = psn(data);                       % recommended default: balanced denoising
results = psn(data, 'conservative');       % remove noise cautiously, keep more of the signal
results = psn(data, 'aggressive');         % remove noise more strongly (best with plenty of data)
results = psn(data, 'compare');            % auto-pick whichever setting best reproduces the data
results = psn(data, 'wiener');             % optimal linear filter, no cutoff (minimizes error)
```

> The MATLAB and Python implementations are feature-for-feature equivalent for the
> denoising algorithm. Python-only conveniences: GPU acceleration (`device`), the
> scikit-learn `PSN` estimator, and the keyword / `opt=` argument forms.
> See [matlab/README.md](matlab/README.md) for detailed MATLAB documentation.

---

## Image Denoising Example

`generate_data` can simulate noisy multi-trial measurements from an image.
PSN denoises the trial-averaged data, recovering the clean image:

```python
import numpy as np
from PIL import Image
from psn import psn, generate_data

# Load an image and simulate noisy multi-trial measurements
img = np.array(Image.open('examples/sunset.jpg').resize((1000, 1000))) / 255.0
H, W, C = img.shape
train_data, _, gt = generate_data(
    true_signal=img, ntrial=3, noise_multiplier=100.0, random_seed=42
)
# train_data shape: (1000, 3000, 3) -> (n_units, n_conditions, n_trials)

# Denoise with PSN
results = psn(train_data, 'wiener')

# Reconstruct images
clean    = np.clip(gt['signal'].T.reshape(H, W, C), 0, 1)
noisy    = np.clip(train_data[:, :, 0].reshape(H, W, C), 0, 1)
denoised = np.clip(results['denoiseddata'].reshape(H, W, C), 0, 1)
```

---

## Configuration

### Presets

Most users only need one of these named presets. They differ in how aggressively
they strip noise, i.e. how much of the data they keep versus discard.

| Preset | What it does | When to use it |
|--------|--------------|----------------|
| *(default)* / `'standard'` | Balanced denoising: a good tradeoff between removing noise and preserving signal. | Start here. |
| `'conservative'` | Removes noise cautiously, keeping more of the original signal. | When discarding real signal is costlier than leaving in some noise. |
| `'aggressive'` | Removes noise more strongly; can recover more but is less stable on small datasets. | When you have plenty of trials/conditions. |
| `'compare'` | Tries two strategies and automatically keeps whichever best reproduces held-out trials. | When you're unsure which setting fits your data. |
| `'wiener'` | The optimal linear filter (minimizes mean-squared error); uses the full data with no dimension cutoff. It shrinks each direction toward zero, trading some bias for lower variance. | When you want the theoretically optimal estimate. |

### Advanced parameters

The presets above are the recommended interface. If you want to build a custom
combination, pass options as a dict, via `opt=`, or as keyword arguments, and a
preset can be combined with any of these (keyword options win):

```python
psn(train_data, {'basis': 'signal', 'device': 'cuda'})   # positional dict
psn(train_data, opt={'basis': 'signal'})                  # opt= keyword
psn(train_data, basis='signal', device='cuda')            # keywords
psn(train_data, 'aggressive', device='cuda')              # preset + keyword override
```

In MATLAB, pass an options struct: `psn(data, opt)`.

The two central knobs are **basis** (which directions in the data to denoise
along) and **criterion** (how many of those directions to keep). The presets set
both for you.

| Parameter | Options | Description |
|-----------|---------|-------------|
| **basis** | `'signal'` (default), `'difference'`, `'pca'`, `'noise'`, `'random'`, `'compare'`, custom matrix | Which set of directions to denoise along. `'signal'` favours directions carrying the most signal; `'difference'` uses the directions of the signal-minus-noise covariance difference (hence the name), which denoises more aggressively. |
| **criterion** | `'max-tradeoff'` (default), `'prediction'`, `'variance'`, `'variance_eigenvalues'`, `'wiener'` | How many directions to keep. `'max-tradeoff'` balances signal kept vs. noise removed; `'variance'` keeps a target fraction of signal; `'wiener'` keeps all with optimal weighting. |
| **threshold_method** | `'global'` (default), `'hybrid'` | Use one cutoff for all units, or a per-unit cutoff on a shared ordering. |
| **alpha** | `0.0`–`1.0` (or `None`) | A single dial from most aggressive (0) to no denoising (1). Overrides `criterion` when set. |
| **variance_threshold** | `0.0`–`1.0` (default `0.99`) | Fraction of signal to keep when `criterion` is `'variance'`/`'variance_eigenvalues'`. |
| **allowable_thresholds** | vector (or `None`) | Restrict the cutoff to these values; a single value forces exactly that many dimensions. |
| **basis_ordering** | `'eigenvalues'` (default), `'signalvariance'` | Order in which directions are considered for keeping. |
| **unit_groups** | `[n_units]` integer labels | Units sharing a label share a cutoff (`'hybrid'` only). |
| **gsn_result** | dict / struct | Reuse precomputed GSN covariances to skip re-estimation. |
| **device** *(Python only)* | `'cpu'` (default), `'cuda'`, `'mps'` | Run on GPU via torch (only when explicitly set). |

---

## Advanced Usage

#### Caching GSN Across Hyperparameter Sweeps

GSN covariance estimation is the expensive step. Pass the cached result back in to
avoid re-running it:

```python
results = psn(train_data)
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    out = psn(train_data, alpha=alpha, gsn_result=results['gsn_result'])
```

#### GPU Acceleration (Python only)

Set `device='cuda'` or `device='mps'` to run the heavy linear algebra on GPU via torch.
Pass data as a numpy array; PSN handles the transfer. GPU pays off mainly at large
`n_units` (~10k+).

```python
results = psn(train_data, device='cuda')
```

#### scikit-learn Estimator (Python only)

```python
from psn import PSN
model = PSN()
denoised = model.fit_transform(train_data)
```

See [SKLEARN_API.md](SKLEARN_API.md) for the full estimator API.

---

## Output Structure

`psn` returns a dictionary (Python) or struct (MATLAB):

```python
results['denoiseddata']      # (n_units, n_conditions) - denoised estimates
results['residuals']         # (n_units, n_conditions, n_trials) - data minus denoised
results['denoiser']          # (n_units, n_units) - denoising matrix
results['unit_means']        # (n_units,) - per-unit means used for centering
results['best_threshold']    # dims retained (scalar, or per-unit array for 'hybrid')
results['fullbasis']         # (n_units, n_dims) - basis vectors
results['signalvar']         # signal variance per dimension
results['noisevar']          # noise variance per dimension
results['objective']         # cumulative objective curve
results['svnv_before']       # (n_units, 2) - signal/noise variance before denoising
results['svnv_after']        # (n_units, 2) - signal/noise variance after denoising
results['gsn_result']        # GSN output (cSb, cNb, ...) for caching
results['recovery_tradeoff'] # diagnostic data behind the recovery figure
```

---

## Citation

```
[Citation information will be added upon publication]
```

## License

MIT. See [LICENSE](LICENSE).

## Contributing

Feedback, bug reports, and contributions are welcome; please open an issue or pull request on GitHub.
