# PSN MATLAB Implementation

MATLAB implementation of **Partitioning Signal and Noise (PSN)**, a method for
denoising multi-trial neural data. See the [main README](../README.md) for an
overview of PSN and the full parameter reference.

> The MATLAB and Python implementations are feature-for-feature equivalent for the
> denoising algorithm. Python-only conveniences: GPU acceleration (`device`), the
> scikit-learn `PSN` estimator, and the keyword / `opt=` argument forms.

---

## Installation

Clone with submodules to include the GSN dependency:

```bash
git clone --recurse-submodules https://github.com/jacob-prince/PSN.git
```

If you've already cloned without submodules:
```bash
cd PSN && git submodule update --init --recursive
```

Verify by adding the toolbox to the path and running the test suite:

```matlab
addpath(genpath('path/to/PSN/matlab'))
test_psn_all_combinations
```

---

## Quick Start

```matlab
cd('path/to/PSN/matlab')
data = randn(50, 100, 5);                 % [n_units x n_conditions x n_trials]

results = psn(data);                       % 'standard' (default)
results = psn(data, 'conservative');       % 99% signal-variance retention
results = psn(data, 'aggressive');         % difference basis, prediction peak
results = psn(data, 'compare');            % best of signal vs difference basis
results = psn(data, 'wiener');             % full-rank matrix Wiener filter

denoised = results.denoiseddata;           % [n_units x n_conditions]
```

---

## Custom Configuration

Pass an options struct as the second argument. Any field overrides the preset default:

```matlab
opt = struct();
opt.basis = 'signal';              % 'signal','difference','pca','noise','random',
                                   %   'compare', or a custom [n_units x D] matrix
opt.criterion = 'max-tradeoff';    % 'max-tradeoff','prediction','variance',
                                   %   'variance_eigenvalues','wiener'
opt.threshold_method = 'global';   % 'global' or 'hybrid' (per-unit thresholds)
opt.variance_threshold = 0.95;
opt.wantverbose = true;
opt.wantfig = true;

results = psn(data, opt);
```

See the [parameter table](../README.md#configuration) in the main README for the
full list of options.

---

## Advanced Usage

#### Caching GSN Across Sweeps

```matlab
results = psn(data);
opt = struct('alpha', 0.3, 'gsn_result', results.gsn_result);
out = psn(data, opt);
```

#### Constraining the Threshold

```matlab
opt = struct('allowable_thresholds', [5 10 20]);   % best among these
results = psn(data, opt);

opt = struct('allowable_thresholds', 10);           % force exactly 10 dims
results = psn(data, opt);
```

#### Per-Unit Thresholds (Hybrid)

```matlab
opt = struct('threshold_method', 'hybrid');
results = psn(data, opt);
unique(results.best_threshold)                     % per-unit thresholds
```

#### NaN Handling

PSN supports uneven trials across conditions:

```matlab
data = randn(10, 50, 5);
data(:, 1, 4:5) = NaN;   % condition 1 has only 3 trials
results = psn(data);
```

Each condition must have at least one trial with valid data across all units.

---

## Output Structure

```matlab
results.denoiseddata       % [n_units x n_conditions] - denoised estimates
results.residuals          % [n_units x n_conditions x n_trials] - data minus denoised
results.denoiser           % [n_units x n_units] - denoising matrix
results.unit_means         % [n_units x 1] - per-unit means used for centering
results.best_threshold     % scalar ('global') or [n_units x 1] ('hybrid')
results.fullbasis          % [n_units x n_dims] - basis vectors
results.signalvar          % signal variance per dimension
results.noisevar           % noise variance per dimension
results.objective          % cumulative objective curve
results.svnv_before        % [n_units x 2] - signal/noise variance before denoising
results.svnv_after         % [n_units x 2] - signal/noise variance after denoising
results.gsn_result         % GSN output struct (.cSb, .cNb, ...) for caching
results.recovery_tradeoff  % diagnostic data behind the recovery figure
```

---

## Troubleshooting

- **"performgsn not found"**: Run `git submodule update --init --recursive` to fetch
  the GSN dependency, then restart MATLAB.
- **Path issues**: `psn.m` adds its own `utilities/` and GSN to the path. If calling
  utilities directly, run `addpath(genpath('path/to/PSN/matlab'))` first.

---

## See Also

- [Main README](../README.md) — overview, Python usage, full parameter reference
- [scikit-learn API](../SKLEARN_API.md) — Python estimator docs
- [Examples](../examples/) — Python demos
