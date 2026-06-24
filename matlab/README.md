# PSN MATLAB Implementation

This directory contains the MATLAB implementation of **Partitioning Signal and Noise (PSN)**, a method for denoising multi-trial neural data.

The MATLAB and Python implementations are feature-for-feature equivalent. The only
Python-only additions are GPU acceleration (`device`) and the scikit-learn `PSN`
estimator class.

---

## Quick Start

### Installation

PSN requires the [GSN (Generative Modeling of Signal and Noise)](https://github.com/cvnlab/GSN) library as a dependency, included as a git submodule. Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/jacob-prince/PSN.git
```

If you've already cloned without submodules:
```bash
cd PSN
git submodule update --init --recursive
```

### Verify Installation

Open MATLAB, add the toolbox to the path, and run the test suite:

```matlab
addpath(genpath('path/to/PSN/matlab'))
test_psn_all_combinations   % exercises bases x criteria x threshold methods + NaN handling
```

---

## Basic Usage

### Minimal Example

```matlab
% Generate test data (50 units, 100 conditions, 5 trials)
data = randn(50, 100, 5);

% Apply PSN with default settings ('standard')
results = psn(data);

% Access denoised data
denoised = results.denoiseddata;  % [nunits x nconds]
fprintf('Denoised data shape: [%d x %d]\n', size(denoised));
```

### Using Presets

```matlab
% STANDARD (default): signal basis at the max-tradeoff threshold (global).
%   A deterministic operating point that captures most of the achievable analytic
%   recovery while retaining more signal variance than the prediction peak.
%   basis='signal', criterion='max-tradeoff', threshold_method='global'
results = psn(data, 'standard');

% CONSERVATIVE: prioritizes retaining signal (99% of signal variance).
%   basis='signal', criterion='variance', threshold_method='global', variance_threshold=0.99
results = psn(data, 'conservative');

% AGGRESSIVE: difference basis at the prediction peak. May improve recovery but
%   can be unstable with limited data.
%   basis='difference', criterion='prediction', threshold_method='global'
results = psn(data, 'aggressive');

% COMPARE: build the signal and difference bases (each at its max-tradeoff
%   threshold) and keep whichever has the higher split-half r at that threshold.
results = psn(data, 'compare');

% WIENER: full-rank matrix Wiener filter (no truncation; basis-free).
results = psn(data, 'wiener');
```

---

## Custom Configuration

### Parameter Structure

```matlab
opt = struct();
opt.basis = 'signal';              % 'signal','difference','pca','noise','random',
                                   %   'compare', or a custom [nunits x D] matrix
opt.criterion = 'max-tradeoff';    % 'max-tradeoff' (default),'prediction','variance',
                                   %   'variance_eigenvalues','wiener'
opt.threshold_method = 'global';   % 'global' (default) or 'hybrid'
opt.variance_threshold = 0.95;     % Used when criterion='variance'/'variance_eigenvalues'
opt.wantverbose = true;            % Print progress messages
opt.wantfig = true;                % Display diagnostic figures

results = psn(data, opt);
```

### Key Parameters

#### **basis** - Basis for dimensionality reduction
- `'signal'` (default): eigenvectors of the signal covariance (cSb)
- `'difference'`: eigenvectors of cSb - cNb/ntrials_avg (admits more noise-dominated dims)
- `'compare'`: build signal and difference bases, keep the higher split-half r at each max-tradeoff K
- `'pca'`, `'noise'`, `'random'`: [not recommended]
- `B` (matrix): custom orthonormal basis [nunits x D]
- `'wiener'`: [deprecated] alias for `criterion='wiener'`

#### **criterion** - How to determine the threshold
All criteria below operate on the **analytic recovery** curve - the GSN-covariance-based
estimate of how well the denoised output recovers the true underlying signal out-of-sample,
`cumsum(signal - noise/ntrials)` over the retained dimensions.
- `'max-tradeoff'` (default): the point on the descending limb farthest from the chord
  between the prediction peak and the do-nothing (trial-average) point - keeps recovery
  high while retaining more signal variance than the prediction peak
- `'prediction'`: the peak of the analytic recovery curve
- `'variance'`: retain dimensions until a target fraction of signal variance is reached
- `'variance_eigenvalues'`: retain until a target fraction of the basis eigenvalues is reached
- `'wiener'`: full-rank matrix Wiener filter (no truncation; basis-free)

#### **threshold_method** - How to apply thresholds
- `'global'` (default): single threshold for all units (symmetric denoiser)
- `'hybrid'`: shared global ordering, unit-specific thresholds

#### **alpha** - Interpolate between the prediction peak and do-nothing
- `[]` (default): disabled
- `0`: prediction peak; `1`: retain all signal variance (the trial average / do nothing)
- Overrides `criterion`; does **not** use `variance_threshold`

#### **variance_threshold** - Target signal-variance fraction
- Default: `0.99`; range `0.0`-`1.0`
- Only used when `criterion='variance'` or `'variance_eigenvalues'`

---

## Advanced Features

### Unit Groups

Group units to share the same threshold (applies in `'hybrid'` mode):

```matlab
unit_groups = [1 1 1 2 2 2 3 3 3 3];  % [nunits x 1]

opt = struct();
opt.threshold_method = 'hybrid';
opt.unit_groups = unit_groups;

results = psn(data, opt);
```

Units with the same group ID receive the same threshold (averaged across the group).

### Allowable Thresholds

Restrict which thresholds PSN may choose. PSN selects the **best** threshold among the
allowable values for the chosen criterion (it never evaluates a threshold outside the set):

```matlab
opt = struct();
opt.allowable_thresholds = [1, 5, 10, 15];  % best of these
results = psn(data, opt);

opt.allowable_thresholds = 10;              % force exactly 10 dimensions
```

### Basis Ordering

```matlab
opt = struct();
opt.basis_ordering = 'eigenvalues';      % use eigenvalue magnitudes (default)
% opt.basis_ordering = 'signalvariance'; % measure signal variance empirically
results = psn(data, opt);
```

### Caching GSN Across Sweeps

GSN covariance estimation is the expensive step. Pass a previous `results.gsn_result`
back in to sweep hyperparameters without re-running GSN:

```matlab
results = psn(data, struct('criterion', 'max-tradeoff'));
opt = struct('alpha', 0.3, 'gsn_result', results.gsn_result);
out = psn(data, opt);
```

---

## Handling Missing Data (NaNs)

PSN supports uneven trials across conditions:

```matlab
data = randn(10, 50, 5);
data(:, 1, 4:5) = NaN;  % Condition 1 only has 3 trials
data(:, 2, 5) = NaN;    % Condition 2 only has 4 trials
results = psn(data);
```

**Requirements:**
- Each condition must have at least one trial with valid data across **all units**
- PSN computes the average number of valid trials and uses it in noise/ntrials formulas
- Denoised output contains no NaNs (filled in from available data); residuals preserve input NaN positions

---

## Output Structure

```matlab
results = psn(data, opt);

% Core outputs
results.denoiseddata    % [nunits x nconds] - denoised estimates (PSN's signal estimate)
results.residuals       % [nunits x nconds x ntrials] - data - denoiseddata (noise estimate)
results.unit_means      % [nunits x 1] - per-unit means used for centering
results.denoiser        % [nunits x nunits] - denoising matrix (symmetric for 'global')

% Threshold / basis
results.best_threshold  % scalar ('global') or [nunits x 1] ('hybrid')
results.fullbasis       % [nunits x dims] - basis vectors after global ordering
results.signalvar       % signal variance per dimension
results.noisevar        % noise variance per dimension
results.objective       % cumulative objective curve used for threshold selection

% Diagnostics
results.svnv_before     % [nunits x 2] - signal/noise variance before denoising
results.svnv_after      % [nunits x 2] - signal/noise variance after denoising
results.gsn_result      % GSN output struct (.cSb, .cNb, ...) for reuse / caching
results.recovery_tradeoff % data behind the recovery-vs-signal-retained figure
```

(See the header of `psn.m` for the complete list of returned fields.)

---

## Common Use Cases

### Example 1: Maximum Signal Retention

```matlab
opt = struct();
opt.basis = 'signal';
opt.criterion = 'variance';
opt.variance_threshold = 0.99;   % keep 99% of signal variance
opt.threshold_method = 'global';
results = psn(data, opt);
```

### Example 2: More Aggressive Denoising

```matlab
opt = struct();
opt.basis = 'difference';        % admits more noise-dominated dimensions
opt.criterion = 'prediction';    % peak of the analytic recovery curve
opt.threshold_method = 'global';
results = psn(data, opt);
```

### Example 3: Per-Unit Thresholds (Hybrid)

```matlab
opt = struct();
opt.threshold_method = 'hybrid'; % shared ordering, unit-specific thresholds
results = psn(data, opt);
unique(results.best_threshold)   % per-unit thresholds
```

### Example 4: Visualize Results

```matlab
opt = struct();
opt.wantfig = true;
opt.wantverbose = true;
results = psn(data, opt);
```

---

## Running Tests

The `tests/` directory contains the MATLAB test suite:

```matlab
addpath(genpath('path/to/PSN/matlab'))

test_psn_all_combinations    % bases x criteria x threshold methods + NaN handling
test_allowable_best_among    % best-among-allowable threshold selection
```

MATLAB↔Python numeric equivalence is checked by the harness scripts under `../tests/`.

---

## Troubleshooting

### "performgsn not found" or "Undefined function 'performgsn'"

**Cause**: the GSN submodule is not initialized.

**Solution**:
```bash
cd PSN
git submodule update --init --recursive
```
Then restart MATLAB.

### Path Issues

`psn.m` adds its own `utilities/` and the GSN submodule to the path. If you call utilities
directly (e.g. in tests), add the whole tree first:
```matlab
addpath(genpath('path/to/PSN/matlab'))
```

### "Data must have at least 2 trials"

Ensure data has shape `[nunits x nconds x ntrials]` with `ntrials >= 2`.

### "Each condition must have at least one trial with valid data across all units"

Some condition has no fully-clean trial. Find all-NaN conditions:
```matlab
for cond = 1:size(data, 2)
    trial_data = data(:, cond, :);
    if all(isnan(trial_data(:)))
        fprintf('Condition %d has all NaN trials\n', cond);
    end
end
```

---

## Performance Tips

1. **Start with `'standard'`** for balanced performance.
2. **Cache GSN** (`gsn_result`) when sweeping hyperparameters on the same data.
3. **Constrain the search** with `allowable_thresholds` if you only care about specific dims.
4. **Disable figures** (`opt.wantfig = false`) for faster batch processing.

---

## See Also

- **Main README**: `../README.md` - overview and Python usage
- **scikit-learn API** (Python): `../SKLEARN_API.md`
- **Examples**: `../examples/`
- **Tests**: `tests/`

---

## Contributing

Found a bug or have a feature request? Please open an issue on [GitHub](https://github.com/jacob-prince/PSN).
