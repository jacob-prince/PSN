# PSN MATLAB Implementation

This directory contains the MATLAB implementation of **Partitioning Signal and Noise (PSN)**, a method for denoising multi-trial neural data.

---

## Quick Start

### Installation

PSN requires the [GSN (Generative Modeling of Signal and Noise)](https://github.com/cvnlab/GLMsingle) library as a dependency. Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/jacob-prince/PSN.git
```

If you've already cloned without submodules:
```bash
cd PSN
git submodule update --init --recursive
```

### Verify Installation

Open MATLAB and run the test scripts:

```matlab
cd('path/to/PSN/matlab')

% Test GSN dependency
test_gsn_dependency

% Test PSN functionality
test_psn
```

If both tests pass, PSN is ready to use!

---

## Basic Usage

### Minimal Example

```matlab
% Generate test data (50 units, 100 conditions, 5 trials)
data = randn(50, 100, 5);

% Apply PSN with default settings
results = psn(data);

% Access denoised data
denoised = results.denoiseddata;  % [nunits x nconds]
fprintf('Denoised data shape: [%d x %d]\n', size(denoised));
```

### Using Presets

PSN provides three convenience presets:

```matlab
% CONSERVATIVE: Prioritizes retaining signal
% - Uses signal basis
% - Variance criterion (retains 99% of signal variance)
% - Global thresholds (symmetric denoiser)
results = psn(data, 'conservative');

% STANDARD: Balances signal retention and out-of-sample generalization (default)
% - Uses signal basis
% - Prediction criterion (maximizes signal - noise/ntrials)
% - Hybrid thresholds (global ordering, unit-specific thresholds)
results = psn(data, 'standard');

% AGGRESSIVE: Maximizes out-of-sample generalization
% - Uses difference basis (cSb - cNb/ntrials)
% - Prediction criterion
% - Unit-specific ordering and thresholds
results = psn(data, 'aggressive');
```

---

## Custom Configuration

### Parameter Structure

Create a struct to customize PSN behavior:

```matlab
opt = struct();

% Basis selection
opt.basis = 'signal';              % 'signal', 'difference', 'pca', or custom matrix

% Threshold criterion
opt.criterion = 'prediction';      % 'prediction', 'variance', 'variance_eigenvalues'

% Threshold method
opt.threshold_method = 'hybrid';   % 'global', 'hybrid', 'unit'

% Variance threshold (for criterion='variance')
opt.variance_threshold = 0.95;     % Retain 95% of signal variance

% Verbosity and visualization
opt.wantverbose = true;            % Print progress messages
opt.wantfig = true;                % Display diagnostic figures

% Run PSN
results = psn(data, opt);
```

### Key Parameters

#### **basis** - Basis for dimensionality reduction
- `'signal'` (default): Eigenvectors of signal covariance (cSb)
- `'difference'`: Eigenvectors of cSb - cNb/ntrials_avg (better for noisy data)
- `'pca'`: Eigenvectors of trial-averaged data covariance (not recommended)
- `B` (matrix): Custom orthonormal basis [nunits x D]

#### **criterion** - How to determine the threshold
- `'prediction'` (default): Maximize cumulative signal - noise/ntrials (out-of-sample generalization)
- `'variance'`: Retain dimensions until target fraction of signal variance is reached
- `'variance_eigenvalues'`: Retain until target fraction of basis eigenvalues is reached

#### **threshold_method** - How to apply thresholds
- `'global'`: Single threshold for all units (symmetric denoiser)
- `'hybrid'` (default): Global ordering, unit-specific thresholds
- `'unit'`: Unit-specific ordering and thresholds (most flexible, potentially unstable)

#### **variance_threshold** - Target variance fraction
- Default: `0.99` (retain 99% of variance)
- Range: `0.0` to `1.0`
- Only used when `criterion='variance'` or `'variance_eigenvalues'`

---

## Advanced Features

### Unit Groups

Group units to share the same threshold:

```matlab
% Example: Group units into 3 populations
unit_groups = [1 1 1 2 2 2 3 3 3 3];  % [nunits x 1]

opt = struct();
opt.threshold_method = 'hybrid';  % or 'unit'
opt.unit_groups = unit_groups;

results = psn(data, opt);
```

Units with the same group ID will receive the same threshold (averaged across the group).

### Custom Allowable Thresholds

Constrain which thresholds PSN can use:

```matlab
opt = struct();
opt.allowable_thresholds = [1, 5, 10, 15];  % Only these thresholds allowed

results = psn(data, opt);
```

If the optimal threshold is not in the list, PSN will round to the nearest allowed value.

To force a specific threshold:
```matlab
opt.allowable_thresholds = 10;  % Force exactly 10 dimensions
```

### Basis Ordering

Control how basis vectors are initially ordered:

```matlab
opt = struct();
opt.basis_ordering = 'eigenvalues';     % Use eigenvalue magnitudes (default)
% opt.basis_ordering = 'signalvariance'; % Measure signal variance empirically

results = psn(data, opt);
```

---

## Handling Missing Data (NaNs)

PSN supports uneven trials across conditions:

```matlab
% Example: Some conditions have fewer trials
data = randn(10, 50, 5);
data(:, 1, 4:5) = NaN;  % Condition 1 only has 3 trials
data(:, 2, 5) = NaN;    % Condition 2 only has 4 trials

% PSN will handle this automatically
results = psn(data);
```

**Requirements:**
- Each condition must have at least one trial with valid data across **all units**
- PSN will compute average number of trials and use it in noise/ntrials formulas
- Denoised output will NOT contain NaNs (filled in based on available data)
- Residuals will preserve NaN positions from input

---

## Output Structure

The `results` struct contains:

```matlab
results = psn(data, opt);

% Primary outputs
results.denoiseddata    % [nunits x nconds] - Denoised estimates
results.residuals       % [nunits x nconds x ntrials] - data - denoiseddata

% Diagnostics
results.thresholds      % [nunits x 1] - Number of dimensions retained per unit
results.basis           % [nunits x D] - Basis vectors used
results.signal_cov      % [nunits x nunits] - Estimated signal covariance
results.noise_cov       % [nunits x nunits] - Estimated noise covariance

% GSN outputs (from performgsn.m)
results.gsn             % Full GSN results struct
```

---

## Common Use Cases

### Example 1: Maximum Signal Retention

```matlab
% Retain as much signal as possible (conservative denoising)
opt = struct();
opt.basis = 'signal';
opt.criterion = 'variance';
opt.variance_threshold = 0.99;  % Keep 99% of signal variance
opt.threshold_method = 'global';

results = psn(data, opt);
```

### Example 2: Maximize Cross-Validation Performance

```matlab
% Optimize for out-of-sample generalization
opt = struct();
opt.basis = 'difference';       % Better for noisy data
opt.criterion = 'prediction';   % Maximize signal - noise/ntrials
opt.threshold_method = 'unit';  % Adapt to each unit

results = psn(data, opt);
```

### Example 3: Symmetric Denoiser (Same Transform for All Units)

```matlab
% Use same threshold across all units (symmetric transform)
opt = struct();
opt.threshold_method = 'global';

results = psn(data, opt);

% All units will use the same number of dimensions
unique(results.thresholds)  % Should be a single value
```

### Example 4: Visualize Results

```matlab
% Enable figures to see diagnostic plots
opt = struct();
opt.wantfig = true;
opt.wantverbose = true;

results = psn(data, opt);

% Diagnostic figures will show:
% - Signal and noise eigenspectra
% - Selected thresholds per unit
% - Signal-to-noise ratio per dimension
```

---

## Running Tests

The `tests/` directory contains comprehensive unit tests:

```matlab
cd('path/to/PSN/matlab')

% Run all tests
test_psn                          % Main PSN functionality
test_gsn_dependency               % Verify GSN is working
test_difference_basis_properties  % Test difference basis properties
test_pca_basis                    % Test PCA basis
test_threshold_methods            % Test global/hybrid/unit thresholds
test_heterogeneous_thresholds     % Test unit groups

% Numeric equivalence with Python
test_simulate_equivalence         % Compare data simulation with Python
```

---

## Troubleshooting

### "performgsn not found" or "Undefined function 'performgsn'"

**Cause**: The GSN submodule is not initialized.

**Solution**:
```bash
cd PSN
git submodule update --init --recursive
```

Then restart MATLAB.

### Path Issues

Ensure MATLAB's working directory is `PSN/matlab`:

```matlab
pwd  % Check current directory
cd('path/to/PSN/matlab')
```

### Dimension Errors

**Error**: "Data must have at least 2 trials"

**Cause**: `data` has shape `[nunits x nconds x 1]` or missing third dimension.

**Solution**: Ensure data has `ntrials >= 2`:
```matlab
size(data)  % Should be [nunits x nconds x ntrials] with ntrials >= 2
```

### NaN Handling Errors

**Error**: "Each condition must have at least one trial with valid data across all units"

**Cause**: Some conditions only have NaN trials, or trials with partial NaN units.

**Solution**: Check for conditions where all trials are NaN:
```matlab
% Find problematic conditions
for cond = 1:size(data, 2)
    trial_data = data(:, cond, :);
    if all(isnan(trial_data(:)))
        fprintf('Condition %d has all NaN trials\n', cond);
    end
end
```

---

## Performance Tips

1. **Use appropriate preset**: Start with `'standard'` for balanced performance
2. **Consider data size**: `'unit'` threshold method is more flexible but slower for large datasets
3. **Limit dimensions**: Use `allowable_thresholds` to constrain search space if needed
4. **Disable figures**: Set `opt.wantfig = false` for faster execution in batch processing

---

## See Also

- **Main README**: `../README.md` - Overview and Python usage
- **Python API**: `../SKLEARN_API.md` - Scikit-learn compatible interface
- **Examples**: `../examples/` - Python usage examples
- **Tests**: `tests/` - Comprehensive MATLAB test suite

---

## Contributing

Found a bug or have a feature request? Please open an issue on [GitHub](https://github.com/jacob-prince/PSN).
