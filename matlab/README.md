# PSN MATLAB Implementation

This directory contains the MATLAB implementation of the Partitioning Signal and Noise (PSN) algorithm.

## Dependencies

This MATLAB implementation depends on the GSN (Generative modeling of Signal and Noise) package for the `performgsn` function. The dependency is managed via git submodule.

## Installation

1. **Clone with submodules** (if cloning fresh):
   ```bash
   git clone --recurse-submodules https://github.com/youruser/psn.git
   ```

2. **Initialize submodules** (if already cloned):
   ```bash
   git submodule update --init --recursive
   ```

3. **Verify installation**:
   Open MATLAB and run:
   ```matlab
   cd path/to/psn/matlab
   test_gsn_dependency
   ```

## Usage

The main function is `psn()` which provides an identical interface to the Python PSN implementation:

```matlab
% Basic usage
data = randn(50, 100, 5);  % 50 units, 100 conditions, 5 trials
results = psn(data);

% With options
opt = struct();
opt.cv_mode = 0;  % Leave-one-out cross-validation
opt.cv_threshold_per = 'unit';  % Unit-wise thresholding
opt.denoisingtype = 1;  % Single-trial denoising
results = psn(data, 0, opt);
```

## File Structure

- `psn.m` - Main PSN denoising function
- `test_gsn_dependency.m` - Test script to verify GSN dependency
- `../external/gsn/` - GSN submodule (contains `performgsn.m`)

## Troubleshooting

If you get an error about missing `performgsn`, ensure:
1. Git submodules are properly initialized
2. The GSN submodule is present at `../external/gsn/matlab/`
3. MATLAB can access the file system location

## Algorithm Details

The PSN algorithm works by:
1. Estimating signal and noise covariance matrices using GSN
2. Computing basis vectors (eigenvectors of signal/noise covariances)
3. Selecting optimal number of dimensions via cross-validation or magnitude thresholding
4. Applying denoising matrix to reconstruct cleaner neural responses

For detailed documentation, see the docstring in `psn.m`.
