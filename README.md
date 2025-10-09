**psn** is a library for Partitioning Signal and Noise. 

Note: as of Sept. 2025, PSN is still under development - the API and the algorithm are both subject to change.

# 🚀 Getting Started with PSN

## Python Installation

### Install and Test

Optionally create a conda environment:

```bash
conda create -n psn python=3.9
conda activate psn
```

Clone the repository and install PSN:

```bash
git clone https://github.com/jacob-prince/PSN.git
cd PSN
pip install -e .
```

Test your installation:

```python
import numpy as np
from psn import PSN

# Generate test data (10 units, 25 conditions, 3 trials)
np.random.seed(42)
data = np.random.randn(10, 25, 3)

# Apply PSN denoising using sklearn-style API
denoiser = PSN()
denoised_data = denoiser.fit_transform(data)

print("PSN installation successful!")
print(f"Original shape: {data.shape}")
print(f"Denoised shape: {denoised_data.shape}")
```

### Explore Examples

Run the comprehensive examples:

```bash
python examples/sklearn_api_demo.py
```

## MATLAB Installation

### Clone and Setup

Clone with submodules for the required GSN dependency:

```bash
git clone --recurse-submodules https://github.com/jacob-prince/PSN.git
```

Or if already cloned:
```bash
cd PSN
git submodule update --init --recursive
```

### Test Installation

Open MATLAB and verify everything works:

```matlab
cd('path/to/PSN/matlab')

% Verify GSN dependency
test_gsn_dependency

% Test PSN functionality  
test_psn
```

You should see successful completion messages for both tests.

### Quick Example

```matlab
% Generate test data and run PSN
data = randn(50, 100, 5);  % 50 units, 100 conditions, 5 trials
results = psn(data);
fprintf('PSN completed! Denoised data shape: [%d x %d]\n', size(results.denoiseddata));
```

### Troubleshooting

- **"performgsn not found"**: Run `git submodule update --init --recursive`
- **Path issues**: Ensure MATLAB is in the correct directory
- See `matlab/README.md` for detailed documentation
