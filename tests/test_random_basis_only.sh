#!/bin/bash

# Test only the random basis functionality
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSN_ROOT="$SCRIPT_DIR"
MATLAB_PATH="/Applications/MATLAB_R2022b.app/bin/matlab"
PYTHON_CMD="python3"
TEST_DATA_DIR="$SCRIPT_DIR/psn_equivalence_test_data"

# Create test data directory
mkdir -p "$TEST_DATA_DIR"

echo "Testing random basis functionality..."

# Function to run random basis test
run_random_basis_test() {
    local test_name="test7_random_basis"
    local nvox=15
    local ncond=40
    local ntrial=3
    local V_param="shared_random_basis"
    local cv_mode=-1
    local cv_threshold_per="population"
    local denoisingtype=0
    local mag_frac=0.8
    
    echo "=========================================="
    echo "Testing: $test_name"
    echo "Parameters: $nvox voxels, $ncond conditions, $ntrial trials"
    echo "V=$V_param, cv_mode=$cv_mode, cv_threshold_per=$cv_threshold_per"
    echo "denoisingtype=$denoisingtype, mag_frac=$mag_frac"
    echo "=========================================="
    
    # Generate shared random basis first
    echo "Generating shared random basis..."
    cat > "$TEST_DATA_DIR/generate_shared_random_basis.py" << EOF
import numpy as np
import scipy.io

# Generate deterministic random basis using QR decomposition
# This will be shared between Python and MATLAB implementations

# Set deterministic seed
np.random.seed(123)

# Generate random matrix and orthogonalize
nvox = $nvox
random_matrix = np.random.randn(nvox, nvox)
Q, R = np.linalg.qr(random_matrix)

# Make sure we have a proper orthonormal basis
assert np.allclose(Q @ Q.T, np.eye(nvox), atol=1e-12), "Basis is not orthonormal"

print(f"Generated shared random basis with shape: {Q.shape}")
print(f"Orthogonality check: max(|Q*Q^T - I|) = {np.max(np.abs(Q @ Q.T - np.eye(nvox))):.2e}")

# Save for both Python and MATLAB
np.save('$TEST_DATA_DIR/shared_random_basis.npy', Q)
scipy.io.savemat('$TEST_DATA_DIR/shared_random_basis.mat', {'shared_basis': Q})

print("Shared random basis saved successfully")
EOF

    $PYTHON_CMD "$TEST_DATA_DIR/generate_shared_random_basis.py"
    
    # Generate test data with Python
    cat > "$TEST_DATA_DIR/generate_${test_name}_data.py" << EOF
import numpy as np
import scipy.io
import sys
import os

# Add PSN to path
sys.path.insert(0, '$PSN_ROOT')
from psn.simulate import generate_data

# Set random seed for reproducibility
np.random.seed(42)

# Generate simulated data
print("Generating simulated data...")
print(f"Shape: {$nvox} voxels × {$ncond} conditions × {$ntrial} trials")

train_data, test_data, ground_truth = generate_data(
    nvox=$nvox,
    ncond=$ncond, 
    ntrial=$ntrial,
    signal_decay=1.0,
    noise_decay=1.0,
    noise_multiplier=1.0,
    align_alpha=0.5,
    align_k=min(5, $nvox//2),
    random_seed=42,
    want_fig=False
)

print(f"Generated data shape: {train_data.shape}")
print(f"Data range: [{np.min(train_data):.6f}, {np.max(train_data):.6f}]")
print(f"Data mean: {np.mean(train_data):.6f}")
print(f"Data std: {np.std(train_data):.6f}")

# Check for NaN or inf values
if not np.isfinite(train_data).all():
    raise ValueError("Generated data contains NaN or Inf values")

# Save data for both Python and MATLAB
np.save('$TEST_DATA_DIR/${test_name}_data.npy', train_data)
scipy.io.savemat('$TEST_DATA_DIR/${test_name}_data.mat', {'data': train_data})

print("Test data generated and saved successfully")
EOF

    $PYTHON_CMD "$TEST_DATA_DIR/generate_${test_name}_data.py"
    
    # Run Python PSN
    cat > "$TEST_DATA_DIR/run_python_${test_name}.py" << EOF
import numpy as np
import scipy.io
import sys
import os
import time

# Add PSN to path
sys.path.insert(0, '$PSN_ROOT')
from psn.psn import psn

# Set random seed for reproducibility
np.random.seed(42)

# Load data
data = np.load('$TEST_DATA_DIR/${test_name}_data.npy')
print(f"Running Python PSN on $test_name data...")
print(f"Data shape: {data.shape}")

# Load shared random basis
shared_basis = np.load('$TEST_DATA_DIR/shared_random_basis.npy')
print(f"Loaded shared random basis with shape: {shared_basis.shape}")
V_param = shared_basis

# Set options
opt = {
    'cv_mode': $cv_mode,
    'cv_threshold_per': '$cv_threshold_per',
    'denoisingtype': $denoisingtype,
    'mag_frac': $mag_frac
}

# Run PSN with timing
start_time = time.time()
results = psn(data, V=V_param, opt=opt, wantfig=False)
end_time = time.time()

print(f"Python computation completed in {end_time - start_time:.2f} seconds")

# Save results (exclude non-serializable items like function handles)
serializable_results = {}
for key, value in results.items():
    if not callable(value):  # Exclude function handles
        serializable_results[key] = value

np.save('$TEST_DATA_DIR/${test_name}_python_results.npy', serializable_results)

# Convert numpy arrays to MATLAB-compatible format for comparison
matlab_results = {}
for key, value in serializable_results.items():
    if isinstance(value, np.ndarray):
        matlab_results[key] = value
    elif isinstance(value, (int, float, np.integer, np.floating)):
        matlab_results[key] = np.array([[value]])  # Scalar as 1x1 matrix
    elif isinstance(value, dict):
        # Skip nested dictionaries as they're hard to compare
        print(f"Warning: Skipping dictionary field {key} for MATLAB comparison")
        continue
    elif value is not None:
        try:
            matlab_results[key] = np.array(value)
        except:
            print(f"Warning: Could not convert {key} to MATLAB format")

scipy.io.savemat('$TEST_DATA_DIR/${test_name}_python_results.mat', matlab_results)

# Print key results info
print(f"Results summary:")
print(f"Available fields: {list(serializable_results.keys())}")
if 'denoiser' in serializable_results and serializable_results['denoiser'] is not None:
    print(f"  denoiser shape: {serializable_results['denoiser'].shape}")
if 'denoiseddata' in serializable_results and serializable_results['denoiseddata'] is not None:
    print(f"  denoiseddata shape: {serializable_results['denoiseddata'].shape}")
if 'best_threshold' in serializable_results and serializable_results['best_threshold'] is not None:
    if np.isscalar(serializable_results['best_threshold']):
        print(f"  best_threshold: {serializable_results['best_threshold']}")
    else:
        print(f"  best_threshold shape: {serializable_results['best_threshold'].shape}")
        print(f"  best_threshold range: [{np.min(serializable_results['best_threshold'])}, {np.max(serializable_results['best_threshold'])}]")

print("Python results saved successfully")
EOF

    $PYTHON_CMD "$TEST_DATA_DIR/run_python_${test_name}.py"
    
    # Run MATLAB PSN
    cat > "$TEST_DATA_DIR/run_matlab_${test_name}.m" << 'EOF'
try
    % Add PSN matlab directory to path
    addpath('$MATLAB_DIR');
    
    % Set random seed for reproducibility
    rng('default');
    rng(42, 'twister');

    % Load data
    load('$TEST_DATA_DIR/${test_name}_data.mat');
    fprintf('Running MATLAB PSN on ${test_name} data...\n');
    fprintf('Data shape: [%d, %d, %d]\n', size(data));

    % Load shared random basis
    shared_basis_data = load('$TEST_DATA_DIR/shared_random_basis.mat');
    shared_basis = shared_basis_data.shared_basis;
    fprintf('Loaded shared random basis with shape: [%d, %d]\n', size(shared_basis));
    V_param = shared_basis;

    % Set options
    opt = struct();
    opt.cv_mode = $cv_mode;
    opt.cv_threshold_per = '$cv_threshold_per';
    opt.denoisingtype = $denoisingtype;
    opt.mag_frac = $mag_frac;

    % Run PSN with timing
    tic;
    results = psn(data, V_param, opt, false);  % wantfig = false
    elapsed = toc;
    
    fprintf('MATLAB computation completed in %.2f seconds\n', elapsed);

    % Save results
    save('$TEST_DATA_DIR/${test_name}_matlab_results.mat', '-struct', 'results');

    % Print key results info
    fprintf('Results summary:\n');
    if isfield(results, 'denoiser') && ~isempty(results.denoiser)
        fprintf('  denoiser shape: [%d, %d]\n', size(results.denoiser));
    end
    if isfield(results, 'denoiseddata') && ~isempty(results.denoiseddata)
        fprintf('  denoiseddata shape: [%s]\n', num2str(size(results.denoiseddata)));
    end
    if isfield(results, 'best_threshold') && ~isempty(results.best_threshold)
        if isscalar(results.best_threshold)
            fprintf('  best_threshold: %d\n', results.best_threshold);
        else
            fprintf('  best_threshold shape: [%s]\n', num2str(size(results.best_threshold)));
            fprintf('  best_threshold range: [%d, %d]\n', min(results.best_threshold), max(results.best_threshold));
        end
    end

    fprintf('MATLAB results saved successfully\n');
    
    % Success
    exit(0);
catch ME
    fprintf('Error in MATLAB PSN:\n');
    fprintf('Message: %s\n', ME.message);
    fprintf('Identifier: %s\n', ME.identifier);
    for i = 1:length(ME.stack)
        fprintf('  File: %s, Function: %s, Line: %d\n', ...
                ME.stack(i).file, ME.stack(i).name, ME.stack(i).line);
    end
    exit(1);
end
EOF

    # Replace placeholders in MATLAB script
    sed -i.bak "s|\$MATLAB_DIR|$PSN_ROOT/matlab|g; s|\$TEST_DATA_DIR|$TEST_DATA_DIR|g; s|\$cv_mode|$cv_mode|g; s|\$cv_threshold_per|$cv_threshold_per|g; s|\$denoisingtype|$denoisingtype|g; s|\$mag_frac|$mag_frac|g; s|\${test_name}|$test_name|g" "$TEST_DATA_DIR/run_matlab_${test_name}.m"
    
    "$MATLAB_PATH" -nosplash -nodesktop -r "try; run('$TEST_DATA_DIR/run_matlab_${test_name}.m'); catch ME; disp('Error running MATLAB:'); disp(ME.message); disp(ME.stack); exit(1); end"
    
    # Create comparison script
    cat > "$TEST_DATA_DIR/compare_${test_name}.py" << EOF
import numpy as np
import scipy.io
import sys

# Paths from shell script
test_data_dir = "$TEST_DATA_DIR"
test_name = "$test_name"

# Load results
try:
    python_results = scipy.io.loadmat(f'{test_data_dir}/{test_name}_python_results.mat')
    matlab_results = scipy.io.loadmat(f'{test_data_dir}/{test_name}_matlab_results.mat')
except FileNotFoundError as e:
    print(f"Error loading results files: {e}")
    sys.exit(1)

# Filter out MATLAB internal fields
matlab_internal_fields = ['__header__', '__version__', '__globals__']
for field in matlab_internal_fields:
    python_results.pop(field, None)
    matlab_results.pop(field, None)

print("Python fields:", sorted(python_results.keys()))
print("MATLAB fields:", sorted(matlab_results.keys()))
print("")

# Compare core fields
tolerance = 1e-9 # Increased tolerance slightly for floating point differences
min_correlation = 0.999999
all_passed = True

fields_to_compare = ['denoiser', 'denoiseddata', 'best_threshold']

for field in fields_to_compare:
    if field in python_results and field in matlab_results:
        py_val = python_results[field].squeeze()
        mat_val = matlab_results[field].squeeze()

        print(f"Comparing {field}:")
        print(f"  Python shape: {py_val.shape}, MATLAB shape: {mat_val.shape}")
        print(f"  Python dtype: {py_val.dtype}, MATLAB dtype: {mat_val.dtype}")
        
        if field == 'best_threshold':
            print("  Adjusting for MATLAB's 1-based indexing by subtracting 1 from MATLAB results.")
            mat_val = mat_val - 1

        # Squeeze might result in 0-dim array for scalars
        if py_val.ndim > 0:
            print(f"  Python range: [{np.min(py_val):.6f}, {np.max(py_val):.6f}]")
        else:
            print(f"  Python value: {py_val}")
        if mat_val.ndim > 0:
            print(f"  MATLAB range: [{np.min(mat_val):.6f}, {np.max(mat_val):.6f}]")
        else:
            print(f"  MATLAB value: {mat_val}")


        if py_val.shape != mat_val.shape:
            print(f"  ✗ FAIL - Shape mismatch")
            all_passed = False
            continue

        max_diff = np.max(np.abs(py_val - mat_val))
        print(f"  Max absolute difference: {max_diff:.2e}")

        if max_diff < tolerance:
            print(f"  ✓ PASS - Within tolerance")
        else:
            print(f"  ✗ FAIL - Exceeds tolerance")
            all_passed = False
            # Show some actual values for debugging
            if py_val.size <= 10:
                print(f"    Python: {py_val}")
                print(f"    MATLAB: {mat_val}")
            else:
                # compare correlation for larger matrices
                if py_val.ndim > 1:
                    py_flat = py_val.flatten()
                    mat_flat = mat_val.flatten()
                else:
                    py_flat = py_val
                    mat_flat = mat_val
                
                if len(py_flat) == len(mat_flat) and np.std(py_flat) > 0 and np.std(mat_flat) > 0:
                    corr = np.corrcoef(py_flat, mat_flat)[0, 1]
                    print(f"  Correlation: {corr:.6f}")
                    if corr < min_correlation:
                        print(f"  ✗ FAIL - Correlation below threshold")
                    else:
                        print(f"  ✓ PASS - Correlation is high")

        print("")
    else:
        print(f"Field {field} missing in one implementation (Python: {field in python_results}, MATLAB: {field in matlab_results})")
        all_passed = False
        print("")

if all_passed:
    print("Comparison PASSED")
    sys.exit(0)
else:
    print("Comparison FAILED")
    sys.exit(1)
EOF

    # Compare results
    $PYTHON_CMD "$TEST_DATA_DIR/compare_${test_name}.py"
}

# Run the test
if run_random_basis_test; then
    echo "SUCCESS: Random basis test passed!"
    exit 0
else
    echo "FAILURE: Random basis test failed!"
    exit 1
fi
