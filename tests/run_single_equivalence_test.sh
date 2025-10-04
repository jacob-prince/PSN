#!/bin/bash

# run_single_equivalence_test.sh
#
# Simple wrapper to run a single PSN equivalence test
#
# Usage: ./run_single_equivalence_test.sh <test_number>
# Example: ./run_single_equivalence_test.sh 10

set -e

TEST_NUM="$1"

if [[ -z "$TEST_NUM" ]]; then
    echo "Usage: $0 <test_number>"
    echo "Example: $0 10  # Run test 10 (truncate)"
    exit 1
fi

# Source the main test script to get access to functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSN_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_DIR="$PSN_ROOT/psn"
MATLAB_DIR="$PSN_ROOT/matlab"
TEST_DATA_DIR="$SCRIPT_DIR/psn_equivalence_test_data"

# MATLAB configuration
MATLAB_PATH="/Applications/MATLAB_R2022b.app/bin/matlab"
PYTHON_CMD="python3"
TOLERANCE="1e-10"
MIN_CORRELATION="0.999999"

mkdir -p "$TEST_DATA_DIR"

# Source the run_psn_equivalence_test function from the main script
# Extract just the function definitions
source <(sed -n '/^run_psn_equivalence_test()/,/^}/p' "$SCRIPT_DIR/test_psn_matlab_python_equivalence.sh")

# Run the requested test
case $TEST_NUM in
    1)
        echo "=== Test 1: Signal covariance basis, unit-wise CV ==="
        run_psn_equivalence_test "test1_signal_unit_cv" 20 50 3 0 0 "unit" 0 0.95
        ;;
    2)
        echo "=== Test 2: Signal covariance basis, population CV ==="
        run_psn_equivalence_test "test2_signal_pop_cv" 20 50 3 0 0 "population" 0 0.95
        ;;
    3)
        echo "=== Test 3: Signal covariance basis, magnitude thresholding ==="
        run_psn_equivalence_test "test3_signal_mag" 20 50 3 0 -1 "population" 0 0.9
        ;;
    4)
        echo "=== Test 4: Transformed signal covariance basis ==="
        run_psn_equivalence_test "test4_transformed_signal" 15 40 3 1 0 "unit" 0 0.95
        ;;
    5)
        echo "=== Test 5: Noise covariance basis ==="
        run_psn_equivalence_test "test5_noise_basis" 15 40 3 2 0 "population" 0 0.95
        ;;
    6)
        echo "=== Test 6: PCA basis ==="
        run_psn_equivalence_test "test6_pca_basis" 15 40 3 3 0 "unit" 0 0.95
        ;;
    7)
        echo "=== Test 7: Random basis ==="
        run_psn_equivalence_test "test7_random_basis" 15 40 3 "shared_random_basis" -1 "population" 0 0.8
        ;;
    8)
        echo "=== Test 8: Single-trial denoising ==="
        run_psn_equivalence_test "test8_single_trial" 15 30 4 0 1 "population" 1 0.95
        ;;
    9)
        echo "=== Test 9: Small dataset edge case ==="
        run_psn_equivalence_test "test9_small_data" 5 10 2 0 0 "population" 0 0.95
        ;;
    10)
        echo "=== Test 10: Truncate functionality ==="
        run_psn_equivalence_test "test10_truncate" 15 30 4 0 0 "population" 0 0.95 2
        ;;
    *)
        echo "Error: Invalid test number $TEST_NUM. Must be 1-10."
        exit 1
        ;;
esac

echo ""
echo "Test $TEST_NUM completed!"
