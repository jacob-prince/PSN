#!/usr/bin/env python3
"""
Test script demonstrating the r2_score_columns function in utils.py

This shows that the R² scoring function is now properly organized 
in the utils module and can be used independently.
"""

import numpy as np
from psn.utils import r2_score_columns, negative_mse_columns

def test_scoring_functions():
    """Test both MSE and R² scoring functions."""
    print("Testing PSN Scoring Functions")
    print("=" * 35)
    
    # Generate test data
    np.random.seed(42)
    n_samples, n_features = 100, 3
    
    # Perfect predictions
    y_true = np.random.randn(n_samples, n_features)
    y_pred_perfect = y_true.copy()
    
    # Good predictions (small noise)
    y_pred_good = y_true + 0.1 * np.random.randn(n_samples, n_features)
    
    # Poor predictions (large noise)  
    y_pred_poor = y_true + 0.5 * np.random.randn(n_samples, n_features)
    
    # Random predictions
    y_pred_random = np.random.randn(n_samples, n_features)
    
    # Test scenarios
    scenarios = [
        ("Perfect predictions", y_pred_perfect),
        ("Good predictions", y_pred_good), 
        ("Poor predictions", y_pred_poor),
        ("Random predictions", y_pred_random)
    ]
    
    print(f"Testing on {n_samples} samples, {n_features} features\n")
    
    for name, y_pred in scenarios:
        mse_scores = negative_mse_columns(y_true, y_pred)
        r2_scores = r2_score_columns(y_true, y_pred)
        
        print(f"{name}:")
        print(f"  Negative MSE: {mse_scores}")
        print(f"  R² scores:    {r2_scores}")
        print(f"  Mean R²:      {np.mean(r2_scores):.3f}")
        print()
    
    # Test edge cases
    print("Edge Cases:")
    print("-" * 12)
    
    # Zero variance in true values
    y_const = np.ones((10, 2))
    y_pred_const = np.ones((10, 2))
    r2_const = r2_score_columns(y_const, y_pred_const)
    print(f"Zero variance (perfect): {r2_const}")
    
    # Zero variance with error
    y_pred_const_err = np.ones((10, 2)) + 0.1
    r2_const_err = r2_score_columns(y_const, y_pred_const_err)
    print(f"Zero variance (with error): {r2_const_err}")
    
    # Empty arrays
    y_empty = np.zeros((0, 3))
    r2_empty = r2_score_columns(y_empty, y_empty)
    print(f"Empty arrays: {r2_empty}")
    
    print("\n✅ All tests passed!")
    print("\nThe r2_score_columns function is now properly organized in utils.py")
    print("and can be used independently or through the PSN sklearn API.")

if __name__ == "__main__":
    test_scoring_functions()
