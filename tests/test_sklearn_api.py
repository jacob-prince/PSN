#!/usr/bin/env python3
"""
Quick test script to verify PSN sklearn API implementation.
"""

import numpy as np
import sys
import os

# Add the PSN package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_functionality():
    """Test basic PSN sklearn API functionality."""
    print("Testing basic PSN sklearn API functionality...")
    
    # Import
    from psn import PSN, psn
    print("✓ Import successful")
    
    # Generate test data
    np.random.seed(42)
    data = np.random.randn(10, 20, 3)  # Small test data
    print("✓ Test data generated")
    
    # Test basic instantiation
    denoiser = PSN(verbose=False, wantfig=False)
    print("✓ PSN instantiation successful")
    
    # Test parameter validation
    try:
        invalid_denoiser = PSN(basis='invalid', verbose=False, wantfig=False)
        invalid_denoiser.fit(data)  # This should trigger validation
        assert False, "Parameter validation should have failed"
    except ValueError:
        print("✓ Parameter validation successful")
    
    # Test fitting
    denoiser.fit(data)
    print("✓ Fitting successful")
    
    # Test transform with 3D data
    denoised_3d = denoiser.transform(data)
    assert denoised_3d.shape == data.shape
    print("✓ 3D transform successful")
    
    # Test transform with 2D data (trial-averaged)
    trial_avg = np.mean(data, axis=2)
    denoised_2d = denoiser.transform(trial_avg)
    assert denoised_2d.shape == trial_avg.shape
    print("✓ 2D transform successful")
    
    # Test fit_transform
    denoiser2 = PSN(verbose=False, wantfig=False)
    denoised_fit_transform = denoiser2.fit_transform(data)
    assert denoised_fit_transform.shape == trial_avg.shape
    print("✓ fit_transform successful")
    
    # Test different parameter combinations
    denoiser3 = PSN(basis='pca', cv='population', scoring='r2', 
                   mag_threshold=0.90, verbose=False, wantfig=False)
    denoiser3.fit(data)
    print("✓ Alternative parameters successful")
    
    # Test magnitude thresholding
    denoiser4 = PSN(cv=None, mag_threshold=0.95, verbose=False, wantfig=False)
    denoiser4.fit(data)
    print("✓ Magnitude thresholding successful")
    
    # Test scoring
    score = denoiser.score(data)
    assert isinstance(score, (int, float))
    print("✓ Scoring successful")
    
    # Test get_params and set_params
    params = denoiser.get_params()
    assert isinstance(params, dict)
    denoiser.set_params(verbose=True)
    print("✓ Parameter getting/setting successful")
    
    print("\n🎉 All tests passed! PSN sklearn API is working correctly.")


def test_api_compatibility():
    """Test that both APIs give similar results."""
    print("\nTesting API compatibility...")
    
    from psn import PSN, psn
    
    # Generate test data
    np.random.seed(42)
    data = np.random.randn(15, 30, 4)
    
    # Functional API
    opt = {
        'cv_mode': 0,
        'cv_threshold_per': 'population',
        'denoisingtype': 0
    }
    results_func = psn(data, V=0, opt=opt, wantfig=False)
    
    # Sklearn API
    denoiser = PSN(basis='signal', cv='population', verbose=False, wantfig=False)
    denoiser.fit(data)
    denoised_sklearn = denoiser.transform(np.mean(data, axis=2))
    
    # Compare denoising matrices
    diff = np.mean(np.abs(results_func['denoiser'] - denoiser.denoiser_))
    print(f"Denoiser matrix difference: {diff:.8f}")
    
    # Compare denoised data
    diff_data = np.mean(np.abs(results_func['denoiseddata'] - denoised_sklearn))
    print(f"Denoised data difference: {diff_data:.8f}")
    
    # Assert that results are equivalent or close enough
    assert diff < 1e-4, f"Denoiser matrices differ by {diff}"
    assert diff_data < 1e-4, f"Denoised data differs by {diff_data}"
    
    print("✓ APIs produce equivalent results")


if __name__ == "__main__":
    print("PSN Sklearn API Test Suite")
    print("=" * 40)
    
    success1 = test_basic_functionality()
    success2 = test_api_compatibility()
    
    if success1 and success2:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)
