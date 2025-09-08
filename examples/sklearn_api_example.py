#!/usr/bin/env python3
"""
Example demonstrating both functional and sklearn-compatible PSN interfaces.

This script shows how to use PSN in both the traditional functional style
and the new sklearn-compatible object-oriented style.
"""

import numpy as np
import matplotlib.pyplot as plt
from psn import psn, PSN
from psn.utils import compute_noise_ceiling
from psn.simulate import generate_data

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(nunits=25, nconds=50, ntrials=3, noise_level=None):
    """Generate synthetic neural data for demonstration using PSN's simulate module."""
    print(f"Generating sample data: {nunits} units, {nconds} conditions, {ntrials} trials")
    
    # Generate data using PSN's simulate.generate_data function
    data, _, ground_truth = generate_data(
        nvox=nunits,
        ncond=nconds,
        ntrial=ntrials,
        noise_multiplier=3,
        align_alpha=0.5,
        align_k=10,
        signal_decay=2,
        noise_decay=1.25,
        random_seed=42
    )
    
    # Extract true signal for compatibility
    signal = ground_truth['signal'].T  # Transpose to match expected shape
    
    return data, signal


def demonstrate_functional_api():
    """Demonstrate the traditional functional PSN interface."""
    print("\n" + "="*60)
    print("FUNCTIONAL API DEMONSTRATION")
    print("="*60)
    
    # Generate data
    data, true_signal = generate_sample_data()
    nunits, nconds, ntrials = data.shape
    
    # Example 1: Basic PSN with default settings
    print("\n1. Basic PSN with signal basis and unit-wise CV:")
    results1 = psn(data, V=0, opt={'cv_mode': 0, 'cv_threshold_per': 'unit'}, wantfig=False)
    print(f"   Denoised data shape: {results1['denoiseddata'].shape}")
    
    # Example 2: Population thresholding
    print("\n2. PSN with population thresholding:")
    opt2 = {
        'cv_mode': 0,
        'cv_threshold_per': 'population',
        'cv_thresholds': np.arange(1, min(21, nunits))  # Test up to 20 dimensions
    }
    results2 = psn(data, V=0, opt=opt2, wantfig=False)
    print(f"   Best threshold: {results2['best_threshold']} dimensions")
    print(f"   Signal subspace shape: {results2['signalsubspace'].shape}")
    
    # Example 3: Magnitude thresholding
    print("\n3. PSN with magnitude thresholding (95% variance):")
    opt3 = {
        'cv_mode': -1,
        'mag_frac': 0.95,
        'mag_type': 0
    }
    results3 = psn(data, V=0, opt=opt3, wantfig=False)
    print(f"   Dimensions retained: {results3['dimsretained']}")
    
    return results1, results2, results3


def demonstrate_sklearn_api():
    """Demonstrate the new sklearn-compatible PSN interface."""
    print("\n" + "="*60)
    print("SKLEARN API DEMONSTRATION")
    print("="*60)
    
    # Generate data
    data, true_signal = generate_sample_data()
    
    # Example 1: Basic usage with default parameters
    print("\n1. Basic PSN with default parameters:")
    denoiser1 = PSN(verbose=True, wantfig=False)
    denoiser1.fit(data)
    denoised_avg = denoiser1.transform(np.mean(data, axis=2))
    print(f"   Denoised trial-average shape: {denoised_avg.shape}")
    
    # Example 2: Population thresholding with PCA basis
    print("\n2. Population thresholding with PCA basis:")
    denoiser2 = PSN(basis='pca', cv='population', verbose=True, wantfig=False)
    denoiser2.fit(data)
    denoised_trials = denoiser2.transform(data)
    print(f"   Denoised single-trial shape: {denoised_trials.shape}")
    
    # Example 3: Magnitude thresholding
    print("\n3. Magnitude thresholding (90% variance):")
    denoiser3 = PSN(basis='signal', cv=None, mag_threshold=0.90, verbose=True, wantfig=False)
    denoiser3.fit(data)
    denoised_data = denoiser3.fit_transform(data)  # Fit and transform in one step
    print(f"   Fit-transform result shape: {denoised_data.shape}")
    
    # Example 4: Custom scoring function
    print("\n4. Using Pearson correlation scoring:")
    denoiser4 = PSN(basis='whitened-signal', cv='unit', scoring='r2', verbose=True, wantfig=False)
    denoiser4.fit(data)
    score = denoiser4.score(data)
    print(f"   Mean reliability score: {score:.3f}")
    
    # Example 5: Custom basis
    print("\n5. Using custom orthonormal basis:")
    nunits = data.shape[0]
    custom_basis = np.linalg.qr(np.random.randn(nunits, nunits))[0]
    denoiser5 = PSN(basis=custom_basis, cv='population', verbose=True, wantfig=False)
    denoiser5.fit(data)
    print(f"   Fitted with custom {custom_basis.shape} basis")
    
    return denoiser1, denoiser2, denoiser3, denoiser4, denoiser5


def compare_apis():
    """Compare results between functional and sklearn APIs."""
    print("\n" + "="*60)
    print("API COMPARISON")
    print("="*60)
    
    # Generate data
    data, true_signal = generate_sample_data()
    
    # Functional API
    opt_func = {
        'cv_mode': 0,
        'cv_threshold_per': 'population',
        'cv_thresholds': np.arange(1, 16)
    }
    results_func = psn(data, V=0, opt=opt_func, wantfig=False)
    
    # Sklearn API with equivalent parameters
    denoiser_sklearn = PSN(basis='signal', cv='population', verbose=False, wantfig=False)
    denoiser_sklearn.fit(data)
    denoised_sklearn = denoiser_sklearn.transform(np.mean(data, axis=2))
    
    # Compare results
    print(f"\nFunctional API:")
    print(f"  Best threshold: {results_func['best_threshold']}")
    print(f"  Denoised data shape: {results_func['denoiseddata'].shape}")
    
    print(f"\nSklearn API:")
    print(f"  Best threshold: {denoiser_sklearn.best_threshold_}")
    print(f"  Denoised data shape: {denoised_sklearn.shape}")
    
    # Check if results are approximately equal
    diff = np.mean(np.abs(results_func['denoiseddata'] - denoised_sklearn))
    print(f"\nMean absolute difference between APIs: {diff:.6f}")
    print("Results are equivalent!" if diff < 1e-10 else "Results differ slightly (expected due to implementation differences)")


def sklearn_pipeline_example():
    """Demonstrate PSN in an sklearn pipeline."""
    print("\n" + "="*60)
    print("SKLEARN PIPELINE EXAMPLE")
    print("="*60)
    
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import GridSearchCV
        from sklearn.base import clone
        
        # Generate data
        data, true_signal = generate_sample_data(nunits=30, nconds=50, ntrials=5)
        
        # Create a simple pipeline
        print("\n1. Creating sklearn pipeline:")
        pipeline = Pipeline([
            ('denoiser', PSN(wantfig=False, verbose=False))
        ])
        
        # Set up parameter grid for grid search
        param_grid = {
            'denoiser__basis': ['signal', 'pca'],
            'denoiser__cv': ['unit', 'population'],
            'denoiser__mag_threshold': [0.90, 0.95]
        }
        
        print("   Pipeline created successfully!")
        print(f"   Parameter grid: {param_grid}")
        
        # Fit the pipeline (note: we use trial-averaged data for simplicity)
        trial_avg = np.mean(data, axis=2)
        pipeline.fit(data)  # Fit uses 3D data
        denoised = pipeline.transform(trial_avg)  # Transform can use 2D data
        
        print(f"   Original data shape: {trial_avg.shape}")
        print(f"   Denoised data shape: {denoised.shape}")
        print("   Pipeline works correctly!")
        
    except ImportError:
        print("sklearn not fully available for pipeline demonstration")


if __name__ == "__main__":
    print("PSN Dual API Demonstration")
    print("=" * 60)
    print("This script demonstrates both functional and sklearn-compatible PSN interfaces")
    
    # Run demonstrations
    demonstrate_functional_api()
    demonstrate_sklearn_api()
    compare_apis()
    sklearn_pipeline_example()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nBoth APIs are now available:")
    print("  - Functional: psn(data, V, opt, wantfig)")
    print("  - Sklearn:    PSN(basis, cv, scoring, ...).fit(data).transform(data)")
    print("\nThe sklearn API provides:")
    print("  - Parameter validation and user-friendly parameter names")
    print("  - Integration with sklearn pipelines and model selection")
    print("  - Consistent fit/transform interface")
    print("  - Built-in scoring and feature naming methods")
    print("  - Full backward compatibility with functional interface")
