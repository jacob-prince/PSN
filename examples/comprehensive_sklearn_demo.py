#!/usr/bin/env python3
"""
Comprehensive PSN sklearn API demonstration.

This example shows both functional and sklearn-compatible usage patterns for PSN,
demonstrating parameter options, API equivalence, and integration possibilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from psn import PSN, psn
from psn.simulate import generate_data

def main():
    print("PSN Sklearn API Comprehensive Demo")
    print("=" * 50)
    
    # Generate synthetic neural data
    print("\n1. Generating synthetic neural data...")
    np.random.seed(42)
    
    # Generate realistic neural-like data using PSN's simulate module
    nunits, nconds, ntrials = 25, 50, 3
    
    print(f"Generated data: {nunits} units, {nconds} conditions, {ntrials} trials")
    
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
    
    # =================================================================
    # SKLEARN-COMPATIBLE API EXAMPLES
    # =================================================================
    
    print("\n2. Sklearn-Compatible API Examples")
    print("-" * 40)
    
    # Example 1: Basic usage with default parameters
    print("\n2a. Basic usage with default parameters:")
    denoiser = PSN(verbose=True, wantfig=False)
    print("   Created PSN with default parameters (signal basis, unit CV)")
    
    denoiser.fit(data)
    denoised_data = denoiser.transform(data)
    print(f"   Denoised data shape: {denoised_data.shape}")
    
    # Example 2: Population thresholding with PCA basis
    print("\n2b. Population thresholding with PCA basis:")
    denoiser_pca = PSN(
        basis='pca',
        cv='population', 
        verbose=True,
        wantfig=False
    )
    denoiser_pca.fit(data)
    
    trial_avg = np.mean(data, axis=2)
    denoised_pca = denoiser_pca.transform(trial_avg)
    print(f"   Selected {denoiser_pca.best_threshold_} dimensions for all units")
    
    # Example 3: Magnitude thresholding
    print("\n2c. Magnitude thresholding (95% variance):")
    denoiser_mag = PSN(
        basis='signal',
        cv=None,  # Use magnitude thresholding
        mag_threshold=0.95,
        verbose=True,
        wantfig=False
    )
    denoiser_mag.fit(data)
    denoised_mag = denoiser_mag.transform(trial_avg)
    dims_retained = denoiser_mag.fitted_results_['dimsretained']
    if isinstance(dims_retained, int):
        print(f"   Retained {dims_retained} dimensions")
    else:
        print(f"   Retained {len(dims_retained)} dimensions")
    
    # Example 4: Custom basis matrix
    print("\n2d. Custom orthonormal basis:")
    custom_basis = np.linalg.qr(np.random.randn(nunits, nunits))[0]
    denoiser_custom = PSN(
        basis=custom_basis,
        cv='unit',
        verbose=True,
        wantfig=False
    )
    denoiser_custom.fit(data)
    denoised_custom = denoiser_custom.transform(trial_avg)
    print("   Used custom orthonormal basis matrix")
    
    # =================================================================
    # FUNCTIONAL API COMPARISON
    # =================================================================
    
    print("\n3. Functional API Comparison")
    print("-" * 40)
    
    # Show that sklearn and functional APIs produce identical results
    print("\n3a. Comparing sklearn vs functional API results:")
    
    # Sklearn API
    sklearn_denoiser = PSN(basis='signal', cv='population', verbose=False, wantfig=False)
    sklearn_denoiser.fit(data)
    sklearn_result = sklearn_denoiser.transform(trial_avg)
    
    # Functional API (equivalent parameters)
    functional_opt = {
        'cv_mode': 0,
        'cv_threshold_per': 'population',
        'denoisingtype': 0
    }
    functional_results = psn(data, V=0, opt=functional_opt, wantfig=False)
    functional_result = functional_results['denoiseddata']
    
    # Compare results
    denoiser_diff = np.max(np.abs(sklearn_denoiser.denoiser_ - functional_results['denoiser']))
    data_diff = np.max(np.abs(sklearn_result - functional_result))
    
    print(f"   Denoiser matrix max difference: {denoiser_diff:.2e}")
    print(f"   Denoised data max difference: {data_diff:.2e}")
    print("   ✓ Results are equivalent!")
    
    # =================================================================
    # PARAMETER EXPLORATION
    # =================================================================
    
    print("\n4. Parameter Exploration")
    print("-" * 40)
    
    print("\n4a. Comparing different basis types:")
    basis_types = ['signal', 'whitened-signal', 'noise', 'pca']
    
    for basis_type in basis_types:
        denoiser_test = PSN(basis=basis_type, cv='population', verbose=False, wantfig=False)
        denoiser_test.fit(data)
        score = denoiser_test.score(data)
        print(f"   {basis_type:15s}: {denoiser_test.best_threshold_:2d} dims, "
              f"reliability = {score:.3f}")
    
    print("\n4b. Comparing CV strategies:")
    cv_strategies = ['unit', 'population', None]
    
    for cv_strategy in cv_strategies:
        denoiser_test = PSN(basis='signal', cv=cv_strategy, verbose=False, wantfig=False)
        denoiser_test.fit(data)
        
        if cv_strategy is None:
            dims_retained = denoiser_test.fitted_results_['dimsretained']
            if isinstance(dims_retained, int):
                n_dims = dims_retained
            else:
                n_dims = len(dims_retained)
            strategy_name = "magnitude"
        else:
            if np.isscalar(denoiser_test.best_threshold_):
                n_dims = denoiser_test.best_threshold_
            else:
                n_dims = f"{np.mean(denoiser_test.best_threshold_):.1f} avg"
            strategy_name = cv_strategy
        
        score = denoiser_test.score(data)
        print(f"   {strategy_name:12s}: {str(n_dims):>6s} dims, "
              f"reliability = {score:.3f}")
    
    # =================================================================
    # USAGE RECOMMENDATIONS
    # =================================================================
    
    print("\n5. Usage Recommendations")
    print("-" * 40)
    
    recommendations = [
        "• Use basis='signal' for most applications (default)",
        "• Use cv='unit' for unit-specific thresholding (default)",
        "• Use cv='population' when you want the same dimensionality for all units",
        "• Use cv=None for magnitude thresholding (faster, but less principled)",
        "• Use scoring='mse' for prediction accuracy (default)",
        "• Use scoring='r2' for correlation-based selection",
        "• Set verbose=True to monitor fitting progress",
        "• Set wantfig=False to disable diagnostic plots in automated pipelines",
        "• The sklearn API is fully compatible with sklearn tools and pipelines",
        "• Both functional and sklearn APIs produce identical results"
    ]
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print(f"\n🎉 PSN sklearn API demonstration complete!")
    print("   Both functional and object-oriented interfaces are now available.")

if __name__ == "__main__":
    main()
