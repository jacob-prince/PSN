#!/usr/bin/env python3
"""
Demonstration of PSN sklearn-compatible API

This script shows how to use the new PSN class with sklearn-style fit/transform
methods alongside the existing functional API.
"""

import numpy as np
import matplotlib.pyplot as plt
from psn import PSN, psn

def main():
    # Generate synthetic neural data
    np.random.seed(42)
    nunits, nconds, ntrials = 50, 100, 8
    
    # Create data with some signal structure
    signal = np.random.randn(nunits, nconds)
    noise = np.random.randn(nunits, nconds, ntrials) * 0.5
    data = signal[:, :, np.newaxis] + noise
    
    print("PSN Sklearn API Demonstration")
    print("=" * 50)
    print(f"Data shape: {data.shape} (units, conditions, trials)")
    
    # 1. Basic usage with default parameters
    print("\n1. Basic usage (signal basis, unit CV):")
    denoiser = PSN()
    denoiser.fit(data)
    denoised_basic = denoiser.transform(data)
    print(f"   Denoised shape: {denoised_basic.shape}")
    
    # 2. Population thresholding with different basis
    print("\n2. Population thresholding with PCA basis:")
    denoiser_pop = PSN(basis='pca', cv='population', verbose=True, wantfig=False)
    denoiser_pop.fit(data)
    denoised_pop = denoiser_pop.transform(data)
    print(f"   Selected {denoiser_pop.best_threshold_} dimensions for all units")
    
    # 3. Magnitude thresholding (no cross-validation)
    print("\n3. Magnitude thresholding (95% variance):")
    denoiser_mag = PSN(basis='whitened-signal', cv=None, mag_threshold=0.95, wantfig=False)
    denoiser_mag.fit(data)
    denoised_mag = denoiser_mag.transform(data)
    print(f"   Retained {denoiser_mag.fitted_results_['dimsretained']} dimensions")
    
    # 4. Custom scoring function (R²)
    print("\n4. Custom scoring (R²):")
    denoiser_r = PSN(basis='signal', cv='population', scoring='r2', wantfig=False)
    denoiser_r.fit(data)
    denoised_r = denoiser_r.transform(data)
    
    # 5. Custom basis matrix
    print("\n5. Custom orthonormal basis:")
    custom_basis = np.linalg.qr(np.random.randn(nunits, nunits))[0]
    denoiser_custom = PSN(basis=custom_basis, cv='unit', wantfig=False)
    denoiser_custom.fit(data)
    denoised_custom = denoiser_custom.transform(data)
    
    # 5b. Unit groups example
    print("\n5b. Unit groups (first 10 units share threshold):")
    unit_groups = np.concatenate([np.zeros(10, dtype=int), np.arange(10, nunits)])  # First 10 units share, rest individual
    denoiser_groups = PSN(basis='signal', cv='unit', unit_groups=unit_groups, wantfig=False)
    denoiser_groups.fit(data)
    denoised_groups = denoiser_groups.transform(data)
    print(f"   Unit groups shape: {unit_groups.shape}, unique groups: {len(np.unique(unit_groups))}")
    
    # 6. Demonstrate fit_transform
    print("\n6. Fit and transform in one step:")
    denoised_fit_transform = PSN(wantfig=False).fit_transform(data)
    print(f"   fit_transform result shape: {denoised_fit_transform.shape}")
    
    # 7. Transform different data shapes
    print("\n7. Transform different data shapes:")
    trial_avg = np.mean(data, axis=2)  # 2D trial-averaged data
    denoised_2d = denoiser.transform(trial_avg)
    print(f"   2D input {trial_avg.shape} -> 2D output {denoised_2d.shape}")
    
    single_trial = data[:, :, 0]  # Single trial
    denoised_single = denoiser.transform(single_trial)
    print(f"   Single trial {single_trial.shape} -> {denoised_single.shape}")
    
    # 8. Sklearn compatibility features
    print("\n8. Sklearn compatibility:")
    print(f"   Feature names: {denoiser.get_feature_names_out()[:5]}...")
    print(f"   Score (noise ceiling): {denoiser.score(data):.3f}")
    
    # 9. Compare with functional API
    print("\n9. Comparison with functional API:")
    functional_results = psn(data, V=0, opt={'cv_mode': 0, 'cv_threshold_per': 'unit'}, wantfig=False)
    
    # Check if results are equivalent
    sklearn_trial_avg = np.mean(denoised_basic, axis=2)
    functional_denoised = functional_results['denoiseddata']
    
    mse = np.mean((sklearn_trial_avg - functional_denoised) ** 2)
    print(f"   MSE between sklearn and functional APIs: {mse:.2e}")
    print(f"   Results are {'equivalent' if mse < 1e-10 else 'different'}")
    
    # 10. Parameter getting and setting
    print("\n10. Parameter management:")
    params = denoiser.get_params()
    print(f"   Current parameters: {list(params.keys())}")
    
    # Change parameters
    denoiser.set_params(cv='population', scoring='r2')
    print(f"   Updated cv: {denoiser.cv}, scoring: {denoiser.scoring}")
    
    print("\n" + "=" * 50)
    print("All demonstrations completed successfully!")
    
    # Optional: Show parameter combinations
    print("\nParameter combinations summary:")
    print("- basis: 'signal', 'whitened-signal', 'noise', 'pca', 'random', or custom matrix")
    print("- cv: 'unit' (per-unit thresholds), 'population' (same threshold), None (magnitude)")
    print("- scoring: 'mse', 'r2' (coefficient of determination), or custom function (for CV modes)")
    print("- mag_threshold: 0.0-1.0 (variance fraction for magnitude mode)")
    print("- unit_groups: array specifying which units share thresholds (for 'unit' mode)")
    print("- verbose: True/False")
    print("- wantfig: True/False")
    print("- gsn_kwargs: dict of additional GSN parameters")

if __name__ == "__main__":
    main()
