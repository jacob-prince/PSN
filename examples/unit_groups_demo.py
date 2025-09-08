#!/usr/bin/env python3
"""
Demonstration of PSN unit_groups functionality

This script shows how to use the unit_groups parameter to specify
which units should share the same CV threshold.
"""

import numpy as np
from psn import PSN

def main():
    # Generate synthetic data
    np.random.seed(42)
    nunits, nconds, ntrials = 20, 50, 6
    
    # Create data with some signal structure
    signal = np.random.randn(nunits, nconds)
    noise = np.random.randn(nunits, nconds, ntrials) * 0.3
    data = signal[:, :, np.newaxis] + noise
    
    print("PSN Unit Groups Demonstration")
    print("=" * 40)
    print(f"Data shape: {data.shape} (units, conditions, trials)")
    
    # 1. Default behavior (each unit gets its own threshold)
    print("\n1. Default unit-wise thresholding:")
    denoiser_default = PSN(cv='unit', verbose=False, wantfig=False)
    denoiser_default.fit(data)
    thresholds_default = denoiser_default.best_threshold_
    print(f"   Each unit gets own threshold: {thresholds_default}")
    print(f"   Unique thresholds: {len(np.unique(thresholds_default))}")
    
    # 2. Group units by pairs
    print("\n2. Pairing units (every 2 units share threshold):")
    unit_groups_pairs = np.repeat(np.arange(nunits // 2), 2)
    denoiser_pairs = PSN(cv='unit', unit_groups=unit_groups_pairs, verbose=False, wantfig=False)
    denoiser_pairs.fit(data)
    thresholds_pairs = denoiser_pairs.best_threshold_
    print(f"   Unit groups: {unit_groups_pairs}")
    print(f"   Thresholds: {thresholds_pairs}")
    print(f"   Unique thresholds: {len(np.unique(thresholds_pairs))}")
    
    # 3. Group by anatomical regions (example)
    print("\n3. Anatomical grouping (4 regions of 5 units each):")
    unit_groups_regions = np.repeat(np.arange(4), 5)
    denoiser_regions = PSN(cv='unit', unit_groups=unit_groups_regions, verbose=False, wantfig=False)
    denoiser_regions.fit(data)
    thresholds_regions = denoiser_regions.best_threshold_
    print(f"   Unit groups (regions): {unit_groups_regions}")
    print(f"   Thresholds per region: {thresholds_regions}")
    print(f"   Unique thresholds: {len(np.unique(thresholds_regions))}")
    
    # 4. Mixed grouping (some units grouped, others individual)
    print("\n4. Mixed grouping:")
    unit_groups_mixed = np.array([0, 0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15])
    denoiser_mixed = PSN(cv='unit', unit_groups=unit_groups_mixed, verbose=False, wantfig=False)
    denoiser_mixed.fit(data)
    thresholds_mixed = denoiser_mixed.best_threshold_
    print(f"   Unit groups: {unit_groups_mixed}")
    print(f"   Thresholds: {thresholds_mixed}")
    
    # Show which units share thresholds
    unique_groups = np.unique(unit_groups_mixed)
    print("   Groups that share thresholds:")
    for group_id in unique_groups:
        units_in_group = np.where(unit_groups_mixed == group_id)[0]
        if len(units_in_group) > 1:
            print(f"     Group {group_id}: units {units_in_group} -> threshold {thresholds_mixed[units_in_group[0]]}")
    
    # 5. Compare denoising performance
    print("\n5. Comparing denoising performance:")
    
    # Apply denoisers
    denoised_default = denoiser_default.transform(data)
    denoised_pairs = denoiser_pairs.transform(data)
    denoised_regions = denoiser_regions.transform(data)
    
    # Compute noise ceiling for comparison
    from psn.utils import compute_noise_ceiling
    nc_original, *_ = compute_noise_ceiling(data)
    nc_default, *_ = compute_noise_ceiling(denoised_default)
    nc_pairs, *_ = compute_noise_ceiling(denoised_pairs)
    nc_regions, *_ = compute_noise_ceiling(denoised_regions)
    
    print(f"   Original data noise ceiling: {np.mean(nc_original):.2f}%")
    print(f"   Default grouping: {np.mean(nc_default):.2f}%")
    print(f"   Pair grouping: {np.mean(nc_pairs):.2f}%")
    print(f"   Region grouping: {np.mean(nc_regions):.2f}%")
    
    print("\n" + "=" * 40)
    print("Key insights:")
    print("- unit_groups allows flexible threshold sharing")
    print("- Fewer groups = more regularization")
    print("- Use anatomical/functional knowledge to group units")
    print("- Compare performance to choose optimal grouping")

if __name__ == "__main__":
    main()
