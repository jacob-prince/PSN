#!/usr/bin/env python3
"""
Demonstration of PSN unit_groups functionality

This script shows how to use the unit_groups parameter to specify
which units should share the same CV threshold.
"""

import numpy as np
from psn import PSN
from psn.utils import split_half_reliability_3d

def main():
    # Generate synthetic data
    np.random.seed(42)
    nunits, nconds, ntrials = 20, 50, 6
    
    # Create data with some signal structure
    signal = np.random.randn(nunits, nconds)
    noise = 0.5 * np.random.randn(nunits, nconds, ntrials)
    data = signal[:, :, np.newaxis] + noise
    
    print("PSN Unit Groups Demonstration")
    print("=" * 40)
    print(f"Data shape: {data.shape} (units, conditions, trials)")
    
    # Example 1: Default unit-wise thresholding (each unit gets its own threshold)
    print("\n1. Default unit-wise thresholding:")
    denoiser_default = PSN(basis='signal', cv='unit', verbose=False, wantfig=False)
    denoiser_default.fit(data)
    denoised_default = denoiser_default.transform(data)
    print(f"   Each unit gets own threshold: {denoiser_default.best_threshold_}")
    print(f"   Unique thresholds: {len(np.unique(denoiser_default.best_threshold_))}")
    
    # Example 2: Pairing units (every 2 units share the same threshold)
    print("\n2. Pairing units (every 2 units share threshold):")
    unit_groups_pairs = np.repeat(np.arange(nunits // 2), 2)[:nunits]
    denoiser_pairs = PSN(basis='signal', cv='unit', unit_groups=unit_groups_pairs, verbose=False, wantfig=False)
    denoiser_pairs.fit(data)
    denoised_pairs = denoiser_pairs.transform(data)
    print(f"   Unit groups: {unit_groups_pairs}")
    print(f"   Thresholds: {denoiser_pairs.best_threshold_}")
    print(f"   Unique thresholds: {len(np.unique(denoiser_pairs.best_threshold_))}")
    
    # Example 3: Anatomical grouping (4 regions of 5 units each)
    print("\n3. Anatomical grouping (4 regions of 5 units each):")
    unit_groups_regions = np.repeat(np.arange(4), nunits // 4)[:nunits]
    denoiser_regions = PSN(basis='signal', cv='unit', unit_groups=unit_groups_regions, verbose=False, wantfig=False)
    denoiser_regions.fit(data)
    denoised_regions = denoiser_regions.transform(data)
    print(f"   Unit groups (regions): {unit_groups_regions}")
    print(f"   Thresholds per region: {denoiser_regions.best_threshold_}")
    print(f"   Unique thresholds: {len(np.unique(denoiser_regions.best_threshold_))}")
    
    # Example 4: Mixed grouping (some units grouped, others individual)
    print("\n4. Mixed grouping:")
    unit_groups_mixed = np.arange(nunits)
    unit_groups_mixed[:3] = 0  # First 3 units share threshold
    unit_groups_mixed[6:8] = 4  # Units 6-7 share threshold  
    unit_groups_mixed[12:14] = 9  # Units 12-13 share threshold
    denoiser_mixed = PSN(basis='signal', cv='unit', unit_groups=unit_groups_mixed, verbose=False, wantfig=False)
    denoiser_mixed.fit(data)
    print(f"   Unit groups: {unit_groups_mixed}")
    print(f"   Thresholds: {denoiser_mixed.best_threshold_}")
    
    # Show which groups share thresholds
    unique_groups, group_counts = np.unique(unit_groups_mixed, return_counts=True)
    shared_groups = unique_groups[group_counts > 1]
    if len(shared_groups) > 0:
        print("   Groups that share thresholds:")
        for group in shared_groups:
            units_in_group = np.where(unit_groups_mixed == group)[0]
            threshold = denoiser_mixed.best_threshold_[units_in_group[0]]
            print(f"     Group {group}: units {units_in_group} -> threshold {threshold}")
    
    # Example 5: Compare denoising performance
    print("\n5. Comparing denoising performance:")
    denoised_default = denoiser_default.transform(data)
    denoised_pairs = denoiser_pairs.transform(data)
    denoised_regions = denoiser_regions.transform(data)
    
    # Compute split-half reliability for comparison
    rel_original = split_half_reliability_3d(data)
    rel_default = split_half_reliability_3d(denoised_default)
    rel_pairs = split_half_reliability_3d(denoised_pairs)
    rel_regions = split_half_reliability_3d(denoised_regions)
    
    print(f"   Original data reliability: {np.mean(rel_original):.3f}")
    print(f"   Default grouping: {np.mean(rel_default):.3f}")
    print(f"   Pair grouping: {np.mean(rel_pairs):.3f}")
    print(f"   Region grouping: {np.mean(rel_regions):.3f}")
    
    print("\n" + "=" * 40)
    print("Key insights:")
    print("- unit_groups allows flexible threshold sharing")
    print("- Fewer groups = more regularization")
    print("- Use anatomical/functional knowledge to group units")
    print("- Compare performance to choose optimal grouping")

if __name__ == "__main__":
    main()
