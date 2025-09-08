"""
Simple example demonstrating PSN (Partitioning Signal and Noise) usage.

This example shows how to:
1. Generate synthetic neural data 
2. Apply PSN denoising
3. Compare raw vs denoised data
"""

import numpy as np
import matplotlib.pyplot as plt
import psn
from psn.simulate import generate_data

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic neural data using PSN's simulate module
    nunits = 25  # Number of neural units
    nconds = 50  # Number of experimental conditions  
    ntrials = 3  # Number of trials per condition
    
    print(f"Generating synthetic data: {nunits} units × {nconds} conditions × {ntrials} trials")
    
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
    
    print("\nApplying PSN denoising...")
    
    # Apply PSN denoising with default settings
    results = psn.psn(
        data,
        V=0,  # Use signal covariance eigenvectors
        opt={
            'cv_mode': 0,  # Leave-one-out cross-validation
            'cv_threshold_per': 'population',  # Single threshold for all units
            'denoisingtype': 0  # Trial-averaged denoising
        },
        wantfig=True  # Generate diagnostic plots
    )
    
    # Print results summary
    print(f"\nDenoising Results:")
    print(f"- Denoiser matrix shape: {results['denoiser'].shape}")
    print(f"- Denoised data shape: {results['denoiseddata'].shape}")
    print(f"- Number of basis dimensions: {results['fullbasis'].shape[1]}")
    
    if 'best_threshold' in results:
        print(f"- Optimal threshold: {results['best_threshold']} dimensions")
    
    # Simple comparison of raw vs denoised data
    raw_data = np.mean(data, axis=2)  # Trial-averaged raw data
    denoised_data = results['denoiseddata']
    
    # Compute correlation between raw and denoised for each unit
    correlations = []
    for unit in range(nunits):
        corr = np.corrcoef(raw_data[unit], denoised_data[unit])[0, 1]
        correlations.append(corr)
    
    print(f"- Mean correlation between raw and denoised: {np.mean(correlations):.3f}")
    print(f"- Correlation range: [{np.min(correlations):.3f}, {np.max(correlations):.3f}]")
    
    # Display plots
    plt.show()
    
    print("\nExample completed successfully!")
    print("Check the diagnostic plots for detailed denoising analysis.")

if __name__ == "__main__":
    main()
