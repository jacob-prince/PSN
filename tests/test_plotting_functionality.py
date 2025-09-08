"""Test the plotting functionality of psn to identify errors in diagnostic figures.

This test focuses specifically on the plotting/visualization components of the PSN library
to catch any errors that might occur during figure generation.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
from psn import psn


@pytest.fixture
def plotting_test_data():
    """Create test data specifically designed to stress test plotting functions."""
    np.random.seed(42)
    
    nunits = 15
    nconds = 20
    ntrials = 4
    
    # Create data with diverse characteristics to test plotting edge cases
    unit_means = np.random.randn(nunits) * 3
    signal = np.random.randn(nunits, nconds) * 2
    noise = np.random.randn(nunits, nconds, ntrials) * 0.8
    
    data = np.zeros((nunits, nconds, ntrials))
    for t in range(ntrials):
        data[:, :, t] = unit_means[:, np.newaxis] + signal + noise[:, :, t]
    
    return data


class TestPlottingFunctionality:
    """Test class for GSN plotting functionality."""

    def test_basic_plotting_functionality(self, plotting_test_data):
        """Test basic plotting functionality with default settings."""
        data = plotting_test_data
        
        # Test with wantfig=True to trigger plotting
        results = psn(data, V=0, wantfig=True)
        plt.close('all')  # Clean up figures
        
        # Check that results are returned successfully
        assert isinstance(results, dict)
        assert 'plot' in results  # Should have plot function

    def test_plotting_all_basis_types(self, plotting_test_data):
        """Test plotting functionality across all basis types."""
        data = plotting_test_data
        basis_types = [0, 1, 2, 3, 4]
        
        for V in basis_types:
            results = psn(data, V=V, wantfig=True)
            plt.close('all')  # Clean up figures
            assert isinstance(results, dict)

    def test_plotting_different_cv_modes(self, plotting_test_data):
        """Test plotting with different cross-validation modes."""
        data = plotting_test_data
        
        cv_configs = [
            {'cv_mode': 0, 'cv_threshold_per': 'population'},
            {'cv_mode': 0, 'cv_threshold_per': 'unit'},
            {'cv_mode': 1, 'cv_threshold_per': 'population'},
            {'cv_mode': 1, 'cv_threshold_per': 'unit'},
            {'cv_mode': -1, 'mag_frac': 0.1}  # Magnitude thresholding
        ]
        
        for config in cv_configs:
            results = psn(data, V=0, opt=config, wantfig=True)
            plt.close('all')  # Clean up figures
            assert isinstance(results, dict)

    def test_plotting_different_denoising_types(self, plotting_test_data):
        """Test plotting with different denoising types."""
        data = plotting_test_data
        
        denoising_configs = [
            {'denoisingtype': 0},  # Trial-averaged
            {'denoisingtype': 1}   # Single-trial
        ]
        
        for config in denoising_configs:
            results = psn(data, V=0, opt=config, wantfig=True)
            plt.close('all')  # Clean up figures
            assert isinstance(results, dict)

    def test_plotting_with_custom_basis(self, plotting_test_data):
        """Test plotting with custom user-supplied basis."""
        data = plotting_test_data
        nunits = data.shape[0]
        
        # Create custom orthonormal basis
        custom_basis = np.linalg.qr(np.random.randn(nunits, nunits))[0]
        results = psn(data, V=custom_basis, wantfig=True)
        plt.close('all')  # Clean up figures
        assert isinstance(results, dict)

    def test_plotting_edge_cases(self):
        """Test plotting with edge case data."""
        
        # Test with very small data
        small_data = np.random.randn(3, 4, 2)
        results = psn(small_data, V=0, wantfig=True)
        plt.close('all')
        assert isinstance(results, dict)
        
        # Test with zero-mean data
        zero_mean_data = np.random.randn(8, 10, 3)
        zero_mean_data = zero_mean_data - np.mean(zero_mean_data, axis=(1, 2), keepdims=True)
        results = psn(zero_mean_data, V=0, wantfig=True)
        plt.close('all')
        assert isinstance(results, dict)
        
        # Test with extreme values
        extreme_data = np.random.randn(6, 8, 3) * 100  # Very large values
        results = psn(extreme_data, V=0, wantfig=True)
        plt.close('all')
        assert isinstance(results, dict)

    def test_regenerate_visualization(self, plotting_test_data):
        """Test the regenerate visualization functionality."""
        data = plotting_test_data
        
        # First run without figures
        results = psn(data, V=0, wantfig=False)
        
        # Test regenerating visualization
        results['plot']()  # Should regenerate the visualization
        plt.close('all')
        
        # Test with test data
        test_data = plotting_test_data
        results['plot'](test_data=test_data)
        plt.close('all')
        
        assert callable(results['plot'])

    def test_plotting_error_handling(self, plotting_test_data):
        """Test error handling in plotting functions."""
        data = plotting_test_data
        
        # Test with invalid cv_thresholds that might cause indexing errors
        opt = {
            'cv_thresholds': [1, 2, 100],  # 100 is larger than data dimensions
            'cv_mode': 0
        }
        
        results = psn(data, V=0, opt=opt, wantfig=True)
        plt.close('all')
        assert isinstance(results, dict)

    def test_cross_validation_scores_plotting(self, plotting_test_data):
        """Test specific cross-validation scores plotting functionality."""
        data = plotting_test_data
        
        # Use specific options that should generate CV scores
        opt = {
            'cv_mode': 0,
            'cv_threshold_per': 'unit',
            'cv_thresholds': np.arange(1, 11)  # Test first 10 dimensions
        }
        
        results = psn(data, V=0, opt=opt, wantfig=True)
        
        # Check that cv_scores exist and have expected shape
        if 'cv_scores' in results and results['cv_scores'] is not None:
            assert results['cv_scores'].shape[0] > 0
        
        plt.close('all')
        assert isinstance(results, dict)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
