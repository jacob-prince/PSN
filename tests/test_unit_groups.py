"""
Test unit_groups functionality in psn.py
"""
import numpy as np
import pytest
from psn import psn


class TestUnitGroups:
    """Test the unit_groups functionality."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.nunits = 10
        self.nconds = 20
        self.ntrials = 4
        
        # Create synthetic data with signal structure
        self.data = np.random.randn(self.nunits, self.nconds, self.ntrials)
        signal = np.random.randn(3, self.nconds)
        signal_weights = np.random.randn(self.nunits, 3)
        for t in range(self.ntrials):
            self.data[:, :, t] += signal_weights @ signal * 0.3
    
    def test_default_unit_groups_unit_mode(self):
        """Test that unit_groups defaults to np.arange(nunits) in unit mode."""
        opt = {'cv_threshold_per': 'unit', 'cv_mode': 0}
        results = psn(self.data, V=0, opt=opt, wantfig=False)
        
        expected = np.arange(self.nunits)
        np.testing.assert_array_equal(opt['unit_groups'], expected)
    
    def test_default_unit_groups_population_mode(self):
        """Test that unit_groups defaults to all zeros in population mode."""
        opt = {'cv_threshold_per': 'population', 'cv_mode': 0}
        results = psn(self.data, V=0, opt=opt, wantfig=False)
        
        expected = np.zeros(self.nunits, dtype=int)
        np.testing.assert_array_equal(opt['unit_groups'], expected)
    
    def test_custom_unit_groups_pairs(self):
        """Test custom unit_groups that group units in pairs."""
        unit_groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        opt = {
            'cv_threshold_per': 'unit', 
            'cv_mode': 0,
            'unit_groups': unit_groups,
            'cv_thresholds': [1, 2, 3, 4, 5]
        }
        results = psn(self.data, V=0, opt=opt, wantfig=False)
        
        # Verify that paired units have the same threshold
        for i in range(0, self.nunits, 2):
            if i+1 < self.nunits:
                assert results['best_threshold'][i] == results['best_threshold'][i+1], \
                    f"Paired units {i} and {i+1} should have same threshold"
    
    def test_all_units_one_group(self):
        """Test that all units in one group behave consistently."""
        unit_groups = np.zeros(self.nunits, dtype=int)
        opt = {
            'cv_threshold_per': 'unit',
            'cv_mode': 0,
            'unit_groups': unit_groups,
            'cv_thresholds': [1, 2, 3, 4, 5]
        }
        results = psn(self.data, V=0, opt=opt, wantfig=False)
        
        # All units should have the same threshold
        unique_thresholds = np.unique(results['best_threshold'])
        assert len(unique_thresholds) == 1, "All units should have same threshold when in one group"
    
    def test_unit_groups_validation_wrong_length(self):
        """Test that wrong length unit_groups raises an error."""
        with pytest.raises(ValueError, match="unit_groups must have length"):
            opt = {
                'cv_threshold_per': 'unit',
                'unit_groups': np.array([0, 1, 2])  # Wrong length
            }
            psn(self.data, V=0, opt=opt, wantfig=False)
    
    def test_unit_groups_validation_negative_values(self):
        """Test that negative unit_groups values raise an error."""
        with pytest.raises(ValueError, match="unit_groups must contain only non-negative integers"):
            opt = {
                'cv_threshold_per': 'unit',
                'unit_groups': np.array([0, 1, -1] + [0]*(self.nunits-3))
            }
            psn(self.data, V=0, opt=opt, wantfig=False)
    
    def test_unit_groups_validation_population_mode(self):
        """Test that non-zero unit_groups in population mode raises an error."""
        with pytest.raises(ValueError, match="When cv_threshold_per='population'"):
            opt = {
                'cv_threshold_per': 'population',
                'unit_groups': np.array([0, 1] + [0]*(self.nunits-2))
            }
            psn(self.data, V=0, opt=opt, wantfig=False)
    
    def test_unit_groups_with_cv_mode_1(self):
        """Test unit_groups functionality with cv_mode=1."""
        unit_groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        opt = {
            'cv_threshold_per': 'unit',
            'cv_mode': 1,
            'unit_groups': unit_groups,
            'cv_thresholds': [1, 2, 3, 4, 5]
        }
        results = psn(self.data, V=0, opt=opt, wantfig=False)
        
        # Verify that paired units have the same threshold
        for i in range(0, self.nunits, 2):
            if i+1 < self.nunits:
                assert results['best_threshold'][i] == results['best_threshold'][i+1], \
                    f"Paired units {i} and {i+1} should have same threshold in cv_mode=1"
    
    def test_single_unit_per_group(self):
        """Test that each unit in its own group works correctly."""
        unit_groups = np.arange(self.nunits)
        opt = {
            'cv_threshold_per': 'unit',
            'cv_mode': 0,
            'unit_groups': unit_groups,
            'cv_thresholds': [1, 2, 3, 4, 5]
        }
        results = psn(self.data, V=0, opt=opt, wantfig=False)
        
        # This should be equivalent to default unit mode behavior
        assert len(results['best_threshold']) == self.nunits
        assert hasattr(results, 'cv_scores') or 'cv_scores' in results
    
    def test_large_groups(self):
        """Test with larger number of groups."""
        # Create more complex grouping
        nunits = 20
        data = np.random.randn(nunits, self.nconds, self.ntrials)
        
        # Group units: [0,1,2], [3,4], [5,6,7,8], [9], [10,11], [12,13,14,15,16], [17,18,19]
        unit_groups = np.array([0,0,0,1,1,2,2,2,2,3,4,4,5,5,5,5,5,6,6,6])
        
        opt = {
            'cv_threshold_per': 'unit',
            'cv_mode': 0,
            'unit_groups': unit_groups,
            'cv_thresholds': [1, 2, 3, 4, 5]
        }
        results = psn(data, V=0, opt=opt, wantfig=False)
        
        # Check that units in the same group have the same threshold
        unique_groups = np.unique(unit_groups)
        for group in unique_groups:
            group_units = np.where(unit_groups == group)[0]
            if len(group_units) > 1:
                group_thresholds = results['best_threshold'][group_units]
                assert np.all(group_thresholds == group_thresholds[0]), \
                    f"Units in group {group} should have same threshold"
    
    def test_unit_groups_with_magnitude_thresholding(self):
        """Test that unit_groups work with magnitude thresholding."""
        unit_groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        opt = {
            'cv_mode': -1,
            'mag_frac': 0.95,
            'unit_groups': unit_groups
        }
        results = psn(self.data, V=0, opt=opt, wantfig=False)
        
        # Should run without error and return valid results
        assert 'denoiser' in results
        assert results['denoiser'].shape == (self.nunits, self.nunits)
    
    def test_unit_groups_modify_in_place(self):
        """Test that unit_groups are properly set in the options dict."""
        opt = {'cv_threshold_per': 'unit', 'cv_mode': 0}
        original_opt = opt.copy()
        
        results = psn(self.data, V=0, opt=opt, wantfig=False)
        
        # Check that unit_groups was added to opt
        assert 'unit_groups' in opt
        assert len(opt['unit_groups']) == self.nunits
        
        # Check that it wasn't in the original
        assert 'unit_groups' not in original_opt


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
