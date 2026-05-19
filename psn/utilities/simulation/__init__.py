"""Simulation utilities for PSN."""

# Note: simulate_data.py exports its own functions directly
# Import them here for convenience

try:
    from .simulate_data import generate_data, generate_heterogeneous_populations
    __all__ = ['generate_data', 'generate_heterogeneous_populations']
except ImportError:
    # If simulate_data is not available, just pass
    __all__ = []
