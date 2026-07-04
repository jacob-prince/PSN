"""PSN (Partitioning Signal and Noise) - Neural data denoising package."""

from .psn import psn
from .sklearn_api import PSN
from .utils import (
    compute_noise_ceiling,
    compute_r2,
    make_orthonormal,
    perform_gsn,
)
from .version import version as __version__

# Import visualization if available
try:
    from .utilities.plotting.visualization import plot_diagnostic_figures
    _has_visualization = True
except ImportError:
    _has_visualization = False

# Import simulation if available
try:
    from .utilities.simulation.simulate_data import (
        generate_data,
        generate_heterogeneous_populations,
    )
    _has_simulation = True
except ImportError:
    _has_simulation = False

__all__ = [
    '__version__',
    'psn',
    'PSN',
    'perform_gsn',
    'compute_noise_ceiling',
    'make_orthonormal',
    'compute_r2',
]

if _has_visualization:
    __all__.append('plot_diagnostic_figures')

if _has_simulation:
    __all__.extend(['generate_data', 'generate_heterogeneous_populations'])
