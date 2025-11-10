"""
FEL-CA: Flux Equality Law Cellular Automaton

Reference implementation of the deterministic flux-equality cellular automaton
for coherent wave dynamics.
"""

__version__ = "1.0.0"
__author__ = "Cagri Aksay"
__email__ = "cagri@aksay.co"

from .simulator import FELSimulator
from .simulator_int import FELSimulatorInt
from .utils import (
    build_rotation_lut,
    compute_energy,
    compute_helicity,
    spectral_analysis
)

__all__ = [
    'FELSimulator',
    'FELSimulatorInt',
    'build_rotation_lut',
    'compute_energy',
    'compute_helicity',
    'spectral_analysis',
]

