# FEL-CA: Flux Equality Law Cellular Automaton

**Deterministic Flux-Equality Cellular Automaton for Coherent Wave Dynamics**

This repository contains the reference implementation of the FEL-CA model described in:
> Aksay, C. "A Deterministic Integer Cellular Automaton with Lattice Continuity: The Flux Equality Law." (2025).

## Supplementary Videos

**Video S1: Random-Field Self-Organization**

https://github.com/user-attachments/assets/a366af5c-79c8-46fe-af97-19d8e421bebd

- Evolution from uniform random initialization showing emergence of long-lived localized flux packets
- Demonstrates Wolfram Class IV–like dynamics with κ=0
- 3D visualization of mesoscale structure formation and persistent coherent entities

**Video S2: Two-Source Interference** 

https://github.com/user-attachments/assets/00a2da49-b9eb-45aa-b049-e06bdc6a7b9a

- Coherent interference between two sources with Δφ = π/2 phase offset
- Shows characteristic diagonal fringe patterns and deterministic superposition
- 3D energy distribution evolution demonstrating wave-like behavior under integer arithmetic

## Overview

The FEL-CA enforces the **Flux Equality Law (FEL)**: for any closed boundary, total flux entering equals total flux leaving. The **Stream–Cancel–Rotate (SCR)** rule implements this through:
1. **Stream**: Flux propagates weighted by normalized components
2. **Cancel**: Opposing streams cancel to form a twist vector
3. **Rotate**: Surviving flux rotates around the twist axis

Key features:
- ✅ Deterministic and reproducible (bitwise-identical with fixed seed)
- ✅ Integer and floating-point implementations
- ✅ Low numerical dispersion (ε_phase ≤ 1% for λ/Δx ≥ 32)
- ✅ Exact conservation of total flux magnitude
- ✅ GPU-accelerated (CUDA/MPS)

## Installation

```bash
# Clone repository
git clone https://github.com/cagriaksay/fel-ca.git
cd fel-ca

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 2.0 (with CUDA or MPS support)
- NumPy
- Matplotlib (for visualization)

## Quick Start

```python
from fel_ca import FELSimulator

# Create 128³ lattice
sim = FELSimulator(N=128, device='cuda')

# Initialize with sine wave
sim.init_sine_wave(wavelength=32, direction='x')

# Run simulation
for _ in range(1000):
    sim.step()
    
# Get flux magnitude
magnitude = sim.get_magnitude()
```

## Repository Structure

```
FEL/
├── fel_ca/                 # Core simulation library
│   ├── __init__.py
│   ├── simulator.py        # Base FEL simulator (float)
│   ├── simulator_int.py    # Integer (Q1.31) implementation
│   └── utils.py            # Utilities (LUT, metrics)
├── experiments/            # Paper experiment scripts
│   ├── 01_single_sine.py
│   ├── 02_two_sine_interference.py
│   ├── 03_conservation.py
│   ├── 04_walking_boundary.py
│   ├── 05_dispersion.py
│   └── run_all.py
├── viewers/                # Visualization scripts
│   ├── view_kappa_sweep.py
│   ├── view_relaxation.py
│   └── view_two_sine.py
├── docs/                   # Documentation
│   └── API.md
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## Experiments

All experiments from the paper are reproducible:

```bash
# Run all experiments
python experiments/run_all.py

# Run specific experiment
python experiments/01_single_sine.py
python experiments/04_walking_boundary.py
```

### Experiment List
1. **Single-Sine Propagation** - Coherent wave propagation
2. **Two-Sine Interference** - Deterministic superposition
3. **Conservation Metrics** - Long-term E(t) and H(t) stability
4. **Walking-Boundary FEL** - Global flux equality verification
5. **Dispersion Analysis** - ω(k) curves and phase error


## License

**Academic Use Only** - Commercial use prohibited.

See [LICENSE](LICENSE) file for full terms.

For commercial licensing inquiries: cagri@aksay.co

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Please note: All contributions must be for academic/research purposes only.

## Contact

Cagri Aksay - cagri@aksay.co

GitHub: [@cagriaksay](https://github.com/cagriaksay)

