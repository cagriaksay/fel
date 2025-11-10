# FEL-CA Repository Structure

This document provides an overview of the complete FEL-CA public repository.

## Directory Tree

```
FEL/
├── README.md                    # Main documentation
├── LICENSE                      # Academic use only license
├── CONTRIBUTING.md              # Contribution guidelines
├── STRUCTURE.md                 # This file
├── setup.py                     # Package installation
├── requirements.txt             # Dependencies
├── .gitignore                   # Git ignore rules
│
├── fel_ca/                      # Core library
│   ├── __init__.py              # Package exports
│   ├── simulator.py             # Float implementation (main)
│   ├── simulator_int.py         # Integer Q1.31 implementation
│   └── utils.py                 # Helper functions
│
├── experiments/                 # Paper experiments
│   ├── __init__.py
│   ├── 01_single_sine.py        # Single-sine propagation
│   ├── 02_two_sine_interference.py  # Two-source interference
│   ├── 03_conservation.py       # E(t) and H(t) tracking
│   ├── 04_walking_boundary.py   # FEL verification (CRITICAL)
│   ├── 05_dispersion.py         # ω(k) dispersion analysis
│   └── run_all.py               # Master experiment runner
│
└── docs/                        # Documentation
    ├── API.md                   # Complete API reference
    └── QUICKSTART.md            # Quick start guide
```

---

## Component Overview

### Core Library (`fel_ca/`)

#### `simulator.py` - Main Floating-Point Simulator
- **Class**: `FELSimulator`
- **Purpose**: Primary implementation using float32/float64
- **Features**:
  - Stream-Cancel-Rotate (SCR) rule
  - Rodrigues rotation with LUT (0.1° resolution)
  - L1-weighted streaming
  - Periodic boundaries
  - GPU acceleration (CUDA/MPS)
- **Key Methods**:
  - `step()` - Execute one SCR step
  - `init_sine_wave()` - Initialize plane wave
  - `init_two_sources()` - Two coherent sources
  - `get_energy()` - Total energy E
  - `compute_flux_through_boundary()` - Boundary flux Φ(B)

#### `simulator_int.py` - Integer Implementation
- **Class**: `FELSimulatorInt`
- **Purpose**: Deterministic Q1.31 fixed-point arithmetic
- **Features**:
  - Integer streaming weights (2³⁰ scale)
  - Float LUT rotation (deterministic)
  - Bitwise-identical results
  - ~40% slower than float, ~1.8×10⁻⁷ drift rate
- **Interface**: Identical to `FELSimulator`

#### `utils.py` - Utility Functions
- `build_rotation_lut()` - Rotation table generator
- `compute_energy(F)` - Energy calculation
- `compute_helicity(F)` - Helicity-like invariant
- `spectral_analysis()` - 2D FFT spectrum
- `measure_phase_speed()` - c_phase from trajectory
- `create_spherical_boundary()` - Sphere for FEL test
- `create_box_boundary()` - Box boundary

---

### Experiments (`experiments/`)

All experiments are **self-contained** with:
- Command-line interface
- Argument parsing
- Progress bars (tqdm)
- Automatic output directories
- PNG plots and numpy data files

#### 01. Single-Sine Propagation
**Purpose**: Validate coherent propagation and energy conservation

**Outputs**:
- Energy over time
- Relative error |ΔE|/E₀
- Space-time diagram
- **Claim**: E conserved within 10⁻⁷

**Runtime**: ~2 minutes (N=128, 1000 steps, GPU)

---

#### 02. Two-Sine Interference
**Purpose**: Demonstrate deterministic superposition

**Configurations**:
- Phase = 0 (stationary pattern)
- Phase = π/2 (drifting fringes)

**Outputs**:
- Interference pattern evolution (9 frames)
- Final interference map
- Energy stability

**Runtime**: ~1 minute per phase (N=128, 500 steps)

---

#### 03. Conservation Metrics
**Purpose**: Long-term E(t) and H(t) stability

**Outputs**:
- Energy E(t) over 1M steps
- Helicity H(t) over 1M steps
- Relative errors (log scale)
- **Claim**: Both conserved within 10⁻⁷

**Runtime**: ~30 minutes (N=128, 1M steps, GPU)

**Quick Mode**: 10k steps (~1 minute)

---

#### 04. Walking-Boundary FEL Verification ⚠️ **CRITICAL**
**Purpose**: Validate Flux Equality Law at all scales

**Method**: Measure Φ(B,t) = Σ F·n̂ through spherical boundaries of increasing radius

**Outputs**:
- Flux through r = [10, 15, 20, 25, 30] over time
- Maximum |Φ| vs radius
- Pass/fail for each boundary
- **Claim**: |Φ(B,t)| < 10⁻⁶ for all B, all t

**Importance**: This is the **empirical test** of the paper's central claim. If this fails, FEL is violated.

**Runtime**: ~5 minutes (N=128, 500 steps)

---

#### 05. Dispersion Analysis
**Purpose**: Quantify numerical dispersion

**Method**: Measure ω(k) for wavelengths λ = [8, 12, 16, 24, 32, 48, 64]

**Outputs**:
- ω(k) dispersion relation
- Phase speed c_phase vs λ
- Phase error ε_phase vs λ
- k_measured vs k_theory
- **Claim**: ε_phase ≤ 1% for λ/Δx ≥ 32

**Runtime**: ~10 minutes (7 wavelengths, 500 steps each)

---

#### Master Runner (`run_all.py`)
**Purpose**: Execute all 5 experiments sequentially

**Modes**:
- `--quick`: Reduced parameters (~5 minutes total)
- Default: Full paper parameters (~2 hours)

**Usage**:
```bash
python experiments/run_all.py --quick --device cuda
```

---

## Documentation (`docs/`)

### API.md
Complete API reference:
- All class methods
- All utility functions
- Example code snippets
- Return types and parameters

### QUICKSTART.md
Quick start guide:
- Installation
- 5-minute tutorial
- Complete examples (interference, conservation, FEL test)
- Tips and troubleshooting

---

## Key Design Decisions

### 1. Minimal Dependencies
Only essential packages:
- `torch` (core simulation)
- `numpy` (data handling)
- `matplotlib` (plotting)
- `scipy` (spectral analysis)
- `tqdm` (progress bars)

### 2. Self-Contained Experiments
Each experiment:
- Runs independently
- Has CLI interface
- Generates publication-ready plots
- Saves raw data (.npy)
- Prints pass/fail verdicts

### 3. Dual Implementation
- **Float**: Fast, standard precision
- **Integer**: Deterministic, bitwise-identical

### 4. GPU-First
- CUDA (NVIDIA)
- MPS (Apple Silicon)
- CPU fallback

### 5. Academic Focus
- License: Academic use only
- Citation required
- No commercial use
- Research-oriented

---

## Reproducibility

All experiments are **fully reproducible**:

1. **Fixed seeds**: Set `torch.manual_seed(42)`
2. **Deterministic mode**: `torch.use_deterministic_algorithms(True)`
3. **Integer mode**: Use `FELSimulatorInt` for bitwise-identical results
4. **Documented parameters**: All experiment scripts have default args matching paper
5. **Git commit hash**: Tag release with paper submission

---

## Usage Patterns

### Quick Test
```bash
python experiments/run_all.py --quick
```

### Single Experiment
```bash
python experiments/04_walking_boundary.py --N 128 --steps 500
```

### Programmatic
```python
from fel_ca import FELSimulator

sim = FELSimulator(N=128, device='cuda')
sim.init_sine_wave(wavelength=32, direction='x', polarization='y')

for _ in range(1000):
    sim.step()

print(f"Energy: {sim.get_energy():.6e}")
```

---

## File Sizes

Approximate sizes:

```
fel_ca/simulator.py          ~6 KB   (300 lines)
fel_ca/simulator_int.py      ~5 KB   (250 lines)
fel_ca/utils.py              ~8 KB   (350 lines)

experiments/01_single_sine.py             ~6 KB
experiments/02_two_sine_interference.py   ~8 KB
experiments/03_conservation.py            ~7 KB
experiments/04_walking_boundary.py        ~10 KB  (CRITICAL)
experiments/05_dispersion.py              ~8 KB
experiments/run_all.py                    ~5 KB

docs/API.md                  ~10 KB
docs/QUICKSTART.md           ~8 KB

Total codebase:              ~80 KB (without results)
```

---

## Performance Benchmarks

**Hardware**: NVIDIA RTX 3090 (24GB VRAM)

| Experiment               | N   | Steps  | Time    |
|--------------------------|-----|--------|---------|
| Single-Sine              | 128 | 1,000  | 2 min   |
| Two-Sine (both phases)   | 128 | 1,000  | 3 min   |
| Conservation (full)      | 128 | 1M     | 30 min  |
| Walking-Boundary         | 128 | 500    | 5 min   |
| Dispersion (7 λ)         | 128 | 3,500  | 10 min  |
| **All experiments (full)** |     |        | **~2 hours** |
| **Quick mode**           |     |        | **~5 min** |

---

## Next Steps for Public Release

1. ✅ Core library implemented
2. ✅ All 5 critical experiments
3. ✅ Documentation (README, API, QUICKSTART)
4. ✅ License (academic use only)
5. ⏳ Test on clean environment
6. ⏳ Create GitHub repository
7. ⏳ Tag release (v1.0.0) with paper submission
8. ⏳ Add Zenodo DOI for archival

---

## Paper Integration

### Required Figures
1. **Fig. 1**: Single-sine space-time → `01_single_sine.py`
2. **Fig. 2**: Interference pattern → `02_two_sine_interference.py`
3. **Fig. 3**: Conservation E(t), H(t) → `03_conservation.py`
4. **Fig. 4**: Walking-boundary Φ(B) → `04_walking_boundary.py`
5. **Fig. 5**: Dispersion ω(k) → `05_dispersion.py`

### Required Tables
- **Table 1**: Comparative table (manual, in paper)
- **Table 2**: Dispersion data → `05_dispersion.py` output
- **Table 3**: Conservation bounds → `03_conservation.py` output

---

## Status: ✅ COMPLETE

All components implemented and ready for public release.

**Next Action**: Test on clean environment and push to GitHub.

