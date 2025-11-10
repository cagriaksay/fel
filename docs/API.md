# FEL-CA API Documentation

## Core Simulators

### `FELSimulator` (Floating-Point)

Main simulator class for floating-point FEL-CA.

```python
from fel_ca import FELSimulator

sim = FELSimulator(
    N=128,                    # Lattice size (N³)
    device='cuda',            # 'cuda', 'mps', or 'cpu'
    rotation_rate=0.01,       # κ coefficient (radians per unit twist)
    dtype=torch.float32       # torch.float32 or torch.float64
)
```

#### Methods

##### `step()`
Execute one SCR (Stream-Cancel-Rotate) step.

```python
sim.step()
```

##### `init_sine_wave(wavelength, direction, amplitude, polarization)`
Initialize with sinusoidal plane wave.

```python
sim.init_sine_wave(
    wavelength=32.0,      # λ/Δx (wavelength in voxels)
    direction='x',        # Propagation direction ('x', 'y', or 'z')
    amplitude=1.0,        # Wave amplitude
    polarization='y'      # Flux direction ('x', 'y', or 'z')
)
```

##### `init_two_sources(positions, amplitude, phase_offset, width)`
Initialize with two Gaussian point sources.

```python
sim.init_two_sources(
    positions=((20, 64, 64), (108, 64, 64)),  # (x1,y1,z1), (x2,y2,z2)
    amplitude=1.0,                             # Source amplitude
    phase_offset=0.0,                          # Phase difference (radians)
    width=3.0                                  # Gaussian width (voxels)
)
```

##### `get_magnitude()`
Get flux magnitude |F| at each voxel.

```python
magnitude = sim.get_magnitude()  # Returns (N,N,N) tensor
```

##### `get_energy()`
Compute total energy E = Σ|F|².

```python
energy = sim.get_energy()  # Returns float
```

##### `get_flux()`
Get current flux field F.

```python
flux = sim.get_flux()  # Returns (N,N,N,3) tensor
```

##### `compute_flux_through_boundary(boundary_faces, normals)`
Compute net flux Φ(B) = Σ F·n̂ through boundary.

```python
from fel_ca.utils import create_spherical_boundary

faces, normals = create_spherical_boundary(
    center=(64, 64, 64),
    radius=20,
    N=128
)
flux = sim.compute_flux_through_boundary(faces, normals)
```

---

### `FELSimulatorInt` (Integer Q1.31)

Integer fixed-point implementation for deterministic simulation.

```python
from fel_ca import FELSimulatorInt

sim = FELSimulatorInt(
    N=128,
    device='cuda',
    rotation_rate=0.01,
    weight_scale=2**30      # Fixed-point scale for streaming weights
)
```

**Note**: Interface identical to `FELSimulator`. Uses Q1.31 fixed-point arithmetic for streaming, float LUT for rotation.

---

## Utility Functions

### `build_rotation_lut(resolution_deg, device, dtype)`

Build rotation lookup table for Rodrigues formula.

```python
from fel_ca import build_rotation_lut

lut = build_rotation_lut(
    resolution_deg=0.1,      # Angular resolution (degrees)
    device='cuda',
    dtype=torch.float32
)
```

Returns dict with keys: `'cos'`, `'sin'`, `'one_minus_cos'`, `'resolution'`, `'num_angles'`.

---

### `compute_energy(F)`

Compute total energy E = Σ|F|².

```python
from fel_ca import compute_energy

E = compute_energy(flux_field)  # flux_field: (N,N,N,3)
```

---

### `compute_helicity(F)`

Compute helicity-like invariant H = Σ F·(∇×F).

```python
from fel_ca import compute_helicity

H = compute_helicity(flux_field)
```

Uses finite-difference curl on periodic lattice.

---

### `spectral_analysis(F, component, slice_axis, slice_idx)`

Perform 2D FFT on a slice for spatial spectrum analysis.

```python
from fel_ca import spectral_analysis

freqs, power = spectral_analysis(
    F=flux_field,
    component=1,        # Analyze Fy component
    slice_axis=0,       # Slice perpendicular to x
    slice_idx=64        # At x=64
)
```

Returns:
- `freqs`: Frequency array (cycles/voxel)
- `power`: 2D power spectrum

---

### Boundary Creation

#### `create_spherical_boundary(center, radius, N)`

Create spherical boundary for flux measurement.

```python
from fel_ca.utils import create_spherical_boundary

faces, normals = create_spherical_boundary(
    center=(64, 64, 64),
    radius=20,
    N=128
)
```

Returns:
- `faces`: (M, 3) integer coordinates of boundary faces
- `normals`: (M, 3) outward unit normals

#### `create_box_boundary(corner1, corner2)`

Create axis-aligned box boundary.

```python
from fel_ca.utils import create_box_boundary

faces, normals = create_box_boundary(
    corner1=(20, 20, 20),
    corner2=(108, 108, 108)
)
```

---

## Experiments

All experiments can be imported and run programmatically:

```python
from experiments.exp1_single_sine import run_single_sine_experiment

results = run_single_sine_experiment(
    N=128,
    wavelength=32.0,
    n_steps=1000,
    device='cuda',
    output_dir='results/01_single_sine'
)
```

Or run from command line:

```bash
python experiments/01_single_sine.py --N 128 --wavelength 32.0 --steps 1000
```

### Available Experiments

1. **01_single_sine.py** - Single-sine propagation
2. **02_two_sine_interference.py** - Two-source interference
3. **03_conservation.py** - Long-term E(t) and H(t) tracking
4. **04_walking_boundary.py** - FEL verification at all scales
5. **05_dispersion.py** - ω(k) dispersion analysis

---

## Example Workflow

```python
import torch
from fel_ca import FELSimulator, compute_energy, compute_helicity

# Create simulator
sim = FELSimulator(N=128, device='cuda', rotation_rate=0.01)

# Initialize
sim.init_sine_wave(wavelength=32.0, direction='x', polarization='y')

# Track conservation
E0 = sim.get_energy()
H0 = compute_helicity(sim.get_flux())

# Run simulation
for t in range(1000):
    sim.step()
    
    if t % 100 == 0:
        E_t = sim.get_energy()
        H_t = compute_helicity(sim.get_flux())
        print(f"t={t}: E={E_t:.6e}, H={H_t:.6e}")

# Final analysis
E_final = sim.get_energy()
print(f"Energy drift: {abs(E_final - E0) / E0:.6e}")
```

---

## Performance Tips

1. **Device Selection**:
   - Use `device='cuda'` for NVIDIA GPUs
   - Use `device='mps'` for Apple Silicon GPUs
   - Use `device='cpu'` as fallback

2. **Grid Size**:
   - Start with N=64 for quick tests
   - Use N=128 for paper-quality results
   - N=256+ requires significant memory (>8GB VRAM)

3. **Determinism**:
   - Use `FELSimulatorInt` for bitwise-identical results
   - Set torch seed: `torch.manual_seed(42)`
   - Use deterministic algorithms: `torch.use_deterministic_algorithms(True)`

4. **Long Runs**:
   - Save checkpoints periodically
   - Use `save_interval` in conservation experiments
   - Monitor memory usage for large grids

---

## Citation

If you use this code in research, please cite:

```bibtex
@article{aksay2025fel,
  title={Deterministic Flux-Equality Cellular Automaton for Coherent Wave Dynamics},
  author={Aksay, Cagri},
  journal={Entropy},
  year={2025},
  publisher={MDPI}
}
```

