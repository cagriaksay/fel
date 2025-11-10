# FEL-CA Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/cagriaksay/fel-ca.git
cd fel-ca

# Install dependencies
pip install -r requirements.txt

# Optional: Install in editable mode
pip install -e .
```

## 5-Minute Tutorial

### 1. Basic Simulation

```python
from fel_ca import FELSimulator

# Create 64³ lattice on GPU
sim = FELSimulator(N=64, device='cuda')

# Initialize sine wave
sim.init_sine_wave(wavelength=16, direction='x', polarization='y')

# Run 100 steps
for _ in range(100):
    sim.step()

# Get results
magnitude = sim.get_magnitude()
energy = sim.get_energy()
print(f"Final energy: {energy:.6e}")
```

### 2. Run Paper Experiments

```bash
# Quick test (5 minutes)
python experiments/run_all.py --quick

# Full paper experiments (2 hours on GPU)
python experiments/run_all.py --device cuda
```

### 3. Single Experiment

```bash
# Walking-boundary FEL verification (CRITICAL)
python experiments/04_walking_boundary.py --N 128 --steps 500

# Dispersion analysis
python experiments/05_dispersion.py --wavelengths 8 12 16 24 32 48 64
```

---

## Example: Two-Source Interference

```python
from fel_ca import FELSimulator
import matplotlib.pyplot as plt

# Create simulator
sim = FELSimulator(N=128, device='cuda')

# Two coherent sources
sim.init_two_sources(
    positions=((30, 64, 64), (98, 64, 64)),  # Opposite sides
    amplitude=1.0,
    phase_offset=0.0,  # In-phase (stationary pattern)
    width=5.0
)

# Run until interference forms
for t in range(500):
    sim.step()
    
    if t % 100 == 0:
        print(f"Step {t}")

# Visualize center slice
magnitude = sim.get_magnitude()
center_slice = magnitude[:, :, 64].cpu().numpy()

plt.figure(figsize=(10, 8))
plt.imshow(center_slice, cmap='hot', origin='lower')
plt.colorbar(label='|F|')
plt.title('Interference Pattern at t=500')
plt.xlabel('X (voxels)')
plt.ylabel('Y (voxels)')
plt.savefig('interference.png', dpi=150)
print("Saved: interference.png")
```

---

## Example: Conservation Tracking

```python
from fel_ca import FELSimulator, compute_energy, compute_helicity

sim = FELSimulator(N=128, device='cuda')
sim.init_sine_wave(wavelength=32, direction='x', polarization='y')

# Initial values
F = sim.get_flux()
E0 = compute_energy(F)
H0 = compute_helicity(F)

print(f"Initial: E={E0:.6e}, H={H0:.6e}")

# Run 10,000 steps
for t in range(10000):
    sim.step()
    
    if t % 1000 == 0:
        F = sim.get_flux()
        E_t = compute_energy(F)
        H_t = compute_helicity(F)
        
        dE = abs(E_t - E0) / E0
        dH = abs(H_t - H0) / abs(H0 + 1e-10)
        
        print(f"t={t:5d}: |ΔE|/E={dE:.2e}, |ΔH|/|H|={dH:.2e}")

# Check paper claim: within 10⁻⁷
F_final = sim.get_flux()
E_final = compute_energy(F_final)
rel_error = abs(E_final - E0) / E0

if rel_error <= 1e-7:
    print(f"✅ PASS: Energy conserved within 10⁻⁷ ({rel_error:.2e})")
else:
    print(f"⚠️  Energy drift: {rel_error:.2e}")
```

---

## Example: Walking-Boundary FEL Test

```python
from fel_ca import FELSimulator
from fel_ca.utils import create_spherical_boundary

sim = FELSimulator(N=128, device='cuda')
sim.init_sine_wave(wavelength=32, direction='x', polarization='y')

# Create boundaries at different scales
center = (64, 64, 64)
radii = [10, 15, 20, 25, 30]

boundaries = {}
for r in radii:
    faces, normals = create_spherical_boundary(center, r, 128)
    boundaries[r] = {
        'faces': faces.to('cuda'),
        'normals': normals.to('cuda')
    }
    print(f"Boundary r={r}: {len(faces)} faces")

# Run and measure flux
print("\nMeasuring flux through boundaries...")
for t in range(500):
    sim.step()
    
    if t % 100 == 0:
        print(f"\nt={t}:")
        for r in radii:
            flux = sim.compute_flux_through_boundary(
                boundaries[r]['faces'],
                boundaries[r]['normals']
            )
            print(f"  r={r:2d}: Φ(B) = {flux:+.6e}")

# Check FEL: Φ(B,t) should be ~0 for all B
print("\n" + "="*50)
for r in radii:
    flux = sim.compute_flux_through_boundary(
        boundaries[r]['faces'],
        boundaries[r]['normals']
    )
    
    if abs(flux) < 1e-6:
        print(f"✅ r={r}: Φ = {flux:+.6e} (FEL holds)")
    else:
        print(f"⚠️  r={r}: Φ = {flux:+.6e} (FEL violation)")
```

---

## Tips

### Device Selection

```python
import torch

# Auto-detect best device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

sim = FELSimulator(N=128, device=device)
```

### Integer Mode (Deterministic)

```python
from fel_ca import FELSimulatorInt

# Bitwise-identical results across runs
sim = FELSimulatorInt(N=128, device='cuda')

# Set seed for reproducibility
import torch
torch.manual_seed(42)

# Enable deterministic algorithms
torch.use_deterministic_algorithms(True)
```

### Memory Management

```python
# For large grids, monitor memory
import torch

sim = FELSimulator(N=256, device='cuda')  # ~2GB VRAM

# Clear cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Batch Processing

```python
# Run multiple experiments
wavelengths = [8, 16, 32, 64]

for lam in wavelengths:
    sim = FELSimulator(N=128, device='cuda')
    sim.init_sine_wave(wavelength=lam, direction='x', polarization='y')
    
    for t in range(1000):
        sim.step()
    
    energy = sim.get_energy()
    print(f"λ={lam}: E={energy:.6e}")
```

---

## Next Steps

1. **Read the Paper**: See `docs/paper.pdf` (when available)
2. **API Documentation**: See `docs/API.md`
3. **Run Experiments**: `python experiments/run_all.py --quick`
4. **Explore Examples**: Check `examples/` directory
5. **Contribute**: Open issues or PRs on GitHub

---

## Common Issues

### CUDA Out of Memory

```python
# Reduce grid size
sim = FELSimulator(N=64, device='cuda')  # Instead of N=128

# Or use CPU
sim = FELSimulator(N=128, device='cpu')
```

### MPS Not Available (macOS)

```python
# Fallback to CPU
try:
    sim = FELSimulator(N=128, device='mps')
except RuntimeError:
    sim = FELSimulator(N=128, device='cpu')
```

### Slow Performance

1. Use GPU (`device='cuda'` or `'mps'`)
2. Reduce grid size (N=64 instead of N=128)
3. Reduce number of steps
4. Use `--quick` flag for experiments

---

## Contact

Questions? cagri@aksay.co

GitHub: https://github.com/cagriaksay/fel-ca

