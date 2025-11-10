#!/usr/bin/env python3
"""
Run all experiments for the FEL paper with exact parameters used in the paper.

This script reproduces all figures and results mentioned in the paper:
- Experiment 1: Single-Sine Propagation (energy drift figure)
- Experiment 2: Two-Sine Interference (90-degree phase)
- Experiment 3: Random-Field Relaxation (seed=123, κ=0)
- Experiment 4: Walking-Boundary (κ=0 only)
- Experiment 5: Benchmarks (CPU vs GPU, determinism)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import importlib.util

def load_experiment(name, filepath):
    """Dynamically load experiment module."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detect_device():
    """Try GPU first, fall back to CPU."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def main():
    print("="*70)
    print("FEL PAPER EXPERIMENTS - EXACT PARAMETERS")
    print("="*70)
    print()
    
    device = detect_device()
    print(f"Using device: {device.upper()}")
    print()
    
    experiments_dir = Path(__file__).parent
    
    # ========================================================================
    # Experiment 1: Single-Sine Propagation
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: Single-Sine Propagation")
    print("="*70)
    print("Parameters:")
    print("  N = 128³")
    print("  wavelength = 32.0")
    print("  n_steps = 10,000")
    print("  κ = 0")
    print()
    
    exp1 = load_experiment('exp1', experiments_dir / '01_single_sine.py')
    exp1.run_single_sine_experiment(
        N=128,
        wavelength=32.0,
        n_steps=10000,
        device=device,
        output_dir='results/01_single_sine'
    )
    
    # ========================================================================
    # Experiment 2: Two-Sine Interference (90-degree phase)
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: Two-Sine Interference (90-degree phase)")
    print("="*70)
    print("Parameters:")
    print("  N = 128³")
    print("  wavelength = 32.0")
    print("  phase_offset = π/2 (1.571)")
    print("  n_steps = 155")
    print("  κ = 0")
    print()
    
    exp2 = load_experiment('exp2', experiments_dir / '02_two_sine_interference.py')
    exp2.run_two_sine_experiment(
        N=128,
        wavelength=32.0,
        phase_offset=np.pi/2,  # 90 degrees
        n_steps=155,
        device=device,
        output_dir='results/02_two_sine',
        save_snapshots=True
    )
    
    # ========================================================================
    # Experiment 3: Random-Field Relaxation
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 3: Random-Field Relaxation")
    print("="*70)
    print("Parameters:")
    print("  N = 128³")
    print("  n_steps = 500")
    print("  κ = 0")
    print("  seed = 123")
    print()
    
    exp3 = load_experiment('exp3', experiments_dir / '03_random_field_relaxation.py')
    exp3.run_random_field_relaxation_experiment(
        N=128,
        n_steps=500,
        rotation_rate=0.0,  # κ = 0
        device=device,
        output_dir='results/03_random_field_relaxation',
        save_snapshots=True,
        seed=123
    )
    
    # ========================================================================
    # Experiment 4: Walking-Boundary (κ=0 only)
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 4: Walking-Boundary FEL Verification (κ=0)")
    print("="*70)
    print("Parameters:")
    print("  N = 128³")
    print("  box_sizes = [10, 20, 30, 40, 50]")
    print("  n_steps = 10")
    print("  warmup_steps = 20")
    print("  κ = 0")
    print()
    
    exp4 = load_experiment('exp4', experiments_dir / '04_walking_boundary.py')
    exp4.run_walking_boundary_experiment(
        N=128,
        box_sizes=[10, 20, 30, 40, 50],
        n_steps=10,
        warmup_steps=20,
        rotation_rate=0.0,  # κ = 0
        device=device,
        output_dir='results/04_walking_boundary'
    )
    
    # ========================================================================
    # Experiment 5: Benchmarks
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 5: Integer vs Float vs CPU vs GPU Benchmarks")
    print("="*70)
    print("Parameters:")
    print("  N = 64³")
    print("  n_steps = 100")
    print("  κ = 0.01")
    print("  seed = 42")
    print()
    
    exp5 = load_experiment('exp5', experiments_dir / '05_benchmarks.py')
    exp5.run_benchmark_experiment(
        N=64,
        n_steps=100,
        rotation_rate=0.01,  # κ = 0.01
        seed=42,
        output_dir='results/05_benchmarks'
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print()
    print("Generated figures:")
    print("  - results/01_single_sine/snapshot_after_emit.png")
    print("  - results/01_single_sine/snapshot_after.png")
    print("  - results/02_two_sine/phase_1.571/interference_evolution.png")
    print("  - results/02_two_sine/phase_1.571/screen_pattern.png")
    print("  - results/03_random_field_relaxation/relaxation_evolution.png")
    print("  - results/04_walking_boundary/walking_boundary.png")
    print("  - results/05_benchmarks/benchmark_results.json")
    print()
    print("All results saved to: results/")
    print("="*70)


if __name__ == '__main__':
    main()

