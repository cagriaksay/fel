#!/usr/bin/env python3
"""
Create a two-picture figure from single sine experiment showing energy drift.
This will be used in the paper to show actual measured drift values.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

# Load energy history
energy_file = Path('results/01_single_sine/energy_history.npy')
if not energy_file.exists():
    print(f"Error: {energy_file} not found. Please run the single sine experiment first.")
    sys.exit(1)

energy_history = np.load(energy_file)
timesteps = np.arange(len(energy_history))

print(f"Energy history shape: {energy_history.shape}")
print(f"Total steps: {len(energy_history)}")

# Calculate per-step drift (relative change per step)
# Use energy after emission as reference (skip first 25 steps of emission)
emission_duration = 25
if len(energy_history) > emission_duration:
    E_ref = energy_history[emission_duration]  # Energy after emission stops
    propagation_energies = energy_history[emission_duration:]
    propagation_timesteps = timesteps[emission_duration:]
    
    # Calculate per-step relative drift
    per_step_drift = np.abs(np.diff(propagation_energies)) / (propagation_energies[:-1] + 1e-10)
    
    print(f"\nEnergy after emission (t={emission_duration}): {E_ref:.6e}")
    print(f"Final energy (t={len(energy_history)-1}): {energy_history[-1]:.6e}")
    print(f"Total relative drift: {abs(energy_history[-1] - E_ref) / E_ref:.2e}")
    print(f"\nPer-step drift statistics (after emission):")
    print(f"  Max per-step drift: {np.max(per_step_drift):.2e}")
    print(f"  Mean per-step drift: {np.mean(per_step_drift):.2e}")
    print(f"  Median per-step drift: {np.median(per_step_drift):.2e}")
    print(f"  95th percentile: {np.percentile(per_step_drift, 95):.2e}")
    print(f"  99th percentile: {np.percentile(per_step_drift, 99):.2e}")
    
    # Create two-picture figure for paper
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: Energy over time (after emission)
    ax = axes[0]
    ax.plot(propagation_timesteps, propagation_energies, 'b-', linewidth=1.5, label='Energy E(t)')
    ax.axhline(E_ref, color='r', linestyle='--', linewidth=1, alpha=0.7, label=f'Reference (t={emission_duration})')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Energy E(t)', fontsize=12)
    ax.set_title('Energy Conservation (Stable Sine Wave)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Right panel: Per-step relative drift (log scale)
    ax = axes[1]
    ax.semilogy(propagation_timesteps[1:], per_step_drift, 'r-', linewidth=0.8, alpha=0.7, label='Per-step drift')
    ax.axhline(np.median(per_step_drift), color='g', linestyle='--', linewidth=1.5, 
               label=f'Median: {np.median(per_step_drift):.2e}')
    ax.axhline(1.8e-7, color='orange', linestyle=':', linewidth=1.5, 
               label='1.8×10⁻⁷ (paper claim)', alpha=0.8)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('|ΔE| / E (per step)', fontsize=12)
    ax.set_title('Per-Step Energy Drift', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path('results/01_single_sine')
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'sine_energy_drift.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved figure: {output_path / 'sine_energy_drift.png'}")
    
    # Also create a version with both panels side by side for paper
    plt.close()
    
else:
    print("Error: Energy history too short")

