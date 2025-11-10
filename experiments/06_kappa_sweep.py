#!/usr/bin/env python3
"""
Experiment 6: Rotation Coefficient (κ) Sweep

Sweeps the rotation coefficient κ from 0 to 1.0 in 0.05 increments to determine
the stability window and phase speed dependence.

This is a CRITICAL experiment for understanding the model's parameter space.

Paper Reference: Section "Rotation Coefficient Sweep"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from felca import FELSimulator


def run_kappa_sweep_experiment(
    N: int = 64,
    kappa_values: np.ndarray = None,
    n_steps: int = 100,
    random_init_amplitude: float = 1.0,
    device: str = 'cuda',
    output_dir: str = 'results/06_kappa_sweep'
):
    """
    Run κ sweep experiment with random initialization.
    
    Args:
        N: Lattice size (N³)
        kappa_values: Array of κ values to test
        n_steps: Number of steps per κ
        random_init_amplitude: Amplitude of random initial flux
        device: 'cuda', 'mps', or 'cpu'
        output_dir: Output directory
    """
    if kappa_values is None:
        kappa_values = np.arange(0.0, 1.05, 0.05)  # 0 to 1.0 in 0.05 increments
    
    print("="*70)
    print("EXPERIMENT 6: Rotation Coefficient (κ) Sweep")
    print("="*70)
    print(f"Grid: {N}³")
    print(f"κ range: {kappa_values[0]:.2f} to {kappa_values[-1]:.2f} ({len(kappa_values)} values)")
    print(f"Steps per κ: {n_steps}")
    print(f"Initialization: Random field (amplitude={random_init_amplitude})")
    print(f"Device: {device}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Storage: [κ_idx, time, z, y, x]
    all_magnitudes = []
    all_energies = []
    
    # Metrics per κ
    final_energies = []
    energy_stds = []
    energy_avgs = []
    max_magnitudes = []
    mean_magnitudes = []
    
    print("Running κ sweep...")
    print()
    
    for kappa_idx, kappa in enumerate(tqdm(kappa_values, desc="κ values")):
        # Create simulator with this κ
        sim = FELSimulator(
            N=N,
            device=device,
            rotation_rate=kappa
        )
        
        # Random initialization (same seed for consistency across κ)
        torch.manual_seed(137)  # Different seed to test consistency
        sim.F_current = random_init_amplitude * (
            torch.randn((N, N, N, 3), device=device, dtype=torch.float32)
        )
        
        # Normalize to reasonable energy
        E0 = sim.get_energy()
        if E0 > 0:
            # Scale to E ≈ N³ for visibility
            sim.F_current *= np.sqrt(N**3 / E0)
        
        # Storage for this κ
        magnitudes_k = []
        energies_k = []
        
        # Initial state
        mag = sim.get_magnitude().cpu().numpy()
        magnitudes_k.append(mag)
        energies_k.append(sim.get_energy())
        
        # Run simulation
        for t in range(n_steps):
            sim.step()
            
            mag = sim.get_magnitude().cpu().numpy()
            magnitudes_k.append(mag)
            energies_k.append(sim.get_energy())
        
        # Store
        all_magnitudes.append(np.array(magnitudes_k))
        all_energies.append(np.array(energies_k))
        
        # Compute metrics
        energies_array = np.array(energies_k)
        final_energies.append(energies_array[-1])
        energy_stds.append(np.std(energies_array))
        energy_avgs.append(np.mean(energies_array))  # Average energy over time
        max_magnitudes.append(np.max(magnitudes_k[-1]))
        mean_magnitudes.append(np.mean(magnitudes_k[-1]))
        
        if kappa_idx % 5 == 0:  # Print every 5th value
            print(f"  κ={kappa:.3f}: E_final={energies_array[-1]:.4f}, "
                  f"E_std={np.std(energies_array):.4e}, "
                  f"max|F|={np.max(magnitudes_k[-1]):.4f}")
    
    # Convert to arrays
    all_magnitudes = np.array(all_magnitudes)  # (n_kappa, n_steps+1, N, N, N)
    all_energies = np.array(all_energies)      # (n_kappa, n_steps+1)
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'κ':<8} {'E_final':<10} {'E_avg':<10} {'E_std':<12} {'max|F|':<10} {'mean|F|':<10}")
    print("-" * 70)
    for i, kappa in enumerate(kappa_values):
        print(f"{kappa:<8.3f} {final_energies[i]:<10.4f} {energy_avgs[i]:<10.4f} "
              f"{energy_stds[i]:<12.4e} "
              f"{max_magnitudes[i]:<10.4f} {mean_magnitudes[i]:<10.4f}")
    print()
    
    # Stability analysis
    print("Stability Analysis:")
    print("-" * 40)
    
    # Define stability: energy doesn't explode, no NaN/Inf
    stable_mask = np.array(final_energies) < 10.0  # Energy doesn't explode
    stable_mask &= np.isfinite(final_energies)
    stable_mask &= np.array(energy_stds) < 1.0  # Not too chaotic
    
    if np.any(stable_mask):
        stable_kappas = kappa_values[stable_mask]
        print(f"Stable κ range: {stable_kappas[0]:.3f} to {stable_kappas[-1]:.3f}")
        print(f"Number of stable κ: {np.sum(stable_mask)} / {len(kappa_values)}")
    else:
        print("⚠️  WARNING: No stable κ values found!")
    
    # Find optimal κ (stable with reasonable final energy)
    if np.any(stable_mask):
        stable_energies = np.array(final_energies)[stable_mask]
        # Optimal: energy close to 1.0 (normalized initial)
        energy_deviation = np.abs(stable_energies - 1.0)
        optimal_idx_in_stable = np.argmin(energy_deviation)
        optimal_kappa = stable_kappas[optimal_idx_in_stable]
        print(f"Optimal κ (closest to E=1.0): {optimal_kappa:.3f}")
    
    print()
    
    # Save data
    np.save(output_path / 'kappa_values.npy', kappa_values)
    np.save(output_path / 'all_magnitudes.npy', all_magnitudes)
    np.save(output_path / 'all_energies.npy', all_energies)
    np.save(output_path / 'final_energies.npy', final_energies)
    np.save(output_path / 'energy_avgs.npy', energy_avgs)
    np.save(output_path / 'energy_stds.npy', energy_stds)
    np.save(output_path / 'max_magnitudes.npy', max_magnitudes)
    np.save(output_path / 'mean_magnitudes.npy', mean_magnitudes)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Energy over time for all κ
    ax = axes[0, 0]
    for i, kappa in enumerate(kappa_values[::2]):  # Plot every other for clarity
        ax.plot(all_energies[i*2], label=f'κ={kappa:.2f}', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Energy E(t)')
    ax.set_title('Energy Evolution for Different κ')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Final energy vs κ
    ax = axes[0, 1]
    ax.plot(kappa_values, final_energies, 'o-', markersize=6)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Initial E')
    ax.set_xlabel('κ (rotation coefficient)')
    ax.set_ylabel('Final Energy E(T)')
    ax.set_title('Final Energy vs κ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy stability (std) vs κ
    ax = axes[1, 0]
    ax.semilogy(kappa_values, energy_stds, 'o-', markersize=6, color='red')
    ax.set_xlabel('κ (rotation coefficient)')
    ax.set_ylabel('Energy Std Dev')
    ax.set_title('Energy Stability vs κ')
    ax.grid(True, alpha=0.3)
    
    # Max magnitude vs κ
    ax = axes[1, 1]
    ax.plot(kappa_values, max_magnitudes, 'o-', markersize=6, color='purple', label='max|F|')
    ax.plot(kappa_values, mean_magnitudes, 's-', markersize=6, color='orange', label='mean|F|')
    ax.set_xlabel('κ (rotation coefficient)')
    ax.set_ylabel('|F|')
    ax.set_title('Flux Magnitude vs κ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'kappa_sweep_summary.png', dpi=300)
    print(f"Saved: {output_path / 'kappa_sweep_summary.png'}")
    
    # Plot heatmap: κ vs time
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Average magnitude over space for each (κ, t)
    spatial_avg = np.mean(all_magnitudes, axis=(2, 3, 4))  # (n_kappa, n_steps+1)
    
    im = ax.imshow(
        spatial_avg,
        aspect='auto',
        origin='lower',
        cmap='hot',
        extent=[0, n_steps, kappa_values[0], kappa_values[-1]],
        interpolation='nearest'
    )
    ax.set_xlabel('Time Step')
    ax.set_ylabel('κ (rotation coefficient)')
    ax.set_title('Spatially-Averaged |F| vs κ and Time')
    plt.colorbar(im, ax=ax, label='⟨|F|⟩')
    plt.tight_layout()
    plt.savefig(output_path / 'kappa_time_heatmap.png', dpi=300)
    print(f"Saved: {output_path / 'kappa_time_heatmap.png'}")
    
    print()
    print(f"All results saved to: {output_path}")
    print("="*70)
    
    return {
        'kappa_values': kappa_values,
        'all_magnitudes': all_magnitudes,
        'all_energies': all_energies,
        'final_energies': final_energies,
        'energy_stds': energy_stds,
        'stable_mask': stable_mask
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='κ Sweep Experiment')
    parser.add_argument('--N', type=int, default=64, help='Lattice size')
    parser.add_argument('--kappa-min', type=float, default=0.0, help='Minimum κ')
    parser.add_argument('--kappa-max', type=float, default=1.0, help='Maximum κ')
    parser.add_argument('--kappa-step', type=float, default=0.05, help='κ step size')
    parser.add_argument('--steps', type=int, default=100, help='Steps per κ')
    parser.add_argument('--amplitude', type=float, default=1.0, help='Random init amplitude')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--output', type=str, default='results/06_kappa_sweep')
    
    args = parser.parse_args()
    
    # Generate κ values
    kappa_values = np.arange(args.kappa_min, args.kappa_max + args.kappa_step/2, args.kappa_step)
    
    results = run_kappa_sweep_experiment(
        N=args.N,
        kappa_values=kappa_values,
        n_steps=args.steps,
        random_init_amplitude=args.amplitude,
        device=args.device,
        output_dir=args.output
    )

