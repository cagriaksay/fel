#!/usr/bin/env python3
"""
Experiment 3: Random-Field Relaxation

Demonstrates how random initialization evolves toward smooth, near-static fields.
Local cancellations damp incoherent components; no large-scale coherence emerges.

Paper Reference: Section "Random-Field Relaxation", Figure: fig:random
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from felca import FELSimulator


def run_random_field_relaxation_experiment(
    N: int = 128,
    n_steps: int = 500,
    rotation_rate: float = 0.615,
    device: str = 'cuda',
    output_dir: str = 'results/03_random_field_relaxation',
    save_snapshots: bool = True,
    seed: int = 42
):
    """
    Run random field relaxation experiment.
    
    Args:
        N: Lattice size (N³)
        n_steps: Number of simulation steps
        rotation_rate: Rotation coefficient κ
        device: 'cuda', 'mps', or 'cpu'
        output_dir: Output directory for results
        save_snapshots: Whether to save snapshot images
        seed: Random seed for initialization
    """
    print("="*70)
    print("EXPERIMENT 3: Random-Field Relaxation")
    print("="*70)
    print(f"Grid: {N}³")
    print(f"Steps: {n_steps}")
    print(f"Rotation rate (κ): {rotation_rate}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create simulator
    sim = FELSimulator(
        N=N,
        device=device,
        rotation_rate=rotation_rate
    )
    
    # Initialize with uniform random field F(x,y,z) ∈ [-1,1]³
    print("Initializing with uniform random field F ∈ [-1,1]³...")
    np.random.seed(seed)
    sim.F_current = torch.tensor(
        np.random.uniform(-1.0, 1.0, size=(N, N, N, 3)),
        device=device,
        dtype=torch.float32
    )
    
    # Compute initial energy
    initial_energy = torch.sum(sim.F_current**2).item()
    print(f"Initial energy: {initial_energy:.3e}")
    print()
    
    # Track metrics over time
    timesteps = []
    energies = []
    energy_drifts = []
    flux_magnitudes = []  # Average |F| per voxel
    
    # Calculate target timesteps for relaxation evolution (9 frames evenly distributed)
    # Include t=0 (initial) and then evenly distribute remaining 8 frames
    num_evolution_frames = 9
    target_times = [0] + [int(1 + i * (n_steps - 1) / (num_evolution_frames - 1)) for i in range(1, num_evolution_frames)]
    target_times[-1] = n_steps  # Ensure last frame is at the final timestep
    print(f"Target timesteps for relaxation evolution: {target_times}")
    print()
    
    # Store slices for evolution visualization
    relaxation_slices = []
    relaxation_times = []
    relaxation_energies = []
    
    print(f"Running {n_steps} steps...")
    for t in tqdm(range(n_steps), desc="Simulation"):
        # Save initial state before first step
        if t == 0 and save_snapshots:
            energy = torch.sum(sim.F_current**2).item()
            slice_data = get_slice_data(sim)
            relaxation_slices.append(slice_data)
            relaxation_times.append(0)
            relaxation_energies.append(energy)
            
            # Save full 3D snapshot for Napari
            snapshot_data = sim.F_current.clone().cpu().numpy()
            np.save(output_path / 'snapshot_t0000.npy', snapshot_data)
        
        sim.step()
        
        # Compute metrics
        energy = torch.sum(sim.F_current**2).item()
        energy_drift = (energy - initial_energy) / initial_energy
        avg_flux_mag = torch.mean(torch.linalg.norm(sim.F_current, dim=-1)).item()
        
        timesteps.append(t + 1)
        energies.append(energy)
        energy_drifts.append(energy_drift)
        flux_magnitudes.append(avg_flux_mag)
        
        # Save snapshot at every step
        if save_snapshots:
            snapshot_data = sim.F_current.clone().cpu().numpy()
            np.save(output_path / f'snapshot_t{t+1:04d}.npy', snapshot_data)
        
        # Save relaxation pattern at target intervals (for 9-frame visualization)
        if save_snapshots and len(relaxation_slices) < num_evolution_frames:
            if (t + 1) in target_times:
                slice_data = get_slice_data(sim)
                relaxation_slices.append(slice_data)
                relaxation_times.append(t + 1)
                relaxation_energies.append(energy)
    
    if save_snapshots:
        if len(relaxation_slices) > 0:
            print(f"Relaxation evolution frames saved at: {relaxation_times}")
        print(f"3D snapshots saved for Napari: {n_steps + 1} snapshots (t=0 to t={n_steps})")
        print()
    
    # Convert to arrays
    timesteps = np.array(timesteps)
    energies = np.array(energies)
    energy_drifts = np.array(energy_drifts)
    flux_magnitudes = np.array(flux_magnitudes)
    
    # Save data
    np.save(output_path / 'timesteps.npy', timesteps)
    np.save(output_path / 'energies.npy', energies)
    np.save(output_path / 'energy_drifts.npy', energy_drifts)
    np.save(output_path / 'flux_magnitudes.npy', flux_magnitudes)
    
    # Print summary
    print()
    print("="*70)
    print("RESULTS: Random-Field Relaxation")
    print("="*70)
    print(f"Initial energy: {initial_energy:.3e}")
    print(f"Final energy: {energies[-1]:.3e}")
    print(f"Energy drift: {energy_drifts[-1]:.6f} ({energy_drifts[-1]*100:.4f}%)")
    print(f"Initial avg |F|: {flux_magnitudes[0]:.6f}")
    print(f"Final avg |F|: {flux_magnitudes[-1]:.6f}")
    print(f"Reduction: {(1 - flux_magnitudes[-1]/flux_magnitudes[0])*100:.2f}%")
    print()
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Energy over time
    ax = axes[0, 0]
    ax.plot(timesteps, energies, 'b-', linewidth=1.5)
    ax.axhline(initial_energy, color='r', linestyle='--', linewidth=1, label='Initial')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Total Energy E(t)')
    ax.set_title('Energy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy drift
    ax = axes[0, 1]
    ax.plot(timesteps, energy_drifts * 100, 'r-', linewidth=1.5)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Energy Drift (%)')
    ax.set_title('Relative Energy Drift')
    ax.grid(True, alpha=0.3)
    
    # Average flux magnitude
    ax = axes[1, 0]
    ax.plot(timesteps, flux_magnitudes, 'g-', linewidth=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Average |F| per Voxel')
    ax.set_title('Flux Magnitude Evolution')
    ax.grid(True, alpha=0.3)
    
    # Energy vs flux magnitude
    ax = axes[1, 1]
    ax.scatter(flux_magnitudes, energies, c=timesteps, cmap='viridis', alpha=0.6, s=10)
    ax.set_xlabel('Average |F| per Voxel')
    ax.set_ylabel('Total Energy')
    ax.set_title('Energy vs Flux Magnitude')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Time Step')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'relaxation_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'relaxation_metrics.png'}")
    
    # Plot relaxation evolution (9 frames)
    if save_snapshots and len(relaxation_slices) == num_evolution_frames:
        plot_relaxation_evolution(relaxation_slices, relaxation_times, relaxation_energies, output_path, n_steps, initial_energy)
    
    print()
    print(f"All results saved to: {output_path}")
    print("="*70)


def get_slice_data(sim):
    """Extract 2D slice data from simulator for visualization."""
    N = sim.N
    center = N // 2
    
    # Take a slice through the center (z = center)
    F_slice = sim.F_current[:, :, center, :].cpu().numpy()
    
    # Compute magnitude
    F_mag = np.linalg.norm(F_slice, axis=-1)
    
    return {
        'F_slice': F_slice,
        'F_mag': F_mag
    }


def plot_relaxation_evolution(relaxation_slices, relaxation_times, relaxation_energies, output_path, n_steps, initial_energy):
    """Plot 9-frame relaxation evolution in a 3x3 grid with individual color scaling and energy annotations."""
    num_frames = len(relaxation_slices)
    if num_frames != 9:
        print(f"Warning: Expected 9 frames, got {num_frames}")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, (slice_data, t, energy) in enumerate(zip(relaxation_slices, relaxation_times, relaxation_energies)):
        ax = axes[i]
        F_mag = slice_data['F_mag']
        
        # Individual color scaling for each frame (0 to 99th percentile)
        vmin = 0
        vmax = np.percentile(F_mag, 99)
        
        im = ax.imshow(F_mag, cmap='hot', vmin=vmin, vmax=vmax, origin='lower')
        
        # Title with time and energy
        title = f't = {t}\nE = {energy:.2e}'
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar to each subplot
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('|F|', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    
    plt.suptitle(f'Random Field Relaxation Evolution ({n_steps} steps)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'relaxation_evolution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'relaxation_evolution.png'}")


def save_snapshot(sim, path, title):
    """Save a 2D slice snapshot of the flux field."""
    N = sim.N
    center = N // 2
    
    # Take a slice through the center (z = center)
    F_slice = sim.F_current[:, :, center, :].cpu().numpy()
    
    # Compute magnitude
    F_mag = np.linalg.norm(F_slice, axis=-1)
    
    # Normalize for visualization
    vmin, vmax = 0, np.percentile(F_mag, 99)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # X component
    im1 = axes[0].imshow(F_slice[:, :, 0], cmap='RdBu', vmin=-1, vmax=1, origin='lower')
    axes[0].set_title(f'{title} - Fx Component')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])
    
    # Y component
    im2 = axes[1].imshow(F_slice[:, :, 1], cmap='RdBu', vmin=-1, vmax=1, origin='lower')
    axes[1].set_title(f'{title} - Fy Component')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1])
    
    # Magnitude
    im3 = axes[2].imshow(F_mag, cmap='hot', vmin=vmin, vmax=vmax, origin='lower')
    axes[2].set_title(f'{title} - |F| Magnitude')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Random-Field Relaxation Experiment')
    parser.add_argument('--N', type=int, default=128, help='Lattice size')
    parser.add_argument('--steps', type=int, default=500, help='Number of steps')
    parser.add_argument('--k', type=float, default=0.615, help='Rotation rate (kappa)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--output', type=str, default='results/03_random_field_relaxation')
    parser.add_argument('--no-snapshots', action='store_true', help='Skip snapshot saving')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
    
    args = parser.parse_args()
    
    run_random_field_relaxation_experiment(
        N=args.N,
        n_steps=args.steps,
        rotation_rate=args.k,
        device=args.device,
        output_dir=args.output,
        save_snapshots=not args.no_snapshots,
        seed=args.seed
    )

