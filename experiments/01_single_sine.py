#!/usr/bin/env python3
"""
Experiment 1: Single-Sine Propagation

Validates coherent wave propagation with minimal dispersion and amplitude decay.

Paper Reference: Section 4.1, Figure: fig:sine
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from felca import FELSimulator


def run_single_sine_experiment(
    N: int = 128,
    wavelength: float = 32.0,
    n_steps: int = 1000,
    device: str = 'cuda',
    output_dir: str = 'results/01_single_sine'
):
    """
    Run single sine wave propagation experiment.
    
    Args:
        N: Lattice size (N³)
        wavelength: Wavelength in voxels (λ/Δx)
        n_steps: Number of simulation steps
        device: 'cuda', 'mps', or 'cpu'
        output_dir: Output directory for results
    """
    print("="*70)
    print("EXPERIMENT 1: Single-Sine Propagation")
    print("="*70)
    print(f"Grid: {N}³")
    print(f"Wavelength: λ/Δx = {wavelength}")
    print(f"Steps: {n_steps}")
    print(f"Device: {device}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create simulator
    sim = FELSimulator(
        N=N,
        device=device,
        rotation_rate=0.0
    )
    
    # Set up emitter (single emitter on left wall, pointing right)
    center = N // 2
    
    # Calculate oscillation axis properly (matching two-photon)
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Point right (+x)
    d = direction / (np.linalg.norm(direction) + 1e-6)
    
    # Create perpendicular basis vectors (matching two-photon logic)
    if abs(d[0]) > 0.9:  # Beam in X direction
        perp1 = np.array([0, 1, 0], dtype=np.float32)  # Y direction
        perp2 = np.array([0, 0, 1], dtype=np.float32)  # Z direction
    elif abs(d[1]) > 0.9:  # Beam in Y direction
        perp1 = np.array([1, 0, 0], dtype=np.float32)
        perp2 = np.array([0, 0, 1], dtype=np.float32)
    else:  # Beam in Z direction
        perp1 = np.array([1, 0, 0], dtype=np.float32)
        perp2 = np.array([0, 1, 0], dtype=np.float32)
    
    # Make perpendicular
    perp1 = perp1 - np.dot(perp1, d) * d
    perp1 = perp1 / (np.linalg.norm(perp1) + 1e-6)
    perp2 = np.cross(d, perp1)
    perp2 = perp2 / (np.linalg.norm(perp2) + 1e-6)
    
    # Oscillation axis (polarization_angle=0 means oscillate in perp1 direction)
    polarization_angle = 0.0  # flat polarization
    pol_rad = polarization_angle * np.pi / 180.0
    oscillation_axis = np.cos(pol_rad) * perp1 + np.sin(pol_rad) * perp2
    oscillation_axis = oscillation_axis / (np.linalg.norm(oscillation_axis) + 1e-6)
    
    emitter = {
        'base_center': np.array([0.0, center, center], dtype=np.float32),  # Left wall, center
        'direction': d,
        'oscillation_axis': oscillation_axis,
        'oscillation_amplitude': 2.0,  # Oscillate laterally
        'emission_amplitude': 2.0,
        'oscillation_speed': 2 * np.pi / 10.0,  # Oscillation period (10 steps per cycle, matching two-photon)
        'inward_bias': 0.0,  # No bias - emit straight
        'beam_width': 1.0,
        'phase': 0.0,
        'step_count': 0,
    }
    
    # Create coordinate meshgrid (FELSimulator uses [N, N, N, 3] shape)
    i = torch.arange(N, device=device, dtype=torch.float32)
    j = torch.arange(N, device=device, dtype=torch.float32)
    k = torch.arange(N, device=device, dtype=torch.float32)
    I, J, K = torch.meshgrid(i, j, k, indexing='ij')  # I=x, J=y, K=z (for FELSimulator [x,y,z,component])
    
    print("Using emitter-based wave generation...")
    print(f"Emitter: pos={tuple(emitter['base_center'])}, dir={tuple(emitter['direction'])}")
    print()
    
    # Storage for analysis
    energy_history = []
    
    # Initial energy
    E0 = sim.get_energy()
    energy_history.append(E0)
    print(f"Initial energy: {E0:.6e}")
    print()
    print("Running simulation...")
    
    # Emit for first 25 steps, then let wave propagate
    emission_duration = 25
    print(f"Emission: {emission_duration} steps, then propagate for {n_steps - emission_duration} steps")
    
    # Run simulation with emitter
    for t in tqdm(range(n_steps)):
        # Only emit for first emission_duration steps
        should_emit = (t < emission_duration)
        
        if should_emit:
            # Emit from emitter (add flux before physics step)
            base_center = emitter['base_center']
            direction = emitter['direction']
            oscillation_axis = emitter['oscillation_axis']
            osc_amplitude = emitter['oscillation_amplitude']
            emission_amplitude = emitter['emission_amplitude']
            oscillation_speed = emitter['oscillation_speed']
            inward_bias = emitter['inward_bias']
            beam_width = emitter['beam_width']
            phase = emitter['phase']
            step_count = emitter['step_count']
            
            # Calculate the emission offset from center (oscillation)
            current_offset_magnitude = osc_amplitude * np.sin(oscillation_speed * step_count + phase)
            emitter_offset = current_offset_magnitude * oscillation_axis
            emitter_pos = base_center + emitter_offset
            
            # Convert to torch
            emitter_pos_t = torch.tensor(emitter_pos, device=device, dtype=torch.float32)
            
            # Distance from current emitter position
            # I, J, K are [x, y, z] (meshgrid with indexing='ij'), emitter_pos is [x, y, z]
            dx = I - emitter_pos_t[0]  # I is X dimension
            dy = J - emitter_pos_t[1]  # J is Y dimension
            dz = K - emitter_pos_t[2]  # K is Z dimension
            dist = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-6)
            
            # Gaussian envelope around emitter position
            envelope = emission_amplitude * torch.exp(-dist**2 / (2 * beam_width**2))
            
            # Emission direction: forward + inward
            forward_x = direction[0]
            forward_y = direction[1]
            forward_z = direction[2]
            
            # Inward points back toward center (for oscillating emitters)
            inward_x = -emitter_offset[0]
            inward_y = -emitter_offset[1]
            inward_z = -emitter_offset[2]
            
            # Normalize inward
            inward_norm = np.sqrt(inward_x**2 + inward_y**2 + inward_z**2 + 1e-6)
            inward_x /= inward_norm
            inward_y /= inward_norm
            inward_z /= inward_norm
            
            # Combine forward + inward bias
            flux_x = (1.0 - inward_bias) * forward_x + inward_bias * inward_x
            flux_y = (1.0 - inward_bias) * forward_y + inward_bias * inward_y
            flux_z = (1.0 - inward_bias) * forward_z + inward_bias * inward_z
            
            # Normalize combined direction
            flux_norm = np.sqrt(flux_x**2 + flux_y**2 + flux_z**2 + 1e-6)
            flux_x /= flux_norm
            flux_y /= flux_norm
            flux_z /= flux_norm
            
            # Add to flux field (FELSimulator uses [N, N, N, 3] shape)
            sim.F_current[..., 0] += envelope * flux_x
            sim.F_current[..., 1] += envelope * flux_y
            sim.F_current[..., 2] += envelope * flux_z
            
            # Update step count for next iteration
            emitter['step_count'] = step_count + 1
        
        # Track energy BEFORE step (measure state at start of timestep)
        E_t = sim.get_energy()
        energy_history.append(E_t)
        
        # Physics step (always run, even when not emitting)
        sim.step()
        
        # Save snapshot right after emission stops (after step)
        if t == emission_duration:
            F_after_emit = sim.F_current.cpu().numpy()
            E_after_emit = sim.get_energy()  # Measure after step for snapshot
    
    # Final energy and snapshot (after)
    E_final = sim.get_energy()
    F_final = sim.F_current.cpu().numpy()
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Initial energy: {E0:.6e}")
    print(f"After emission (t={emission_duration}): {E_after_emit:.6e}")
    print(f"Final energy (t={n_steps}):   {E_final:.6e}")
    if E0 > 0:
        print(f"Relative change: {abs(E_final - E0) / E0:.6e}")
        print(f"Conservation (|ΔE|/E₀): {abs(E_final - E0) / E0:.2e}")
        rel_error = abs(E_final - E0) / E0
    else:
        print(f"Energy increase: {E_final:.6e}")
        rel_error = float('inf')
    print()
    
    # Check conservation claim: within 10⁻⁷ (only if E0 > 0)
    if E0 > 0 and abs(E_final - E0) / E0 <= 1e-7:
        print("✅ PASS: Energy conserved within 10⁻⁷")
    else:
        print("⚠️  WARNING: Energy drift exceeds 10⁻⁷")
    print()
    
    # Save data
    np.save(output_path / 'energy_history.npy', np.array(energy_history))
    
    # Plot energy conservation
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Convert to numpy array for easier manipulation
    energy_array = np.array(energy_history)
    timesteps = np.arange(len(energy_history))
    
    # Identify outliers (more than 3 std dev from mean after emission)
    if E0 == 0 and E_after_emit > 0:
        after_emit_energies = energy_array[emission_duration:]
        mean_after = np.mean(after_emit_energies)
        std_after = np.std(after_emit_energies)
        outlier_mask = np.abs(energy_array - mean_after) > 3 * std_after
        outlier_indices = np.where(outlier_mask)[0]
        if len(outlier_indices) > 0:
            print(f"⚠️  Detected {len(outlier_indices)} outlier(s) at step(s): {outlier_indices}")
            print(f"   Outlier values: {energy_array[outlier_indices]}")
    
    # Energy over time
    ax = axes[0]
    ax.plot(timesteps, energy_array, 'b-', linewidth=0.5, label='Energy')
    if E0 == 0 and E_after_emit > 0 and len(outlier_indices) > 0:
        # Mark outliers
        ax.scatter(timesteps[outlier_indices], energy_array[outlier_indices], 
                  color='red', s=50, zorder=5, label='Outliers', marker='x')
    ax.axvline(emission_duration, color='g', linestyle='--', alpha=0.5, label=f'Emission stops (t={emission_duration})')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Energy E(t)')
    ax.set_title('Energy Conservation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Relative error from energy after emission (or from E0 if E0 > 0)
    ax = axes[1]
    if E0 > 0:
        # Use E0 as reference
        E_ref = E0
        rel_error = np.abs(energy_array - E_ref) / E_ref
        ax.semilogy(timesteps, rel_error, 'r-', linewidth=0.5)
        ax.axhline(1e-7, color='k', linestyle='--', label='10⁻⁷ threshold')
        ax.set_ylabel('|ΔE| / E₀')
        ax.set_title('Relative Energy Error (from initial)')
    else:
        # Use energy after emission as reference
        E_ref = E_after_emit
        if E_ref > 0:
            rel_error = np.abs(energy_array - E_ref) / E_ref
            # Filter out outliers from the main plot line (but show them separately)
            if len(outlier_indices) > 0:
                # Plot main line excluding outliers
                normal_mask = ~outlier_mask
                ax.semilogy(timesteps[normal_mask], rel_error[normal_mask], 'r-', linewidth=0.5, label='Normal')
                # Plot outliers separately
                ax.semilogy(timesteps[outlier_indices], rel_error[outlier_indices], 
                           'rx', markersize=8, label='Outliers', zorder=5)
            else:
                ax.semilogy(timesteps, rel_error, 'r-', linewidth=0.5)
            ax.axvline(emission_duration, color='g', linestyle='--', alpha=0.5, label=f'Emission stops (t={emission_duration})')
            ax.axhline(1e-7, color='k', linestyle='--', label='10⁻⁷ threshold')
            ax.set_ylabel('|ΔE| / E(after emission)')
            ax.set_title('Relative Energy Error (from after emission)')
            ax.legend()
        else:
            # Fallback: show absolute energy
            ax.plot(timesteps, energy_array, 'r-', linewidth=0.5)
            ax.set_ylabel('Energy E(t)')
            ax.set_title('Energy Over Time (from zero)')
    ax.set_xlabel('Time Step')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'energy_conservation.png', dpi=300)
    print(f"Saved: {output_path / 'energy_conservation.png'}")
    
    # Save before/after snapshots (2D slice through center)
    center_y = N // 2
    center_z = N // 2
    
    # Check all components to see which has the signal
    Fx_after_emit = F_after_emit[:, :, center_z, 0]
    Fy_after_emit = F_after_emit[:, :, center_z, 1]
    Fz_after_emit = F_after_emit[:, :, center_z, 2]
    Fx_after = F_final[:, :, center_z, 0]
    Fy_after = F_final[:, :, center_z, 1]
    Fz_after = F_final[:, :, center_z, 2]
    
    print(f"Fx range: [{Fx_after_emit.min():.4f}, {Fx_after_emit.max():.4f}]")
    print(f"Fy range: [{Fy_after_emit.min():.4f}, {Fy_after_emit.max():.4f}]")
    print(f"Fz range: [{Fz_after_emit.min():.4f}, {Fz_after_emit.max():.4f}]")
    
    # Use Fx (emitter points in +x, so Fx should have the signal)
    # Calculate global min/max for Fx (symmetric for RdBu)
    vmin_fx = min(Fx_after_emit.min(), Fx_after.min())
    vmax_fx = max(Fx_after_emit.max(), Fx_after.max())
    vmax_fx_abs = max(abs(vmin_fx), abs(vmax_fx))
    vmin_fx_sym = -vmax_fx_abs
    vmax_fx_sym = vmax_fx_abs
    
    print(f"Fx color scale (symmetric): vmin={vmin_fx_sym:.4f}, vmax={vmax_fx_sym:.4f}")
    
    # After emission: Fx component at x-y plane (center z)
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(
        Fx_after_emit,
        cmap='RdBu',
        origin='lower',
        interpolation='bilinear',
        aspect='auto',
        vmin=vmin_fx_sym,
        vmax=vmax_fx_sym
    )
    ax.set_xlabel('Y Position (voxels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('X Position (voxels)', fontsize=12, fontweight='bold')
    ax.set_title(f'After Emission: Wave Packet (t={emission_duration})', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Flux Fx')
    plt.tight_layout()
    plt.savefig(output_path / 'snapshot_after_emit.png', dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'snapshot_after_emit.png'}")
    
    # Final: Fx component at x-y plane (center z)
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(
        Fx_after,
        cmap='RdBu',
        origin='lower',
        interpolation='bilinear',
        aspect='auto',
        vmin=vmin_fx_sym,
        vmax=vmax_fx_sym
    )
    ax.set_xlabel('Y Position (voxels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('X Position (voxels)', fontsize=12, fontweight='bold')
    ax.set_title(f'After: Propagated Wave (t={n_steps})', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Flux Fx')
    plt.tight_layout()
    plt.savefig(output_path / 'snapshot_after.png', dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'snapshot_after.png'}")
    
    print()
    print(f"All results saved to: {output_path}")
    print("="*70)
    
    return {
        'E0': E0,
        'E_final': E_final,
        'rel_error': abs(E_final - E0) / E0 if E0 > 0 else float('inf'),
        'energy_history': energy_history
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Single-Sine Propagation Experiment')
    parser.add_argument('--N', type=int, default=128, help='Lattice size')
    parser.add_argument('--wavelength', type=float, default=32.0, help='Wavelength (λ/Δx)')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--output', type=str, default='results/01_single_sine')
    
    args = parser.parse_args()
    
    results = run_single_sine_experiment(
        N=args.N,
        wavelength=args.wavelength,
        n_steps=args.steps,
        device=args.device,
        output_dir=args.output
    )

