#!/usr/bin/env python3
"""
Experiment 2: Two-Sine Interference

Demonstrates deterministic superposition and stable interference patterns.

Paper Reference: Section 4.2, Figure: fig:interference
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from felca import FELSimulator


def run_two_sine_experiment(
    N: int = 128,
    wavelength: float = 32.0,
    phase_offset: float = 0.0,
    n_steps: int = 500,
    device: str = 'cuda',
    output_dir: str = 'results/02_two_sine',
    save_snapshots: bool = True
):
    """
    Run two-source interference experiment.
    
    Args:
        N: Lattice size (N³)
        wavelength: Wavelength in voxels
        phase_offset: Phase difference between sources (0 or π/2)
        n_steps: Number of simulation steps
        device: 'cuda', 'mps', or 'cpu'
        output_dir: Output directory
    """
    print("="*70)
    print("EXPERIMENT 2: Two-Sine Interference")
    print("="*70)
    print(f"Grid: {N}³")
    print(f"Wavelength: λ/Δx = {wavelength}")
    print(f"Phase offset: Δφ = {phase_offset:.3f} rad ({phase_offset*180/np.pi:.1f}°)")
    print(f"Steps: {n_steps}")
    print(f"Device: {device}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    if phase_offset == 0:
        output_path = output_path / 'phase_0'
    else:
        output_path = output_path / f'phase_{phase_offset:.3f}'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create simulator
    sim = FELSimulator(N=N, device=device, rotation_rate=0.0)
    
    # Set up two emitters: one at back (z), one at right (x)
    # Emitter 1: at z=10 (back/bottom of XZ picture), pointing +z (toward center)
    # Emitter 2: at x=N-10 (right side of XZ picture), pointing -x (left toward center)
    print("Setting up two emitter-based sources (orthogonal)...")
    z1 = 10  # Back wall (bottom of XZ picture)
    x2 = N - 10  # Right wall (right side of XZ picture)
    center = N // 2
    
    # Define diagonal screen in XZ plane (spanning all Y)
    # Diagonal from (x=80, z=0) to (x=127, z=48) in XZ space
    screen_x0, screen_z0 = 80, 0  # Start point
    screen_x1, screen_z1 = 127, 48  # End point
    
    # Create coordinate meshgrid (FELSimulator uses [N, N, N, 3] shape)
    i = torch.arange(N, device=device, dtype=torch.float32)
    j = torch.arange(N, device=device, dtype=torch.float32)
    k = torch.arange(N, device=device, dtype=torch.float32)
    I, J, K = torch.meshgrid(i, j, k, indexing='ij')  # I=x, J=y, K=z
    
    # Emitter 1: at z=10, pointing +z (toward center)
    direction1 = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Point +z
    d1 = direction1 / (np.linalg.norm(direction1) + 1e-6)
    
    # Create perpendicular basis vectors (matching exp 1 logic)
    if abs(d1[0]) > 0.9:  # Beam in X direction
        perp1_1 = np.array([0, 1, 0], dtype=np.float32)  # Y direction
        perp2_1 = np.array([0, 0, 1], dtype=np.float32)  # Z direction
    elif abs(d1[1]) > 0.9:  # Beam in Y direction
        perp1_1 = np.array([1, 0, 0], dtype=np.float32)
        perp2_1 = np.array([0, 0, 1], dtype=np.float32)
    else:  # Beam in Z direction
        perp1_1 = np.array([1, 0, 0], dtype=np.float32)  # X direction
        perp2_1 = np.array([0, 1, 0], dtype=np.float32)  # Y direction
    
    # Make perpendicular
    perp1_1 = perp1_1 - np.dot(perp1_1, d1) * d1
    perp1_1 = perp1_1 / (np.linalg.norm(perp1_1) + 1e-6)
    perp2_1 = np.cross(d1, perp1_1)
    perp2_1 = perp2_1 / (np.linalg.norm(perp2_1) + 1e-6)
    
    # Oscillation axis (polarization_angle=0 means oscillate in perp1 direction)
    polarization_angle = 0.0  # flat polarization
    pol_rad = polarization_angle * np.pi / 180.0
    oscillation_axis1 = np.cos(pol_rad) * perp1_1 + np.sin(pol_rad) * perp2_1
    oscillation_axis1 = oscillation_axis1 / (np.linalg.norm(oscillation_axis1) + 1e-6)
    
    emitter1 = {
        'base_center': np.array([center, center, z1], dtype=np.float32),
        'direction': d1,
        'oscillation_axis': oscillation_axis1,
        'oscillation_amplitude': 2.0,
        'emission_amplitude': 2.0,
        'oscillation_speed': 2 * np.pi / 10.0,  # Match exp 1: 10 steps per cycle
        'inward_bias': 0.0,
        'beam_width': 1.0,  # Match exp 1: tighter beam
        'phase': 0.0,  # First emitter has phase 0
        'step_count': 0,
    }
    
    # Emitter 2: at x=N-10 (right side of XZ picture), pointing -x (left toward center)
    direction2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # Point -x (left)
    d2 = direction2 / (np.linalg.norm(direction2) + 1e-6)
    
    # Create perpendicular basis vectors (matching exp 1 logic)
    if abs(d2[0]) > 0.9:  # Beam in X direction
        perp1_2 = np.array([0, 1, 0], dtype=np.float32)  # Y direction
        perp2_2 = np.array([0, 0, 1], dtype=np.float32)  # Z direction
    elif abs(d2[1]) > 0.9:  # Beam in Y direction
        perp1_2 = np.array([1, 0, 0], dtype=np.float32)
        perp2_2 = np.array([0, 0, 1], dtype=np.float32)
    else:  # Beam in Z direction
        perp1_2 = np.array([1, 0, 0], dtype=np.float32)  # X direction
        perp2_2 = np.array([0, 1, 0], dtype=np.float32)  # Y direction
    
    # Make perpendicular
    perp1_2 = perp1_2 - np.dot(perp1_2, d2) * d2
    perp1_2 = perp1_2 / (np.linalg.norm(perp1_2) + 1e-6)
    perp2_2 = np.cross(d2, perp1_2)
    perp2_2 = perp2_2 / (np.linalg.norm(perp2_2) + 1e-6)
    
    # Rotate polarization by 90 degrees for emitter 2 to match emitter 1
    polarization_angle2 = 90.0  # 90 degrees rotation
    pol_rad2 = polarization_angle2 * np.pi / 180.0
    oscillation_axis2 = np.cos(pol_rad2) * perp1_2 + np.sin(pol_rad2) * perp2_2
    oscillation_axis2 = oscillation_axis2 / (np.linalg.norm(oscillation_axis2) + 1e-6)
    
    emitter2 = {
        'base_center': np.array([x2, center, center], dtype=np.float32),  # Right wall (right side of XZ picture)
        'direction': d2,
        'oscillation_axis': oscillation_axis2,
        'oscillation_amplitude': 2.0,
        'emission_amplitude': 2.0,
        'oscillation_speed': 2 * np.pi / 10.0,  # Match exp 1: 10 steps per cycle
        'inward_bias': 0.0,
        'beam_width': 1.0,  # Match exp 1: tighter beam
        'phase': phase_offset,  # Second emitter has phase offset
        'step_count': 0,
    }
    
    print(f"Emitter 1: pos={tuple(emitter1['base_center'])}, dir={tuple(emitter1['direction'])}, phase=0")
    print(f"Emitter 2: pos={tuple(emitter2['base_center'])}, dir={tuple(emitter2['direction'])}, phase={phase_offset:.3f}")
    print(f"Screen: diagonal from (x={screen_x0}, z={screen_z0}) to (x={screen_x1}, z={screen_z1}) in XZ plane")
    
    # Storage
    interference_slices = []  # Save XY slices at center Z
    interference_times = []  # Timesteps when interference slices were saved
    screen_patterns = []  # 2D interference patterns on diagonal screen (diagonal_pos × Y)
    screen_pattern_times = []  # Timesteps when patterns were saved
    
    # Storage for full 3D snapshots (for Napari)
    if save_snapshots:
        print("Will save full 3D snapshots for Napari viewing")
    
    # Screen parameters (coordinates already defined above)
    screen_dx = screen_x1 - screen_x0
    screen_dz = screen_z1 - screen_z0
    screen_length = np.sqrt(screen_dx**2 + screen_dz**2)
    
    # Screen normal vector (perpendicular to diagonal, pointing "forward")
    # Diagonal direction: (screen_dx, 0, screen_dz)
    # Normal in XZ plane: (-screen_dz, 0, screen_dx) normalized
    screen_normal_norm = np.sqrt(screen_dx**2 + screen_dz**2)
    screen_normal = np.array([-screen_dz / screen_normal_norm, 0.0, screen_dx / screen_normal_norm])  # (nx, ny, nz)
    
    # Perpendicular direction for offset (to create two parallel lines)
    perp_norm = screen_length
    perp_dx = -screen_dz / perp_norm  # Perpendicular X component
    perp_dz = screen_dx / perp_norm   # Perpendicular Z component
    
    # Two back-to-back planes: use integer offset of 1 voxel on either side of diagonal
    line_offset = 1.0
    
    # Sample points along diagonal
    num_samples = int(screen_length) + 1
    t_values = np.linspace(0, 1, num_samples)
    
    print(f"Screen: diagonal from ({screen_x0}, {screen_z0}) to ({screen_x1}, {screen_z1})")
    print(f"  Length: {screen_length:.1f}, Samples: {num_samples}")
    
    # Enumerate all screen indices that will be checked
    print(f"\nScreen indices to be checked (all {num_samples} points, integer indices only):")
    screen_indices = []
    for i, t in enumerate(t_values):
        x_center = int(np.round(screen_x0 + t * screen_dx))
        z_center = int(np.round(screen_z0 + t * screen_dz))
        x1_idx = int(np.round(x_center + line_offset * perp_dx))
        z1_idx = int(np.round(z_center + line_offset * perp_dz))
        x2_idx = int(np.round(x_center - line_offset * perp_dx))
        z2_idx = int(np.round(z_center - line_offset * perp_dz))
        x1_idx = np.clip(x1_idx, 0, N-1)
        z1_idx = np.clip(z1_idx, 0, N-1)
        x2_idx = np.clip(x2_idx, 0, N-1)
        z2_idx = np.clip(z2_idx, 0, N-1)
        screen_indices.append((x1_idx, z1_idx, x2_idx, z2_idx))
        print(f"  Point {i:3d}: plane1 (x={x1_idx:3d}, z={z1_idx:3d}), plane2 (x={x2_idx:3d}, z={z2_idx:3d})")
    print(f"  (Each point spans all Y values 0-{N-1}, total {num_samples * 2 * N} voxels checked)")
    print()
    
    # Create two parallel diagonal lines
    screen_points_line1 = []  # First line
    screen_points_line2 = []  # Second line (offset)
    
    for t in t_values:
        # Main diagonal point
        x_center = screen_x0 + t * screen_dx
        z_center = screen_z0 + t * screen_dz
        
        # Two parallel points
        x1 = x_center + line_offset * perp_dx
        z1 = z_center + line_offset * perp_dz
        x2 = x_center - line_offset * perp_dx
        z2 = z_center - line_offset * perp_dz
        
        # Clamp to grid bounds and round to integer voxel positions
        x1_idx = int(np.round(np.clip(x1, 0, N-1)))
        z1_idx = int(np.round(np.clip(z1, 0, N-1)))
        x2_idx = int(np.round(np.clip(x2, 0, N-1)))
        z2_idx = int(np.round(np.clip(z2, 0, N-1)))
        
        screen_points_line1.append((x1_idx, z1_idx))
        screen_points_line2.append((x2_idx, z2_idx))
    
    # Combine both lines (remove duplicates)
    screen_points = list(set(screen_points_line1 + screen_points_line2))
    
    # Initialize accumulated screen pattern (1D along diagonal, spanning all Y)
    accumulated_screen_pattern = np.zeros((num_samples, N))  # (position along diagonal, Y position)
    
    print(f"Screen: Diagonal in XZ plane from ({screen_x0}, z={screen_z0}) to ({screen_x1}, z={screen_z1})")
    print(f"  Two parallel lines (offset ±{line_offset} voxels) to avoid gaps")
    print(f"  {len(screen_points)} unique screen points, spanning all Y positions")
    print(f"  Screen length: {screen_length:.1f} voxels, {num_samples} sample points")
    print(f"Grid center: ({center}, {center}, {center})")
    print(f"Emitter 1: ({center}, {center}, {z1}) pointing +z")
    print(f"Emitter 2: ({x2}, {center}, {center}) pointing -x")
    print()
    
    # Initial screen pattern is zero (accumulated pattern starts empty)
    
    # Emit for first 25 steps, then let waves propagate and interfere
    emission_duration = min(15, n_steps - 5)  # Emit for 15 steps or until 5 steps before end
    print(f"Emission: {emission_duration} steps, then propagate for {n_steps - emission_duration} steps")
    print()
    
    # Calculate target timesteps for interference evolution (9 frames evenly distributed)
    # Distribute from t=1 to t=n_steps-1 (last step)
    num_evolution_frames = 9
    target_times = [1 + int(i * (n_steps - 2) / (num_evolution_frames - 1)) for i in range(num_evolution_frames)]
    # Ensure last frame is at the final timestep
    target_times[-1] = n_steps - 1
    print(f"Target timesteps for interference evolution: {target_times}")
    
    # Run simulation
    for t in tqdm(range(n_steps)):
        t_int = int(t)  # Ensure t is an integer
        # Only emit for first emission_duration steps
        should_emit = (t_int < emission_duration)
        
        if should_emit:
            # Emit from both emitters
            for emitter in [emitter1, emitter2]:
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
        
        # Measure flux along diagonal screen BEFORE step (to capture flux that's about to cross)
        # Get flux magnitude field
        F_mag = sim.get_magnitude()  # (N, N, N) = (X, Y, Z) - magnitude at each voxel
        F = sim.F_current  # (N, N, N, 3) - flux vector field
        
        # Sample along diagonal using the two parallel lines
        screen_pattern = np.zeros((num_samples, N))  # (position along diagonal, Y position)
        
        # For each point along the diagonal
        for i, t in enumerate(t_values):
            # Main diagonal point (round to integer)
            x_center = int(np.round(screen_x0 + t * screen_dx))
            z_center = int(np.round(screen_z0 + t * screen_dz))
            
            # Two back-to-back planes (one on each side of diagonal)
            # Plane 1: offset in +normal direction (integer offset)
            x1_idx = int(np.round(x_center + line_offset * perp_dx))
            z1_idx = int(np.round(z_center + line_offset * perp_dz))
            # Plane 2: offset in -normal direction (integer offset)
            x2_idx = int(np.round(x_center - line_offset * perp_dx))
            z2_idx = int(np.round(z_center - line_offset * perp_dz))
            
            # Clamp to grid bounds
            x1_idx = np.clip(x1_idx, 0, N-1)
            z1_idx = np.clip(z1_idx, 0, N-1)
            x2_idx = np.clip(x2_idx, 0, N-1)
            z2_idx = np.clip(z2_idx, 0, N-1)
            
            # Sample flux magnitude from both planes (spanning all Y)
            # Measure all flux at screen location (magnitude, not just crossing component)
            # FIXED: F_mag shape is (X, Y, Z) but we need to access as F_mag[z, y, x] for screen in XZ plane
            # Screen coordinates are (x, z) in XZ plane, but flux field is indexed as (X, Y, Z)
            # So we access F_mag[z_idx, :, x_idx] to get all Y at screen position (x, z)
            flux_mag1 = F_mag[z1_idx, :, x1_idx].cpu().numpy()  # (Y,) flux magnitude at plane 1: F_mag[z, y, x]
            flux_mag2 = F_mag[z2_idx, :, x2_idx].cpu().numpy()  # (Y,) flux magnitude at plane 2: F_mag[z, y, x]
            screen_pattern[i, :] = flux_mag1 + flux_mag2  # Sum both planes
            
            # Zero out flux at both planes (absorb flux that touches screen)
            # This prevents flux from continuing past the screen
            # F shape is (X, Y, Z, 3), so we need F[z, y, x, :] to match
            F[z1_idx, :, x1_idx, :] = 0.0  # Zero flux at plane 1
            F[z2_idx, :, x2_idx, :] = 0.0  # Zero flux at plane 2
        
        # Add to accumulated pattern (additive screen - sum of all flux at each step)
        accumulated_screen_pattern += screen_pattern
        
        # Save full 3D snapshot at every step (for Napari)
        if save_snapshots:
            snapshot_data = sim.F_current.clone().cpu().numpy()
            np.save(output_path / f'snapshot_t{t_int+1:04d}.npy', snapshot_data)
        
        # Now do the physics step (after measuring and zeroing flux at screen)
        sim.step()
        
        # Debug: check when flux reaches screen (after step)
        if t_int in [50, 100, 125, 150, 154] or t_int == n_steps - 1:
            max_at_step = screen_pattern.max()
            max_accumulated = accumulated_screen_pattern.max()
            # Get post-step flux for debugging
            F_mag_post = sim.get_magnitude()
            max_field = float(torch.max(F_mag_post).item())
            # Find where max flux is in the field
            max_idx = torch.argmax(F_mag_post.view(-1))
            max_pos = np.unravel_index(max_idx.item(), (N, N, N))
            # Check center of diagonal screen
            mid_i = num_samples // 2
            x_mid = int(screen_x0 + 0.5 * screen_dx)
            z_mid = int(screen_z0 + 0.5 * screen_dz)
            y_mid = N // 2
            val_at_center = float(F_mag_post[x_mid, y_mid, z_mid].item())
            total_screen_flux = screen_pattern.sum()
            print(f"  t={t_int}: Screen pattern max: {max_at_step:.6e}, Accumulated: {max_accumulated:.6e}, Total: {total_screen_flux:.6e}")
            print(f"    Field max: {max_field:.6e} at (x={max_pos[0]}, y={max_pos[1]}, z={max_pos[2]}), Screen center (x={x_mid}, y={y_mid}, z={z_mid}): {val_at_center:.6e}")
            # At final step, check flux at several screen points and neighbors
            if t_int == n_steps - 1:
                print(f"  Final step: Checking flux at screen points and neighbors...")
                for check_i in [0, num_samples//4, num_samples//2, 3*num_samples//4, num_samples-1]:
                    t_check = t_values[check_i]
                    x_check = int(np.round(screen_x0 + t_check * screen_dx))
                    z_check = int(np.round(screen_z0 + t_check * screen_dz))
                    val_at_point = float(F_mag_post[x_check, y_mid, z_check].item())
                    # Check neighbors
                    val_xp1 = float(F_mag_post[min(x_check+1, N-1), y_mid, z_check].item()) if x_check+1 < N else 0.0
                    val_xm1 = float(F_mag_post[max(x_check-1, 0), y_mid, z_check].item()) if x_check-1 >= 0 else 0.0
                    val_zp1 = float(F_mag_post[x_check, y_mid, min(z_check+1, N-1)].item()) if z_check+1 < N else 0.0
                    val_zm1 = float(F_mag_post[x_check, y_mid, max(z_check-1, 0)].item()) if z_check-1 >= 0 else 0.0
                    print(f"    Screen point {check_i} (x={x_check}, z={z_check}): flux={val_at_point:.6e}")
                    print(f"      Neighbors: x+1={val_xp1:.6e}, x-1={val_xm1:.6e}, z+1={val_zp1:.6e}, z-1={val_zm1:.6e}")
        
        # Save snapshot right after emission stops (after step)
        if save_snapshots and t_int == emission_duration:
            # Sum all Y slices to get XZ projection (looking along Y axis to see waves along Z)
            magnitude = sim.get_magnitude()  # (N, N, N) = (X, Y, Z)
            slice_after_emit = torch.sum(magnitude, dim=1).cpu().numpy()  # Sum along Y axis -> (X, Z)
            E_after_emit = sim.get_energy()  # Measure after step for snapshot
        
        # Save interference pattern at regular intervals for evolution
        # Save if this timestep matches one of our target times and we haven't saved it yet
        if t_int > 0 and len(interference_slices) < num_evolution_frames:
            # Check if this is a target timestep
            is_target = t_int in target_times
            if is_target:
                # Sum all Y slices to get XZ projection
                magnitude = sim.get_magnitude()  # (N, N, N) = (X, Y, Z)
                slice_xz = torch.sum(magnitude, dim=1).cpu().numpy()  # Sum along Y axis -> (X, Z)
                interference_slices.append(slice_xz)
                interference_times.append(t_int)  # Store the actual timestep as int
                if len(interference_slices) <= 3:  # Debug first few
                    print(f"  Saved interference slice {len(interference_slices)-1} at t={t_int}")
    
    # Save final accumulated pattern
    if save_snapshots:
        screen_patterns.append(accumulated_screen_pattern.copy())  # Final accumulated pattern
        screen_pattern_times.append(n_steps - 1)
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Steps: {n_steps}")
    if len(interference_times) > 0:
        print(f"Interference evolution frames saved at: {interference_times}")
    if phase_offset == 0:
        print("Expected: Stationary interference pattern")
    else:
        print("Expected: Drifting interference fringes")
    print()
    
    # Save data
    np.save(output_path / 'interference_slices.npy', np.array(interference_slices))
    np.save(output_path / 'screen_patterns.npy', np.array(screen_patterns))
    
    # Plot final accumulated interference pattern on screen
    if save_snapshots and len(screen_patterns) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        pattern = screen_patterns[0]  # Final accumulated pattern
        # Pattern is 2D: (num_samples along diagonal, Y position)
        # Transpose to swap axes: (Y position, diagonal position)
        im = ax.imshow(pattern.T, cmap='hot', origin='lower', aspect='auto', interpolation='bilinear')
        ax.set_xlabel('Position Along Diagonal Screen')
        ax.set_ylabel('Y Position (voxels)')
        ax.set_title(f'Accumulated Energy on Diagonal Screen (t={screen_pattern_times[0]}, Δφ = {phase_offset*180/np.pi:.1f}°)')
        plt.colorbar(im, ax=ax, label='Accumulated Energy')
        plt.tight_layout()
        plt.savefig(output_path / 'screen_pattern.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path / 'screen_pattern.png'}")
    
    # Plot interference pattern evolution
    n_frames = min(9, len(interference_slices))
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes[:n_frames]):
        idx = i
        if idx < len(interference_slices):
            im = ax.imshow(interference_slices[idx], cmap='hot', origin='lower')
            # Draw diagonal screen line in XZ projection
            ax.plot([screen_x0, screen_x1], [screen_z0, screen_z1], 'w-', linewidth=2, alpha=0.8, linestyle='--', label='Screen')
            # Use the actual timestep when this frame was saved
            t_step = interference_times[idx] if idx < len(interference_times) else (idx + 1)
            ax.set_title(f't = {t_step}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.axis('off')
    
    for ax in axes[n_frames:]:
        ax.axis('off')
    
    plt.suptitle(f'Interference Pattern Evolution (Δφ = {phase_offset*180/np.pi:.1f}°)')
    plt.tight_layout()
    plt.savefig(output_path / 'interference_evolution.png', dpi=300)
    print(f"Saved: {output_path / 'interference_evolution.png'}")
    
    # Save snapshots if requested
    if save_snapshots:
        # Save snapshot after emission stops
        if 'slice_after_emit' in locals():
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(slice_after_emit, cmap='hot', origin='lower', interpolation='bilinear')
            # Draw diagonal screen line in XZ projection
            ax.plot([screen_x0, screen_x1], [screen_z0, screen_z1], 'w-', linewidth=2, alpha=0.8, linestyle='--', label='Screen')
            ax.set_title(f'After Emission (t={emission_duration}, Δφ = {phase_offset*180/np.pi:.1f}°)')
            ax.set_xlabel('X (voxels)')
            ax.set_ylabel('Z (voxels)')
            plt.colorbar(im, ax=ax, label='|F|')
            plt.tight_layout()
            plt.savefig(output_path / 'snapshot_after_emit.png', dpi=300)
            print(f"Saved: {output_path / 'snapshot_after_emit.png'}")
        
    if save_snapshots:
        print(f"3D snapshots saved for Napari: {n_steps} snapshots (t=1 to t={n_steps})")
    
    print()
    print(f"All results saved to: {output_path}")
    print("="*70)
    
    return {
        'interference_slices': interference_slices,
        'screen_patterns': screen_patterns
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Two-Sine Interference Experiment')
    parser.add_argument('--N', type=int, default=128, help='Lattice size')
    parser.add_argument('--wavelength', type=float, default=32.0, help='Wavelength')
    parser.add_argument('--phase', type=float, default=0.0, 
                       help='Phase offset in radians (0 or π/2=1.571)')
    parser.add_argument('--steps', type=int, default=20, help='Number of steps')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--output', type=str, default='results/02_two_sine')
    parser.add_argument('--both-phases', action='store_true', 
                       help='Run both phase=0 and phase=π/2')
    parser.add_argument('--save-snapshots', action='store_true', default=True,
                       help='Save snapshot images (default: True)')
    parser.add_argument('--no-snapshots', dest='save_snapshots', action='store_false',
                       help='Disable snapshot saving')
    
    args = parser.parse_args()
    
    if args.both_phases:
        print("Running both phase configurations...\n")
        
        # Phase = 0 (stationary)
        print("\n>>> Phase Offset: 0 (stationary pattern)\n")
        run_two_sine_experiment(
            N=args.N,
            wavelength=args.wavelength,
            phase_offset=0.0,
            n_steps=args.steps,
            device=args.device,
            output_dir=args.output,
            save_snapshots=args.save_snapshots
        )
        
        # Phase = π/2 (drifting)
        print("\n>>> Phase Offset: π/2 (drifting fringes)\n")
        run_two_sine_experiment(
            N=args.N,
            wavelength=args.wavelength,
            phase_offset=np.pi/2,
            n_steps=args.steps,
            device=args.device,
            output_dir=args.output,
            save_snapshots=args.save_snapshots
        )
    else:
        run_two_sine_experiment(
            N=args.N,
            wavelength=args.wavelength,
            phase_offset=args.phase,
            n_steps=args.steps,
            device=args.device,
            output_dir=args.output,
            save_snapshots=args.save_snapshots
        )

