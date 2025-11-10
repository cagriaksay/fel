#!/usr/bin/env python3
"""
View Two-Sine Interference snapshots in Napari.

Loads all saved snapshots and displays them as a time series.
Supports viewing different phase configurations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import napari
from glob import glob


def view_two_sine_snapshots(data_dir: str = 'results/02_two_sine', phase: str = None):
    """
    Load and view two-sine interference snapshots in Napari.
    
    Args:
        data_dir: Base directory containing phase subdirectories
        phase: Specific phase to view ('0', '0.785', '1.571', or None for all)
    """
    data_path = Path(data_dir)
    
    # Determine which phase directories to load
    if phase is not None:
        phase_dirs = [data_path / f'phase_{phase}']
    else:
        # Find all phase directories
        phase_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('phase_')])
    
    if not phase_dirs:
        print(f"No phase directories found in {data_path}")
        return
    
    print(f"Found {len(phase_dirs)} phase configuration(s)")
    
    # Load snapshots from first phase directory (or specified one)
    phase_dir = phase_dirs[0]
    snapshot_files = sorted(glob(str(phase_dir / 'snapshot_t*.npy')))
    
    if not snapshot_files:
        print(f"No snapshot files found in {phase_dir}")
        return
    
    print(f"Loading snapshots from: {phase_dir.name}")
    print(f"Found {len(snapshot_files)} snapshots")
    print("Loading snapshots...")
    
    # Load all snapshots
    snapshots = []
    for i, f in enumerate(snapshot_files):
        data = np.load(f)
        # Data shape is (N, N, N, 3) - flux vector field
        # Compute magnitude for visualization
        F_mag = np.linalg.norm(data, axis=-1)  # (N, N, N)
        snapshots.append(F_mag)
        if i == 0 or i == len(snapshot_files) - 1 or (i + 1) % 20 == 0:
            print(f"  Loaded {Path(f).name}: shape {data.shape}, |F| range [{F_mag.min():.4f}, {F_mag.max():.4f}]")
    
    if len(snapshot_files) > 2:
        print(f"  ... ({len(snapshot_files) - 2} more snapshots) ...")
    
    # Stack into time series: (time, N, N, N)
    time_series = np.stack(snapshots, axis=0)
    print(f"\nTime series shape: {time_series.shape}")
    
    # Get grid size from first snapshot
    first_snapshot_full = np.load(snapshot_files[0])
    N = first_snapshot_full.shape[0]
    
    # Calculate contrast limits (handle case where all values might be zero)
    max_val = np.max(time_series)
    if max_val > 0:
        contrast_max = np.percentile(time_series[time_series > 0], 99) if np.any(time_series > 0) else max_val
        contrast_limits = (0, float(contrast_max))
    else:
        # All zeros - set a default range
        contrast_limits = (0, 1.0)
        print("Warning: All values are zero, using default contrast limits")
    
    print(f"Contrast limits: {contrast_limits}")
    
    # Extract phase from directory name for title
    phase_str = phase_dir.name.replace('phase_', '')
    try:
        phase_deg = float(phase_str) * 180 / np.pi
        title = f"Two-Sine Interference (Δφ = {phase_deg:.1f}°)"
    except:
        title = f"Two-Sine Interference ({phase_dir.name})"
    
    # Create Napari viewer
    viewer = napari.Viewer(title=title)
    
    # Add as 3D volume time series (only flux magnitude)
    # Time series shape is (time, z, y, x)
    viewer.add_image(
        time_series,
        name='Flux Magnitude |F|',
        colormap='hot',
        contrast_limits=contrast_limits,
        scale=(1, 1, 1, 1),  # (t, z, y, x) - one unit per dimension
        axis_labels=['time', 'z', 'y', 'x'],
        blending='additive',
        opacity=0.8
    )
    
    # Create wireframe box around the volume
    # Define the 8 corners of the box (in z, y, x order for Napari)
    corners = np.array([
        [0, 0, 0],         # front bottom left
        [N-1, 0, 0],       # front bottom right
        [N-1, N-1, 0],     # front top right
        [0, N-1, 0],       # front top left
        [0, 0, N-1],       # back bottom left
        [N-1, 0, N-1],     # back bottom right
        [N-1, N-1, N-1],   # back top right
        [0, N-1, N-1]      # back top left
    ], dtype=float)
    
    # Define connections for the 12 edges
    connections = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # front face
        [4, 5], [5, 6], [6, 7], [7, 4],  # back face
        [0, 4], [1, 5], [2, 6], [3, 7]   # connecting edges
    ])
    
    # Create all line segments for the wireframe
    line_segments = []
    for edge in connections:
        line_segments.append([corners[edge[0]], corners[edge[1]]])
    
    # Add all edges as a single shapes layer
    wireframe_lines = np.array(line_segments)  # (12, 2, 3)
    
    viewer.add_shapes(
        wireframe_lines,
        shape_type='line',
        name='Wireframe',
        edge_width=1,
        edge_color='white',
        opacity=0.8,
        visible=True
    )
    
    # Set to 3D display mode - this will show 3D volume and only time as slider
    viewer.dims.ndisplay = 3
    
    print("\nNapari viewer opened!")
    print("Controls:")
    print(f"  - Time slider: Navigate through {len(snapshot_files)} snapshots")
    print("  - 3D view: Rotate and zoom the volume")
    print("  - Adjust contrast limits in layer controls")
    print(f"\nPhase configuration: {phase_dir.name}")
    if len(phase_dirs) > 1:
        print(f"  (Other phases available: {[d.name for d in phase_dirs[1:]]})")
    
    napari.run()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='View Two-Sine Interference in Napari')
    parser.add_argument('--data-dir', type=str, default='results/02_two_sine',
                       help='Base directory containing phase subdirectories')
    parser.add_argument('--phase', type=str, default=None,
                       help='Specific phase to view (0, 0.785, 1.571, or None for first available)')
    
    args = parser.parse_args()
    
    view_two_sine_snapshots(args.data_dir, args.phase)

