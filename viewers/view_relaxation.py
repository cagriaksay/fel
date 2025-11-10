#!/usr/bin/env python3
"""
View Random Field Relaxation snapshots in Napari.

Loads all saved snapshots and displays them as a time series.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import napari
from glob import glob


def view_relaxation_snapshots(data_dir: str = 'results/03_random_field_relaxation'):
    """Load and view relaxation snapshots in Napari."""
    data_path = Path(data_dir)
    
    # Find all snapshot files
    snapshot_files = sorted(glob(str(data_path / 'snapshot_t*.npy')))
    
    if not snapshot_files:
        print(f"No snapshot files found in {data_path}")
        return
    
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
    
    # Create Napari viewer
    viewer = napari.Viewer(title="Random Field Relaxation (k=0)")
    
    # Add as 3D volume time series (only flux magnitude)
    # Time series shape is (time, z, y, x) = (9, N, N, N)
    viewer.add_image(
        time_series,
        name='Flux Magnitude |F|',
        colormap='hot',
        contrast_limits=(0, np.percentile(time_series, 99)),
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
    
    napari.run()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='View Random Field Relaxation in Napari')
    parser.add_argument('--data-dir', type=str, default='results/03_random_field_relaxation',
                       help='Directory containing snapshot files')
    
    args = parser.parse_args()
    
    view_relaxation_snapshots(args.data_dir)

