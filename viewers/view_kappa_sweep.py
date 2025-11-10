#!/usr/bin/env python3
"""
Napari viewer for κ sweep experiment.

Visualizes flux magnitude for different κ values with a slider.
"""

import sys
from pathlib import Path
import numpy as np
import napari


def view_kappa_sweep(data_dir: str = 'results/06_kappa_sweep'):
    """
    Launch Napari viewer for κ sweep data.
    
    Args:
        data_dir: Directory containing sweep results
    """
    data_path = Path(data_dir)
    
    # Load data
    print("Loading κ sweep data...")
    kappa_values = np.load(data_path / 'kappa_values.npy')
    all_magnitudes = np.load(data_path / 'all_magnitudes.npy')
    
    # all_magnitudes shape: (n_kappa, n_steps+1, N, N, N)
    n_kappa, n_steps, N, _, _ = all_magnitudes.shape
    
    print(f"Loaded data:")
    print(f"  κ values: {len(kappa_values)} ({kappa_values[0]:.3f} to {kappa_values[-1]:.3f})")
    print(f"  Time steps: {n_steps}")
    print(f"  Grid size: {N}³")
    print()
    
    # Reshape for Napari: (n_kappa, n_steps, Z, Y, X)
    # Napari will interpret first dim as slider
    data = all_magnitudes  # Already in correct shape
    
    print("Launching Napari...")
    print("Use the slider at the bottom to change κ")
    print()
    
    # Create viewer
    viewer = napari.Viewer(ndisplay=3)  # Force 3D display
    
    # Add as image layer with κ slider
    layer = viewer.add_image(
        data,
        name='Flux Magnitude |F|',
        colormap='hot',
        contrast_limits=[0, np.percentile(data, 99)],
        rendering='mip',  # Maximum intensity projection for 3D
        axis_labels=['κ', 'time', 'z', 'y', 'x']
    )
    
    # Create κ value lookup
    def get_kappa_for_slice(kappa_idx):
        if 0 <= kappa_idx < len(kappa_values):
            return kappa_values[kappa_idx]
        return 0.0
    
    # Add text overlay showing current κ
    @viewer.bind_key('k')
    def print_kappa(viewer):
        """Press 'k' to print current κ value."""
        kappa_idx = viewer.dims.current_step[0]
        time_idx = viewer.dims.current_step[1]
        kappa = get_kappa_for_slice(kappa_idx)
        print(f"κ = {kappa:.3f}, time = {time_idx}")
    
    # Print instructions
    print("="*70)
    print("Napari Controls:")
    print("="*70)
    print("  - Left sidebar sliders: Navigate κ and time")
    print("  - First slider: Change κ value (0.00 to 1.00)")
    print("  - Second slider: Time step (0 to {})".format(n_steps-1))
    print("  - Mouse: Rotate 3D view")
    print("  - Scroll: Zoom in/out")
    print("  - Press 'k': Print current κ and time to console")
    print("  - Top-right buttons: Toggle 2D/3D view")
    print("="*70)
    print()
    print(f"Initial: κ = {kappa_values[0]:.3f}")
    print()
    print("TIP: Use 'mip' rendering (already set) to see the full 3D volume")
    
    # Start viewer
    napari.run()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='View κ Sweep Results in Napari')
    parser.add_argument('data_dir', type=str, nargs='?', 
                       default='../results/06_kappa_sweep',
                       help='Directory containing sweep results')
    
    args = parser.parse_args()
    
    # Check if data exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_path}")
        print()
        print("Run the experiment first:")
        print(f"  python experiments/06_kappa_sweep.py")
        sys.exit(1)
    
    if not (data_path / 'kappa_values.npy').exists():
        print(f"Error: κ sweep data not found in {data_path}")
        print()
        print("Run the experiment first:")
        print(f"  python experiments/06_kappa_sweep.py")
        sys.exit(1)
    
    view_kappa_sweep(args.data_dir)

