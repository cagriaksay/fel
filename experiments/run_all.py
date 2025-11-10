#!/usr/bin/env python3
"""
Run all FEL-CA paper experiments sequentially.

This script runs all 4 critical experiments needed for the paper:
1. Single-Sine Propagation (includes energy conservation)
2. Two-Sine Interference
3. Walking-Boundary FEL Verification
4. Dispersion Analysis

Use --quick for a fast sanity check (reduced parameters).
"""

import sys
import time
from pathlib import Path

# Import all experiment modules
import importlib.util

def load_experiment(name, filepath):
    """Dynamically load experiment module."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_all_experiments(quick_mode=False, device='cuda', output_base='results'):
    """
    Run all experiments.
    
    Args:
        quick_mode: If True, use reduced parameters for fast testing
        device: 'cuda', 'mps', or 'cpu'
        output_base: Base directory for all results
    """
    print("="*70)
    print("FEL-CA PAPER EXPERIMENTS")
    print("="*70)
    print()
    
    if quick_mode:
        print("⚡ QUICK MODE: Running with reduced parameters")
        print("   (For full paper results, run without --quick)")
        print()
    
    start_time = time.time()
    
    experiments_dir = Path(__file__).parent
    
    # ========================================================================
    # Experiment 1: Single-Sine Propagation
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1/5: Single-Sine Propagation")
    print("="*70 + "\n")
    
    exp1 = load_experiment('exp1', experiments_dir / '01_single_sine.py')
    
    if quick_mode:
        exp1.run_single_sine_experiment(
            N=64, wavelength=16.0, n_steps=200, device=device,
            output_dir=f'{output_base}/01_single_sine'
        )
    else:
        exp1.run_single_sine_experiment(
            N=128, wavelength=32.0, n_steps=1000, device=device,
            output_dir=f'{output_base}/01_single_sine'
        )
    
    # ========================================================================
    # Experiment 2: Two-Sine Interference
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2/5: Two-Sine Interference")
    print("="*70 + "\n")
    
    exp2 = load_experiment('exp2', experiments_dir / '02_two_sine_interference.py')
    
    if quick_mode:
        # Run only phase=0
        exp2.run_two_sine_experiment(
            N=64, wavelength=16.0, phase_offset=0.0, n_steps=200, device=device,
            output_dir=f'{output_base}/02_two_sine'
        )
    else:
        # Run both phase configurations
        print("\n>>> Phase Offset: 0 (stationary pattern)\n")
        exp2.run_two_sine_experiment(
            N=128, wavelength=32.0, phase_offset=0.0, n_steps=500, device=device,
            output_dir=f'{output_base}/02_two_sine'
        )
        
        print("\n>>> Phase Offset: π/2 (drifting fringes)\n")
        import numpy as np
        exp2.run_two_sine_experiment(
            N=128, wavelength=32.0, phase_offset=np.pi/2, n_steps=500, device=device,
            output_dir=f'{output_base}/02_two_sine'
        )
    
    # ========================================================================
    # Experiment 3: Walking-Boundary FEL Verification
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 3/4: Walking-Boundary FEL Verification ⚠️  CRITICAL")
    print("="*70 + "\n")
    
    exp3 = load_experiment('exp3', experiments_dir / '04_walking_boundary.py')
    
    if quick_mode:
        exp3.run_walking_boundary_experiment(
            N=64, radii=[8, 12, 16, 20], n_steps=200, device=device,
            output_dir=f'{output_base}/04_walking_boundary'
        )
    else:
        exp3.run_walking_boundary_experiment(
            N=128, radii=[10, 15, 20, 25, 30], n_steps=500, device=device,
            output_dir=f'{output_base}/04_walking_boundary'
        )
    
    # ========================================================================
    # Experiment 4: Dispersion Analysis
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 4/4: Dispersion Analysis")
    print("="*70 + "\n")
    
    exp4 = load_experiment('exp4', experiments_dir / '05_dispersion.py')
    
    if quick_mode:
        exp4.run_dispersion_experiment(
            N=64, wavelengths=[8, 12, 16, 24], n_steps=200, device=device,
            output_dir=f'{output_base}/05_dispersion'
        )
    else:
        exp4.run_dispersion_experiment(
            N=128, wavelengths=[8, 12, 16, 24, 32, 48, 64], n_steps=500, device=device,
            output_dir=f'{output_base}/05_dispersion'
        )
    
    # ========================================================================
    # Summary
    # ========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Results saved to: {output_base}/")
    print()
    print("Next steps:")
    print("  1. Review results in each experiment subdirectory")
    print("  2. Check PNG plots for visual verification")
    print("  3. Verify claims:")
    print("     - FEL holds (Φ(B,t) ≈ 0 for all boundaries)")
    print("     - Phase error ≤ 1% for λ/Δx ≥ 32")
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run all FEL-CA paper experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (reduced parameters, ~5 minutes)
  python run_all.py --quick
  
  # Full paper experiments (~2 hours on GPU)
  python run_all.py --device cuda
  
  # Run on CPU
  python run_all.py --device cpu --quick
        """
    )
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with reduced parameters')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'mps', 'cpu'],
                       help='Device to run on')
    parser.add_argument('--output', type=str, default='results',
                       help='Base output directory')
    
    args = parser.parse_args()
    
    run_all_experiments(
        quick_mode=args.quick,
        device=args.device,
        output_base=args.output
    )

