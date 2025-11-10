#!/usr/bin/env python3
"""
Experiment 5: Integer vs Float vs CPU vs GPU Benchmarks

Compares performance and determinism across:
- Integer (Q1.31) vs Float32 implementations
- CPU vs GPU (MPS/CUDA) devices
- Verifies bitwise determinism with same seed

Paper Reference: Section "Integer vs. Float Benchmarks"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from felca import FELSimulator, FELSimulatorInt


def run_simulation_with_seed(
    sim_class,
    N: int,
    n_steps: int,
    device: str,
    seed: int,
    rotation_rate: float = 0.01
):
    """
    Run simulation with a specific seed and return final state.
    
    Returns:
        tuple: (F_final, time_per_step_ms, metrics_dict)
    """
    # Create simulator
    if sim_class == FELSimulatorInt:
        sim = sim_class(N=N, device=device, rotation_rate=rotation_rate)
    else:
        sim = sim_class(N=N, device=device, rotation_rate=rotation_rate, dtype=torch.float32)
    
    # Initialize with random field using seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    elif device == 'mps':
        # MPS doesn't have manual_seed, but we use numpy seed
        pass
    
    sim.F_current = torch.tensor(
        np.random.uniform(-1.0, 1.0, size=(N, N, N, 3)),
        device=device,
        dtype=torch.float32
    )
    
    # Store initial state
    F_initial = sim.F_current.clone()
    initial_energy = torch.sum(sim.F_current**2).item()
    
    # Warmup (if GPU)
    if device != 'cpu':
        for _ in range(10):
            sim.step()
        sim.F_current = F_initial.clone()
    
    # Benchmark and run
    torch.cuda.synchronize() if device == 'cuda' else None
    torch.mps.synchronize() if device == 'mps' else None
    
    start_time = time.time()
    for _ in range(n_steps):
        sim.step()
    
    torch.cuda.synchronize() if device == 'cuda' else None
    torch.mps.synchronize() if device == 'mps' else None
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_step_ms = (total_time / n_steps) * 1000
    
    # Get final state
    F_final = sim.F_current.clone().cpu()
    final_energy = torch.sum(sim.F_current**2).item()
    
    # Compute metrics
    F_diff = F_final - F_initial.cpu()
    component_drift_max = torch.max(torch.abs(F_diff)).item()
    component_drift_2norm = torch.linalg.norm(F_diff).item()
    component_drift_2norm_rel = component_drift_2norm / (torch.linalg.norm(F_initial.cpu()).item() + 1e-10)
    energy_drift = (final_energy - initial_energy) / (initial_energy + 1e-10)
    
    metrics = {
        'time_per_step_ms': time_per_step_ms,
        'total_time': total_time,
        'component_drift_max': component_drift_max,
        'component_drift_2norm': component_drift_2norm,
        'component_drift_2norm_rel': component_drift_2norm_rel,
        'energy_drift': energy_drift,
        'initial_energy': initial_energy,
        'final_energy': final_energy
    }
    
    return F_final, time_per_step_ms, metrics


def compare_determinism(F1, F2, name1, name2):
    """Compare two final states for bitwise determinism."""
    diff = F1 - F2
    max_diff = torch.max(torch.abs(diff)).item()
    mean_diff = torch.mean(torch.abs(diff)).item()
    rms_diff = torch.sqrt(torch.mean(diff**2)).item()
    relative_diff = rms_diff / (torch.sqrt(torch.mean(F1**2)).item() + 1e-10)
    
    # Check bitwise equality
    is_bitwise_equal = torch.allclose(F1, F2, rtol=0, atol=0)
    num_different = torch.sum(torch.abs(diff) > 1e-10).item()
    total_elements = F1.numel()
    
    return {
        'is_bitwise_equal': is_bitwise_equal,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'rms_diff': rms_diff,
        'relative_diff': relative_diff,
        'num_different': num_different,
        'total_elements': total_elements,
        'percent_different': (num_different / total_elements) * 100
    }


def run_benchmark_experiment(
    N: int = 128,
    n_steps: int = 200,
    rotation_rate: float = 0.01,
    seed: int = 42,
    output_dir: str = 'results/05_benchmarks'
):
    """
    Run comprehensive benchmark experiment comparing CPU vs GPU and determinism.
    
    Args:
        N: Lattice size (N³)
        n_steps: Number of steps to benchmark
        rotation_rate: Rotation coefficient κ
        seed: Random seed for determinism testing
        output_dir: Output directory for results
    """
    print("="*70)
    print("EXPERIMENT 5: Integer vs Float vs CPU vs GPU Benchmarks")
    print("="*70)
    print(f"Grid: {N}³")
    print(f"Steps: {n_steps}")
    print(f"Rotation rate (κ): {rotation_rate}")
    print(f"Seed: {seed}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine available devices (need CPU and at least one GPU)
    gpu_device = None
    if torch.cuda.is_available():
        gpu_device = 'cuda'
    elif torch.backends.mps.is_available():
        gpu_device = 'mps'
    
    if gpu_device is None:
        print("⚠️  No GPU available. Will only benchmark CPU.")
        devices = ['cpu']
    else:
        devices = ['cpu', gpu_device]
        print(f"Available devices: CPU and {gpu_device.upper()}")
    print()
    
    results = {}
    determinism_results = {}
    
    # Test Float32 implementation
    print("="*70)
    print("FLOAT32 IMPLEMENTATION")
    print("="*70)
    
    float_results = {}
    for device in devices:
        print(f"\nRunning Float32 on {device.upper()}...")
        try:
            F_final, time_ms, metrics = run_simulation_with_seed(
                FELSimulator, N, n_steps, device, seed, rotation_rate
            )
            float_results[device] = {
                'F_final': F_final,
                'metrics': metrics
            }
            print(f"  ✓ Time: {time_ms:.2f} ms/step")
            print(f"  ✓ Component drift: {metrics['component_drift_max']:.6f}")
            print(f"  ✓ Energy drift: {metrics['energy_drift']*100:.6f}%")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare CPU vs GPU for Float32
    if 'cpu' in float_results and gpu_device and gpu_device in float_results:
        print(f"\nComparing Float32: CPU vs {gpu_device.upper()}...")
        det = compare_determinism(
            float_results['cpu']['F_final'],
            float_results[gpu_device]['F_final'],
            'CPU', gpu_device.upper()
        )
        determinism_results['float32_cpu_vs_gpu'] = det
        print(f"  Max difference: {det['max_diff']:.2e}")
        print(f"  RMS difference: {det['rms_diff']:.2e}")
        print(f"  Relative difference: {det['relative_diff']:.2e}")
        if det['is_bitwise_equal']:
            print(f"  ✅ BITWISE IDENTICAL")
        else:
            print(f"  ⚠️  {det['num_different']}/{det['total_elements']} elements differ ({det['percent_different']:.4f}%)")
    
    results['float32'] = float_results
    
    # Test Integer implementation
    print("\n" + "="*70)
    print("INTEGER (Q1.31) IMPLEMENTATION")
    print("="*70)
    
    int_results = {}
    for device in devices:
        print(f"\nRunning Integer on {device.upper()}...")
        try:
            F_final, time_ms, metrics = run_simulation_with_seed(
                FELSimulatorInt, N, n_steps, device, seed, rotation_rate
            )
            int_results[device] = {
                'F_final': F_final,
                'metrics': metrics
            }
            print(f"  ✓ Time: {time_ms:.2f} ms/step")
            print(f"  ✓ Component drift: {metrics['component_drift_max']:.6f}")
            print(f"  ✓ Energy drift: {metrics['energy_drift']*100:.6f}%")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare CPU vs GPU for Integer
    if 'cpu' in int_results and gpu_device and gpu_device in int_results:
        print(f"\nComparing Integer: CPU vs {gpu_device.upper()}...")
        det = compare_determinism(
            int_results['cpu']['F_final'],
            int_results[gpu_device]['F_final'],
            'CPU', gpu_device.upper()
        )
        determinism_results['int32_cpu_vs_gpu'] = det
        print(f"  Max difference: {det['max_diff']:.2e}")
        print(f"  RMS difference: {det['rms_diff']:.2e}")
        print(f"  Relative difference: {det['relative_diff']:.2e}")
        if det['is_bitwise_equal']:
            print(f"  ✅ BITWISE IDENTICAL")
        else:
            print(f"  ⚠️  {det['num_different']}/{det['total_elements']} elements differ ({det['percent_different']:.4f}%)")
    
    results['int32'] = int_results
    
    # Print summary table
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"{'Implementation':<15} {'Device':<8} {'Time/Step (ms)':<15} {'Drift (max)':<12} {'Energy Drift':<15}")
    print("-"*70)
    
    for impl_name, impl_results in results.items():
        for device, data in impl_results.items():
            m = data['metrics']
            print(f"{impl_name:<15} {device:<8} {m['time_per_step_ms']:>14.2f} "
                  f"{m['component_drift_max']:>11.2f} {m['energy_drift']*100:>14.6f}%")
    
    print("="*70)
    
    # Determinism summary
    if determinism_results:
        print("\n" + "="*70)
        print("DETERMINISM TEST (CPU vs GPU with same seed)")
        print("="*70)
        for test_name, det in determinism_results.items():
            impl = test_name.split('_')[0]
            print(f"\n{impl.upper()}:")
            if det['is_bitwise_equal']:
                print(f"  ✅ BITWISE IDENTICAL between CPU and GPU")
            else:
                print(f"  Max diff: {det['max_diff']:.2e}")
                print(f"  RMS diff: {det['rms_diff']:.2e}")
                print(f"  Relative diff: {det['relative_diff']:.2e}")
                print(f"  Different elements: {det['num_different']}/{det['total_elements']} ({det['percent_different']:.4f}%)")
        print("="*70)
    
    # Save results
    import json
    save_dict = {
        'N': N,
        'n_steps': n_steps,
        'rotation_rate': rotation_rate,
        'seed': seed,
        'performance': {},
        'determinism': {}
    }
    
    for impl_name, impl_results in results.items():
        save_dict['performance'][impl_name] = {}
        for device, data in impl_results.items():
            save_dict['performance'][impl_name][device] = data['metrics']
    
    for test_name, det in determinism_results.items():
        save_dict['determinism'][test_name] = {
            'is_bitwise_equal': bool(det['is_bitwise_equal']),
            'max_diff': det['max_diff'],
            'rms_diff': det['rms_diff'],
            'relative_diff': det['relative_diff'],
            'num_different': int(det['num_different']),
            'total_elements': int(det['total_elements']),
            'percent_different': det['percent_different']
        }
    
    with open(output_path / 'benchmark_results.json', 'w') as f:
        json.dump(save_dict, f, indent=2, default=str)
    
    # Create visualizations
    if len(results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Performance comparison
        ax = axes[0, 0]
        devices_list = []
        float_times = []
        int_times = []
        
        for device in devices:
            if device in results.get('float32', {}):
                devices_list.append(device)
                float_times.append(results['float32'][device]['metrics']['time_per_step_ms'])
            if device in results.get('int32', {}):
                int_times.append(results['int32'][device]['metrics']['time_per_step_ms'])
        
        x = np.arange(len(devices_list))
        width = 0.35
        if float_times:
            ax.bar(x - width/2, float_times, width, label='Float32', alpha=0.7, color='blue')
        if int_times:
            ax.bar(x + width/2, int_times, width, label='Integer', alpha=0.7, color='orange')
        ax.set_ylabel('Time per Step (ms)')
        ax.set_title('Performance: CPU vs GPU')
        ax.set_xticks(x)
        ax.set_xticklabels(devices_list)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Component drift
        ax = axes[0, 1]
        float_drifts = []
        int_drifts = []
        for device in devices_list:
            if device in results.get('float32', {}):
                float_drifts.append(results['float32'][device]['metrics']['component_drift_max'])
            if device in results.get('int32', {}):
                int_drifts.append(results['int32'][device]['metrics']['component_drift_max'])
        
        if float_drifts:
            ax.bar(x - width/2, float_drifts, width, label='Float32', alpha=0.7, color='blue')
        if int_drifts:
            ax.bar(x + width/2, int_drifts, width, label='Integer', alpha=0.7, color='orange')
        ax.set_ylabel('Max Component Drift')
        ax.set_title('Conservation: Component Drift')
        ax.set_xticks(x)
        ax.set_xticklabels(devices_list)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Determinism comparison
        ax = axes[1, 0]
        if determinism_results:
            test_names = []
            max_diffs = []
            for test_name, det in determinism_results.items():
                test_names.append(test_name.replace('_', ' ').title())
                max_diffs.append(det['max_diff'])
            ax.bar(test_names, max_diffs, alpha=0.7, color=['green' if d['is_bitwise_equal'] else 'red' 
                   for d in determinism_results.values()])
            ax.set_ylabel('Max Difference')
            ax.set_title('Determinism: CPU vs GPU (log scale)')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Speedup
        ax = axes[1, 1]
        if 'cpu' in results.get('float32', {}) and gpu_device and gpu_device in results.get('float32', {}):
            cpu_time = results['float32']['cpu']['metrics']['time_per_step_ms']
            gpu_time = results['float32'][gpu_device]['metrics']['time_per_step_ms']
            speedup_float = cpu_time / gpu_time
            
            if 'cpu' in results.get('int32', {}) and gpu_device in results.get('int32', {}):
                cpu_time_int = results['int32']['cpu']['metrics']['time_per_step_ms']
                gpu_time_int = results['int32'][gpu_device]['metrics']['time_per_step_ms']
                speedup_int = cpu_time_int / gpu_time_int
                
                ax.bar(['Float32', 'Integer'], [speedup_float, speedup_int], alpha=0.7, color=['blue', 'orange'])
            else:
                ax.bar(['Float32'], [speedup_float], alpha=0.7, color='blue')
            
            ax.axhline(1.0, color='k', linestyle='--', linewidth=0.5)
            ax.set_ylabel(f'Speedup (CPU/{gpu_device.upper()})')
            ax.set_title('GPU Speedup')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {output_path / 'benchmark_comparison.png'}")
    
    print()
    print(f"All results saved to: {output_path}")
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Integer vs Float vs CPU vs GPU Benchmarks')
    parser.add_argument('--N', type=int, default=128, help='Lattice size')
    parser.add_argument('--steps', type=int, default=200, help='Number of steps to benchmark')
    parser.add_argument('--k', type=float, default=0.01, help='Rotation rate (kappa)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for determinism testing')
    parser.add_argument('--output', type=str, default='results/05_benchmarks')
    
    args = parser.parse_args()
    
    run_benchmark_experiment(
        N=args.N,
        n_steps=args.steps,
        rotation_rate=args.k,
        seed=args.seed,
        output_dir=args.output
    )

