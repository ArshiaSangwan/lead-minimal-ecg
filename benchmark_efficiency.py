#!/usr/bin/env python3
"""
Model Efficiency Analysis
=========================

Reports model size, FLOPs, memory usage, and inference time.

Usage:
    python benchmark_efficiency.py --model resnet1d --n_leads 3
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.model import get_model, count_parameters


def count_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    Estimate FLOPs for a forward pass.
    
    Note: This is an approximation. For exact counts, use specialized tools
    like fvcore or thop.
    """
    from functools import reduce
    import operator
    
    total_flops = 0
    
    def flops_hook(module, input, output):
        nonlocal total_flops
        
        if isinstance(module, nn.Conv1d):
            # FLOPs = 2 * K * Cin * Cout * L_out
            batch_size, in_channels, in_length = input[0].shape
            out_channels, _, kernel_size = module.weight.shape
            out_length = output.shape[2]
            
            flops = 2 * kernel_size * in_channels * out_channels * out_length * batch_size
            total_flops += flops
            
        elif isinstance(module, nn.Linear):
            # FLOPs = 2 * in_features * out_features
            batch_size = input[0].shape[0]
            flops = 2 * module.in_features * module.out_features * batch_size
            total_flops += flops
            
        elif isinstance(module, nn.BatchNorm1d):
            # FLOPs ≈ 2 * num_features * length (mean + var computation)
            batch_size = input[0].shape[0]
            length = input[0].shape[2] if len(input[0].shape) > 2 else 1
            flops = 4 * module.num_features * length * batch_size
            total_flops += flops
    
    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear, nn.BatchNorm1d)):
            hooks.append(module.register_forward_hook(flops_hook))
    
    # Forward pass
    x = torch.randn(1, *input_shape)
    if next(model.parameters()).is_cuda:
        x = x.cuda()
    
    model.eval()
    with torch.no_grad():
        model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return total_flops


def measure_memory(model: nn.Module, input_shape: Tuple[int, ...], batch_size: int = 32) -> Dict:
    """Measure GPU memory usage during inference."""
    
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    model = model.cuda()
    model.eval()
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    x = torch.randn(batch_size, *input_shape).cuda()
    
    # Warm up
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure
    with torch.no_grad():
        _ = model(x)
    
    torch.cuda.synchronize()
    
    memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    memory_reserved = torch.cuda.memory_reserved() / 1024**2
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    return {
        'memory_allocated_mb': memory_allocated,
        'memory_reserved_mb': memory_reserved,
        'peak_memory_mb': peak_memory,
        'batch_size': batch_size,
    }


def measure_inference_time(
    model: nn.Module, 
    input_shape: Tuple[int, ...],
    batch_sizes: List[int] = [1, 8, 32, 128],
    n_warmup: int = 10,
    n_iterations: int = 100,
    use_amp: bool = True
) -> Dict:
    """Measure inference time for different batch sizes."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, *input_shape).to(device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(n_warmup):
                if use_amp and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        _ = model(x)
                else:
                    _ = model(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            
            with torch.no_grad():
                if use_amp and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        _ = model(x)
                else:
                    _ = model(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        results[batch_size] = {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'median_ms': float(np.median(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'samples_per_second': float(batch_size / (np.mean(times) / 1000)),
        }
    
    return results


def get_model_size(model: nn.Module) -> Dict:
    """Get model size in different metrics."""
    
    n_params = count_parameters(model)
    
    # Size in MB (assuming float32)
    size_mb_fp32 = n_params * 4 / 1024**2
    size_mb_fp16 = n_params * 2 / 1024**2
    
    # Count by layer type
    layer_counts = {}
    for name, module in model.named_modules():
        layer_type = type(module).__name__
        if layer_type not in layer_counts:
            layer_counts[layer_type] = 0
        layer_counts[layer_type] += 1
    
    return {
        'total_parameters': n_params,
        'trainable_parameters': n_params,  # All are trainable in our case
        'size_mb_fp32': size_mb_fp32,
        'size_mb_fp16': size_mb_fp16,
        'layer_counts': layer_counts,
    }


def benchmark_model(
    model_name: str = 'resnet1d',
    n_leads: int = 12,
    seq_length: int = 5000,
    n_classes: int = 5,
    **model_kwargs
) -> Dict:
    """Run full benchmark on a model configuration."""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name} ({n_leads} leads)")
    print(f"{'='*60}")
    
    # Build model
    model = get_model(model_name, n_leads=n_leads, n_classes=n_classes, **model_kwargs)
    input_shape = (n_leads, seq_length)
    
    # Model size
    print("\n📏 Model Size:")
    size_info = get_model_size(model)
    print(f"   Parameters: {size_info['total_parameters']:,}")
    print(f"   Size (FP32): {size_info['size_mb_fp32']:.2f} MB")
    print(f"   Size (FP16): {size_info['size_mb_fp16']:.2f} MB")
    
    # FLOPs
    print("\n Computational Cost:")
    flops = count_flops(model, input_shape)
    print(f"   FLOPs: {flops:,}")
    print(f"   GFLOPs: {flops / 1e9:.3f}")
    
    # Memory
    print("\n Memory Usage:")
    memory_info = measure_memory(model, input_shape, batch_size=32)
    if 'error' not in memory_info:
        print(f"   Allocated: {memory_info['memory_allocated_mb']:.2f} MB")
        print(f"   Peak: {memory_info['peak_memory_mb']:.2f} MB")
    else:
        print(f"   {memory_info['error']}")
        memory_info = {}
    
    # Inference time
    print("\n  Inference Time:")
    time_info = measure_inference_time(model, input_shape)
    for batch_size, times in time_info.items():
        print(f"   Batch {batch_size:3d}: {times['mean_ms']:.2f}±{times['std_ms']:.2f} ms "
              f"({times['samples_per_second']:.1f} samples/sec)")
    
    # Aggregate results
    results = {
        'model_name': model_name,
        'n_leads': n_leads,
        'seq_length': seq_length,
        'n_classes': n_classes,
        'size': size_info,
        'flops': flops,
        'gflops': flops / 1e9,
        'memory': memory_info,
        'inference_time': time_info,
    }
    
    return results


def benchmark_all_configurations(
    output_dir: str = 'outputs/benchmarks'
) -> List[Dict]:
    """Benchmark all lead configurations."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lead configurations
    configs = [
        {'name': '12-lead', 'n_leads': 12},
        {'name': '6-lead', 'n_leads': 6},
        {'name': '3-lead', 'n_leads': 3},
        {'name': '2-lead', 'n_leads': 2},
        {'name': '1-lead', 'n_leads': 1},
    ]
    
    all_results = []
    
    for config in configs:
        results = benchmark_model(
            model_name='resnet1d',
            n_leads=config['n_leads'],
            seq_length=5000,
            n_classes=5
        )
        results['config_name'] = config['name']
        all_results.append(results)
    
    # Save results
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Generate summary table
    generate_efficiency_table(all_results, output_dir)
    
    return all_results


def generate_efficiency_table(results: List[Dict], output_dir: Path):
    """Generate LaTeX table comparing model efficiency."""
    
    print("\n" + "="*70)
    print(" EFFICIENCY COMPARISON")
    print("="*70)
    
    # Print ASCII table
    print(f"\n{'Config':<12} {'Params':>12} {'GFLOPs':>10} {'Size (MB)':>12} {'Time (ms)':>12} {'Throughput':>12}")
    print("-" * 70)
    
    for r in results:
        config = r.get('config_name', f"{r['n_leads']}-lead")
        params = r['size']['total_parameters']
        gflops = r['gflops']
        size_mb = r['size']['size_mb_fp32']
        
        # Use batch_size=32 timing
        time_ms = r['inference_time'].get(32, {}).get('mean_ms', 0)
        throughput = r['inference_time'].get(32, {}).get('samples_per_second', 0)
        
        print(f"{config:<12} {params:>12,} {gflops:>10.3f} {size_mb:>12.2f} {time_ms:>12.2f} {throughput:>12.1f}")
    
    # LaTeX table
    latex = r"""\begin{table}[htbp]
\centering
\caption{Computational Efficiency by Lead Configuration. Throughput measured on RTX 4090 with batch size 32 and mixed precision.}
\label{tab:efficiency}
\begin{tabular}{lccccc}
\toprule
\textbf{Config} & \textbf{Parameters} & \textbf{GFLOPs} & \textbf{Size (MB)} & \textbf{Time (ms)} & \textbf{Throughput} \\
\midrule
"""
    
    for r in results:
        config = r.get('config_name', f"{r['n_leads']}-lead")
        params = r['size']['total_parameters']
        gflops = r['gflops']
        size_mb = r['size']['size_mb_fp32']
        time_ms = r['inference_time'].get(32, {}).get('mean_ms', 0)
        throughput = r['inference_time'].get(32, {}).get('samples_per_second', 0)
        
        # Format parameters (e.g., 155K, 1.2M)
        if params >= 1e6:
            params_str = f"{params/1e6:.1f}M"
        else:
            params_str = f"{params/1e3:.0f}K"
        
        latex += f"{config} & {params_str} & {gflops:.2f} & {size_mb:.1f} & {time_ms:.1f} & {throughput:.0f}/s \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_dir / 'efficiency_table.tex', 'w') as f:
        f.write(latex)
    
    print(f"\n LaTeX table saved to {output_dir / 'efficiency_table.tex'}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark model efficiency")
    
    parser.add_argument("--model", type=str, default="resnet1d",
                        help="Model architecture")
    parser.add_argument("--n_leads", type=int, default=12,
                        help="Number of ECG leads")
    parser.add_argument("--all", action="store_true",
                        help="Benchmark all configurations")
    parser.add_argument("--output", type=str, default="outputs/benchmarks",
                        help="Output directory")
    
    args = parser.parse_args()
    
    if args.all:
        benchmark_all_configurations(args.output)
    else:
        results = benchmark_model(args.model, n_leads=args.n_leads)
        
        # Save
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / f'benchmark_{args.model}_{args.n_leads}lead.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
