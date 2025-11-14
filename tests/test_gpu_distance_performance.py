"""
Test GPU distance computation performance across different dataset sizes.

This script benchmarks the performance of GPU-accelerated distance computation
for hierarchical clustering with varying numbers of contexts.
"""

import numpy as np
import time
import json
from ragboost.context_index import build_context_index


def generate_synthetic_data(num_contexts, num_docs_per_context=20, total_docs=1000):
    """
    Generate synthetic retrieval data for testing.
    
    Args:
        num_contexts: Number of contexts/queries to generate
        num_docs_per_context: Number of retrieved documents per context
        total_docs: Total number of unique documents in the corpus
    
    Returns:
        List of document ID lists (top-k retrieval results)
    """
    topk_doc_ids = []
    for _ in range(num_contexts):
        # Generate random document IDs with some overlap for realistic scenarios
        doc_ids = np.random.choice(total_docs, size=num_docs_per_context, replace=False).tolist()
        topk_doc_ids.append(doc_ids)
    return topk_doc_ids


def benchmark_gpu_distance(num_contexts, linkage_method='average', alpha=0.005, num_runs=3, skip_cpu=False):
    """
    Benchmark GPU distance computation for a given number of contexts.
    
    Args:
        num_contexts: Number of contexts to test
        linkage_method: Hierarchical clustering linkage method
        alpha: Weight for position term in distance calculation
        num_runs: Number of runs to average over
        skip_cpu: Skip CPU testing (useful for very large datasets)
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {num_contexts} contexts")
    print(f"{'='*60}")
    
    # Generate synthetic data
    print(f"Generating synthetic data for {num_contexts} contexts...")
    topk_doc_ids = generate_synthetic_data(num_contexts)
    
    gpu_times = []
    cpu_times = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # Test GPU
        print("  Testing GPU...")
        start = time.perf_counter()
        try:
            result_gpu = build_context_index(
                topk_doc_ids,
                linkage_method=linkage_method,
                use_gpu=True,
                alpha=alpha
            )
            gpu_time = time.perf_counter() - start
            gpu_times.append(gpu_time)
            print(f"    GPU time: {gpu_time:.4f}s")
        except Exception as e:
            print(f"    GPU failed: {e}")
            gpu_times.append(None)
        
        # Test CPU (skip for very large datasets)
        if skip_cpu:
            print("  Skipping CPU test (too large)")
        else:
            print("  Testing CPU...")
            start = time.perf_counter()
            try:
                result_cpu = build_context_index(
                    topk_doc_ids,
                    linkage_method=linkage_method,
                    use_gpu=False,
                    alpha=alpha
                )
                cpu_time = time.perf_counter() - start
                cpu_times.append(cpu_time)
                print(f"    CPU time: {cpu_time:.4f}s")
            except Exception as e:
                print(f"    CPU failed: {e}")
                cpu_times.append(None)
    
    # Calculate statistics
    valid_gpu_times = [t for t in gpu_times if t is not None]
    valid_cpu_times = [t for t in cpu_times if t is not None]
    
    result = {
        "num_contexts": num_contexts,
        "num_runs": num_runs,
        "linkage_method": linkage_method,
        "alpha": alpha,
    }
    
    if valid_gpu_times:
        result["gpu_mean"] = np.mean(valid_gpu_times)
        result["gpu_std"] = np.std(valid_gpu_times)
        result["gpu_min"] = np.min(valid_gpu_times)
        result["gpu_max"] = np.max(valid_gpu_times)
    else:
        result["gpu_mean"] = None
        
    if valid_cpu_times:
        result["cpu_mean"] = np.mean(valid_cpu_times)
        result["cpu_std"] = np.std(valid_cpu_times)
        result["cpu_min"] = np.min(valid_cpu_times)
        result["cpu_max"] = np.max(valid_cpu_times)
    else:
        result["cpu_mean"] = None
    
    if valid_gpu_times and valid_cpu_times:
        result["speedup"] = result["cpu_mean"] / result["gpu_mean"]
    else:
        result["speedup"] = None
    
    # Print summary
    print(f"\n{'-'*60}")
    print(f"Summary for {num_contexts} contexts:")
    if result["gpu_mean"] is not None:
        print(f"  GPU: {result['gpu_mean']:.4f}s ± {result['gpu_std']:.4f}s")
    if result["cpu_mean"] is not None:
        print(f"  CPU: {result['cpu_mean']:.4f}s ± {result['cpu_std']:.4f}s")
    if result["speedup"] is not None:
        print(f"  Speedup: {result['speedup']:.2f}x")
    print(f"{'-'*60}")
    
    return result


def main():
    """Run comprehensive GPU distance computation benchmarks."""
    
    # Test sizes: 64, 128, 512, 4k, 8k, 12k, 100k
    test_sizes = [64, 128, 512, 4096, 8192, 12288, 102400]
    
    print("="*60)
    print("GPU Distance Computation Performance Benchmark")
    print("="*60)
    print(f"Test sizes: {test_sizes}")
    print(f"Linkage method: average")
    print(f"Alpha: 0.005")
    print(f"Number of runs per size: 3")
    print("="*60)
    
    all_results = []
    
    for size in test_sizes:
        try:
            # Skip CPU testing for very large datasets (100k)
            skip_cpu = size >= 100000
            result = benchmark_gpu_distance(
                num_contexts=size,
                linkage_method='average',
                alpha=0.005,
                num_runs=3,
                skip_cpu=skip_cpu
            )
            all_results.append(result)
        except Exception as e:
            print(f"\nFailed to benchmark {size} contexts: {e}")
            all_results.append({
                "num_contexts": size,
                "error": str(e)
            })
    
    # Save results to JSON
    output_file = "gpu_distance_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Benchmark Complete!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Print summary table
    print("\n" + "="*60)
    print("Summary Table")
    print("="*60)
    print(f"{'Size':>10} {'GPU (s)':>12} {'CPU (s)':>12} {'Speedup':>10}")
    print("-"*60)
    
    for result in all_results:
        if "error" in result:
            print(f"{result['num_contexts']:>10} {'ERROR':>12} {'ERROR':>12} {'N/A':>10}")
        else:
            gpu_str = f"{result['gpu_mean']:.4f}" if result['gpu_mean'] is not None else "N/A"
            cpu_str = f"{result['cpu_mean']:.4f}" if result['cpu_mean'] is not None else "N/A"
            speedup_str = f"{result['speedup']:.2f}x" if result['speedup'] is not None else "N/A"
            print(f"{result['num_contexts']:>10} {gpu_str:>12} {cpu_str:>12} {speedup_str:>10}")
    
    print("="*60)


if __name__ == "__main__":
    main()
