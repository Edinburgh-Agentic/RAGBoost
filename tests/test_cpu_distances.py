#!/usr/bin/env python3
"""
Testing and benchmarking suite for optimized CPU distance computation.
"""

import numpy as np
import time
from ragboost.context_index.compute_distance_cpu import (
    prepare_contexts_for_cpu,
    compute_distance_optimized,
    compute_distance_matrix_cpu_optimized,
)

from test_utils import generate_contexts


def compute_distance_naive(context_i, context_j, alpha):
    """
    Naive implementation using dict/set (for comparison).
    This is the old/slow method.
    """
    if len(context_i) == 0 or len(context_j) == 0:
        return 1.0
    
    # Build position maps
    pos_i = {chunk_id: idx for idx, chunk_id in enumerate(context_i)}
    pos_j = {chunk_id: idx for idx, chunk_id in enumerate(context_j)}
    
    # Find intersection
    set_i = set(context_i)
    set_j = set(context_j)
    intersection = set_i & set_j
    
    # Overlap term
    max_size = max(len(context_i), len(context_j))
    overlap_term = 1.0 - (len(intersection) / max_size)
    
    # Position term
    if len(intersection) == 0:
        position_term = 0.0
    else:
        position_diff_sum = sum(abs(pos_i[k] - pos_j[k]) for k in intersection)
        position_term = alpha * (position_diff_sum / len(intersection))
    
    return overlap_term + position_term


def verify_correctness(contexts, alpha=0.005, num_test=50):
    """Verify optimized implementation matches naive implementation."""
    print(f"\n{'='*70}")
    print(f"CORRECTNESS VERIFICATION")
    print(f"{'='*70}")
    
    test_contexts = contexts[:num_test]
    n = len(test_contexts)
    num_pairs = n * (n - 1) // 2
    
    print(f"Testing {num_pairs:,} pairs...")
    
    # Prepare data for optimized version
    chunk_ids, original_positions, lengths, offsets = prepare_contexts_for_cpu(test_contexts)
    
    # Compute with both methods
    naive_results = []
    optimized_results = []
    
    for i in range(n):
        for j in range(i + 1, n):
            naive_dist = compute_distance_naive(test_contexts[i], test_contexts[j], alpha)
            optimized_dist = compute_distance_optimized(
                chunk_ids, original_positions, lengths, offsets, i, j, alpha
            )
            
            naive_results.append(naive_dist)
            optimized_results.append(optimized_dist)
    
    # Compare
    naive_results = np.array(naive_results)
    optimized_results = np.array(optimized_results)
    
    diffs = np.abs(naive_results - optimized_results)
    max_diff = diffs.max()
    mean_diff = diffs.mean()
    
    print(f"\nResults:")
    print(f"  Max difference: {max_diff:.10f}")
    print(f"  Mean difference: {mean_diff:.10f}")
    print(f"  Naive range: [{naive_results.min():.6f}, {naive_results.max():.6f}]")
    print(f"  Optimized range: [{optimized_results.min():.6f}, {optimized_results.max():.6f}]")
    
    if max_diff < 1e-6:
        print(f"\n✓ PASSED: Results match within tolerance")
        return True
    else:
        print(f"\n✗ FAILED: Results differ by more than tolerance")
        return False


def benchmark_single_threaded(contexts, alpha=0.005, sample_size=1000):
    """Benchmark single-threaded performance: naive vs optimized."""
    print(f"\n{'='*70}")
    print(f"SINGLE-THREADED BENCHMARK")
    print(f"{'='*70}")
    
    sample_contexts = contexts[:sample_size]
    n = len(sample_contexts)
    num_pairs = n * (n - 1) // 2
    
    print(f"Sample size: {n} contexts ({num_pairs:,} pairs)")
    
    # Naive method
    print(f"\n[1/2] Naive method (dict/set)...")
    start = time.time()
    
    for i in range(n):
        for j in range(i + 1, n):
            _ = compute_distance_naive(sample_contexts[i], sample_contexts[j], alpha)
    
    time_naive = time.time() - start
    rate_naive = num_pairs / time_naive
    
    print(f"  Time: {time_naive:.3f}s")
    print(f"  Rate: {rate_naive:,.0f} pairs/sec")
    
    # Optimized method
    print(f"\n[2/2] Optimized method (merge)...")
    prep_start = time.time()
    chunk_ids, original_positions, lengths, offsets = prepare_contexts_for_cpu(sample_contexts)
    prep_time = time.time() - prep_start
    
    start = time.time()
    for i in range(n):
        for j in range(i + 1, n):
            _ = compute_distance_optimized(
                chunk_ids, original_positions, lengths, offsets, i, j, alpha
            )
    
    time_optimized = time.time() - start
    rate_optimized = num_pairs / time_optimized
    
    print(f"  Prep time: {prep_time:.3f}s")
    print(f"  Compute time: {time_optimized:.3f}s")
    print(f"  Rate: {rate_optimized:,.0f} pairs/sec")
    
    # Summary
    speedup = rate_optimized / rate_naive
    time_savings = (time_naive - time_optimized) / time_naive * 100
    
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"{'='*70}")
    print(f"Speedup: {speedup:.2f}x faster")
    print(f"Time savings: {time_savings:.1f}%")
    print(f"For {num_pairs:,} pairs:")
    print(f"  Naive: {time_naive:.1f}s")
    print(f"  Optimized: {time_optimized:.1f}s (+ {prep_time:.1f}s prep)")


def test_full_pipeline(num_contexts=5000):
    """Test the full multi-threaded pipeline."""
    print(f"\n{'='*70}")
    print(f"FULL PIPELINE TEST")
    print(f"{'='*70}")
    
    # Generate contexts
    print("\n[1/2] Generating contexts...")
    contexts = generate_contexts(num_contexts, avg_chunks=20, seed=42)
    
    # Compute matrix
    print("\n[2/2] Computing distance matrix...")
    condensed = compute_distance_matrix_cpu_optimized(
        contexts, 
        alpha=0.005, 
        num_workers=None,  # Use all cores
        batch_size=1000
    )
    
    # Verify result
    print(f"\n{'='*70}")
    print(f"Distance Matrix Statistics:")
    print(f"{'='*70}")
    print(f"Shape: condensed vector of length {len(condensed):,}")
    print(f"Min: {condensed.min():.6f}")
    print(f"Max: {condensed.max():.6f}")
    print(f"Mean: {condensed.mean():.6f}")
    print(f"Median: {np.median(condensed):.6f}")
    print(f"Std: {condensed.std():.6f}")
    
    # Check for issues
    if condensed.max() > 3.0:
        print(f"\n⚠ WARNING: Max distance seems high")
    elif np.any(condensed < 0):
        print(f"\n⚠ WARNING: Found negative distances")
    else:
        print(f"\n✓ Distance range looks correct!")
    
    return condensed


def main():
    """Run all tests."""
    print("="*70)
    print(" OPTIMIZED CPU DISTANCE COMPUTATION - TEST SUITE")
    print("="*70)
    
    # Generate test data
    print("\nGenerating test data...")
    contexts = generate_contexts(10000, avg_chunks=20, seed=42)
    
    # Test 1: Correctness
    print("\n" + "="*70)
    print("TEST 1: Correctness Verification")
    print("="*70)
    if not verify_correctness(contexts, num_test=50):
        print("\n✗ Correctness test failed! Aborting.")
        return
    
    # Test 2: Single-threaded benchmark
    print("\n" + "="*70)
    print("TEST 2: Single-threaded Performance")
    print("="*70)
    benchmark_single_threaded(contexts, sample_size=1000)
    
    # Test 3: Full pipeline
    print("\n" + "="*70)
    print("TEST 3: Full Multi-threaded Pipeline")
    print("="*70)
    condensed = test_full_pipeline(num_contexts=5000)
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main()