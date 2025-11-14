#!/usr/bin/env python3
"""
Testing and verification suite for GPU distance computation.
"""

import numpy as np
import cupy as cp
import time
from ragboost.context_index.compute_distance_gpu import (
    get_gpu_info, 
    distance_kernel,
    prepare_contexts_for_gpu,
    compute_distance_matrix_gpu
)
from test_utils import generate_contexts

def compute_distance_cpu_detailed(context_i, context_j, alpha):
    """CPU reference implementation with detailed breakdown."""
    if len(context_i) == 0 or len(context_j) == 0:
        return {
            'distance': 1.0,
            'overlap_term': 1.0,
            'position_term': 0.0,
            'intersection_size': 0,
            'max_size': max(len(context_i), len(context_j)) if len(context_i) > 0 or len(context_j) > 0 else 0,
            'avg_pos_diff': 0.0,
            'intersection': set()
        }
    
    pos_i = {chunk_id: idx for idx, chunk_id in enumerate(context_i)}
    pos_j = {chunk_id: idx for idx, chunk_id in enumerate(context_j)}
    
    set_i = set(context_i)
    set_j = set(context_j)
    intersection = set_i & set_j
    
    max_size = max(len(context_i), len(context_j))
    overlap_term = 1.0 - (len(intersection) / max_size)
    
    if len(intersection) == 0:
        position_term = 0.0
        avg_pos_diff = 0.0
    else:
        position_diff_sum = sum(abs(pos_i[k] - pos_j[k]) for k in intersection)
        avg_pos_diff = position_diff_sum / len(intersection)
        position_term = alpha * avg_pos_diff
    
    distance = overlap_term + position_term
    
    return {
        'distance': distance,
        'overlap_term': overlap_term,
        'position_term': position_term,
        'intersection_size': len(intersection),
        'max_size': max_size,
        'avg_pos_diff': avg_pos_diff,
        'intersection': intersection
    }

def compute_distance_cpu(context_i, context_j, alpha):
    """CPU reference implementation for distance calculation."""
    if len(context_i) == 0 or len(context_j) == 0:
        return 1.0
    
    pos_i = {chunk_id: idx for idx, chunk_id in enumerate(context_i)}
    pos_j = {chunk_id: idx for idx, chunk_id in enumerate(context_j)}
    
    set_i = set(context_i)
    set_j = set(context_j)
    intersection = set_i & set_j
    
    max_size = max(len(context_i), len(context_j))
    overlap_term = 1.0 - (len(intersection) / max_size)
    
    if len(intersection) == 0:
        position_term = 0.0
    else:
        position_diff_sum = sum(abs(pos_i[k] - pos_j[k]) for k in intersection)
        position_term = alpha * (position_diff_sum / len(intersection))
    
    return overlap_term + position_term


def analyze_high_distances(contexts, alpha, threshold=1.0, max_show=10):
    """Find and analyze pairs with distance > threshold."""
    print(f"\n{'='*70}")
    print(f"Analyzing pairs with distance > {threshold}")
    print(f"{'='*70}")
    
    n = len(contexts)
    high_distance_pairs = []
    
    print(f"Checking {n * (n-1) // 2:,} pairs...")
    
    for i in range(n):
        for j in range(i + 1, n):
            result = compute_distance_cpu_detailed(contexts[i], contexts[j], alpha)
            if result['distance'] > threshold:
                high_distance_pairs.append({
                    'i': i,
                    'j': j,
                    **result
                })
    
    print(f"\nFound {len(high_distance_pairs)} pairs with distance > {threshold}")
    
    if len(high_distance_pairs) == 0:
        print("✓ No pairs exceed threshold")
        return high_distance_pairs
    
    # Sort by distance
    high_distance_pairs.sort(key=lambda x: x['distance'], reverse=True)
    
    # Show top examples
    print(f"\nTop {min(max_show, len(high_distance_pairs))} pairs:")
    for idx, pair in enumerate(high_distance_pairs[:max_show]):
        i, j = pair['i'], pair['j']
        print(f"\n  [{idx+1}] Pair ({i}, {j}):")
        print(f"      Distance: {pair['distance']:.6f}")
        print(f"      Overlap term: {pair['overlap_term']:.6f}")
        print(f"      Position term: {pair['position_term']:.6f}")
        print(f"      Intersection size: {pair['intersection_size']}/{pair['max_size']}")
        print(f"      Avg position diff: {pair['avg_pos_diff']:.2f}")
        print(f"      Context lengths: {len(contexts[i])}, {len(contexts[j])}")
        
        # Show why distance is high
        if pair['intersection_size'] == 0:
            print(f"      → No overlap (distance = 1.0)")
        elif pair['position_term'] > 0.1:
            print(f"      → High position differences! (position_term = {pair['position_term']:.6f})")
            print(f"      → Shared chunks are at very different positions")
            
            # Show position details for shared chunks
            if len(pair['intersection']) <= 5:
                print(f"      → Shared chunks and positions:")
                pos_i = {chunk_id: idx for idx, chunk_id in enumerate(contexts[i])}
                pos_j = {chunk_id: idx for idx, chunk_id in enumerate(contexts[j])}
                for chunk_id in sorted(pair['intersection']):
                    diff = abs(pos_i[chunk_id] - pos_j[chunk_id])
                    print(f"         Chunk {chunk_id}: pos_i={pos_i[chunk_id]}, pos_j={pos_j[chunk_id]}, diff={diff}")
    
    # Statistics
    distances = [p['distance'] for p in high_distance_pairs]
    overlap_terms = [p['overlap_term'] for p in high_distance_pairs]
    position_terms = [p['position_term'] for p in high_distance_pairs]
    
    print(f"\n  Statistics of high-distance pairs:")
    print(f"    Distance range: [{min(distances):.6f}, {max(distances):.6f}]")
    print(f"    Avg distance: {np.mean(distances):.6f}")
    print(f"    Avg overlap term: {np.mean(overlap_terms):.6f}")
    print(f"    Avg position term: {np.mean(position_terms):.6f}")
    print(f"    Max position term: {max(position_terms):.6f}")
    
    return high_distance_pairs


def verify_gpu_implementation(contexts, alpha, num_test=20):
    """Verify GPU matches CPU."""
    print(f"\n{'='*70}")
    print(f"VERIFICATION: GPU vs CPU (n={num_test}, alpha={alpha})")
    print(f"{'='*70}")
    
    test_contexts = contexts[:num_test]
    n_test = len(test_contexts)
    
    # Prepare
    chunk_ids, original_positions, lengths, offsets = prepare_contexts_for_gpu(test_contexts)
    
    d_chunk_ids = cp.asarray(chunk_ids)
    d_original_positions = cp.asarray(original_positions)
    d_lengths = cp.asarray(lengths)
    d_offsets = cp.asarray(offsets)
    
    # CPU
    cpu_results = []
    for i in range(n_test):
        for j in range(i + 1, n_test):
            cpu_results.append(compute_distance_cpu(test_contexts[i], test_contexts[j], alpha))
    
    # GPU
    test_pairs = n_test * (n_test - 1) // 2
    d_test_output = cp.zeros(test_pairs, dtype=cp.float32)
    
    threads_per_block = 256
    blocks = (test_pairs + threads_per_block - 1) // threads_per_block
    
    distance_kernel(
        (blocks,), (threads_per_block,),
        (d_chunk_ids, d_original_positions, d_lengths, d_offsets,
         d_test_output, n_test, 0, n_test, np.float32(alpha))
    )
    
    cp.cuda.Stream.null.synchronize()
    gpu_results = cp.asnumpy(d_test_output)
    
    # Check for errors
    error_mask = gpu_results < 0
    if np.any(error_mask):
        print(f"\n⚠ ERROR: GPU kernel returned error sentinels!")
        error_codes = np.unique(gpu_results[error_mask])
        error_names = {
            -999.0: "Invalid (i,j) indices",
            -998.0: "Batch boundary error",
            -997.0: "Invalid length",
            -996.0: "Offset inconsistency",
            -995.0: "Invalid position value",
            -994.0: "Position diff >= max_context_len",
            -993.0: "Avg pos diff >= max_context_len or negative",
            -989.0: "Invalid chunk ID (negative or > 1M)",
            -988.0: "Overflow in position_diff_sum",
            -987.0: "Invalid intersection size",
            -986.0: "Invalid max_size",
            -985.0: "overlap_term out of range [0,1]",
            -984.0: "NaN/Inf in overlap_term",
            -983.0: "NaN/Inf in avg_pos_diff",
            -982.0: "Invalid position_term (>100 or <0)",
            -981.0: "NaN/Inf in position_term",
            -980.0: "NaN/Inf in final_distance",
            -979.0: "Unreasonable final_distance (>101 or <0)",
        }
        for code in error_codes:
            count = np.sum(gpu_results == code)
            name = error_names.get(code, "Unknown error")
            print(f"  Code {code}: {name} ({count} occurrences)")
        
        # Show details of first error
        first_error_idx = np.where(error_mask)[0][0]
        i, j = 0, 0
        remaining = first_error_idx
        for row in range(n_test):
            pairs_in_row = n_test - row - 1
            if remaining < pairs_in_row:
                i = row
                j = row + 1 + remaining
                break
            remaining -= pairs_in_row
        
        print(f"\n  First error at index {first_error_idx}: pair ({i}, {j})")
        print(f"  Context[{i}] length: {len(test_contexts[i])}")
        print(f"  Context[{j}] length: {len(test_contexts[j])}")
        print(f"  CPU result: {cpu_results[first_error_idx]:.6f}")
        
        return False
    
    # Check for distances > 1
    cpu_results = np.array(cpu_results)
    high_distances_gpu = gpu_results > 1.0
    high_distances_cpu = cpu_results > 1.0
    
    if np.any(high_distances_gpu) or np.any(high_distances_cpu):
        print(f"\n⚠ WARNING: Found distances > 1.0!")
        print(f"  GPU: {np.sum(high_distances_gpu)} pairs")
        print(f"  CPU: {np.sum(high_distances_cpu)} pairs")
        
        # Print details of pairs with distance > 1
        print(f"\n  Pairs with distance > 1.0:")
        pair_idx = 0
        count_shown = 0
        for i in range(n_test):
            for j in range(i + 1, n_test):
                if gpu_results[pair_idx] > 1.0 or cpu_results[pair_idx] > 1.0:
                    # Compute intersection details
                    set_i = set(test_contexts[i])
                    set_j = set(test_contexts[j])
                    intersection = set_i & set_j
                    
                    print(f"    Pair ({i}, {j}):")
                    print(f"      Lengths: {len(test_contexts[i])}, {len(test_contexts[j])}")
                    print(f"      Intersection size: {len(intersection)}")
                    print(f"      CPU distance: {cpu_results[pair_idx]:.6f}")
                    print(f"      GPU distance: {gpu_results[pair_idx]:.6f}")
                    
                    count_shown += 1
                    if count_shown >= 5:
                        remaining = np.sum(gpu_results[pair_idx:] > 1.0)
                        if remaining > 0:
                            print(f"    ... and {remaining} more pairs")
                        break
                pair_idx += 1
            if count_shown >= 5:
                break
    
    # Compare
    diffs = np.abs(cpu_results - gpu_results)
    
    max_diff = diffs.max()
    mean_diff = diffs.mean()
    
    print(f"\nResults:")
    print(f"  Max difference: {max_diff:.10f}")
    print(f"  Mean difference: {mean_diff:.10f}")
    print(f"  CPU range: [{cpu_results.min():.6f}, {cpu_results.max():.6f}]")
    print(f"  GPU range: [{gpu_results.min():.6f}, {gpu_results.max():.6f}]")
    
    tolerance = 1e-4
    within_tol = np.sum(diffs < tolerance)
    pct = 100.0 * within_tol / len(diffs)
    
    print(f"  Within {tolerance}: {within_tol}/{len(diffs)} ({pct:.2f}%)")
    
    if pct < 99.9:
        print(f"\n⚠ FAILED: Only {pct:.2f}% within tolerance")
        return False
    else:
        print(f"\n✓ PASSED")
        return True


def main():
    """Main testing function."""
    num_contexts = 100000
    avg_length = 20
    alpha = 0.001
    batch_rows = 5000
    
    print("="*70)
    print(" GPU DISTANCE COMPUTATION TEST SUITE")
    print("="*70)
    
    # GPU info
    gpu_info = get_gpu_info(0)
    free, total = cp.cuda.Device().mem_info
    print(f"\nGPU: {gpu_info['name']}")
    print(f"Memory: {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free")
    
    # Generate
    print(f"\n[1/4] Generating contexts...")
    start = time.time()
    contexts = generate_contexts(num_contexts, avg_length, seed=42)
    print(f"✓ Generated in {time.time() - start:.1f}s")
    
    # Analyze high distances on small sample
    print(f"\n[2/4] Analyzing high distances on sample...")
    sample_size = 100
    high_pairs = analyze_high_distances(contexts[:sample_size], alpha, threshold=1.0, max_show=10)
    
    if len(high_pairs) > 0:
        print(f"\n⚠ WARNING: Found {len(high_pairs)} pairs with distance > 1.0")
        print(f"  This means position differences are VERY large!")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Verify
    print(f"\n[3/4] Verifying GPU implementation...")
    for test_alpha in [0.001, 0.002, 0.005]:
        for test_size in [20, 50, 100]:
            if not verify_gpu_implementation(contexts, test_alpha, num_test=test_size):
                print(f"\n✗ FAILED at n={test_size}, alpha={test_alpha}")
                return
    
    print(f"\n✓ All verification passed!")
    
    # Compute
    print(f"\n[4/4] Computing full distance matrix...")
    condensed_matrix = compute_distance_matrix_gpu(contexts, alpha, batch_rows)
    
    if condensed_matrix is None:
        print("\n✗ Computation failed due to errors")
        return

    # Stats
    print(f"\n{'='*70}")
    print("Distance Matrix Statistics:")
    print(f"{'='*70}")
    print(f"Shape: {condensed_matrix.shape}")
    print(f"Min: {condensed_matrix.min():.6f}")
    print(f"Max: {condensed_matrix.max():.6f}")
    print(f"Mean: {condensed_matrix.mean():.6f}")
    print(f"Median: {np.median(condensed_matrix):.6f}")
    print(f"Std: {condensed_matrix.std():.6f}")
    
    if condensed_matrix.max() > 3.0:
        print(f"\n⚠ WARNING: Max distance seems high")
    else:
        print(f"\n✓ Distance range looks correct!")
    
    print(f"\n✓ Ready for scipy.cluster.hierarchy.linkage()")


if __name__ == "__main__":
    main()