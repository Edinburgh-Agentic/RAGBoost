"""
Test for verifying that all members in a group share a common prefix.

This test validates the InterContextScheduler's grouping mechanism by:
1. Creating synthetic test cases with known prefix patterns
2. Building context index and scheduling contexts
3. Verifying that all members within each group share a common prefix
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragboost.context_index import build_context_index
from ragboost.context_ordering.inter_scheduler import InterContextScheduler


def find_common_prefix(sequences: List[List[int]]) -> List[int]:
    """
    Find the longest common prefix among a list of sequences.
    
    Args:
        sequences: List of sequences (each is a list of integers)
        
    Returns:
        The longest common prefix
    """
    if not sequences:
        return []
    
    if len(sequences) == 1:
        return sequences[0]
    
    # Find minimum length
    min_len = min(len(seq) for seq in sequences)
    
    # Find common prefix
    common = []
    for i in range(min_len):
        val = sequences[0][i]
        if all(seq[i] == val for seq in sequences):
            common.append(val)
        else:
            break
    
    return common


def verify_group_prefix_sharing(
    groups: List[Tuple[float, List[int]]],
    contexts: List[List[int]],
    verbose: bool = True
) -> Tuple[bool, List[dict]]:
    """
    Verify that all members in each group share a common prefix.
    
    Args:
        groups: List of (score, group_indices) tuples
        contexts: List of all contexts (reordered)
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (all_passed, issues_list)
    """
    issues = []
    total_groups = len(groups)
    multi_item_groups = 0
    passed_groups = 0
    
    if verbose:
        print("\n" + "="*80)
        print("VERIFYING GROUP PREFIX SHARING")
        print("="*80)
    
    for group_id, (score, group_indices) in enumerate(groups):
        if len(group_indices) <= 1:
            # Single-item groups trivially share prefix with themselves
            if verbose:
                print(f"Group {group_id}: size=1, trivially passes")
            continue
        
        multi_item_groups += 1
        
        # Get contexts for all members in this group
        group_contexts = [contexts[idx] for idx in group_indices]
        
        # Find common prefix
        common_prefix = find_common_prefix(group_contexts)
        
        # Check if there is a common prefix
        if len(common_prefix) > 0:
            passed_groups += 1
            if verbose:
                print(f"✓ Group {group_id}: size={len(group_indices)}, "
                      f"common_prefix={common_prefix[:5]}{'...' if len(common_prefix) > 5 else ''} "
                      f"(len={len(common_prefix)})")
                if verbose and len(group_indices) <= 5:
                    for idx in group_indices:
                        print(f"    Context {idx}: {contexts[idx][:10]}{'...' if len(contexts[idx]) > 10 else ''}")
        else:
            issues.append({
                'group_id': group_id,
                'size': len(group_indices),
                'score': score,
                'query_indices': group_indices,
                'contexts': group_contexts,
                'common_prefix_length': 0
            })
            if verbose:
                print(f"✗ Group {group_id}: size={len(group_indices)}, NO COMMON PREFIX!")
                for idx in group_indices:
                    print(f"    Context {idx}: {contexts[idx][:10]}{'...' if len(contexts[idx]) > 10 else ''}")
    
    all_passed = len(issues) == 0
    
    if verbose:
        print("\n" + "="*80)
        print(f"Total groups: {total_groups}")
        print(f"Multi-item groups: {multi_item_groups}")
        print(f"Passed: {passed_groups}/{multi_item_groups}")
        print(f"Failed: {len(issues)}/{multi_item_groups}")
        print("="*80)
    
    return all_passed, issues


def test_simple_synthetic_case():
    """
    Test with a simple synthetic case where we know the expected grouping.
    All contexts have the same length (simulating top-k retrieval in RAG).
    """
    print("\n" + "="*80)
    print("TEST 1: Simple Synthetic Case (Same Length)")
    print("="*80)
    
    # Create synthetic contexts with known prefix patterns
    # All contexts have 5 documents (simulating top-5 retrieval)
    # Expected groups based on document prefix:
    #   - Group A: Contexts starting with [1, 2]
    #   - Group B: Contexts starting with [3, 4]
    #   - Group C: Contexts starting with [5, 6]
    contexts = [
        [1, 2, 3, 4, 5],    # Group A
        [1, 2, 3, 6, 7],    # Group A
        [1, 2, 4, 8, 9],    # Group A
        [3, 4, 5, 6, 7],    # Group B
        [3, 4, 5, 8, 9],    # Group B
        [3, 4, 6, 10, 11],  # Group B
        [5, 6, 7, 8, 9],    # Group C
        [5, 6, 7, 12, 13],  # Group C
    ]
    
    print(f"Created {len(contexts)} synthetic contexts (all length {len(contexts[0])})")
    for i, ctx in enumerate(contexts):
        print(f"  Context {i}: {ctx}")
    
    # Build context index
    print("\nBuilding context index...")
    clustering_result = build_context_index(contexts, use_gpu=False, alpha=0.005)
    
    # Schedule contexts
    print("\nScheduling contexts...")
    scheduler = InterContextScheduler()
    scheduled_reordered, scheduled_originals, index_mapping, groups = scheduler.schedule_contexts(
        clustering_result
    )
    
    print(f"\nCreated {len(groups)} groups:")
    for i, (score, group_indices) in enumerate(groups):
        print(f"  Group {i}: {len(group_indices)} contexts, score={score:.2f}, indices={group_indices}")
    
    # Get the final contexts after scheduler reordering
    # The scheduler updates the tree nodes' doc_ids based on parent prefix
    # We need to extract the final reordered contexts from the tree nodes
    final_contexts = []
    for i in range(len(contexts)):
        # Find the leaf node for this context
        for node_id, node in clustering_result.unique_nodes.items():
            if node.is_leaf and i in node.original_indices:
                final_contexts.append(node.doc_ids)
                break
    
    print(f"\nFinal contexts after scheduler node reordering:")
    for i, ctx in enumerate(final_contexts[:5]):  # Show first 5
        print(f"  Context {i}: {ctx[:10]}{'...' if len(ctx) > 10 else ''}")
    
    # Verify prefix sharing using final contexts
    all_passed, issues = verify_group_prefix_sharing(
        groups, 
        final_contexts,  # Use contexts after scheduler's node reordering
        verbose=True
    )
    
    if all_passed:
        print("\n✅ TEST PASSED: All groups share common prefix!")
        return True
    else:
        print("\n❌ TEST FAILED: Some groups don't share common prefix!")
        return False


def test_real_dataset_sample():
    """
    Test with a sample from real dataset.
    """
    print("\n" + "="*80)
    print("TEST 2: Real Dataset Sample")
    print("="*80)
    
    import json
    
    # Load a sample from real dataset
    dataset_path = '/home/jysc/Demnok/datasets/multihopRAG/mulhoprag_new_queries_top15.jsonl'
    
    try:
        with open(dataset_path, 'r') as f:
            queries = [json.loads(line) for line in f][:50]  # Use first 50 queries
        
        contexts = [q['top_k_doc_id'] for q in queries]
        print(f"Loaded {len(contexts)} queries from dataset")
        
        # Build context index
        print("\nBuilding context index...")
        clustering_result = build_context_index(contexts, use_gpu=False, alpha=0.005)
        
        # Schedule contexts
        print("\nScheduling contexts...")
        scheduler = InterContextScheduler()
        scheduled_reordered, scheduled_originals, index_mapping, groups = scheduler.schedule_contexts(
            clustering_result
        )
        
        print(f"\nCreated {len(groups)} groups")
        
        # Get final contexts after scheduler node reordering
        final_contexts = []
        for i in range(len(contexts)):
            for node_id, node in clustering_result.unique_nodes.items():
                if node.is_leaf and i in node.original_indices:
                    final_contexts.append(node.doc_ids)
                    break
        
        # Verify prefix sharing
        all_passed, issues = verify_group_prefix_sharing(
            groups, 
            final_contexts,  # Use contexts after scheduler's node reordering
            verbose=False  # Less verbose for real data
        )
        
        # Print summary
        print(f"\nGroups with 2+ members:")
        for i, (score, group_indices) in enumerate(groups):
            if len(group_indices) > 1:
                group_contexts = [final_contexts[idx] for idx in group_indices]
                common_prefix = find_common_prefix(group_contexts)
                print(f"  Group {i}: size={len(group_indices)}, "
                      f"common_prefix_len={len(common_prefix)}, "
                      f"score={score:.2f}")
        
        if issues:
            print("\n⚠️  Groups without common prefix:")
            for issue in issues:
                print(f"  Group {issue['group_id']}: size={issue['size']}, score={issue['score']:.2f}")
        
        if all_passed:
            print("\n✅ TEST PASSED: All groups share common prefix!")
            return True
        else:
            print(f"\n❌ TEST FAILED: {len(issues)} groups don't share common prefix!")
            return False
            
    except FileNotFoundError:
        print(f"⚠️  Dataset not found at {dataset_path}, skipping test")
        return None


def test_edge_cases():
    """
    Test edge cases with same-length contexts but different prefix patterns.
    """
    print("\n" + "="*80)
    print("TEST 3: Edge Cases (Same Length Contexts)")
    print("="*80)
    
    # Create edge case contexts - all with same length (5 docs)
    contexts = [
        [1, 5, 6, 7, 8],        # Starts with 1
        [1, 2, 10, 11, 12],     # Starts with 1, 2
        [1, 2, 3, 13, 14],      # Starts with 1, 2, 3
        [2, 15, 16, 17, 18],    # Starts with 2
        [2, 3, 19, 20, 21],     # Starts with 2, 3
        [2, 3, 4, 22, 23],      # Starts with 2, 3, 4
        [1, 2, 3, 4, 24],       # Starts with 1, 2, 3, 4
    ]
    
    print(f"Created {len(contexts)} edge case contexts (all length {len(contexts[0])})")
    for i, ctx in enumerate(contexts):
        print(f"  Context {i}: {ctx}")
    
    # Build context index
    print("\nBuilding context index...")
    clustering_result = build_context_index(contexts, use_gpu=False, alpha=0.005)
    
    # Schedule contexts
    print("\nScheduling contexts...")
    scheduler = InterContextScheduler()
    scheduled_reordered, scheduled_originals, index_mapping, groups = scheduler.schedule_contexts(
        clustering_result
    )
    
    print(f"\nCreated {len(groups)} groups")
    
    # Get final contexts after scheduler node reordering
    final_contexts = []
    for i in range(len(contexts)):
        for node_id, node in clustering_result.unique_nodes.items():
            if node.is_leaf and i in node.original_indices:
                final_contexts.append(node.doc_ids)
                break
    
    # Verify prefix sharing
    all_passed, issues = verify_group_prefix_sharing(
        groups, 
        final_contexts,  # Use contexts after scheduler's node reordering
        verbose=True
    )
    
    if all_passed:
        print("\n✅ TEST PASSED: All groups share common prefix!")
        return True
    else:
        print("\n❌ TEST FAILED: Some groups don't share common prefix!")
        return False


def main():
    """
    Run all tests.
    """
    print("\n" + "="*80)
    print("GROUP PREFIX SHARING TEST SUITE")
    print("="*80)
    
    results = []
    
    # Test 1: Simple synthetic case
    result1 = test_simple_synthetic_case()
    results.append(("Simple Synthetic", result1))
    
    # Test 2: Real dataset sample
    result2 = test_real_dataset_sample()
    if result2 is not None:
        results.append(("Real Dataset", result2))
    
    # Test 3: Edge cases
    result3 = test_edge_cases()
    results.append(("Edge Cases", result3))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    
    for test_name, result in results:
        if result is True:
            print(f"✅ {test_name}: PASSED")
        elif result is False:
            print(f"❌ {test_name}: FAILED")
        else:
            print(f"⚠️  {test_name}: SKIPPED")
    
    all_passed = all(r in [True, None] for _, r in results)
    any_failed = any(r is False for _, r in results)
    
    if all_passed and not any_failed:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n💥 SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())
