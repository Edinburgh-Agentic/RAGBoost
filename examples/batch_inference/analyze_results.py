#!/usr/bin/env python3
"""
Analyze the RAGBoost planner results on the real dataset.
"""

import json
from collections import defaultdict
from pathlib import Path


def analyze_results(results_path='real_dataset_output.jsonl'):
    """Analyze the planner output and generate detailed statistics."""
    
    print("="*80)
    print("RAGBoost Planner Results Analysis")
    print("="*80)
    
    # Load results (new format with grouped items)
    groups = []
    all_queries = []
    
    with open(results_path, 'r') as f:
        for line in f:
            group = json.loads(line)
            groups.append(group)
            # Extract individual queries from items
            for item in group.get('items', []):
                query_with_group = {
                    'group_id': group['group_id'],
                    'group_size': group['group_size'],
                    'group_score': group['group_score'],
                    'qid': item['qid'],
                    'text': item['text'],
                    'top_k_doc_id': item.get('top_k_doc_id', item.get('reordered_docs', [])),
                    'orig_top_k_doc_id': item.get('orig_top_k_doc_id', item.get('original_docs', []))
                }
                all_queries.append(query_with_group)
    
    print(f"\nTotal queries processed: {len(all_queries)}")
    print(f"Total groups created: {len(groups)}")
    
    # Group size distribution
    print("\n" + "="*80)
    print("Group Size Distribution")
    print("="*80)
    
    size_dist = defaultdict(int)
    for group in groups:
        size = group['group_size']
        size_dist[size] += 1
    
    for size in sorted(size_dist.keys(), reverse=True):
        count = size_dist[size]
        print(f"  Groups with {size:3d} queries: {count:3d} groups")
    
    # Document sharing analysis
    print("\n" + "="*80)
    print("Document Sharing Analysis (Top 10 Groups)")
    print("="*80)
    
    group_sharing = []
    for group in groups:
        if group['group_size'] <= 1:
            continue
        
        # Find shared documents from items
        all_doc_sets = [set(item.get('orig_top_k_doc_id', item.get('original_docs', []))) 
                       for item in group.get('items', [])]
        if not all_doc_sets:
            continue
            
        common_docs = set.intersection(*all_doc_sets)
        
        # Calculate statistics
        total_docs = sum(len(item.get('orig_top_k_doc_id', item.get('original_docs', []))) 
                        for item in group.get('items', []))
        shared_doc_accesses = len(common_docs) * group['group_size']
        sharing_ratio = shared_doc_accesses / total_docs if total_docs > 0 else 0
        
        group_sharing.append({
            'group_id': group['group_id'],
            'size': group['group_size'],
            'common_docs': len(common_docs),
            'total_docs': total_docs,
            'sharing_ratio': sharing_ratio,
            'items': group.get('items', [])
        })
    
    # Sort by sharing ratio
    group_sharing.sort(key=lambda x: x['sharing_ratio'], reverse=True)
    
    for i, gs in enumerate(group_sharing[:10]):
        print(f"\nGroup {gs['group_id']} (Size: {gs['size']} queries):")
        print(f"  Common documents: {gs['common_docs']}")
        print(f"  Total document accesses: {gs['total_docs']}")
        print(f"  Sharing ratio: {gs['sharing_ratio']:.2%}")
        print(f"  Sample queries:")
        for j, item in enumerate(gs['items'][:2]):
            print(f"    [{j+1}] qid {item['qid']}: {item['text'][:70]}...")
    
    # Overall statistics
    print("\n" + "="*80)
    print("Overall Statistics")
    print("="*80)
    
    total_shared = sum(gs['common_docs'] * gs['size'] for gs in group_sharing)
    total_accesses = sum(gs['total_docs'] for gs in group_sharing)
    overall_sharing = total_shared / total_accesses if total_accesses > 0 else 0
    
    print(f"Total shared document accesses: {total_shared}")
    print(f"Total document accesses: {total_accesses}")
    print(f"Overall sharing ratio: {overall_sharing:.2%}")
    
    # Cache hit potential
    print("\n" + "="*80)
    print("Potential Cache Benefits")
    print("="*80)
    
    multi_query_groups = [gs for gs in group_sharing if gs['size'] > 1]
    if multi_query_groups:
        avg_group_size = sum(gs['size'] for gs in multi_query_groups) / len(multi_query_groups)
        avg_common_docs = sum(gs['common_docs'] for gs in multi_query_groups) / len(multi_query_groups)
        
        print(f"Groups with multiple queries: {len(multi_query_groups)}")
        print(f"Average group size: {avg_group_size:.2f} queries")
        print(f"Average common documents per group: {avg_common_docs:.2f}")
        
        # Estimate cache hits
        cache_hit_potential = sum(
            gs['common_docs'] * (gs['size'] - 1) for gs in multi_query_groups
        )
        print(f"Potential cache hits (assuming sequential execution): {cache_hit_potential}")
        print(f"Cache hit rate: {cache_hit_potential / total_accesses:.2%}")
    
    # Show example of a high-sharing group
    if group_sharing:
        print("\n" + "="*80)
        print("Example: Highest Sharing Group")
        print("="*80)
        
        best_group = group_sharing[0]
        print(f"Group {best_group['group_id']} with {best_group['size']} queries")
        print(f"Sharing ratio: {best_group['sharing_ratio']:.2%}")
        
        # Find common docs
        all_doc_sets = [set(q['original_docs']) for q in best_group['queries']]
        common_docs = set.intersection(*all_doc_sets)
        
        print(f"\nCommon documents (IDs): {sorted(list(common_docs))[:10]}")
        print(f"\nQueries in this group:")
        for i, q in enumerate(best_group['queries'][:5]):
            print(f"\n  Query {i+1} (qid {q['qid']}):")
            print(f"    Text: {q['text'][:100]}...")
            print(f"    Original docs: {q['original_docs'][:5]}...")
            print(f"    Reordered docs: {q['reordered_docs'][:5]}...")
        
        if len(best_group['queries']) > 5:
            print(f"\n  ... and {len(best_group['queries']) - 5} more queries")


def main():
    results_path = Path(__file__).parent / "real_dataset_output.jsonl"
    
    if not results_path.exists():
        print(f"Error: {results_path} not found.")
        print("Please run 'python run_real_dataset.py' first.")
        return
    
    analyze_results(str(results_path))
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
