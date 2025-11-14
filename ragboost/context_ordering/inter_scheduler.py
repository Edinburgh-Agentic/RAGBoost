"""
Inter-Context Scheduler (RAGBoost Paper Algorithm)

This module implements the scheduling algorithm described in the RAGBoost paper:
1. Reuses search paths obtained during context ordering (no redundant tree lookups)
2. Groups contexts by the first element of their search path, naturally separating cache regions
3. Sorts contexts within each group by path length in descending order

This avoids the O(N log M) tree rescanning overhead of existing methods.
"""

from typing import List, Tuple, Any, Dict
from collections import defaultdict


class InterContextScheduler:
    """
    Schedules context execution using search-path-based grouping.
    
    This scheduler:
    - Reuses search paths obtained during context ordering (no redundant tree lookups)
    - Groups contexts by the first element of their search path
    - Sorts contexts within each group by path length descending
    
    Time complexity: O(N) grouping + O(N log N) sorting over N contexts
    (Independent of tree size M, unlike traditional O(N log M) + O(N log N) methods)
    """
    
    def schedule_contexts(
        self, 
        clustering_result: Any
    ) -> Tuple[List, List, List, List]:
        """
        Schedule contexts using search-path-based grouping and sorting.
        
        Args:
            clustering_result: An object containing `search_paths`, `reordered_prompts`,
                             and `original_prompts` from the clustering stage

        Returns:
            A tuple of (scheduled_reordered, scheduled_originals, final_index_mapping, 
                       all_groups_with_info)
        """
        reordered_contexts = clustering_result.reordered_prompts
        original_contexts = clustering_result.original_prompts
        search_paths = clustering_result.search_paths

        # Step 1: Group contexts by the first child index in their search path
        # This naturally separates cache regions - O(N) complexity
        groups_by_root = self._group_by_root_prefix(search_paths)
        
        # Step 2: Sort by path length in descending order within each group
        # Ensures longer prefix matches execute before shorter ones
        sorted_groups = self._sort_groups_by_path_length(
            groups_by_root, search_paths, reordered_contexts
        )
        
        # Step 3: Calculate scores for groups (for compatibility with existing interface)
        all_groups_with_info = []
        for group_indices in sorted_groups:
            # Score based on common prefix length * group size
            score = self._calculate_group_score(reordered_contexts, group_indices)
            all_groups_with_info.append((score, group_indices))
        
        # Step 4: Sort groups by score, then by first index for deterministic ordering
        # (higher score is better, lower first index for ties)
        all_groups_with_info.sort(key=lambda x: (-x[0], x[1][0] if x[1] else float('inf')))
        
        # Step 5: Create final ordering
        final_index_mapping = [idx for score, group in all_groups_with_info for idx in group]
        
        scheduled_reordered = [reordered_contexts[i] for i in final_index_mapping]
        scheduled_originals = [original_contexts[i] for i in final_index_mapping]
        
        return scheduled_reordered, scheduled_originals, final_index_mapping, all_groups_with_info

    def _group_by_root_prefix(
        self, 
        search_paths: List[List[int]]
    ) -> Dict[int, List[int]]:
        """
        Group contexts by the first child index in their search path.
        
        This O(N) operation naturally separates contexts into cache regions
        based on which child of the root they belong to. The first element of
        the search path is the child index (0, 1, 2, ...) indicating which
        child of the root to follow.
        
        Args:
            search_paths: List of search paths (child indices) for each context
            
        Returns:
            Dictionary mapping first child index to list of context indices
        """
        groups = defaultdict(list)
        
        for context_idx, path in enumerate(search_paths):
            if len(path) >= 1:
                # Group by first child index (which child of root)
                group_key = path[0]
                groups[group_key].append(context_idx)
            else:
                # Empty path - group separately
                groups[-1].append(context_idx)
        
        return groups

    def _sort_groups_by_path_length(
        self, 
        groups_by_root: Dict[int, List[int]], 
        search_paths: List[List[int]],
        contexts: List[List[int]]
    ) -> List[List[int]]:
        """
        Sort contexts within each group by path length in descending order.
        
        This ensures longer prefix matches execute before shorter ones,
        maximizing cache reuse under tight KV budgets, as described in the paper.
        
        Total complexity: O(N log N) across all groups
        
        Args:
            groups_by_root: Groups of context indices by root prefix
            search_paths: Search paths for each context
            contexts: Reordered contexts (unused in simplified version)
            
        Returns:
            List of sorted groups
        """
        sorted_groups = []
        
        for root_prefix, group_indices in groups_by_root.items():
            # Sort by path length in descending order, with index as tiebreaker
            sorted_group = sorted(
                group_indices,
                key=lambda idx: (-len(search_paths[idx]), idx)
            )
            sorted_groups.append(sorted_group)
        
        return sorted_groups

    def _calculate_group_score(
        self, 
        contexts: List[List[int]], 
        group_indices: List[int]
    ) -> float:
        """
        Calculate a score for a group based on common prefix length and group size.
        
        Args:
            contexts: List of all contexts
            group_indices: Indices of contexts in this group
            
        Returns:
            Score for the group (higher is better)
        """
        if len(group_indices) <= 1:
            return 0

        group_contexts = [contexts[idx] for idx in group_indices]
        
        try:
            min_len = min(len(c) for c in group_contexts)
        except (ValueError, TypeError):
            return 0

        if min_len == 0:
            return 0
            
        # Calculate common prefix length across all group members
        common_prefix_len = 0
        for i in range(min_len):
            first_item = group_contexts[0][i]
            if all(c[i] == first_item for c in group_contexts[1:]):
                common_prefix_len += 1
            else:
                break
        
        # Score = common_prefix_length * group_size
        return common_prefix_len * len(group_indices)
