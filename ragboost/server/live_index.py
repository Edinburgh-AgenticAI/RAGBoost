"""
Live Context Index

Dynamic context index supporting search, insertion, updates, and eviction.
Implements the algorithms described in the RAGBoost paper.

Inherits from ContextIndex to provide:
1. Initial construction and clustering
2. Intra-context reordering  
3. Inter-context scheduling
4. Then becomes live for dynamic updates

Key Design for Request Tracking:
- request_id is ONLY stored on leaf nodes (actual requests)
- Parent/intermediate nodes do NOT have request_id values
- Eviction removes entire request when all tokens are evicted
- Tree is automatically pruned when branches become empty
"""

import time
import uuid
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import deque

from ..context_index.index_construction import ContextIndex
from ..context_index.tree_nodes import ClusterNode
from ..context_ordering import InterContextScheduler
from .metadata import NodeMetadata
from .eviction_heap import EvictionHeap


def compute_prefix_length(list1: List[int], list2: List[int]) -> int:
    """Compute the length of common prefix between two lists."""
    length = 0
    for a, b in zip(list1, list2):
        if a == b:
            length += 1
        else:
            break
    return length


class LiveContextIndex(ContextIndex):
    """
    Live context index with dynamic updates and request tracking.
    
    Workflow:
    1. Build initial index: build_and_schedule() -> constructs tree, reorders, schedules
    2. Go live: enable dynamic search, insert, update, evict operations
    3. Track requests: each leaf node has a unique request_id
    4. Accumulate tokens: update input+output tokens when requests complete
    5. Evict on capacity: when heap exceeds max_tokens, evict LRU requests
    
    Key invariants:
    - Only leaf nodes have request_id values
    - Parent/intermediate nodes do NOT have request_id
    - Eviction heap and context index remain synchronized
    - Tree is automatically pruned when branches become empty
    
    Supports:
    - Context search: O(|C| · log n)
    - Node traversal: O(h)
    - Context insertion: O(1) or O(|C|)
    - Token updates: O(1)
    - Eviction: O(k · h) for k nodes
    """
    
    def __init__(self, alpha: float = 0.005, use_gpu: bool = False,
                 linkage_method: str = "average", batch_size: int = 10000,
                 max_tokens: Optional[int] = None):
        """
        Initialize live context index.
        
        Args:
            alpha: Distance computation parameter
            use_gpu: Whether to use GPU for distance computation
            linkage_method: Linkage method for hierarchical clustering
            batch_size: Batch size for distance computation
            max_tokens: Maximum token capacity (triggers eviction when exceeded)
        """
        # Initialize parent ContextIndex
        super().__init__(alpha=alpha, use_gpu=use_gpu,
                        linkage_method=linkage_method, batch_size=batch_size)
        
        # Additional components for live operations
        self.metadata: Dict[int, NodeMetadata] = {}
        self.eviction_heap = EvictionHeap(max_tokens=max_tokens)
        self.inter_scheduler = InterContextScheduler()
        
        # Request tracking
        self._request_to_node: Dict[str, int] = {}  # request_id -> node_id
        self._max_tokens = max_tokens
        self._next_request_counter: int = 0  # Counter for auto-generating request_ids
        
        # Track if index is live
        self.is_live = False
        self.initial_result = None
        self.scheduled_result = None
        
        # Tree structure aliases (for backwards compatibility)
        self.nodes: Dict[int, ClusterNode] = {}
        self.root_id: Optional[int] = None
        self.next_node_id: int = 0
        
        # Statistics for live operations
        self.live_stats = {
            'total_searches': 0,
            'total_insertions': 0,
            'total_updates': 0,
            'total_evictions': 0,
            'total_token_updates': 0,
            'total_search_time_us': 0,
            'total_traversal_time_us': 0,
        }
    
    @property
    def max_tokens(self) -> Optional[int]:
        """Get maximum token capacity."""
        return self._max_tokens
    
    @max_tokens.setter
    def max_tokens(self, value: int):
        """Set maximum token capacity."""
        self._max_tokens = value
        self.eviction_heap.max_tokens = value
    
    def build_and_schedule(self, contexts: List[List[int]], 
                          initial_tokens_per_context: int = 0) -> Dict:
        """
        Build index, reorder contexts, schedule execution, and go live.
        
        This is the main entry point that combines:
        1. fit_transform() - build tree and reorder contexts
        2. Inter-context scheduling - optimize execution order
        3. Initialize live metadata - prepare for dynamic updates
        
        Args:
            contexts: List of contexts (each is a list of document IDs)
            initial_tokens_per_context: Initial token count for each context
            
        Returns:
            Dictionary with scheduled results, index info, and request_id mapping
        """
        print("=" * 80)
        print("BUILDING LIVE CONTEXT INDEX")
        print("=" * 80)
        
        # Step 1: Build static index (clustering + reordering)
        print("\n1. Building static index...")
        self.initial_result = self.fit_transform(contexts)
        
        print(f"   ✓ Built tree with {self.initial_result.stats['total_nodes']} nodes")
        print(f"   ✓ Leaf nodes: {self.initial_result.stats['leaf_nodes']}")
        
        # Step 2: Inter-context scheduling
        print("\n2. Scheduling contexts for optimal execution...")
        scheduled_reordered, scheduled_originals, final_mapping, groups = \
            self.inter_scheduler.schedule_contexts(self.initial_result)
        
        print(f"   ✓ Created {len(groups)} execution groups")
        
        self.scheduled_result = {
            'scheduled_reordered': scheduled_reordered,
            'scheduled_originals': scheduled_originals,
            'final_mapping': final_mapping,
            'groups': groups,
            'clustering_result': self.initial_result
        }
        
        # Step 3: Initialize live metadata (auto-generates request_ids)
        print("\n3. Initializing live metadata...")
        request_id_mapping, request_ids_ordered = self._initialize_live_metadata(initial_tokens_per_context)
        
        print(f"   ✓ Initialized {len(self.metadata)} nodes with metadata")
        print(f"   ✓ Eviction heap size: {len(self.eviction_heap)}")
        print(f"   ✓ Auto-assigned {len(request_id_mapping)} request IDs")
        
        # Add request_id mapping to result (dict and ordered list)
        self.scheduled_result['request_id_mapping'] = request_id_mapping
        self.scheduled_result['request_ids'] = request_ids_ordered  # Ordered list matching input contexts
        
        # Step 4: Mark as live
        self.is_live = True
        
        print("\n" + "=" * 80)
        print("✓ INDEX IS NOW LIVE - Ready for dynamic operations")
        print("=" * 80 + "\n")
        
        return self.scheduled_result
    
    def _initialize_live_metadata(self, initial_tokens_per_context: int) -> Tuple[Dict[str, int], List[str]]:
        """
        Initialize metadata for all nodes after static index is built.
        
        Auto-generates request_id for each leaf node during construction.
        Returns both a mapping dict and an ordered list of request_ids.
        
        Args:
            initial_tokens_per_context: Initial token count for each context
            
        Returns:
            Tuple of:
                - Dictionary mapping request_id -> node_id for all leaf nodes
                - List of request_ids in same order as input contexts
        """
        if not self.initial_result:
            raise RuntimeError("Must call fit_transform() before initializing metadata")
        
        unique_nodes = self.initial_result.unique_nodes
        request_id_mapping = {}  # request_id -> node_id
        
        # Set up node aliases
        self.nodes = unique_nodes
        
        # Find root
        for node_id, node in unique_nodes.items():
            if hasattr(node, 'is_root') and node.is_root:
                self.root_id = node_id
                break
        
        # Set next node ID
        self.next_node_id = max(unique_nodes.keys()) + 1 if unique_nodes else 0
        
        # Counter for auto-generating request_ids
        leaf_counter = 0
        
        # Track original context index -> request_id mapping
        original_index_to_request_id = {}
        
        # Initialize metadata for all nodes
        for node_id, node in unique_nodes.items():
            search_path = self._compute_search_path(node_id)
            
            # Determine if this is a leaf node
            is_leaf = hasattr(node, 'is_leaf') and node.is_leaf
            
            # Compute token counts (only leaf nodes have initial tokens)
            if is_leaf:
                total_tokens = initial_tokens_per_context
                # Auto-generate request_id for leaf nodes using UUID
                request_id = f"req-{uuid.uuid4().hex[:12]}"
                leaf_counter += 1
                
                # Track which original context index this leaf represents
                if hasattr(node, 'original_indices') and node.original_indices:
                    for orig_idx in node.original_indices:
                        original_index_to_request_id[orig_idx] = request_id
            else:
                # Internal node: no direct tokens, no request_id
                total_tokens = 0
                request_id = None
            
            # Compute extra tokens (beyond parent)
            parent_tokens = 0
            if node.parent is not None and node.parent in self.metadata:
                parent_tokens = self.metadata[node.parent].total_tokens
            extra_tokens = max(0, total_tokens - parent_tokens)
            
            # Create metadata with auto-generated request_id for leaf nodes
            metadata = NodeMetadata(
                node_id=node_id,
                total_tokens=total_tokens,
                extra_tokens=extra_tokens,
                search_path=search_path,
                is_leaf=is_leaf,
                doc_ids=node.doc_ids if hasattr(node, 'doc_ids') else None,
                request_id=request_id
            )
            
            self.metadata[node_id] = metadata
            
            # Add leaf nodes to eviction heap and tracking
            if is_leaf and request_id:
                self._request_to_node[request_id] = node_id
                self.eviction_heap.push(metadata)
                request_id_mapping[request_id] = node_id
        
        self.next_node_id = max(self.nodes.keys()) + 1 if self.nodes else 0
        self._next_request_counter = leaf_counter  # Track for future inserts
        
        # Build ordered list of request_ids matching original context order
        num_contexts = len(original_index_to_request_id)
        request_ids_ordered = [
            original_index_to_request_id.get(i) for i in range(num_contexts)
        ]
        
        return request_id_mapping, request_ids_ordered
    
    # =========================================================================
    # Token Tracking (For SGLang Integration)
    # =========================================================================
    
    def update_tokens(self, request_id: str, num_tokens: int) -> Dict[str, Any]:
        """
        Update token count for a request.
        
        THIS IS THE ENDPOINT SGLANG SHOULD CALL TO UPDATE THE EVICTION HEAP.
        
        When SGLang processes a request, call this to update the token count.
        If the heap exceeds max_tokens, eviction is triggered automatically.
        
        Args:
            request_id: The unique request identifier (from build/insert response)
            num_tokens: Total tokens for this request (input + output)
            
        Returns:
            Dictionary with update results and any eviction that occurred
        """
        result = {
            'request_id': request_id,
            'num_tokens': num_tokens,
            'updated': False,
            'eviction_triggered': False,
            'evicted_requests': []
        }
        
        # Find the node for this request
        node_id = self._request_to_node.get(request_id)
        if node_id is None:
            return result
        
        metadata = self.metadata.get(node_id)
        if metadata is None:
            return result
        
        # Update token count
        # For leaf nodes, extra_tokens = total_tokens - parent_tokens
        # When we update total_tokens, extra_tokens changes by the same delta
        old_tokens = metadata.total_tokens
        old_extra = metadata.extra_tokens
        delta = num_tokens - old_tokens
        
        if delta > 0:
            metadata.add_tokens(delta)
        elif delta < 0:
            metadata.remove_tokens(abs(delta))
        
        # extra_tokens also changes by delta (parent tokens stay the same)
        new_extra = metadata.extra_tokens
        extra_delta = new_extra - old_extra
        
        metadata.update_access_time()
        self.eviction_heap.update_access_time(node_id)
        
        # Update heap's total token tracking (heap tracks extra_tokens, not total)
        self.eviction_heap._total_tokens += extra_delta
        
        result['updated'] = True
        result['previous_tokens'] = old_tokens
        result['current_tokens'] = num_tokens
        result['extra_tokens'] = new_extra
        self.live_stats['total_token_updates'] += 1
        
        # Log the token update for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"Token update: request_id={request_id}, node_id={node_id}, "
            f"total: {old_tokens} -> {num_tokens}, extra: {old_extra} -> {new_extra}, "
            f"heap_total={self.eviction_heap.total_tokens()}, "
            f"max_tokens={self._max_tokens}"
        )
        
        # Check if eviction is needed
        if self.eviction_heap.needs_eviction():
            tokens_to_evict = self.eviction_heap.tokens_to_evict()
            eviction_result = self.evict(tokens_to_evict)
            
            result['eviction_triggered'] = True
            result['evicted_requests'] = eviction_result.get('evicted_request_ids', [])
            result['tokens_evicted'] = eviction_result.get('tokens_evicted', 0)
            result['heap_tokens_after'] = self.eviction_heap.total_tokens()
        
        return result
    
    def touch(self, request_id: str) -> bool:
        """
        Update access time for a request (LRU sync with SGLang).
        
        Call this when SGLang accesses a cached prefix to keep
        the eviction heap's LRU ordering in sync.
        
        Args:
            request_id: The unique request identifier
            
        Returns:
            True if request was found and touched, False otherwise
        """
        node_id = self._request_to_node.get(request_id)
        if node_id is None:
            return False
        
        metadata = self.metadata.get(node_id)
        if metadata is None:
            return False
        
        # Update access time
        old_time = metadata.last_access_time
        metadata.update_access_time()
        self.eviction_heap.update_access_time(node_id)
        
        self.live_stats['total_touches'] = self.live_stats.get('total_touches', 0) + 1
        
        # Log the touch for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"Touch: request_id={request_id}, node_id={node_id}, "
            f"access_time: {old_time:.3f} -> {metadata.last_access_time:.3f}, "
            f"heap_size={len(self.eviction_heap)}, total_tokens={self.eviction_heap.total_tokens()}"
        )
        
        return True
    
    def touch_batch(self, request_ids: List[str]) -> Dict[str, Any]:
        """
        Update access time for multiple requests at once.
        
        Args:
            request_ids: List of request IDs to touch
            
        Returns:
            Dictionary with touched and not_found counts
        """
        touched = 0
        not_found = []
        
        for request_id in request_ids:
            if self.touch(request_id):
                touched += 1
            else:
                not_found.append(request_id)
        
        return {
            'touched': touched,
            'not_found': not_found
        }
    
    def get_request_node(self, request_id: str) -> Optional[int]:
        """
        Get the node_id for a request.
        
        Args:
            request_id: The unique request identifier
            
        Returns:
            node_id if found, None otherwise
        """
        return self._request_to_node.get(request_id)
    
    def get_all_request_ids(self) -> Set[str]:
        """Get all tracked request IDs."""
        return set(self._request_to_node.keys())
    
    def search(self, context: List[int], update_access: bool = True) -> Tuple[List[int], int, int]:
        """
        Search for best matching node using greedy descent.
        
        Algorithm:
        1. Start at root
        2. At each level, select child with minimum distance
        3. Stop at leaf or when all children are equidistant
        4. Return search path and matched node
        
        Complexity: O(|C| · log n)
        
        Args:
            context: Query context (list of document IDs)
            update_access: Whether to update LRU timestamp
            
        Returns:
            Tuple of (search_path, matched_node_id, shared_prefix_length)
        """
        start_time = time.perf_counter()
        
        if self.root_id is None:
            return ([], -1, 0)
        
        current_id = self.root_id
        search_path = []
        
        while current_id is not None:
            current_node = self.nodes[current_id]
            
            # If leaf, we're done
            if current_node.is_leaf:
                break
            
            # If no children, stop here
            if not current_node.children:
                break
            
            # Find child with minimum distance
            min_distance = float('inf')
            best_child_idx = None
            distances = []
            
            for idx, child_id in enumerate(current_node.children):
                if child_id not in self.nodes:
                    continue
                
                child_node = self.nodes[child_id]
                child_docs = self.metadata[child_id].doc_ids or child_node.doc_ids
                
                # Compute prefix length (shared tokens)
                prefix_len = compute_prefix_length(context, child_docs)
                distance = len(context) - prefix_len  # Distance = non-matching tokens
                
                distances.append(distance)
                
                if distance < min_distance:
                    min_distance = distance
                    best_child_idx = idx
            
            # Check if all children are equidistant
            if distances and all(d == distances[0] for d in distances):
                # All equidistant, stop here (longest shared prefix found)
                break
            
            # Descend to best child
            if best_child_idx is not None:
                search_path.append(best_child_idx)
                current_id = current_node.children[best_child_idx]
            else:
                break
        
        # Compute shared prefix length with matched node
        matched_node = self.nodes[current_id]
        matched_docs = self.metadata[current_id].doc_ids or matched_node.doc_ids
        shared_prefix_length = compute_prefix_length(context, matched_docs) if matched_docs else 0
        
        # Update access time
        if update_access and current_id in self.metadata:
            self.metadata[current_id].update_access_time()
            self.eviction_heap.update_access_time(current_id)
        
        # Update statistics
        elapsed_us = (time.perf_counter() - start_time) * 1_000_000
        self.live_stats['total_searches'] += 1
        self.live_stats['total_search_time_us'] += elapsed_us
        
        return (search_path, current_id, shared_prefix_length)
    
    def traverse(self, search_path: List[int]) -> Optional[ClusterNode]:
        """
        Traverse to a node using its search path.
        
        Complexity: O(h) where h is tree height
        
        Args:
            search_path: List of child indices from root
            
        Returns:
            ClusterNode at the end of the path, or None if invalid
        """
        start_time = time.perf_counter()
        
        if self.root_id is None:
            return None
        
        current_id = self.root_id
        
        for child_idx in search_path:
            if current_id not in self.nodes:
                return None
            
            current_node = self.nodes[current_id]
            
            if not current_node.children or child_idx >= len(current_node.children):
                return None
            
            current_id = current_node.children[child_idx]
        
        # Update statistics
        elapsed_us = (time.perf_counter() - start_time) * 1_000_000
        self.live_stats['total_traversal_time_us'] += elapsed_us
        
        return self.nodes.get(current_id)
    
    def insert(self, context: List[int], search_path: List[int], 
               total_tokens: int = 0) -> Tuple[int, List[int], str]:
        """
        Insert a new context into the index.
        
        Two cases:
        1. Matched internal node: Append as child - O(1)
        2. Matched leaf: Insert as sibling (child of leaf's parent) - O(1)
        
        Auto-generates a unique request_id for the new leaf node.
        
        Args:
            context: New context to insert
            search_path: Search path from search() operation
            total_tokens: Initial token count
            
        Returns:
            Tuple of (new_node_id, new_search_path, request_id)
        """
        start_time = time.perf_counter()
        
        # Find matched node
        matched_node = self.traverse(search_path)
        
        if matched_node is None:
            # Invalid path, insert at root
            matched_node = self.nodes[self.root_id]
            search_path = []
        
        matched_id = matched_node.node_id
        
        if matched_node.is_leaf:
            # Case 2: Matched a leaf, insert as sibling (child of leaf's parent)
            new_node_id, new_search_path, request_id = self._insert_at_leaf(
                context, matched_node, search_path, total_tokens
            )
        else:
            # Case 1: Matched internal node, append as child
            new_node_id, new_search_path, request_id = self._insert_at_internal(
                context, matched_node, search_path, total_tokens
            )
        
        # Update statistics
        elapsed_us = (time.perf_counter() - start_time) * 1_000_000
        self.live_stats['total_insertions'] += 1
        
        return (new_node_id, new_search_path, request_id)
    
    def _insert_at_internal(self, context: List[int], parent_node: ClusterNode,
                           search_path: List[int], total_tokens: int) -> Tuple[int, List[int], str]:
        """Insert new context as child of internal node."""
        # Auto-generate request_id using UUID
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        
        # Create new leaf node
        new_node = ClusterNode(
            node_id=self.next_node_id,
            content=context,
            children=[],
            parent=parent_node.node_id,
            original_indices={self.next_node_id}
        )
        
        self.nodes[self.next_node_id] = new_node
        parent_node.add_child(self.next_node_id)
        
        # Create metadata with auto-generated request_id
        parent_tokens = self.metadata[parent_node.node_id].total_tokens
        metadata = NodeMetadata(
            node_id=self.next_node_id,
            total_tokens=total_tokens,
            extra_tokens=max(0, total_tokens - parent_tokens),
            search_path=search_path + [len(parent_node.children) - 1],
            doc_ids=context,
            is_leaf=True,
            request_id=request_id
        )
        
        self.metadata[self.next_node_id] = metadata
        self._request_to_node[request_id] = self.next_node_id
        self.eviction_heap.push(metadata)
        
        new_search_path = search_path + [len(parent_node.children) - 1]
        new_node_id = self.next_node_id
        self.next_node_id += 1
        
        return (new_node_id, new_search_path, request_id)
    
    def _insert_at_leaf(self, context: List[int], leaf_node: ClusterNode,
                       search_path: List[int], total_tokens: int) -> Tuple[int, List[int], str]:
        """
        Insert new context as sibling of the matched leaf node.
        
        Instead of creating a new internal node, we simply insert the new context
        as another child of the leaf's parent node.
        """
        # Auto-generate request_id using UUID
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        
        # Get parent node
        if leaf_node.parent is None:
            # Leaf is directly under root, use root as parent
            parent_node = self.nodes[self.root_id]
            parent_search_path = []
        else:
            parent_node = self.nodes[leaf_node.parent]
            # Parent's search path is the leaf's search path without the last element
            parent_search_path = search_path[:-1] if search_path else []
        
        # Create new leaf node as sibling of matched leaf
        new_leaf = ClusterNode(
            node_id=self.next_node_id,
            content=context,
            children=[],
            parent=parent_node.node_id,
            original_indices={self.next_node_id}
        )
        
        self.nodes[self.next_node_id] = new_leaf
        parent_node.add_child(self.next_node_id)
        new_leaf_id = self.next_node_id
        self.next_node_id += 1
        
        # New search path: parent's path + index of new child
        new_search_path = parent_search_path + [len(parent_node.children) - 1]
        
        # Create metadata for new leaf with auto-generated request_id
        parent_tokens = self.metadata[parent_node.node_id].total_tokens if parent_node.node_id in self.metadata else 0
        new_metadata = NodeMetadata(
            node_id=new_leaf_id,
            total_tokens=total_tokens,
            extra_tokens=max(0, total_tokens - parent_tokens),
            search_path=new_search_path,
            doc_ids=context,
            is_leaf=True,
            request_id=request_id
        )
        self.metadata[new_leaf_id] = new_metadata
        self._request_to_node[request_id] = new_leaf_id
        self.eviction_heap.push(new_metadata)
        
        return (new_leaf_id, new_search_path, request_id)
    
    def update_node(self, search_path: List[int], token_delta: int) -> bool:
        """
        Update a node's token count.
        
        Args:
            search_path: Path to the node
            token_delta: Tokens to add (positive) or remove (negative)
            
        Returns:
            True if update successful, False otherwise
        """
        node = self.traverse(search_path)
        
        if node is None or node.node_id not in self.metadata:
            return False
        
        metadata = self.metadata[node.node_id]
        
        if token_delta > 0:
            metadata.add_tokens(token_delta)
            self.eviction_heap.update_access_time(node.node_id)
        else:
            metadata.remove_tokens(abs(token_delta))
        
        self.live_stats['total_updates'] += 1
        
        return True
    
    def evict(self, num_tokens: int) -> Dict[str, Any]:
        """
        Evict tokens from LRU nodes to sync with SGLang's cache eviction.
        
        THIS IS THE API THAT SGLANG SHOULD CALL WHEN IT EVICTS CACHE.
        
        When SGLang's radix_cache.evict(num_tokens) is called, SGLang should
        also call this method with the same num_tokens to keep our index in sync.
        
        Integration point in SGLang:
            # In scheduler_batch.py or wherever eviction happens
            def evict_from_tree_cache(tree_cache, num_tokens):
                tree_cache.evict(num_tokens)
                
                # Add this line to sync with RAGBoost index
                if hasattr(self, 'ragboost_index') and self.ragboost_index is not None:
                    self.ragboost_index.evict(num_tokens)
        
        Process:
        1. Pop LRU nodes from heap
        2. Decrement token counts
        3. Remove nodes with zero tokens
        4. Recursively delete empty parents
        
        Args:
            num_tokens: Number of tokens to evict (same as SGLang's eviction)
            
        Returns:
            Dictionary with eviction statistics:
                - evicted_node_ids: List of node IDs that were completely evicted
                - evicted_request_ids: List of request_ids that were evicted
                - tokens_evicted: Total number of tokens actually evicted (extra_tokens only)
                - nodes_remaining: Number of nodes still in the index
                - tokens_remaining: Total tokens remaining in the index
        """
        evicted_nodes = []
        evicted_request_ids = []
        tokens_evicted = 0
        
        # Keep evicting whole leaves until we've freed enough tokens
        while tokens_evicted < num_tokens and not self.eviction_heap.is_empty():
            # Get LRU leaf node (only leaves with request_id are in heap)
            lru_metadata = self.eviction_heap.pop()
            
            if lru_metadata is None:
                break
            
            node_id = lru_metadata.node_id
            request_id = lru_metadata.request_id
            
            # Evict this WHOLE leaf - count its extra_tokens as freed
            # (We can't partially evict - either the whole leaf is cached or not)
            leaf_extra_tokens = lru_metadata.extra_tokens
            tokens_evicted += leaf_extra_tokens
            
            # Track evicted request
            if request_id:
                evicted_request_ids.append(request_id)
                if request_id in self._request_to_node:
                    del self._request_to_node[request_id]
            
            evicted_nodes.append(node_id)
            
            # Remove leaf and prune empty parents
            # This may free additional tokens from shared prefixes that are no longer used
            parent_tokens_freed = self._remove_node_and_prune(node_id)
            tokens_evicted += parent_tokens_freed
        
        self.live_stats['total_evictions'] += len(evicted_nodes)
        
        return {
            'evicted_node_ids': evicted_nodes,
            'evicted_request_ids': evicted_request_ids,
            'tokens_evicted': tokens_evicted,
            'nodes_remaining': len(self.nodes),
            'tokens_remaining': self.eviction_heap.total_tokens(),
            'requests_remaining': len(self._request_to_node)
        }
    
    def get_eviction_stats(self) -> Dict[str, Any]:
        """
        Get current eviction-related statistics.
        
        Useful for monitoring and debugging the synchronization between
        SGLang's cache and our index.
        
        Returns:
            Dictionary with current state:
                - total_nodes: Total nodes in the index
                - active_nodes: Nodes with tokens > 0
                - total_tokens: Sum of all tokens in the index
                - max_tokens: Maximum token capacity
                - utilization_pct: Current utilization percentage
                - heap_size: Number of nodes in eviction heap
                - num_requests: Number of tracked requests
                - oldest_access_time: Timestamp of LRU node
        """
        total_tokens = self.eviction_heap.total_tokens()
        stats = {
            'total_nodes': len(self.nodes),
            'active_nodes': len(self.metadata),
            'total_tokens': total_tokens,
            'max_tokens': self._max_tokens,
            'utilization_pct': (total_tokens / self._max_tokens * 100) if self._max_tokens else 0,
            'heap_size': len(self.eviction_heap),
            'num_requests': len(self._request_to_node),
            'oldest_access_time': None
        }
        
        # Get oldest access time (LRU node)
        lru_node = self.eviction_heap.peek()
        if lru_node:
            stats['oldest_access_time'] = lru_node.last_access_time
        
        return stats
    
    def _remove_node(self, node_id: int):
        """
        Remove a node and recursively delete empty parents.
        
        Args:
            node_id: Node to remove
        """
        self._remove_node_and_prune(node_id)
    
    def _remove_node_and_prune(self, node_id: int) -> int:
        """
        Remove a node and recursively delete empty parents.
        
        When a leaf is removed, if its parent has no other children,
        the parent's tokens (shared prefix) can also be freed.
        This continues up the tree.
        
        Args:
            node_id: Node to remove
            
        Returns:
            Total tokens freed from pruned parent nodes (shared prefix tokens)
        """
        if node_id not in self.nodes:
            return 0
        
        tokens_freed = 0
        node = self.nodes[node_id]
        parent_id = node.parent
        
        # Remove from parent's child list
        if parent_id and parent_id in self.nodes:
            parent = self.nodes[parent_id]
            if node_id in parent.children:
                parent.children.remove(node_id)
            
            # Recursively remove empty parents (not root)
            # When parent becomes childless, its tokens (shared prefix) are freed
            if not parent.children and not parent.is_root:
                if parent_id in self.metadata:
                    # Parent's extra_tokens are now freed (no longer shared)
                    tokens_freed += self.metadata[parent_id].extra_tokens
                tokens_freed += self._remove_node_and_prune(parent_id)
        
        # Delete node
        del self.nodes[node_id]
        
        if node_id in self.metadata:
            del self.metadata[node_id]
        
        self.eviction_heap.remove(node_id)
        
        return tokens_freed
    
    def _compute_search_path(self, node_id: int) -> List[int]:
        """Compute search path from root to node."""
        if node_id == self.root_id:
            return []
        
        path = []
        current_id = node_id
        visited = set()
        
        while current_id != self.root_id and current_id is not None:
            if current_id in visited:
                break
            visited.add(current_id)
            
            node = self.nodes.get(current_id)
            if node is None or node.parent is None:
                break
            
            parent = self.nodes.get(node.parent)
            if parent is None:
                break
            
            try:
                child_idx = parent.children.index(current_id)
                path.append(child_idx)
            except (ValueError, AttributeError):
                break
            
            current_id = node.parent
        
        return path[::-1]
    
    def _find_common_prefix(self, list1: List[int], list2: List[int]) -> List[int]:
        """Find common prefix of two lists."""
        prefix = []
        for a, b in zip(list1, list2):
            if a == b:
                prefix.append(a)
            else:
                break
        return prefix
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        avg_search_time = (self.live_stats['total_search_time_us'] / self.live_stats['total_searches'] 
                          if self.live_stats['total_searches'] > 0 else 0)
        avg_traversal_time = (self.live_stats['total_traversal_time_us'] / 
                             (self.live_stats['total_searches'] + self.live_stats['total_updates'])
                             if (self.live_stats['total_searches'] + self.live_stats['total_updates']) > 0 else 0)
        
        return {
            'num_nodes': len(self.nodes),
            'active_nodes': len(self.metadata),
            'total_tokens': self.eviction_heap.total_tokens(),
            'heap_size': len(self.eviction_heap),
            'total_searches': self.live_stats['total_searches'],
            'total_insertions': self.live_stats['total_insertions'],
            'total_updates': self.live_stats['total_updates'],
            'total_evictions': self.live_stats['total_evictions'],
            'avg_search_time_us': avg_search_time,
            'avg_traversal_time_us': avg_traversal_time,
            **self.eviction_heap.get_stats()
        }
