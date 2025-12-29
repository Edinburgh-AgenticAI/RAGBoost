"""
Example: SGLang Integration with RAGBoost HTTP Server

This shows exactly how to integrate SGLang with the RAGBoost index server.

SETUP:
1. Start RAGBoost server: python -m ragboost.server.http_server --port 8765
2. Initialize index (run this once)
3. Add eviction sync to SGLang (see below)
"""

import requests
import time
from ragboost.server.http_client import RAGBoostIndexClient


# ============================================================================
# STEP 1: Start the Server (Run in separate terminal)
# ============================================================================

"""
Terminal 1:
$ python -m ragboost.server.http_server --port 8765

Output:
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     RAGBoost Index Server starting...
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8765
"""


# ============================================================================
# STEP 2: Initialize the Index (Run once at startup)
# ============================================================================

def initialize_index():
    """Initialize the RAGBoost index with your contexts."""
    
    # Your RAG contexts (each context is a list of document IDs)
    contexts = [
        [1, 5, 10, 15, 20],     # Query 1 uses docs 1, 5, 10, 15, 20
        [2, 5, 11, 16, 21],     # Query 2 uses docs 2, 5, 11, 16, 21
        [1, 5, 12, 17, 22],     # Query 3 uses docs 1, 5, 12, 17, 22
        [3, 6, 13, 18, 23],     # Query 4 uses docs 3, 6, 13, 18, 23
        [1, 5, 10, 19, 24],     # Query 5 uses docs 1, 5, 10, 19, 24
    ]
    
    print("Initializing RAGBoost index...")
    
    response = requests.post(
        "http://localhost:8765/build",
        json={
            "contexts": contexts,
            "initial_tokens_per_context": 512,  # Initial tokens per context
            "use_gpu": True,                     # Use GPU if available
            "alpha": 0.005,
            "linkage_method": "average"
        },
        timeout=30.0  # Building can take a while
    )
    
    result = response.json()
    print(f"✓ Index initialized: {result}")
    return result


# ============================================================================
# STEP 3: SGLang Integration - Add This to Your SGLang Code
# ============================================================================

class SGLangScheduler_WithRAGBoost:
    """
    Example SGLang scheduler with RAGBoost HTTP client integration.
    
    This is what your actual SGLang scheduler should look like.
    """
    
    def __init__(self, model_config, cache_config):
        """Initialize scheduler with RAGBoost client."""
        
        # ... your existing SGLang initialization ...
        self.tree_cache = None  # Your RadixCache
        
        # NEW: Initialize RAGBoost HTTP client
        try:
            self.ragboost_client = RAGBoostIndexClient(
                base_url="http://localhost:8765",
                timeout=1.0  # 1 second timeout for low latency
            )
            
            # Check if server is ready
            if self.ragboost_client.is_ready():
                print("✓ RAGBoost index server connected and ready")
            else:
                print("⚠ RAGBoost index server not ready")
                self.ragboost_client = None
        
        except Exception as e:
            print(f"⚠ Could not connect to RAGBoost server: {e}")
            self.ragboost_client = None
    
    def evict_tokens(self, num_tokens):
        """
        Evict tokens from cache.
        
        THIS IS THE METHOD YOU NEED TO MODIFY IN SGLANG.
        Just add the ragboost_client.evict() call.
        """
        
        # 1. Original SGLang eviction
        print(f"SGLang evicting {num_tokens} tokens from tree_cache...")
        # self.tree_cache.evict(num_tokens)  # Your actual eviction code
        
        # 2. NEW: Sync with RAGBoost index (THIS IS THE ONLY LINE TO ADD!)
        if self.ragboost_client is not None:
            try:
                result = self.ragboost_client.evict(num_tokens)
                if result and result.get("status") == "success":
                    print(f"✓ RAGBoost evicted {result['tokens_evicted']} tokens, "
                          f"removed {len(result['evicted_node_ids'])} nodes")
                else:
                    print(f"⚠ RAGBoost eviction failed: {result}")
            except Exception as e:
                print(f"⚠ RAGBoost eviction sync error: {e}")
        else:
            print("⚠ RAGBoost client not available, skipping sync")


# ============================================================================
# MINIMAL INTEGRATION - Even Simpler
# ============================================================================

class SGLangScheduler_Minimal:
    """Absolute minimal integration - just HTTP POST."""
    
    def __init__(self):
        self.ragboost_url = "http://localhost:8765"
        self.tree_cache = None
    
    def evict_tokens(self, num_tokens):
        """Evict with minimal integration."""
        
        # SGLang eviction
        # self.tree_cache.evict(num_tokens)
        
        # RAGBoost sync (ONE LINE!)
        try:
            requests.post(
                f"{self.ragboost_url}/evict",
                json={"num_tokens": num_tokens},
                timeout=1.0
            )
        except Exception:
            pass  # Silent failure - don't block SGLang


# ============================================================================
# COMPLETE WORKING EXAMPLE
# ============================================================================

def complete_example():
    """Complete working example with server interaction."""
    
    print("=" * 80)
    print("RAGBoost HTTP Server Integration Example")
    print("=" * 80)
    print()
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8765/health", timeout=2.0)
        health = response.json()
        print(f"✓ Server is running: {health}")
    except Exception as e:
        print(f"✗ Server is not running: {e}")
        print("\nPlease start the server first:")
        print("  python -m ragboost.server.http_server --port 8765")
        return
    
    print()
    
    # Initialize index if not already initialized
    if health.get("status") != "ready":
        print("Index not initialized, building...")
        initialize_index()
    else:
        print("✓ Index already initialized")
    
    print()
    
    # Create client
    print("Creating RAGBoost client...")
    client = RAGBoostIndexClient("http://localhost:8765")
    
    if not client.is_ready():
        print("✗ Index not ready")
        return
    
    print("✓ Client connected")
    print()
    
    # Get initial stats
    print("--- Initial Stats ---")
    stats = client.get_stats()
    if stats:
        evict_stats = stats['eviction_stats']
        print(f"Total nodes: {evict_stats['total_nodes']}")
        print(f"Total tokens: {evict_stats['total_tokens']}")
    print()
    
    # Simulate eviction (this is what SGLang would do)
    print("--- Simulating Cache Eviction ---")
    num_tokens_to_evict = 1024
    
    print(f"1. SGLang evicts {num_tokens_to_evict} tokens from tree_cache")
    # (In real SGLang: self.tree_cache.evict(num_tokens_to_evict))
    
    print(f"2. Sync with RAGBoost index...")
    result = client.evict(num_tokens_to_evict)
    
    if result and result.get("status") == "success":
        print(f"   ✓ Evicted {result['tokens_evicted']} tokens")
        print(f"   ✓ Removed {len(result['evicted_node_ids'])} nodes")
        print(f"   ✓ Remaining: {result['tokens_remaining']} tokens in {result['nodes_remaining']} nodes")
    else:
        print(f"   ✗ Eviction failed: {result}")
    
    print()
    
    # Get final stats
    print("--- Final Stats ---")
    stats = client.get_stats()
    if stats:
        evict_stats = stats['eviction_stats']
        print(f"Total nodes: {evict_stats['total_nodes']}")
        print(f"Total tokens: {evict_stats['total_tokens']}")
    
    print()
    print("=" * 80)
    print("✓ Example complete!")
    print()
    print("To integrate with SGLang:")
    print("1. Add: from ragboost.server.http_client import RAGBoostIndexClient")
    print("2. In __init__: self.ragboost_client = RAGBoostIndexClient(...)")
    print("3. In evict: self.ragboost_client.evict(num_tokens)")
    print("=" * 80)


# ============================================================================
# WHAT TO ADD TO SGLANG - COPY THIS
# ============================================================================

"""
╔══════════════════════════════════════════════════════════════════════════╗
║  COPY THIS INTO YOUR SGLANG SCHEDULER CODE                              ║
╚══════════════════════════════════════════════════════════════════════════╝

At the top of your file:
────────────────────────────────────────────────────────────────────────────
from ragboost.server.http_client import RAGBoostIndexClient

In your __init__ method:
────────────────────────────────────────────────────────────────────────────
try:
    self.ragboost_client = RAGBoostIndexClient("http://localhost:8765", timeout=1.0)
    if not self.ragboost_client.is_ready():
        self.ragboost_client = None
except:
    self.ragboost_client = None

In your eviction method (wherever you call tree_cache.evict):
────────────────────────────────────────────────────────────────────────────
self.tree_cache.evict(num_tokens)

# Add this line:
if self.ragboost_client:
    self.ragboost_client.evict(num_tokens)

That's it! Just 3 simple additions.
"""


if __name__ == "__main__":
    complete_example()
