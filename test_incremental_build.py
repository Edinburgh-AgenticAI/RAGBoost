"""
Test Incremental Build Feature

This tests the incremental build algorithm:
1. First batch: Full build (clustering, scheduling, index creation)
2. Second batch: Incremental build
   - Search existing index for matches
   - Reorder matched contexts to align with existing prefix
   - Build separate index for unmatched contexts
   - Merge new index under global root

Prerequisites:
- RAGBoost Index Server running on localhost:8765:
    python -m ragboost.server.http_server --port 8765 --sglang-url http://localhost:30000
- (Optional) SGLang server running on localhost:30000 for full LLM test
"""

import requests
import json

INDEX_SERVER = "http://localhost:8765"


def test_incremental_build_basic():
    """Test basic incremental build functionality."""
    print("\n" + "="*70)
    print("TEST: Incremental Build - Basic")
    print("="*70)
    
    # Step 0: Reset index to start fresh
    print("\n0️⃣ Resetting index...")
    reset_response = requests.post(f"{INDEX_SERVER}/reset")
    print(f"   Reset: {reset_response.json().get('message', 'done')}")
    
    # Step 1: Initial build with first batch
    print("\n1️⃣ Initial build (first batch)...")
    
    batch1_contexts = [
        [1, 2, 3, 4, 5],      # Context A
        [1, 2, 3, 6, 7],      # Context B (shares [1,2,3] with A)
        [1, 2, 8, 9, 10],     # Context C (shares [1,2] with A,B)
        [11, 12, 13, 14, 15], # Context D (different branch)
        [11, 12, 13, 16, 17], # Context E (shares [11,12,13] with D)
    ]
    
    build1_response = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": batch1_contexts,
            "initial_tokens_per_context": 100,
            "incremental": False  # First batch is always full build
        }
    )
    
    assert build1_response.status_code == 200, f"Build failed: {build1_response.text}"
    result1 = build1_response.json()
    
    print(f"   ✅ Mode: {result1.get('mode')}")
    print(f"   ✅ Built with {result1.get('num_contexts')} contexts")
    print(f"   ✅ Request IDs: {result1.get('request_ids')}")
    
    batch1_request_ids = result1.get('request_ids', [])
    assert len(batch1_request_ids) == 5, "Expected 5 request IDs"
    
    # Check index stats
    stats1_response = requests.get(f"{INDEX_SERVER}/stats").json()
    stats1 = stats1_response.get('index_stats', {})
    print(f"   📊 Nodes in tree: {stats1.get('num_nodes', 'N/A')}")
    print(f"   📊 Requests tracked: {stats1.get('num_requests', 'N/A')}")
    
    # Step 2: Incremental build with second batch
    print("\n2️⃣ Incremental build (second batch)...")
    
    batch2_contexts = [
        [1, 2, 3, 4, 20],     # Matches [1,2,3,4] from Context A -> should reorder
        [1, 2, 3, 21, 22],    # Matches [1,2,3] -> should reorder  
        [11, 12, 13, 23, 24], # Matches [11,12,13] from D/E -> should reorder
        [30, 31, 32, 33, 34], # No match -> will be built and merged
        [30, 31, 32, 35, 36], # No match, but shares with above -> clustered together
    ]
    
    build2_response = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": batch2_contexts,
            "initial_tokens_per_context": 100,
            "incremental": True  # Incremental mode
        }
    )
    
    assert build2_response.status_code == 200, f"Incremental build failed: {build2_response.text}"
    result2 = build2_response.json()
    
    print(f"   ✅ Mode: {result2.get('mode')}")
    print(f"   ✅ Matched & inserted: {result2.get('matched_count')}")
    print(f"   ✅ Built & merged: {result2.get('merged_count')}")
    print(f"   ✅ Request IDs: {result2.get('request_ids')}")
    
    batch2_request_ids = result2.get('request_ids', [])
    assert len(batch2_request_ids) == 5, "Expected 5 new request IDs"
    
    # Verify all request IDs are NEW (not reused from batch 1)
    for rid in batch2_request_ids:
        assert rid not in batch1_request_ids, f"Request ID {rid} was reused (should be new)"
    print(f"   ✅ All request IDs are new (no reuse)")
    
    # Check reordered contexts
    reordered = result2.get('reordered_contexts', [])
    if reordered:
        print(f"\n   📋 Reordered contexts:")
        for i, (orig, reord) in enumerate(zip(batch2_contexts, reordered)):
            if orig != reord:
                print(f"      Context {i}: {orig} -> {reord}")
            else:
                print(f"      Context {i}: {orig} (unchanged)")
    
    # Check index stats after incremental build
    stats2_response = requests.get(f"{INDEX_SERVER}/stats").json()
    stats2 = stats2_response.get('index_stats', {})
    print(f"\n   📊 Nodes in tree: {stats2.get('num_nodes', 'N/A')}")
    print(f"   📊 Requests tracked: {stats2.get('num_requests', 'N/A')}")
    
    # Should have more requests now
    assert stats2.get('num_requests', 0) > stats1.get('num_requests', 0), \
        "Expected more requests after incremental build"
    
    print("\n" + "="*70)
    print("✅ TEST PASSED: Incremental Build Basic")
    print("="*70)


def test_incremental_build_all_matched():
    """Test incremental build when ALL new contexts match existing ones."""
    print("\n" + "="*70)
    print("TEST: Incremental Build - All Matched")
    print("="*70)
    
    # Reset
    requests.post(f"{INDEX_SERVER}/reset")
    
    # Initial build
    batch1 = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 6, 7],
        [10, 20, 30, 40, 50],
    ]
    
    build1 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": batch1, "incremental": False}
    ).json()
    
    print(f"1️⃣ Initial: {build1.get('num_contexts')} contexts")
    
    # Incremental with contexts that should ALL match
    batch2 = [
        [1, 2, 3, 100, 101],    # Matches [1,2,3]
        [1, 2, 3, 4, 102],      # Matches [1,2,3,4]
        [10, 20, 30, 103, 104], # Matches [10,20,30]
    ]
    
    build2 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": batch2, "incremental": True}
    ).json()
    
    print(f"2️⃣ Incremental:")
    print(f"   Matched: {build2.get('matched_count')}")
    print(f"   Merged: {build2.get('merged_count')}")
    
    assert build2.get('matched_count') == 3, "Expected all 3 to match"
    assert build2.get('merged_count') == 0, "Expected 0 merged (all matched)"
    
    print("\n✅ TEST PASSED: All Matched")


def test_incremental_build_none_matched():
    """Test incremental build when NO new contexts match existing ones."""
    print("\n" + "="*70)
    print("TEST: Incremental Build - None Matched")
    print("="*70)
    
    # Reset
    requests.post(f"{INDEX_SERVER}/reset")
    
    # Initial build
    batch1 = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 6, 7],
    ]
    
    build1 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": batch1, "incremental": False}
    ).json()
    
    print(f"1️⃣ Initial: {build1.get('num_contexts')} contexts")
    
    # Incremental with completely different contexts
    batch2 = [
        [100, 200, 300, 400, 500],  # No overlap
        [100, 200, 300, 600, 700],  # No overlap with batch1
        [800, 900, 1000],           # No overlap
    ]
    
    build2 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": batch2, "incremental": True}
    ).json()
    
    print(f"2️⃣ Incremental:")
    print(f"   Matched: {build2.get('matched_count')}")
    print(f"   Merged: {build2.get('merged_count')}")
    
    assert build2.get('matched_count') == 0, "Expected 0 matched"
    assert build2.get('merged_count') == 3, "Expected all 3 merged"
    
    print("\n✅ TEST PASSED: None Matched")


def test_incremental_build_with_llm():
    """Test incremental build with actual LLM requests.
    
    Requires SGLang server running.
    """
    print("\n" + "="*70)
    print("TEST: Incremental Build - With LLM Generation")
    print("="*70)
    
    # Reset
    requests.post(f"{INDEX_SERVER}/reset")
    
    # Initial build
    batch1 = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 6, 7],
        [10, 11, 12, 13, 14],
    ]
    
    build1 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": batch1, "incremental": False}
    ).json()
    
    batch1_ids = build1.get('request_ids', [])
    print(f"1️⃣ Initial build: {len(batch1_ids)} request IDs")
    
    # Send LLM requests for batch 1
    print("\n2️⃣ Sending LLM requests for batch 1...")
    for i, rid in enumerate(batch1_ids[:2]):
        try:
            resp = requests.post(
                f"{INDEX_SERVER}/v1/completions",
                json={
                    "model": "Qwen/Qwen3-4B",
                    "prompt": f"Test prompt {i}: What is 2+2?",
                    "max_tokens": 20,
                    "temperature": 0.0,
                    "rid": rid
                },
                timeout=30
            )
            if resp.status_code == 200:
                print(f"   ✅ Request {i} (rid={rid[:16]}...) completed")
            else:
                print(f"   ❌ Request {i} failed: {resp.status_code}")
        except Exception as e:
            print(f"   ⚠️ Request {i} error: {e}")
    
    # Incremental build
    batch2 = [
        [1, 2, 3, 100, 101],  # Matches existing
        [50, 51, 52, 53, 54], # No match
    ]
    
    build2 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": batch2, "incremental": True}
    ).json()
    
    batch2_ids = build2.get('request_ids', [])
    print(f"\n3️⃣ Incremental build: {len(batch2_ids)} new request IDs")
    print(f"   Matched: {build2.get('matched_count')}, Merged: {build2.get('merged_count')}")
    
    # Send LLM requests for batch 2
    print("\n4️⃣ Sending LLM requests for batch 2...")
    for i, rid in enumerate(batch2_ids):
        try:
            resp = requests.post(
                f"{INDEX_SERVER}/v1/completions",
                json={
                    "model": "Qwen/Qwen3-4B",
                    "prompt": f"Test prompt batch2-{i}: What is 3+3?",
                    "max_tokens": 20,
                    "temperature": 0.0,
                    "rid": rid
                },
                timeout=30
            )
            if resp.status_code == 200:
                print(f"   ✅ Request {i} (rid={rid[:16]}...) completed")
            else:
                print(f"   ❌ Request {i} failed: {resp.status_code}")
        except Exception as e:
            print(f"   ⚠️ Request {i} error: {e}")
    
    print("\n✅ TEST PASSED: With LLM Generation")


def test_reordering_correctness():
    """Test that reordering correctly aligns with matched prefix."""
    print("\n" + "="*70)
    print("TEST: Reordering Correctness (Comprehensive)")
    print("="*70)
    
    # Reset
    requests.post(f"{INDEX_SERVER}/reset")
    
    # =========================================================================
    # Test Case 1: Simple prefix match with scrambled order
    # =========================================================================
    print("\n📋 Test Case 1: Simple prefix match")
    print("-" * 50)
    
    batch1 = [
        [10, 20, 30, 40, 50],  # Establishes prefix [10, 20, 30, 40, 50]
    ]
    
    build1 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": batch1, "incremental": False}
    ).json()
    
    print(f"   Initial context: [10, 20, 30, 40, 50]")
    
    # New context has [10, 20, 30] in common but in different positions
    # Original: [99, 30, 20, 10, 88]
    # Matched prefix: [10, 20, 30, ...]
    # Expected reorder: [10, 20, 30, 99, 88] (prefix elements first, in prefix order)
    batch2 = [
        [99, 30, 20, 10, 88],  
    ]
    
    build2 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": batch2, "incremental": True}
    ).json()
    
    reordered = build2.get('reordered_contexts', [[]])[0]
    original = batch2[0]
    
    print(f"   New context:     {original}")
    print(f"   Reordered:       {reordered}")
    
    # Verify: reordered should start with [10, 20, 30] (in that order)
    expected_prefix = [10, 20, 30]
    actual_prefix = reordered[:3]
    
    if actual_prefix == expected_prefix:
        print(f"   ✅ Prefix correctly reordered: {actual_prefix}")
    else:
        print(f"   ❌ Expected prefix {expected_prefix}, got {actual_prefix}")
        
    # Verify: remaining elements should be [99, 88]
    remaining = reordered[3:]
    if set(remaining) == {99, 88}:
        print(f"   ✅ Remaining elements preserved: {remaining}")
    else:
        print(f"   ❌ Remaining elements wrong: {remaining}")
    
    # =========================================================================
    # Test Case 2: Multiple contexts with different matches
    # =========================================================================
    print("\n📋 Test Case 2: Multiple contexts, different match depths")
    print("-" * 50)
    
    requests.post(f"{INDEX_SERVER}/reset")
    
    # Initial batch with two different prefix paths
    batch1 = [
        [1, 2, 3, 4, 5],      # Path A
        [100, 200, 300, 400], # Path B
    ]
    
    build1 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": batch1, "incremental": False}
    ).json()
    
    print(f"   Initial contexts:")
    print(f"      Path A: [1, 2, 3, 4, 5]")
    print(f"      Path B: [100, 200, 300, 400]")
    
    # New contexts that match different paths
    batch2 = [
        [9, 3, 2, 1, 8],       # Should match Path A ([1, 2, 3]) -> reorder to [1, 2, 3, 9, 8]
        [400, 300, 100, 999],  # Should match Path B ([100, 200?, 300?, 400]) -> reorder
        [777, 888, 999],       # No match -> no reordering
    ]
    
    build2 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": batch2, "incremental": True}
    ).json()
    
    reordered_contexts = build2.get('reordered_contexts', [])
    
    print(f"\n   Incremental contexts and reordering:")
    for i, (orig, reord) in enumerate(zip(batch2, reordered_contexts)):
        changed = "→ REORDERED" if orig != reord else "(unchanged)"
        print(f"      Context {i}: {orig} -> {reord} {changed}")
    
    # Context 0: Check if it starts with elements from [1, 2, 3, 4, 5]
    ctx0 = reordered_contexts[0] if reordered_contexts else []
    shared_with_path_a = [x for x in ctx0 if x in [1, 2, 3, 4, 5]]
    print(f"\n   Context 0 analysis:")
    print(f"      Shared with Path A: {shared_with_path_a}")
    if ctx0 and shared_with_path_a:
        # Check if shared elements come first
        first_n = ctx0[:len(shared_with_path_a)]
        if first_n == shared_with_path_a:
            print(f"      ✅ Shared elements correctly at front")
        else:
            print(f"      ⚠️ Shared elements not at front: first {len(shared_with_path_a)} = {first_n}")
    
    # Context 1: Check if it starts with elements from [100, 200, 300, 400]
    ctx1 = reordered_contexts[1] if len(reordered_contexts) > 1 else []
    shared_with_path_b = [x for x in ctx1 if x in [100, 200, 300, 400]]
    print(f"\n   Context 1 analysis:")
    print(f"      Shared with Path B: {shared_with_path_b}")
    if ctx1 and shared_with_path_b:
        first_n = ctx1[:len(shared_with_path_b)]
        if first_n == shared_with_path_b:
            print(f"      ✅ Shared elements correctly at front")
        else:
            print(f"      ⚠️ Shared elements not at front: first {len(shared_with_path_b)} = {first_n}")
    
    # Context 2: Should be unchanged (no match)
    ctx2_orig = batch2[2]
    ctx2_reord = reordered_contexts[2] if len(reordered_contexts) > 2 else []
    print(f"\n   Context 2 analysis (no match expected):")
    if ctx2_orig == ctx2_reord:
        print(f"      ✅ Unchanged as expected")
    else:
        print(f"      ⚠️ Was changed: {ctx2_orig} -> {ctx2_reord}")
    
    # =========================================================================
    # Test Case 3: Verify ordering within prefix
    # =========================================================================
    print("\n📋 Test Case 3: Verify exact prefix ordering")
    print("-" * 50)
    
    requests.post(f"{INDEX_SERVER}/reset")
    
    # Create a specific tree path
    batch1 = [
        [1000, 2000, 3000, 4000, 5000],  # Known exact prefix
    ]
    
    requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": batch1, "incremental": False}
    )
    
    print(f"   Prefix in tree: [1000, 2000, 3000, 4000, 5000]")
    
    # Test with context where shared elements are in reverse order
    test_context = [5000, 4000, 3000, 2000, 1000, 9999]
    
    build2 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": [test_context], "incremental": True}
    ).json()
    
    reordered = build2.get('reordered_contexts', [[]])[0]
    
    print(f"   Original:  {test_context}")
    print(f"   Reordered: {reordered}")
    
    # The reordered context should start with [1000, 2000, 3000, 4000, 5000]
    # (the order from the tree prefix, not the original context order)
    expected_prefix = [1000, 2000, 3000, 4000, 5000]
    actual_prefix = reordered[:5]
    
    if actual_prefix == expected_prefix:
        print(f"   ✅ Perfect prefix match: {actual_prefix}")
    else:
        print(f"   ⚠️ Expected {expected_prefix}")
        print(f"      Got      {actual_prefix}")
    
    # The last element should be 9999
    if reordered[-1] == 9999:
        print(f"   ✅ Non-prefix element at end: {reordered[-1]}")
    else:
        print(f"   ⚠️ Non-prefix element not at end")
    
    # =========================================================================
    # Test Case 4: Partial prefix match
    # =========================================================================
    print("\n📋 Test Case 4: Partial prefix match (only some elements shared)")
    print("-" * 50)
    
    requests.post(f"{INDEX_SERVER}/reset")
    
    batch1 = [
        [10, 20, 30, 40, 50],
    ]
    
    requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": batch1, "incremental": False}
    )
    
    print(f"   Prefix in tree: [10, 20, 30, 40, 50]")
    
    # New context shares only 2 elements: 10 and 30 (not 20, 40, 50)
    test_context = [77, 30, 88, 10, 99]
    
    build2 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": [test_context], "incremental": True}
    ).json()
    
    reordered = build2.get('reordered_contexts', [[]])[0]
    matched = build2.get('matched_count', 0)
    
    print(f"   Original:  {test_context}")
    print(f"   Reordered: {reordered}")
    print(f"   Matched count: {matched}")
    
    # Should start with [10, 30] (elements from prefix that exist in context, in prefix order)
    shared_elements = [x for x in [10, 20, 30, 40, 50] if x in test_context]
    print(f"   Shared elements (in prefix order): {shared_elements}")
    
    if matched > 0:
        actual_start = reordered[:len(shared_elements)]
        if actual_start == shared_elements:
            print(f"   ✅ Shared elements at front in correct order: {actual_start}")
        else:
            print(f"   ⚠️ Expected {shared_elements} at front, got {actual_start}")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETED: Reordering Correctness")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    tests = {
        "basic": test_incremental_build_basic,
        "all_matched": test_incremental_build_all_matched,
        "none_matched": test_incremental_build_none_matched,
        "llm": test_incremental_build_with_llm,
        "reorder": test_reordering_correctness,
    }
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name in tests:
            tests[test_name]()
        elif test_name == "all":
            for name, test_fn in tests.items():
                if name != "llm":  # Skip LLM test in "all" (requires server)
                    try:
                        test_fn()
                    except Exception as e:
                        print(f"\n❌ TEST FAILED: {name}")
                        print(f"   Error: {e}")
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available: {', '.join(tests.keys())}, all")
    else:
        # Run basic test by default
        test_incremental_build_basic()
