"""
Test Multi-Turn Conversation Context Deduplication

Tests the deduplication feature where subsequent turns in a conversation
have overlapping documents removed and replaced with reference hints.

Example scenario:
- req-a (turn 1): [4, 3, 1]
- req-b (turn 2, continues req-a): [4, 3, 2]
- Expected: req-b's context deduplicated to [2] with hints for [4, 3]
"""

import requests
import json

INDEX_SERVER = "http://localhost:8765"


def test_basic_deduplication():
    """Test basic two-turn deduplication."""
    print("\n" + "="*70)
    print("TEST: Basic Multi-Turn Deduplication")
    print("="*70)
    
    # Reset
    requests.post(f"{INDEX_SERVER}/reset")
    
    # =========================================================================
    # Turn 1: Initial request with documents [4, 3, 1]
    # =========================================================================
    print("\n1️⃣ Turn 1: req-a with documents [4, 3, 1]")
    
    turn1_contexts = [[4, 3, 1]]
    
    turn1_response = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": turn1_contexts,
            "incremental": False,
            "deduplicate": True,
            "parent_request_ids": [None],  # No parent for turn 1
        }
    )
    
    assert turn1_response.status_code == 200, f"Turn 1 failed: {turn1_response.text}"
    turn1_result = turn1_response.json()
    
    req_a_id = turn1_result["request_ids"][0]
    print(f"   Request ID: {req_a_id}")
    print(f"   Documents: {turn1_contexts[0]}")
    
    # Turn 1 should have no deduplication (it's the first turn)
    if "deduplication" in turn1_result:
        dedup = turn1_result["deduplication"]["results"][0]
        print(f"   Deduplicated docs: {dedup['deduplicated_docs']}")
        print(f"   Overlapping docs: {dedup['overlapping_docs']}")
        assert dedup["overlapping_docs"] == [], "Turn 1 should have no overlapping docs"
        assert dedup["deduplicated_docs"] == turn1_contexts[0], "Turn 1 should keep all docs"
    
    # =========================================================================
    # Turn 2: Continuation with documents [4, 3, 2]
    # Documents [4, 3] overlap with turn 1, should be deduplicated
    # =========================================================================
    print("\n2️⃣ Turn 2: req-b with documents [4, 3, 2] (continues req-a)")
    
    turn2_contexts = [[4, 3, 2]]
    
    turn2_response = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": turn2_contexts,
            "incremental": True,
            "deduplicate": True,
            "parent_request_ids": [req_a_id],  # Continues from turn 1
        }
    )
    
    assert turn2_response.status_code == 200, f"Turn 2 failed: {turn2_response.text}"
    turn2_result = turn2_response.json()
    
    req_b_id = turn2_result["request_ids"][0]
    print(f"   Request ID: {req_b_id}")
    print(f"   Original documents: {turn2_contexts[0]}")
    
    # Check deduplication results
    assert "deduplication" in turn2_result, "Turn 2 should have deduplication results"
    dedup = turn2_result["deduplication"]["results"][0]
    
    print(f"\n   📋 Deduplication Results:")
    print(f"      Original docs:     {dedup['original_docs']}")
    print(f"      Overlapping docs:  {dedup['overlapping_docs']}")
    print(f"      New docs:          {dedup['new_docs']}")
    print(f"      Deduplicated docs: {dedup['deduplicated_docs']}")
    print(f"      Reference hints:   {dedup['reference_hints']}")
    
    # Verify deduplication
    assert set(dedup['overlapping_docs']) == {4, 3}, f"Expected [4, 3] to overlap, got {dedup['overlapping_docs']}"
    assert dedup['new_docs'] == [2], f"Expected new docs [2], got {dedup['new_docs']}"
    assert dedup['deduplicated_docs'] == [2], f"Expected deduplicated [2], got {dedup['deduplicated_docs']}"
    assert len(dedup['reference_hints']) == 2, f"Expected 2 hints, got {len(dedup['reference_hints'])}"
    
    print("\n" + "="*70)
    print("✅ TEST PASSED: Basic Multi-Turn Deduplication")
    print("="*70)


def test_three_turn_conversation():
    """Test deduplication across 3 turns."""
    print("\n" + "="*70)
    print("TEST: Three-Turn Conversation")
    print("="*70)
    
    # Reset
    requests.post(f"{INDEX_SERVER}/reset")
    
    # Turn 1: [10, 20, 30]
    print("\n1️⃣ Turn 1: [10, 20, 30]")
    t1 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": [[10, 20, 30]], "incremental": False, "deduplicate": True}
    ).json()
    req_1 = t1["request_ids"][0]
    print(f"   Request: {req_1}")
    
    # Turn 2: [10, 20, 40] - [10, 20] overlap with turn 1
    print("\n2️⃣ Turn 2: [10, 20, 40] (continues turn 1)")
    t2 = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": [[10, 20, 40]], 
            "incremental": True, 
            "deduplicate": True,
            "parent_request_ids": [req_1]
        }
    ).json()
    req_2 = t2["request_ids"][0]
    dedup_2 = t2["deduplication"]["results"][0]
    
    print(f"   Request: {req_2}")
    print(f"   Overlapping: {dedup_2['overlapping_docs']}")
    print(f"   New: {dedup_2['new_docs']}")
    
    assert set(dedup_2['overlapping_docs']) == {10, 20}
    assert dedup_2['new_docs'] == [40]
    
    # Turn 3: [10, 30, 50] - [10] overlaps with t1, [30] overlaps with t1, [50] is new
    # Note: All overlaps should be from the conversation chain (t1 + t2)
    print("\n3️⃣ Turn 3: [10, 30, 50] (continues turn 2)")
    t3 = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": [[10, 30, 50]], 
            "incremental": True, 
            "deduplicate": True,
            "parent_request_ids": [req_2]
        }
    ).json()
    dedup_3 = t3["deduplication"]["results"][0]
    
    print(f"   Overlapping: {dedup_3['overlapping_docs']}")
    print(f"   New: {dedup_3['new_docs']}")
    
    # [10, 30] were in turn 1, and turn 3's parent is turn 2, 
    # which has parent turn 1, so the chain includes all docs
    assert set(dedup_3['overlapping_docs']) == {10, 30}, f"Expected [10, 30] overlap, got {dedup_3['overlapping_docs']}"
    assert dedup_3['new_docs'] == [50], f"Expected [50] new, got {dedup_3['new_docs']}"
    
    print("\n" + "="*70)
    print("✅ TEST PASSED: Three-Turn Conversation")
    print("="*70)


def test_multiple_parallel_conversations():
    """Test deduplication with multiple parallel conversations."""
    print("\n" + "="*70)
    print("TEST: Multiple Parallel Conversations")
    print("="*70)
    
    # Reset
    requests.post(f"{INDEX_SERVER}/reset")
    
    # Conversation A - Turn 1
    print("\n1️⃣ Conversation A - Turn 1: [100, 101, 102]")
    ca_t1 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": [[100, 101, 102]], "incremental": False, "deduplicate": True}
    ).json()
    ca_req_1 = ca_t1["request_ids"][0]
    
    # Conversation B - Turn 1 (different conversation, same batch)
    print("   Conversation B - Turn 1: [200, 201, 202]")
    cb_t1 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": [[200, 201, 202]], "incremental": True, "deduplicate": True}
    ).json()
    cb_req_1 = cb_t1["request_ids"][0]
    
    # Conversation A - Turn 2: [100, 101, 103] - overlaps with conv A turn 1
    print("\n2️⃣ Conversation A - Turn 2: [100, 101, 103]")
    ca_t2 = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": [[100, 101, 103]], 
            "incremental": True, 
            "deduplicate": True,
            "parent_request_ids": [ca_req_1]  # Points to conversation A's turn 1
        }
    ).json()
    ca_dedup_2 = ca_t2["deduplication"]["results"][0]
    
    print(f"   Overlapping: {ca_dedup_2['overlapping_docs']}")
    print(f"   New: {ca_dedup_2['new_docs']}")
    
    assert set(ca_dedup_2['overlapping_docs']) == {100, 101}
    assert ca_dedup_2['new_docs'] == [103]
    
    # Conversation B - Turn 2: [200, 201, 203] - overlaps with conv B turn 1 only
    print("\n   Conversation B - Turn 2: [200, 201, 203]")
    cb_t2 = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": [[200, 201, 203]], 
            "incremental": True, 
            "deduplicate": True,
            "parent_request_ids": [cb_req_1]  # Points to conversation B's turn 1
        }
    ).json()
    cb_dedup_2 = cb_t2["deduplication"]["results"][0]
    
    print(f"   Overlapping: {cb_dedup_2['overlapping_docs']}")
    print(f"   New: {cb_dedup_2['new_docs']}")
    
    assert set(cb_dedup_2['overlapping_docs']) == {200, 201}
    assert cb_dedup_2['new_docs'] == [203]
    
    print("\n" + "="*70)
    print("✅ TEST PASSED: Multiple Parallel Conversations")
    print("="*70)


def test_no_overlap():
    """Test when there's no overlap between turns."""
    print("\n" + "="*70)
    print("TEST: No Overlap Between Turns")
    print("="*70)
    
    # Reset
    requests.post(f"{INDEX_SERVER}/reset")
    
    # Turn 1: [1, 2, 3]
    t1 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": [[1, 2, 3]], "incremental": False, "deduplicate": True}
    ).json()
    req_1 = t1["request_ids"][0]
    
    # Turn 2: [4, 5, 6] - completely different docs
    t2 = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": [[4, 5, 6]], 
            "incremental": True, 
            "deduplicate": True,
            "parent_request_ids": [req_1]
        }
    ).json()
    dedup_2 = t2["deduplication"]["results"][0]
    
    print(f"   Turn 1 docs: [1, 2, 3]")
    print(f"   Turn 2 docs: [4, 5, 6]")
    print(f"   Overlapping: {dedup_2['overlapping_docs']}")
    print(f"   New: {dedup_2['new_docs']}")
    
    assert dedup_2['overlapping_docs'] == [], "Expected no overlap"
    assert dedup_2['new_docs'] == [4, 5, 6], "All docs should be new"
    assert dedup_2['reference_hints'] == [], "No hints when no overlap"
    
    print("\n" + "="*70)
    print("✅ TEST PASSED: No Overlap Between Turns")
    print("="*70)


def test_complete_overlap():
    """Test when turn 2 is completely overlapping with turn 1."""
    print("\n" + "="*70)
    print("TEST: Complete Overlap (Same Documents)")
    print("="*70)
    
    # Reset
    requests.post(f"{INDEX_SERVER}/reset")
    
    # Turn 1: [1, 2, 3]
    t1 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": [[1, 2, 3]], "incremental": False, "deduplicate": True}
    ).json()
    req_1 = t1["request_ids"][0]
    
    # Turn 2: [1, 2, 3] - exact same docs
    t2 = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": [[1, 2, 3]], 
            "incremental": True, 
            "deduplicate": True,
            "parent_request_ids": [req_1]
        }
    ).json()
    dedup_2 = t2["deduplication"]["results"][0]
    
    print(f"   Turn 1 docs: [1, 2, 3]")
    print(f"   Turn 2 docs: [1, 2, 3]")
    print(f"   Overlapping: {dedup_2['overlapping_docs']}")
    print(f"   New: {dedup_2['new_docs']}")
    print(f"   Deduplicated: {dedup_2['deduplicated_docs']}")
    
    assert set(dedup_2['overlapping_docs']) == {1, 2, 3}, "All docs should overlap"
    assert dedup_2['new_docs'] == [], "No new docs"
    assert dedup_2['deduplicated_docs'] == [], "Deduplicated should be empty"
    assert len(dedup_2['reference_hints']) == 3, "Should have 3 hints"
    
    print("\n" + "="*70)
    print("✅ TEST PASSED: Complete Overlap")
    print("="*70)


def test_batch_deduplication():
    """Test deduplication with multiple contexts in one request."""
    print("\n" + "="*70)
    print("TEST: Batch Deduplication")
    print("="*70)
    
    # Reset
    requests.post(f"{INDEX_SERVER}/reset")
    
    # Turn 1: Two contexts
    # Context 0: [10, 20, 30]
    # Context 1: [40, 50, 60]
    print("\n1️⃣ Turn 1: Two independent contexts")
    t1 = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": [[10, 20, 30], [40, 50, 60]], 
            "incremental": False, 
            "deduplicate": True,
            "parent_request_ids": [None, None]
        }
    ).json()
    req_0 = t1["request_ids"][0]
    req_1 = t1["request_ids"][1]
    
    print(f"   Context 0: [10, 20, 30] -> {req_0}")
    print(f"   Context 1: [40, 50, 60] -> {req_1}")
    
    # Turn 2: Two contexts continuing their respective parents
    # Context 0 continues from req_0: [10, 20, 35] - [10, 20] overlap
    # Context 1 continues from req_1: [40, 50, 65] - [40, 50] overlap
    print("\n2️⃣ Turn 2: Continue both conversations")
    t2 = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": [[10, 20, 35], [40, 50, 65]], 
            "incremental": True, 
            "deduplicate": True,
            "parent_request_ids": [req_0, req_1]
        }
    ).json()
    
    dedup_0 = t2["deduplication"]["results"][0]
    dedup_1 = t2["deduplication"]["results"][1]
    
    print(f"\n   Context 0 deduplication:")
    print(f"      Overlapping: {dedup_0['overlapping_docs']}")
    print(f"      New: {dedup_0['new_docs']}")
    
    print(f"\n   Context 1 deduplication:")
    print(f"      Overlapping: {dedup_1['overlapping_docs']}")
    print(f"      New: {dedup_1['new_docs']}")
    
    assert set(dedup_0['overlapping_docs']) == {10, 20}
    assert dedup_0['new_docs'] == [35]
    
    assert set(dedup_1['overlapping_docs']) == {40, 50}
    assert dedup_1['new_docs'] == [65]
    
    print("\n" + "="*70)
    print("✅ TEST PASSED: Batch Deduplication")
    print("="*70)


def test_custom_hint_template():
    """Test custom reference hint templates."""
    print("\n" + "="*70)
    print("TEST: Custom Hint Template")
    print("="*70)
    
    # Reset
    requests.post(f"{INDEX_SERVER}/reset")
    
    # Turn 1
    t1 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": [[1, 2, 3]], "incremental": False, "deduplicate": True}
    ).json()
    req_1 = t1["request_ids"][0]
    
    # Turn 2 with custom hint template
    custom_template = "📌 Document {doc_id} was already discussed."
    
    t2 = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": [[1, 2, 4]], 
            "incremental": True, 
            "deduplicate": True,
            "parent_request_ids": [req_1],
            "hint_template": custom_template
        }
    ).json()
    dedup_2 = t2["deduplication"]["results"][0]
    
    print(f"   Custom template: {custom_template}")
    print(f"   Generated hints:")
    for hint in dedup_2['reference_hints']:
        print(f"      {hint}")
    
    # Check that hints use the custom template
    for hint in dedup_2['reference_hints']:
        assert "📌" in hint, f"Expected emoji in hint: {hint}"
        assert "was already discussed" in hint, f"Expected custom text in hint: {hint}"
    
    print("\n" + "="*70)
    print("✅ TEST PASSED: Custom Hint Template")
    print("="*70)


def print_scenario_example():
    """Print the concrete scenario from requirements."""
    print("\n" + "="*70)
    print("SCENARIO: Multi-Turn Deduplication Example")
    print("="*70)
    
    # Reset
    requests.post(f"{INDEX_SERVER}/reset")
    
    print("""
📋 Scenario:
   - req-a (turn 1): Documents [4, 3, 1]
   - req-b (turn 2): Documents [4, 3, 2] (continuation of req-a)
   
   Expected:
   - Detect overlap: [4, 3] already in req-a
   - Deduplicate: req-b sends only [2]
   - Add hints: "Refer to Doc 4...", "Refer to Doc 3..."
    """)
    
    # Turn 1: req-a with [4, 3, 1]
    t1 = requests.post(
        f"{INDEX_SERVER}/build",
        json={"contexts": [[4, 3, 1]], "incremental": False, "deduplicate": True}
    ).json()
    req_a = t1["request_ids"][0]
    
    print(f"✅ Turn 1 (req-a): Created with ID {req_a}")
    print(f"   Documents: [4, 3, 1]")
    
    # Turn 2: req-b with [4, 3, 2], continues req-a
    t2 = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": [[4, 3, 2]], 
            "incremental": True, 
            "deduplicate": True,
            "parent_request_ids": [req_a]
        }
    ).json()
    req_b = t2["request_ids"][0]
    dedup = t2["deduplication"]["results"][0]
    
    print(f"\n✅ Turn 2 (req-b): Created with ID {req_b}")
    print(f"   Original documents: [4, 3, 2]")
    print(f"\n   🔄 Deduplication Result:")
    print(f"      Overlapping (removed): {dedup['overlapping_docs']}")
    print(f"      New docs (to send):    {dedup['new_docs']}")
    print(f"\n   📝 Reference Hints:")
    for hint in dedup['reference_hints']:
        print(f"      - {hint}")
    
    print(f"\n   📤 Final context for req-b:")
    print(f"      {dedup['reference_hints']}")
    print(f"      + [Content of Doc 2]")
    
    print("\n" + "="*70)


def test_deduplicate_endpoint():
    """
    Test the standalone /deduplicate endpoint.
    
    This endpoint is designed for Turn 2+ in multi-turn conversations.
    It only performs deduplication - no index build/search operations.
    
    Flow:
    1. Turn 1: Call /build (builds index, registers request)
    2. Turn 2+: Call /deduplicate (just deduplicates, no index ops)
    """
    print("\n" + "="*70)
    print("TEST: Standalone /deduplicate Endpoint")
    print("="*70)
    
    # Reset
    requests.post(f"{INDEX_SERVER}/reset")
    
    # =========================================================================
    # Turn 1: Use /build to create initial index
    # =========================================================================
    print("\n1️⃣ Turn 1: Use /build for initial request [4, 3, 1]")
    
    turn1_response = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": [[4, 3, 1]],
            "incremental": False,
            "deduplicate": True,  # Register in conversation tracker
            "parent_request_ids": [None],
        }
    )
    
    assert turn1_response.status_code == 200, f"Turn 1 failed: {turn1_response.text}"
    turn1_result = turn1_response.json()
    req_a_id = turn1_result["request_ids"][0]
    
    print(f"   Request ID: {req_a_id}")
    print(f"   Mode: {turn1_result['mode']} (index build)")
    
    if "deduplication" in turn1_result:
        print(f"   Is new conversation: {turn1_result['deduplication']['results'][0].get('is_new_conversation', True)}")
    
    # =========================================================================
    # Turn 2: Use /deduplicate (no index operations!)
    # =========================================================================
    print("\n2️⃣ Turn 2: Use /deduplicate for [4, 3, 2] (no index ops)")
    
    turn2_response = requests.post(
        f"{INDEX_SERVER}/deduplicate",
        json={
            "contexts": [[4, 3, 2]],
            "parent_request_ids": [req_a_id],
        }
    )
    
    assert turn2_response.status_code == 200, f"Turn 2 failed: {turn2_response.text}"
    turn2_result = turn2_response.json()
    
    print(f"   Status: {turn2_result['status']}")
    print(f"   Message: {turn2_result['message']}")
    
    result = turn2_result["results"][0]
    req_b_id = result["request_id"]
    
    print(f"\n   📋 Deduplication Results for {req_b_id}:")
    print(f"      Parent: {result['parent_request_id']}")
    print(f"      Original docs:     {result['original_docs']}")
    print(f"      Overlapping docs:  {result['overlapping_docs']}")
    print(f"      New docs:          {result['new_docs']}")
    print(f"      Deduplicated docs: {result['deduplicated_docs']}")
    print(f"      Is new conversation: {result['is_new_conversation']}")
    print(f"      Reference hints:   {result['reference_hints']}")
    
    # Verify deduplication
    assert set(result['overlapping_docs']) == {4, 3}, f"Expected [4, 3] to overlap"
    assert result['new_docs'] == [2], f"Expected new docs [2]"
    assert result['is_new_conversation'] == False, "Should not be new conversation"
    assert result['parent_request_id'] == req_a_id, "Parent should match"
    
    # Check summary
    summary = turn2_result["summary"]
    print(f"\n   📊 Summary:")
    print(f"      Total contexts: {summary['total_contexts']}")
    print(f"      New conversations: {summary['new_conversations']}")
    print(f"      Continued conversations: {summary['continued_conversations']}")
    print(f"      Total docs deduplicated: {summary['total_docs_deduplicated']}")
    
    assert summary['continued_conversations'] == 1
    assert summary['total_docs_deduplicated'] == 2
    
    # =========================================================================
    # Turn 3: Use /deduplicate again
    # =========================================================================
    print("\n3️⃣ Turn 3: Use /deduplicate for [4, 2, 5] (continues turn 2)")
    
    turn3_response = requests.post(
        f"{INDEX_SERVER}/deduplicate",
        json={
            "contexts": [[4, 2, 5]],
            "parent_request_ids": [req_b_id],
        }
    )
    
    assert turn3_response.status_code == 200
    turn3_result = turn3_response.json()
    result3 = turn3_result["results"][0]
    
    print(f"   Overlapping docs:  {result3['overlapping_docs']}")
    print(f"   New docs:          {result3['new_docs']}")
    
    # [4] was in turn 1, [2] was in turn 2, both should be deduped
    assert set(result3['overlapping_docs']) == {4, 2}, f"Expected [4, 2] overlap"
    assert result3['new_docs'] == [5], f"Expected [5] new"
    
    print("\n" + "="*70)
    print("✅ TEST PASSED: Standalone /deduplicate Endpoint")
    print("="*70)


def test_deduplicate_batch():
    """Test batch deduplication with /deduplicate endpoint."""
    print("\n" + "="*70)
    print("TEST: Batch /deduplicate Endpoint")
    print("="*70)
    
    # Reset
    requests.post(f"{INDEX_SERVER}/reset")
    
    # Turn 1: Create two separate conversations
    print("\n1️⃣ Turn 1: Two conversations via /build")
    
    t1 = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": [[1, 2, 3], [10, 20, 30]],  # Two conversations
            "incremental": False,
            "deduplicate": True,
            "parent_request_ids": [None, None],  # Both are turn 1
        }
    ).json()
    
    req_a1 = t1["request_ids"][0]  # Conversation A turn 1
    req_b1 = t1["request_ids"][1]  # Conversation B turn 1
    
    print(f"   Conv A Turn 1: {req_a1} -> [1, 2, 3]")
    print(f"   Conv B Turn 1: {req_b1} -> [10, 20, 30]")
    
    # Turn 2: Batch deduplicate both conversations
    print("\n2️⃣ Turn 2: Batch /deduplicate for both conversations")
    
    t2 = requests.post(
        f"{INDEX_SERVER}/deduplicate",
        json={
            "contexts": [[1, 2, 4], [10, 20, 40]],
            "parent_request_ids": [req_a1, req_b1],
        }
    ).json()
    
    print(f"   Conv A Turn 2: {t2['results'][0]['overlapping_docs']} overlap, {t2['results'][0]['new_docs']} new")
    print(f"   Conv B Turn 2: {t2['results'][1]['overlapping_docs']} overlap, {t2['results'][1]['new_docs']} new")
    
    assert set(t2['results'][0]['overlapping_docs']) == {1, 2}
    assert t2['results'][0]['new_docs'] == [4]
    assert set(t2['results'][1]['overlapping_docs']) == {10, 20}
    assert t2['results'][1]['new_docs'] == [40]
    
    print("\n" + "="*70)
    print("✅ TEST PASSED: Batch /deduplicate Endpoint")
    print("="*70)


def test_deduplicate_mixed_batch():
    """Test batch where some are new conversations and some continue existing."""
    print("\n" + "="*70)
    print("TEST: Mixed Batch /deduplicate (new + continued)")
    print("="*70)
    
    # Reset
    requests.post(f"{INDEX_SERVER}/reset")
    
    # Turn 1: One conversation
    print("\n1️⃣ Turn 1: Start one conversation via /build")
    
    t1 = requests.post(
        f"{INDEX_SERVER}/build",
        json={
            "contexts": [[1, 2, 3]],
            "incremental": False,
            "deduplicate": True,
            "parent_request_ids": [None],
        }
    ).json()
    
    req_a1 = t1["request_ids"][0]
    print(f"   Conv A Turn 1: {req_a1} -> [1, 2, 3]")
    
    # Mixed batch: one continues Conv A, one is new Conv B
    print("\n2️⃣ Mixed batch: Continue Conv A + Start new Conv B")
    
    t2 = requests.post(
        f"{INDEX_SERVER}/deduplicate",
        json={
            "contexts": [[1, 2, 4], [100, 200, 300]],
            "parent_request_ids": [req_a1, None],  # First continues A, second is new
        }
    ).json()
    
    conv_a_result = t2['results'][0]
    conv_b_result = t2['results'][1]
    
    print(f"   Conv A Turn 2:")
    print(f"      Is new: {conv_a_result['is_new_conversation']}")
    print(f"      Overlapping: {conv_a_result['overlapping_docs']}")
    print(f"      New: {conv_a_result['new_docs']}")
    
    print(f"   Conv B Turn 1 (new):")
    print(f"      Is new: {conv_b_result['is_new_conversation']}")
    print(f"      Overlapping: {conv_b_result['overlapping_docs']}")
    print(f"      New: {conv_b_result['new_docs']}")
    
    # Verify
    assert conv_a_result['is_new_conversation'] == False
    assert set(conv_a_result['overlapping_docs']) == {1, 2}
    assert conv_a_result['new_docs'] == [4]
    
    assert conv_b_result['is_new_conversation'] == True
    assert conv_b_result['overlapping_docs'] == []
    assert conv_b_result['new_docs'] == [100, 200, 300]
    
    # Summary should show 1 new, 1 continued
    assert t2['summary']['new_conversations'] == 1
    assert t2['summary']['continued_conversations'] == 1
    
    print("\n" + "="*70)
    print("✅ TEST PASSED: Mixed Batch /deduplicate")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    tests = {
        "basic": test_basic_deduplication,
        "three_turn": test_three_turn_conversation,
        "parallel": test_multiple_parallel_conversations,
        "no_overlap": test_no_overlap,
        "complete_overlap": test_complete_overlap,
        "batch": test_batch_deduplication,
        "custom_hint": test_custom_hint_template,
        "scenario": print_scenario_example,
        "endpoint": test_deduplicate_endpoint,
        "endpoint_batch": test_deduplicate_batch,
        "endpoint_mixed": test_deduplicate_mixed_batch,
    }
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name in tests:
            tests[test_name]()
        elif test_name == "all":
            for name, test_fn in tests.items():
                try:
                    test_fn()
                except Exception as e:
                    print(f"\n❌ TEST FAILED: {name}")
                    print(f"   Error: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available: {', '.join(tests.keys())}, all")
    else:
        # Run scenario example by default
        print_scenario_example()
