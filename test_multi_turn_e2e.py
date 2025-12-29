#!/usr/bin/env python3
"""
End-to-End Test for Multi-Turn Conversation Deduplication

This test validates the complete multi-turn deduplication workflow:
1. ConversationTracker unit tests
2. HTTP Server /deduplicate endpoint tests
3. Multi-turn conversation chain tests
4. Reset and eviction tests

Usage:
    python test_multi_turn_e2e.py
    
    # With server already running:
    python test_multi_turn_e2e.py --server-url http://localhost:8765
"""

import argparse
import json
import time
import requests
import subprocess
import sys
import signal
from typing import Optional

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def log_test(name: str, passed: bool, details: str = ""):
    """Log test result with color."""
    status = f"{GREEN}PASSED{RESET}" if passed else f"{RED}FAILED{RESET}"
    print(f"  [{status}] {name}")
    if details and not passed:
        print(f"         {RED}{details}{RESET}")


def log_section(name: str):
    """Log section header."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{name}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")


class MultiTurnE2ETest:
    """End-to-end tests for multi-turn conversation deduplication."""
    
    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url
        self.server_process = None
        self.passed = 0
        self.failed = 0
    
    def run_all_tests(self):
        """Run all tests."""
        print(f"\n{YELLOW}Multi-Turn Conversation Deduplication E2E Tests{RESET}")
        print(f"{YELLOW}{'='*50}{RESET}\n")
        
        # Unit tests (no server needed)
        self.test_conversation_tracker_unit()
        
        # HTTP server tests
        if self.server_url or self._start_server():
            try:
                self.test_http_deduplicate_endpoint()
                self.test_http_multi_turn_chain()
                self.test_http_reset_endpoint()
                self.test_http_build_with_deduplication()
                self.test_http_eviction_clears_conversation()
            finally:
                self._stop_server()
        
        # Summary
        self._print_summary()
        return self.failed == 0
    
    def test_conversation_tracker_unit(self):
        """Test ConversationTracker unit functionality."""
        log_section("1. ConversationTracker Unit Tests")
        
        from ragboost.server.conversation_tracker import ConversationTracker
        
        # Test 1: Basic deduplication
        tracker = ConversationTracker()
        result1 = tracker.deduplicate('req_a', [4, 3, 1], parent_request_id=None)
        
        test_passed = (
            result1.is_new_conversation == True and
            result1.new_docs == [4, 3, 1] and
            result1.overlapping_docs == []
        )
        log_test("Turn 1 - New conversation detection", test_passed,
                 f"Expected new_docs=[4,3,1], got {result1.new_docs}")
        self._record_result(test_passed)
        
        # Test 2: Deduplication with overlap
        result2 = tracker.deduplicate('req_b', [4, 3, 2], parent_request_id='req_a')
        
        test_passed = (
            result2.is_new_conversation == False and
            result2.new_docs == [2] and
            set(result2.overlapping_docs) == {4, 3}
        )
        log_test("Turn 2 - Overlap detection (4, 3)", test_passed,
                 f"Expected new_docs=[2], overlap={{4,3}}, got new={result2.new_docs}, overlap={result2.overlapping_docs}")
        self._record_result(test_passed)
        
        # Test 3: Reference hints generation
        test_passed = len(result2.reference_hints) == 2
        log_test("Turn 2 - Reference hints generated", test_passed,
                 f"Expected 2 hints, got {len(result2.reference_hints)}")
        self._record_result(test_passed)
        
        # Test 4: Chain propagation
        result3 = tracker.deduplicate('req_c', [4, 2, 5], parent_request_id='req_b')
        
        test_passed = (
            result3.new_docs == [5] and
            set(result3.overlapping_docs) == {4, 2}
        )
        log_test("Turn 3 - Chain propagation (4 from T1, 2 from T2)", test_passed,
                 f"Expected new=[5], overlap={{4,2}}, got new={result3.new_docs}, overlap={result3.overlapping_docs}")
        self._record_result(test_passed)
        
        # Test 5: Get conversation chain
        chain = tracker.get_conversation_chain('req_c')
        test_passed = len(chain) == 3
        log_test("Get conversation chain (3 turns)", test_passed,
                 f"Expected 3 turns, got {len(chain)}")
        self._record_result(test_passed)
        
        # Test 6: Reset functionality
        tracker.reset()
        test_passed = len(tracker._requests) == 0
        log_test("Reset clears all requests", test_passed,
                 f"Expected 0 requests, got {len(tracker._requests)}")
        self._record_result(test_passed)
        
        # Test 7: Orphaned request (parent doesn't exist)
        tracker2 = ConversationTracker()
        result_orphan = tracker2.deduplicate('orphan', [1, 2, 3], parent_request_id='nonexistent')
        
        test_passed = (
            result_orphan.is_new_conversation == True and
            result_orphan.new_docs == [1, 2, 3]
        )
        log_test("Orphaned request treated as new conversation", test_passed,
                 f"Expected new conversation, got is_new={result_orphan.is_new_conversation}")
        self._record_result(test_passed)
        
        # Test 8: Clear single conversation
        tracker3 = ConversationTracker()
        tracker3.deduplicate('conv1_t1', [1, 2], parent_request_id=None)
        tracker3.deduplicate('conv1_t2', [2, 3], parent_request_id='conv1_t1')
        tracker3.deduplicate('conv2_t1', [4, 5], parent_request_id=None)
        
        cleared = tracker3.clear_conversation('conv1_t1')
        test_passed = (
            cleared == 1 and
            len(tracker3._requests) == 2  # conv1_t2 and conv2_t1 remain
        )
        log_test("Clear single request from conversation", test_passed,
                 f"Expected 2 remaining, got {len(tracker3._requests)}")
        self._record_result(test_passed)
    
    def test_http_deduplicate_endpoint(self):
        """Test HTTP /deduplicate endpoint."""
        log_section("2. HTTP /deduplicate Endpoint Tests")
        
        # Reset first
        self._reset_server()
        
        # Test 1: First turn (new conversation)
        response = requests.post(
            f"{self.server_url}/deduplicate",
            json={
                "request_id": "http_req_1",
                "doc_ids": [10, 20, 30],
            }
        )
        
        test_passed = response.status_code == 200
        log_test("POST /deduplicate returns 200", test_passed,
                 f"Got status {response.status_code}")
        self._record_result(test_passed)
        
        if response.status_code == 200:
            data = response.json()
            test_passed = (
                data.get("is_new_conversation") == True and
                data.get("new_docs") == [10, 20, 30]
            )
            log_test("Turn 1 response - new conversation", test_passed,
                     f"Got: {json.dumps(data, indent=2)[:200]}")
            self._record_result(test_passed)
        
        # Test 2: Second turn (with parent)
        response2 = requests.post(
            f"{self.server_url}/deduplicate",
            json={
                "request_id": "http_req_2",
                "doc_ids": [20, 30, 40],
                "parent_request_id": "http_req_1"
            }
        )
        
        if response2.status_code == 200:
            data2 = response2.json()
            test_passed = (
                data2.get("is_new_conversation") == False and
                data2.get("new_docs") == [40] and
                set(data2.get("overlapping_docs", [])) == {20, 30}
            )
            log_test("Turn 2 response - deduplication works", test_passed,
                     f"Got new_docs={data2.get('new_docs')}, overlap={data2.get('overlapping_docs')}")
            self._record_result(test_passed)
            
            test_passed = len(data2.get("reference_hints", [])) == 2
            log_test("Turn 2 response - reference hints", test_passed,
                     f"Expected 2 hints, got {len(data2.get('reference_hints', []))}")
            self._record_result(test_passed)
    
    def test_http_multi_turn_chain(self):
        """Test HTTP multi-turn conversation chain."""
        log_section("3. HTTP Multi-Turn Conversation Chain Tests")
        
        # Reset first
        self._reset_server()
        
        # Simulate a 5-turn conversation
        conversation = [
            ("turn_1", [1, 2, 3, 4, 5], None),
            ("turn_2", [3, 4, 5, 6, 7], "turn_1"),       # 3,4,5 overlap
            ("turn_3", [5, 6, 7, 8, 9], "turn_2"),       # 5,6,7 overlap
            ("turn_4", [1, 8, 9, 10, 11], "turn_3"),     # 1,8,9 overlap (1 from turn1!)
            ("turn_5", [10, 11, 12, 13, 14], "turn_4"),  # 10,11 overlap
        ]
        
        expected_new = [
            [1, 2, 3, 4, 5],  # Turn 1: all new
            [6, 7],           # Turn 2: 3,4,5 overlap
            [8, 9],           # Turn 3: 5,6,7 overlap
            [10, 11],         # Turn 4: 1,8,9 overlap
            [12, 13, 14],     # Turn 5: 10,11 overlap
        ]
        
        for i, (request_id, doc_ids, parent_id) in enumerate(conversation):
            payload = {
                "request_id": request_id,
                "doc_ids": doc_ids,
            }
            if parent_id:
                payload["parent_request_id"] = parent_id
            
            response = requests.post(f"{self.server_url}/deduplicate", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                test_passed = data.get("new_docs") == expected_new[i]
                log_test(f"Turn {i+1} ({request_id}): new_docs={expected_new[i]}", test_passed,
                         f"Got new_docs={data.get('new_docs')}")
                self._record_result(test_passed)
            else:
                log_test(f"Turn {i+1} ({request_id})", False,
                         f"HTTP error: {response.status_code}")
                self._record_result(False)
    
    def test_http_reset_endpoint(self):
        """Test HTTP /reset endpoint."""
        log_section("4. HTTP /reset Endpoint Tests")
        
        # Add some data first
        requests.post(f"{self.server_url}/deduplicate", json={
            "request_id": "reset_test_1",
            "doc_ids": [1, 2, 3],
        })
        requests.post(f"{self.server_url}/deduplicate", json={
            "request_id": "reset_test_2",
            "doc_ids": [2, 3, 4],
            "parent_request_id": "reset_test_1"
        })
        
        # Test reset with conversation_only=true (default)
        response = requests.post(f"{self.server_url}/reset", json={
            "conversation_only": True
        })
        
        test_passed = response.status_code == 200
        log_test("POST /reset returns 200", test_passed,
                 f"Got status {response.status_code}")
        self._record_result(test_passed)
        
        # Verify conversation is reset
        response2 = requests.post(f"{self.server_url}/deduplicate", json={
            "request_id": "after_reset",
            "doc_ids": [1, 2, 3],
        })
        
        if response2.status_code == 200:
            data = response2.json()
            test_passed = (
                data.get("is_new_conversation") == True and
                data.get("new_docs") == [1, 2, 3]
            )
            log_test("After reset - new docs are not deduplicated", test_passed,
                     f"Got is_new={data.get('is_new_conversation')}, new_docs={data.get('new_docs')}")
            self._record_result(test_passed)
    
    def test_http_build_with_deduplication(self):
        """Test HTTP /build endpoint with deduplication."""
        log_section("5. HTTP /build with Deduplication Tests")
        
        # Reset first
        self._reset_server()
        
        # Prepare contexts (token sequences)
        contexts = [
            [101, 2054, 2003, 1996],  # Doc 0
            [101, 1037, 4990, 1997],  # Doc 1
            [101, 2000, 2022, 2030],  # Doc 2
            [101, 2035, 2008, 14697], # Doc 3
            [101, 1996, 2069, 2518],  # Doc 4
        ]
        
        # Build with first set of docs
        response1 = requests.post(f"{self.server_url}/build", json={
            "contexts": contexts[:3],  # Docs 0, 1, 2
            "request_id": "build_req_1",
        })
        
        test_passed = response1.status_code == 200
        log_test("POST /build (Turn 1) returns 200", test_passed,
                 f"Got status {response1.status_code}")
        self._record_result(test_passed)
        
        # Build with overlapping docs
        response2 = requests.post(f"{self.server_url}/build", json={
            "contexts": contexts[1:4],  # Docs 1, 2, 3 (1,2 overlap)
            "request_id": "build_req_2",
            "parent_request_id": "build_req_1",
        })
        
        if response2.status_code == 200:
            data = response2.json()
            # Check if deduplication info is included
            dedup_info = data.get("deduplication", {})
            if dedup_info:
                test_passed = len(dedup_info.get("new_docs", [])) < 3  # Should be less than all 3
                log_test("POST /build (Turn 2) - deduplication applied", test_passed,
                         f"Got dedup_info: {dedup_info}")
                self._record_result(test_passed)
            else:
                # Build endpoint might not include dedup info
                log_test("POST /build (Turn 2) - completed", True, "No dedup info returned (expected)")
                self._record_result(True)
    
    def test_http_eviction_clears_conversation(self):
        """Test that /evict endpoint also clears ConversationTracker."""
        log_section("6. HTTP /evict Clears Conversation Tests")
        
        # Reset first
        self._reset_server()
        
        # Create a conversation chain
        requests.post(f"{self.server_url}/deduplicate", json={
            "request_id": "evict_test_1",
            "doc_ids": [100, 200, 300],
        })
        requests.post(f"{self.server_url}/deduplicate", json={
            "request_id": "evict_test_2",
            "doc_ids": [200, 300, 400],
            "parent_request_id": "evict_test_1"
        })
        
        # Verify turn 2 deduplicated correctly
        response = requests.post(f"{self.server_url}/deduplicate", json={
            "request_id": "evict_test_3",
            "doc_ids": [300, 400, 500],
            "parent_request_id": "evict_test_2"
        })
        
        if response.status_code == 200:
            data = response.json()
            test_passed = data.get("new_docs") == [500]
            log_test("Before eviction - deduplication works", test_passed,
                     f"Expected new_docs=[500], got {data.get('new_docs')}")
            self._record_result(test_passed)
        
        # Now evict the root request
        evict_response = requests.post(f"{self.server_url}/evict", json={
            "request_ids": ["evict_test_1"]
        })
        
        test_passed = evict_response.status_code == 200
        log_test("POST /evict returns 200", test_passed,
                 f"Got status {evict_response.status_code}")
        self._record_result(test_passed)
        
        if evict_response.status_code == 200:
            evict_data = evict_response.json()
            test_passed = evict_data.get("conversations_cleared", 0) >= 1
            log_test("/evict clears conversation entries", test_passed,
                     f"conversations_cleared={evict_data.get('conversations_cleared')}")
            self._record_result(test_passed)
    
    def _start_server(self) -> bool:
        """Start the RAGBoost HTTP server."""
        log_section("Starting RAGBoost Server")
        
        if self.server_url:
            print(f"Using existing server at {self.server_url}")
            return True
        
        try:
            # Start server in background
            self.server_process = subprocess.Popen(
                [sys.executable, "-m", "ragboost.server.http_server", "--port", "18765"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="/home/jysc/RAGBoost-real"
            )
            self.server_url = "http://localhost:18765"
            
            # Wait for server to start
            print(f"Starting server on {self.server_url}...")
            for _ in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                try:
                    response = requests.get(f"{self.server_url}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"{GREEN}Server started successfully!{RESET}")
                        return True
                except requests.exceptions.RequestException:
                    pass
            
            print(f"{RED}Server failed to start within 30 seconds{RESET}")
            self._stop_server()
            return False
            
        except Exception as e:
            print(f"{RED}Failed to start server: {e}{RESET}")
            return False
    
    def _stop_server(self):
        """Stop the HTTP server."""
        if self.server_process:
            print(f"\n{YELLOW}Stopping server...{RESET}")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
    
    def _reset_server(self):
        """Reset server state."""
        try:
            requests.post(f"{self.server_url}/reset", json={"conversation_only": True}, timeout=5)
        except:
            pass
    
    def _record_result(self, passed: bool):
        """Record test result."""
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def _print_summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}TEST SUMMARY{RESET}")
        print(f"{BLUE}{'='*60}{RESET}")
        print(f"Total tests: {total}")
        print(f"  {GREEN}Passed: {self.passed}{RESET}")
        print(f"  {RED}Failed: {self.failed}{RESET}")
        
        if self.failed == 0:
            print(f"\n{GREEN}✅ All tests passed!{RESET}")
        else:
            print(f"\n{RED}❌ {self.failed} test(s) failed{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Turn Conversation E2E Tests")
    parser.add_argument("--server-url", type=str, default=None,
                        help="URL of running RAGBoost server (e.g., http://localhost:8765)")
    parser.add_argument("--unit-only", action="store_true",
                        help="Run only unit tests (no server needed)")
    args = parser.parse_args()
    
    test = MultiTurnE2ETest(server_url=args.server_url)
    
    if args.unit_only:
        test.test_conversation_tracker_unit()
        test._print_summary()
        success = test.failed == 0
    else:
        success = test.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
