"""
RAGBoost Index HTTP Client

Simple client for calling the RAGBoost Index Server from SGLang.
This is what SGLang should use to sync eviction with the remote index.
"""

import logging
from typing import List, Dict, Any, Optional

try:
    import requests
except ImportError:
    raise ImportError(
        "requests library is required for the HTTP client. "
        "Install with: pip install requests"
    )

logger = logging.getLogger(__name__)


class RAGBoostIndexClient:
    """
    Client for RAGBoost Live Index Server.
    
    This is what SGLang should instantiate to communicate with the index server.
    
    Example usage in SGLang:
        # In scheduler initialization:
        self.ragboost_client = RAGBoostIndexClient("http://localhost:8765")
        
        # In eviction code:
        def evict_tokens(self, num_tokens):
            self.tree_cache.evict(num_tokens)
            self.ragboost_client.evict(num_tokens)  # Sync with index
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8765",
        timeout: float = 1.0,
        retry_on_failure: bool = False
    ):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the RAGBoost index server
            timeout: Request timeout in seconds (default 1.0 for low latency)
            retry_on_failure: Whether to retry on network failures
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retry_on_failure = retry_on_failure
        self.session = requests.Session()
    
    def _post(self, endpoint: str, json_data: Dict) -> Optional[Dict]:
        """Make a POST request to the server."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.post(
                url,
                json=json_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            logger.warning(f"RAGBoost index request timed out: {endpoint}")
            return None
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"RAGBoost index request failed: {e}")
            return None
    
    def _get(self, endpoint: str) -> Optional[Dict]:
        """Make a GET request to the server."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            logger.warning(f"RAGBoost index request timed out: {endpoint}")
            return None
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"RAGBoost index request failed: {e}")
            return None
    
    def evict(self, num_tokens: int) -> Optional[Dict[str, Any]]:
        """
        Evict tokens from the index.
        
        THIS IS THE MAIN METHOD THAT SGLANG SHOULD CALL FOR EVICTION SYNC.
        
        Args:
            num_tokens: Number of tokens to evict (same as SGLang's eviction)
        
        Returns:
            Dictionary with eviction results, or None if request failed
        """
        return self._post("/evict", {"num_tokens": num_tokens})
    
    def search(
        self, 
        context: List[int], 
        update_access: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Search for a context in the index."""
        return self._post("/search", {
            "context": context,
            "update_access": update_access
        })
    
    def update_node(
        self, 
        search_path: List[int], 
        token_delta: int
    ) -> Optional[Dict[str, Any]]:
        """Update a node's token count."""
        return self._post("/update", {
            "search_path": search_path,
            "token_delta": token_delta
        })
    
    def insert(
        self, 
        context: List[int], 
        search_path: List[int], 
        total_tokens: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Insert a new context.
        
        Returns a dictionary containing:
        - node_id: The new leaf node ID
        - search_path: Path to the new node
        - request_id: Auto-generated request_id for token tracking
        """
        return self._post("/insert", {
            "context": context,
            "search_path": search_path,
            "total_tokens": total_tokens
        })
    
    # =========================================================================
    # Token Tracking (SGLang Integration)
    # =========================================================================
    
    def update_tokens(
        self,
        request_id: str,
        num_tokens: int
    ) -> Optional[Dict[str, Any]]:
        """
        Update token count for a request.
        
        THIS IS THE MAIN METHOD FOR SGLANG INTEGRATION.
        
        Call this when:
        1. A request starts processing (with initial input tokens)
        2. When generation completes (with total tokens: input + output)
        
        The method:
        - Updates the token count for the given request_id
        - Automatically triggers eviction if capacity is exceeded
        - Returns eviction results if any nodes were evicted
        
        Args:
            request_id: The request ID (from build or insert response)
            num_tokens: Total number of tokens for this request
            
        Returns:
            Dictionary with update result and any eviction info
        """
        return self._post("/update_tokens", {
            "request_id": request_id,
            "num_tokens": num_tokens
        })
    
    def get_requests(self) -> Optional[Dict[str, Any]]:
        """Get all tracked request IDs."""
        return self._get("/requests")
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get index statistics."""
        return self._get("/stats")
    
    def health(self) -> Optional[Dict[str, Any]]:
        """Check server health."""
        return self._get("/health")
    
    def is_ready(self) -> bool:
        """Check if the server is ready."""
        health = self.health()
        return health is not None and health.get("status") == "ready"
    
    def close(self):
        """Close the client session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.close()


# Convenience functions for simple usage

def evict_tokens(num_tokens: int, server_url: str = "http://localhost:8765"):
    """
    Simple function to evict tokens.
    
    For one-off calls without maintaining a client instance.
    """
    try:
        response = requests.post(
            f"{server_url}/evict",
            json={"num_tokens": num_tokens},
            timeout=1.0
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"RAGBoost eviction failed: {e}")
        return None


def update_request_tokens(
    request_id: str,
    input_tokens: int,
    output_tokens: int,
    server_url: str = "http://localhost:8765"
):
    """
    Simple function to update request tokens.
    
    For one-off calls without maintaining a client instance.
    """
    try:
        response = requests.post(
            f"{server_url}/update_request_tokens",
            json={
                "request_id": request_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            },
            timeout=1.0
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"RAGBoost token update failed: {e}")
        return None
