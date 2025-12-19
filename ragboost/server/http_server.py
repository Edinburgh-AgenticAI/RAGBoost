"""
RAGBoost Live Index HTTP Server

A FastAPI-based HTTP server that:
1. Exposes the LiveContextIndex as a REST API
2. Proxies LLM requests to SGLang backend
3. Automatically tracks tokens and triggers eviction

Usage:
    python -m ragboost.server.http_server --port 8765 --max-tokens 1000000 --sglang-url http://localhost:30000

Environment variables (alternative to CLI args):
    RAGBOOST_MAX_TOKENS: Maximum tokens allowed in index
    RAGBOOST_SGLANG_URL: SGLang backend URL (default: http://localhost:30000)
"""

import argparse
import logging
import time
import asyncio
import os
import uuid
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    import uvicorn
    import aiohttp
except ImportError:
    raise ImportError(
        "FastAPI, uvicorn, and aiohttp are required for the HTTP server. "
        "Install with: pip install fastapi uvicorn pydantic aiohttp"
    )

from .live_index import LiveContextIndex

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None
    logger.warning(
        "transformers not installed. Chat template functionality will be unavailable. "
        "Install with: pip install transformers"
    )


# Global state (initialized from env vars or CLI args)
_index: Optional[LiveContextIndex] = None
_max_tokens: Optional[int] = None
_sglang_url: Optional[str] = None
_aiohttp_session: Optional[aiohttp.ClientSession] = None
_tokenizer = None  # AutoTokenizer instance for chat template
_model_name: Optional[str] = None  # Model name for tokenizer


def _init_config():
    """Initialize config from environment variables."""
    global _max_tokens, _sglang_url, _tokenizer, _model_name

    if _max_tokens is None:
        env_max_tokens = os.environ.get("RAGBOOST_MAX_TOKENS")
        if env_max_tokens:
            _max_tokens = int(env_max_tokens)

    if _sglang_url is None:
        _sglang_url = os.environ.get("RAGBOOST_SGLANG_URL", "http://localhost:30000")

    # Initialize tokenizer for chat template if model is specified
    if _tokenizer is None:
        env_model = os.environ.get("RAGBOOST_MODEL_NAME")
        if env_model and AutoTokenizer is not None:
            try:
                _model_name = env_model
                _tokenizer = AutoTokenizer.from_pretrained(_model_name)
                logger.info(f"Loaded tokenizer for chat template: {_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer for {env_model}: {e}")


# Request/Response Models
class BuildIndexRequest(BaseModel):
    """Request to build the index."""

    contexts: List[List[Any]] = Field(
        ..., description="List of contexts (each is a list of document IDs)"
    )
    initial_tokens_per_context: int = Field(
        0, description="Initial token count per context"
    )
    alpha: float = Field(0.005, description="Distance computation parameter")
    use_gpu: bool = Field(False, description="Use GPU for distance computation")
    linkage_method: str = Field("average", description="Linkage method for clustering")


class EvictRequest(BaseModel):
    """Request to evict tokens."""

    num_tokens: int = Field(..., gt=0, description="Number of tokens to evict")


class SearchRequest(BaseModel):
    """Request to search for a context."""

    context: List[Any] = Field(..., description="Query context (list of document IDs)")
    update_access: bool = Field(True, description="Whether to update LRU timestamp")


class UpdateNodeRequest(BaseModel):
    """Request to update a node's token count."""

    search_path: List[int] = Field(..., description="Path to the node")
    token_delta: int = Field(
        ..., description="Tokens to add (positive) or remove (negative)"
    )


class InsertRequest(BaseModel):
    """Request to insert a new context."""

    context: List[Any] = Field(..., description="New context to insert")
    search_path: List[int] = Field(..., description="Search path from search operation")
    total_tokens: int = Field(0, description="Initial token count")


class RegisterRequestModel(BaseModel):
    """DEPRECATED - Request IDs are now auto-generated during build/insert."""

    request_id: str = Field(..., description="Unique identifier for the request")
    node_id: int = Field(..., description="The leaf node ID in the context index")


class UpdateTokensRequest(BaseModel):
    """Request to update token count for a request (SGLang integration)."""

    request_id: str = Field(
        ..., description="The request ID (from build/insert response)"
    )
    num_tokens: int = Field(
        ..., ge=0, description="Total number of tokens for this request"
    )


class TouchRequest(BaseModel):
    """Request to update access time for a request (LRU sync with SGLang)."""

    request_id: str = Field(
        ..., description="The request ID to touch"
    )


class TouchBatchRequest(BaseModel):
    """Request to update access time for multiple requests."""

    request_ids: List[str] = Field(
        ..., description="List of request IDs to touch"
    )


# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global _aiohttp_session

    # Initialize config from environment variables
    _init_config()

    logger.info("RAGBoost Index Server starting...")
    logger.info(f"  max_tokens: {_max_tokens}")
    logger.info(f"  sglang_url: {_sglang_url}")

    _aiohttp_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600))
    yield
    if _aiohttp_session:
        await _aiohttp_session.close()
    logger.info("RAGBoost Index Server shutting down...")


app = FastAPI(
    title="RAGBoost Live Index Server",
    description="HTTP API for RAGBoost LiveContextIndex with SGLang proxy and eviction synchronization",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "RAGBoost Live Index Server",
        "status": "running",
        "index_initialized": _index is not None,
        "timestamp": time.time(),
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    global _max_tokens

    # Ensure config is initialized from env vars
    _init_config()

    if _index is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "message": "Index not initialized. Call POST /build first.",
            },
        )

    stats = _index.get_stats()
    current_tokens = stats.get("total_tokens", 0)

    # max_tokens is guaranteed to be set
    return {
        "status": "ready",
        "eviction_enabled": True,
        "max_tokens": _max_tokens,
        "current_tokens": current_tokens,
        "utilization_pct": (current_tokens / _max_tokens * 100) if _max_tokens else 0,
        "index_stats": stats,
        "timestamp": time.time(),
    }


@app.post("/build")
async def build_index(request: BuildIndexRequest):
    """
    Build and initialize the index.

    This should be called once at startup before any other operations.
    Note: max_tokens is set when the server starts via --max-tokens argument.

    Auto-generates unique request_ids for each leaf node (context).
    The response includes a mapping from request_id -> node_id that can be used
    to track tokens via POST /update_tokens.
    """
    global _index, _max_tokens

    # Ensure config is initialized from env vars
    _init_config()

    if _max_tokens is None:
        raise HTTPException(
            status_code=500,
            detail="Server not configured with max_tokens. Restart server with --max-tokens argument.",
        )

    try:
        logger.info(f"Building index with {len(request.contexts)} contexts...")
        logger.info(f"Using max_tokens from server config: {_max_tokens:,}")

        _index = LiveContextIndex(
            alpha=request.alpha,
            use_gpu=request.use_gpu,
            linkage_method=request.linkage_method,
            max_tokens=_max_tokens,  # Pass max_tokens to index
        )

        result = _index.build_and_schedule(
            contexts=request.contexts,
            initial_tokens_per_context=request.initial_tokens_per_context,
        )

        # Extract request_id mapping for SGLang integration
        request_id_mapping = result.get("request_id_mapping", {})
        # Ordered list of request_ids matching input contexts order
        request_ids = result.get("request_ids", [])

        logger.info(
            f"Index built successfully. Auto-assigned {len(request_id_mapping)} request IDs"
        )

        return {
            "status": "success",
            "message": "Index built successfully",
            "num_contexts": len(request.contexts),
            "max_tokens": _max_tokens,
            "request_id_mapping": request_id_mapping,  # request_id -> node_id (dict)
            "request_ids": request_ids,  # Ordered list matching input contexts
            "stats": _index.get_stats(),
        }

    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evict")
async def evict(request: EvictRequest):
    """
    Evict tokens from the index.

    THIS IS THE MAIN ENDPOINT THAT SGLANG SHOULD CALL FOR CACHE EVICTION SYNC.

    IMPORTANT: The index must be initialized with max_tokens parameter.
    Call POST /build with max_tokens set before using this endpoint.

    When SGLang evicts cache with tree_cache.evict(num_tokens), it should also
    call this endpoint with the same num_tokens value.

    Integration in SGLang:
        import requests

        def evict_tokens(self, num_tokens):
            # SGLang's eviction
            self.tree_cache.evict(num_tokens)

            # Sync with RAGBoost index
            try:
                requests.post(
                    "http://localhost:8765/evict",
                    json={"num_tokens": num_tokens},
                    timeout=1.0
                )
            except Exception as e:
                logger.warning(f"RAGBoost eviction sync failed: {e}")
    """
    global _max_tokens

    # Ensure config is initialized from env vars
    _init_config()

    # Check if server was started with max_tokens
    if _max_tokens is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Server not configured with max_tokens. "
                "Restart server with --max-tokens argument. "
                "Example: python -m ragboost.server.http_server --port 8765 --max-tokens 1000000"
            ),
        )

    # Check if index is initialized
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /build first."
        )

    try:
        # Get current stats before eviction
        current_tokens = _index.eviction_heap.total_tokens()

        # Perform eviction
        result = _index.evict(request.num_tokens)

        # Log eviction details
        logger.info(
            f"Eviction: requested={request.num_tokens}, "
            f"evicted={result['tokens_evicted']}, "
            f"nodes_removed={len(result['evicted_node_ids'])}, "
            f"tokens: {current_tokens:,} -> {result['tokens_remaining']:,}, "
            f"max_allowed={_max_tokens:,}"
        )

        return {
            "status": "success",
            "max_tokens": _max_tokens,
            "tokens_before": current_tokens,
            **result,
        }

    except Exception as e:
        logger.error(f"Error during eviction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search(request: SearchRequest):
    """Search for a context in the index."""
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /build first."
        )

    try:
        search_path, node_id, prefix_length = _index.search(
            context=request.context, update_access=request.update_access
        )

        return {
            "status": "success",
            "search_path": search_path,
            "node_id": node_id,
            "prefix_length": prefix_length,
        }

    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update")
async def update_node(request: UpdateNodeRequest):
    """Update a node's token count."""
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /build first."
        )

    try:
        success = _index.update_node(
            search_path=request.search_path, token_delta=request.token_delta
        )

        return {"status": "success" if success else "failed", "updated": success}

    except Exception as e:
        logger.error(f"Error during update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insert")
async def insert_context(request: InsertRequest):
    """
    Insert a new context into the index.

    Auto-generates a unique request_id for the new leaf node.
    The response includes the request_id that can be used to track tokens
    via POST /update_tokens.
    """
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /build first."
        )

    try:
        node_id, search_path, request_id = _index.insert(
            context=request.context,
            search_path=request.search_path,
            total_tokens=request.total_tokens,
        )

        return {
            "status": "success",
            "node_id": node_id,
            "search_path": search_path,
            "request_id": request_id,  # Auto-generated for SGLang integration
        }

    except Exception as e:
        logger.error(f"Error during insertion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Token Tracking Endpoint (SGLang Integration)
# ============================================================================


@app.post("/update_tokens")
async def update_tokens(request: UpdateTokensRequest):
    """
    Update token count for a request.

    THIS IS THE MAIN ENDPOINT FOR SGLANG INTEGRATION.

    Call this endpoint when:
    1. A request starts processing (with initial input tokens)
    2. When generation completes (with total tokens: input + output)

    The endpoint:
    - Updates the token count for the given request_id
    - Automatically triggers eviction if capacity is exceeded
    - Returns eviction results if any nodes were evicted

    Example SGLang integration:
        # After request completes
        response = requests.post(
            "http://localhost:8765/update_tokens",
            json={
                "request_id": "req-000001",  # From build/insert response
                "num_tokens": 1500           # Total tokens (input + output)
            }
        )
    """
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /build first."
        )

    try:
        result = _index.update_tokens(
            request_id=request.request_id, num_tokens=request.num_tokens
        )

        if not result["updated"]:
            # Request not found - could be:
            # 1. Invalid request_id (never existed)
            # 2. Request was evicted during processing (race condition)
            # Either way, return success - the request is "handled" (either never existed or already evicted)
            logger.warning(
                f"Request ID '{request.request_id}' not found (possibly evicted during processing). "
                f"Returning success anyway."
            )
            return {
                "status": "success",
                "updated": False,
                "message": "Request not found (possibly evicted)",
                "request_id": request.request_id,
                "num_tokens": request.num_tokens,
            }

        logger.info(
            f"Updated tokens for request_id={request.request_id}: "
            f"{result.get('previous_tokens', 0)} -> {result['num_tokens']}"
        )

        if result["eviction_triggered"]:
            logger.info(
                f"Eviction triggered: evicted {len(result['evicted_requests'])} requests, "
                f"{result.get('tokens_evicted', 0)} tokens"
            )

        return {"status": "success", **result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/touch")
async def touch_request(request: TouchRequest):
    """
    Update access time for a request (LRU sync).

    THIS ENDPOINT KEEPS LRU IN SYNC WITH SGLANG.

    Call this when SGLang accesses (hits) a cached prefix.
    This updates the access time so RAGBoost's eviction heap
    stays in sync with SGLang's LRU ordering.

    Example SGLang integration:
        # When a request hits cached prefix
        requests.post(
            "http://localhost:8765/touch",
            json={"request_id": "req-000001"}
        )
    """
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /build first."
        )

    try:
        success = _index.touch(request.request_id)

        if not success:
            raise HTTPException(
                status_code=404, detail=f"Request ID '{request.request_id}' not found."
            )

        return {"status": "success", "request_id": request.request_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error touching request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/touch_batch")
async def touch_batch(request: TouchBatchRequest):
    """
    Update access time for multiple requests at once.

    More efficient than calling /touch multiple times.
    """
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /build first."
        )

    try:
        results = _index.touch_batch(request.request_ids)

        return {
            "status": "success",
            "touched": results["touched"],
            "not_found": results["not_found"]
        }

    except Exception as e:
        logger.error(f"Error touching batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/requests")
async def get_requests():
    """
    Get all tracked request IDs.

    Returns the list of all request_ids currently in the index.
    """
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /build first."
        )

    try:
        request_ids = list(_index.get_all_request_ids())

        return {
            "status": "success",
            "num_requests": len(request_ids),
            "request_ids": request_ids,
        }

    except Exception as e:
        logger.error(f"Error getting requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get index statistics."""
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /build first."
        )

    try:
        stats = _index.get_stats()
        eviction_stats = _index.get_eviction_stats()

        return {
            "status": "success",
            "index_stats": stats,
            "eviction_stats": eviction_stats,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/heap_status")
async def get_heap_status():
    """
    Get detailed eviction heap status for debugging.
    
    Shows:
    - Total tokens in heap
    - Max tokens allowed
    - Number of tracked requests
    - LRU order (oldest first)
    """
    if _index is None:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Call POST /build first."
        )

    try:
        heap = _index.eviction_heap
        
        # Get all requests sorted by access time (oldest first = LRU)
        requests_info = []
        for request_id, node_id in _index._request_to_node.items():
            metadata = _index.metadata.get(node_id)
            if metadata:
                requests_info.append({
                    "request_id": request_id,
                    "node_id": node_id,
                    "tokens": metadata.total_tokens,
                    "last_access_time": metadata.last_access_time,
                })
        
        # Sort by access time (oldest first)
        requests_info.sort(key=lambda x: x["last_access_time"])
        
        return {
            "status": "success",
            "total_tokens": heap.total_tokens(),
            "max_tokens": _index._max_tokens,
            "utilization_pct": (heap.total_tokens() / _index._max_tokens * 100) if _index._max_tokens else 0,
            "num_requests": len(requests_info),
            "heap_size": len(heap),
            "total_touches": _index.live_stats.get('total_touches', 0),
            "total_token_updates": _index.live_stats.get('total_token_updates', 0),
            "lru_order": requests_info,  # Oldest first (will be evicted first)
        }

    except Exception as e:
        logger.error(f"Error getting heap status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_index():
    """Reset the index (for debugging/testing). Note: max_tokens remains set from startup."""
    global _index

    try:
        _index = None
        logger.info("Index reset (max_tokens preserved from startup)")

        return {
            "status": "success",
            "message": "Index reset successfully. Call POST /build to reinitialize.",
            "max_tokens": _max_tokens,
        }

    except Exception as e:
        logger.error(f"Error resetting index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Chat Template Helper
# ============================================================================


def apply_chat_template(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Apply chat template to a prompt using the configured tokenizer.

    Args:
        prompt: The user's message/prompt text
        system_prompt: Optional system prompt to prepend

    Returns:
        The formatted prompt string with chat template applied,
        or the original prompt if no tokenizer is configured.
    """
    if _tokenizer is None:
        return prompt

    # Build messages list
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        # Apply chat template with tokenize=False (return string, not tokens)
        # and add_generation_prompt=True (append the assistant prompt prefix)
        formatted = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return formatted
    except Exception as e:
        logger.warning(f"Failed to apply chat template: {e}. Using raw prompt.")
        return prompt


# ============================================================================
# SGLang Proxy Endpoints
# ============================================================================


@app.post("/v1/completions")
async def proxy_completions(request: Request):
    """
    Proxy /v1/completions to SGLang and auto-update tokens.

    This endpoint:
    1. Forwards the request to SGLang backend
    2. Tracks token usage from the response
    3. Automatically updates the eviction heap
    4. Returns the response to the client

    To associate a request with a context, include 'request_id' in the request body.
    """
    # Ensure config is loaded
    if _sglang_url is None:
        _init_config()

    sglang_url = _sglang_url or os.environ.get(
        "RAGBOOST_SGLANG_URL", "http://localhost:30000"
    )

    if not sglang_url:
        raise HTTPException(
            status_code=503,
            detail="SGLang URL not configured. Set RAGBOOST_SGLANG_URL env var or use --sglang-url.",
        )

    try:
        # Parse request body
        body = await request.json()
        request_id = body.pop("request_id", None)  # Extract request_id if present
        
        # NOTE: We do NOT auto-generate request_id anymore.
        # The client should pass request_id from the /build response.
        # If not provided, RAGBoost token tracking is skipped for this request.
        if not request_id:
            logger.debug("No request_id provided, RAGBoost tracking disabled for this request")

        # Apply chat template if explicitly requested (default False - template should be applied at prompt generation)
        apply_template = body.pop("apply_chat_template", False)  # Default to False
        system_prompt = body.pop("system_prompt", None)  # Optional system prompt

        if apply_template and _tokenizer is not None and "prompt" in body:
            original_prompt = body["prompt"]
            body["prompt"] = apply_chat_template(original_prompt, system_prompt)
            logger.debug("Applied chat template to prompt")

        # Touch the request at the START to update LRU access time
        # This keeps RAGBoost's eviction heap in sync with SGLang's cache access pattern
        # Only touch if request_id exists in the index
        if request_id and _index:
            if _index.touch(request_id):
                logger.debug(f"Touched request_id={request_id}")
            else:
                logger.warning(f"Request ID '{request_id}' not found in index, token tracking disabled")
                request_id = None  # Clear so SGLang won't try to update

        # Pass request_id to SGLang so it can use the same ID for token tracking
        # Only set rid if we have a valid request_id in the index
        if request_id:
            body["rid"] = request_id
            logger.info(f"Proxy: forwarding request with rid={request_id}")
        else:
            logger.info("Proxy: forwarding request without rid (no RAGBoost tracking)")

        # Forward to SGLang
        sglang_api_url = f"{sglang_url}/v1/completions"
        logger.debug(f"Proxying to {sglang_api_url}")

        async with _aiohttp_session.post(sglang_api_url, json=body) as response:
            result = await response.json()

            # Token tracking is handled by SGLang via RAGBOOST_INDEX_URL
            # SGLang calls /update_tokens after request completion
            # SGLang calls /evict after its internal cache eviction
            
            # Add request_id to response for client reference
            if request_id and response.status == 200:
                usage = result.get("usage", {})
                result["_ragboost"] = {
                    "request_id": request_id,
                    "tokens_reported": usage.get("total_tokens", 0),
                }

            return JSONResponse(content=result, status_code=response.status)

    except aiohttp.ClientError as e:
        logger.error(f"Error proxying to SGLang: {e}")
        raise HTTPException(status_code=502, detail=f"SGLang backend error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy_sglang(path: str, request: Request):
    """
    Generic proxy for other SGLang /v1/* endpoints.

    Forwards requests to SGLang backend without modification.
    """
    # Ensure config is loaded
    if _sglang_url is None:
        _init_config()

    sglang_url = _sglang_url or os.environ.get(
        "RAGBOOST_SGLANG_URL", "http://localhost:30000"
    )

    if not sglang_url:
        raise HTTPException(
            status_code=503,
            detail="SGLang URL not configured. Set RAGBOOST_SGLANG_URL env var or use --sglang-url.",
        )

    try:
        target_url = f"{sglang_url}/v1/{path}"

        if request.method == "GET":
            async with _aiohttp_session.get(target_url) as response:
                result = await response.json()
                return JSONResponse(content=result, status_code=response.status)
        else:
            body = await request.json()
            async with _aiohttp_session.post(target_url, json=body) as response:
                result = await response.json()
                return JSONResponse(content=result, status_code=response.status)

    except aiohttp.ClientError as e:
        logger.error(f"Error proxying to SGLang: {e}")
        raise HTTPException(status_code=502, detail=f"SGLang backend error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the HTTP server."""
    parser = argparse.ArgumentParser(
        description="RAGBoost Live Index HTTP Server with SGLang Proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m ragboost.server.http_server --port 8765 --max-tokens 1000000 --sglang-url http://localhost:30000

The server acts as a proxy between your application and SGLang:
  - Your app sends requests to localhost:8765
  - The server forwards them to SGLang and tracks tokens
  - Eviction is triggered automatically when max_tokens is exceeded
        """,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="Maximum tokens allowed in index (REQUIRED for eviction)",
    )
    parser.add_argument(
        "--sglang-url",
        type=str,
        default="http://localhost:30000",
        help="SGLang backend URL (default: http://localhost:30000)",
    )
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name/path for chat template tokenizer (e.g., 'Qwen/Qwen3-32B')",
    )

    args = parser.parse_args()

    # Set environment variables so they propagate to uvicorn workers
    os.environ["RAGBOOST_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["RAGBOOST_SGLANG_URL"] = args.sglang_url.rstrip("/")
    if args.model:
        os.environ["RAGBOOST_MODEL_NAME"] = args.model

    # Also set global config for direct access
    global _max_tokens, _sglang_url, _tokenizer, _model_name
    _max_tokens = args.max_tokens
    _sglang_url = args.sglang_url.rstrip("/")

    # Initialize tokenizer for chat template
    if args.model and AutoTokenizer is not None:
        try:
            _model_name = args.model
            _tokenizer = AutoTokenizer.from_pretrained(_model_name)
            logger.info(f"Loaded tokenizer for chat template: {_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {args.model}: {e}")

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Starting RAGBoost Index Server on {args.host}:{args.port}")
    logger.info(f"Maximum tokens configured: {_max_tokens:,}")
    logger.info(f"SGLang backend URL: {_sglang_url}")

    # Run server
    uvicorn.run(
        "ragboost.server.http_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )
    #     try: requests.post("http://localhost:8765/evict", json={"num_tokens": num_tokens}, timeout=1.0)
    #     except: pass


if __name__ == "__main__":
    main()
