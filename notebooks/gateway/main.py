#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Gateway
-----------

This FastAPI application acts as a reverse proxy and unified entry
point for all underlying micro‑services. It exposes a simple routing
scheme: `/vector/<path>` for the vector search service, `/knowledge/<path>`
for the knowledge graph service and so on. The target service URLs are
supplied via environment variables (`VECTOR_SEARCH_URL`, etc.).

Clients can call the gateway instead of talking directly to each
service, simplifying configuration. The gateway does not modify
responses and supports most HTTP methods transparently.

This service acts as the single entry point for all client requests, routing them
to the appropriate downstream microservice. It also handles cross-cutting concerns
like authentication, rate limiting, and structured logging.
"""

import os
import httpx
import logging
from typing import Dict, Any

from fastapi import FastAPI, Request, Response, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pythonjsonlogger import jsonlogger

# --- Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
API_KEY = os.getenv("API_KEY", "default-secret-key") # Replace with a real secret management system in production
API_KEY_NAME = "X-API-KEY"

# --- Service URLs ---
SERVICE_URLS = {
    "vector": os.getenv("VECTOR_SEARCH_URL", "http://vector_search:8000"),
    "knowledge": os.getenv("KNOWLEDGE_GRAPH_URL", "http://knowledge_graph:8000"),
    "causal": os.getenv("CAUSAL_INFERENCE_URL", "http://causal_inference:8000"),
    "time": os.getenv("TIME_SERIES_URL", "http://time_series:8000"),
    "multi": os.getenv("MULTI_MODAL_URL", "http://multi_modal:8000"),
    "hier": os.getenv("HIER_URL", "http://hierarchical_classification:8000"),
    "rule": os.getenv("RULE_ENGINE_URL", "http://rule_engine:8000"),
    "orch": os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000"),
    "web": os.getenv("WEB_SERVICE_URL", "http://web_service:8000"),
    "shell": os.getenv("SHELL_COMMAND_URL", "http://shell_command:8000"),
}

# --- Structured Logging Setup ---
log = logging.getLogger("api_gateway")
log.setLevel(LOG_LEVEL)
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    '%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d'
)
logHandler.setFormatter(formatter)
log.addHandler(logHandler)
log.propagate = False


# --- Rate Limiting Setup ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Citadel - API Gateway", version="2.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- Security ---
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    """Validates the API key provided in the request header."""
    if api_key == API_KEY:
        return api_key
    else:
        log.warning("Invalid API Key received.", extra={"remote_addr": get_remote_address})
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials",
        )

# --- Generic Proxy Route ---
@app.api_route("/{service}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
@limiter.limit("100/minute") # Example: 100 requests per minute per IP
async def route_request(request: Request, service: str, path: str, api_key: str = Depends(get_api_key)):
    """
    A generic proxy endpoint that routes requests to the correct microservice.
    """
    if service not in SERVICE_URLS:
        log.error(f"Attempted to route to an unknown service: {service}")
        raise HTTPException(status_code=404, detail="Service not found")

    service_url = SERVICE_URLS[service]
    downstream_url = f"{service_url}/{path}"
    
    body = await request.body()
    headers = dict(request.headers)
    
    # Avoid forwarding host header from the original request
    headers.pop("host", None)
    
    log.info(f"Routing request for {service}/{path}", extra={
        "service": service,
        "path": path,
        "method": request.method,
        "client_host": request.client.host
    })

    try:
        async with httpx.AsyncClient() as client:
            rp = await client.request(
                method=request.method,
                url=downstream_url,
                headers=headers,
                params=request.query_params,
                content=body,
                timeout=60.0,
            )
            
            # Copy the response to return to the client
            response = Response(content=rp.content, status_code=rp.status_code, headers=dict(rp.headers))
            return response
            
    except httpx.ConnectError as e:
        log.error(f"Service connection error for {service}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {service}")
    except httpx.TimeoutException as e:
        log.error(f"Service timeout for {service}: {e}", exc_info=True)
        raise HTTPException(status_code=504, detail=f"Gateway Timeout: {service}")
    except Exception as e:
        log.critical(f"An unexpected error occurred while routing to {service}: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Bad Gateway: An internal error occurred.")


@app.get("/health", summary="Health check endpoint", tags=["Monitoring"])
async def health_check():
    """Provides a basic health check of the gateway itself."""
    return {"status": "ok", "gateway": "alive"}
