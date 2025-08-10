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
"""

from __future__ import annotations

import os
from typing import Dict

import httpx
from fastapi import FastAPI, Request, Response, HTTPException


SERVICE_PREFIXES = {
    "vector": "VECTOR_SEARCH_URL",
    "knowledge": "KNOWLEDGE_GRAPH_URL",
    "causal": "CAUSAL_INFERENCE_URL",
    "time": "TIME_SERIES_URL",
    "multi": "MULTI_MODAL_URL",
    "hier": "HIERARCHICAL_CLASSIFICATION_URL",
    "rule": "RULE_ENGINE_URL",
    "orch": "ORCHESTRATOR_URL",
    "web": "WEB_SERVICE_URL",
    "shell": "SHELL_COMMAND_URL",
}


def resolve_service(prefix: str) -> str | None:
    env_var = SERVICE_PREFIXES.get(prefix)
    if not env_var:
        return None
    return os.getenv(env_var)


app = FastAPI(title="API Gateway", version="1.0.0")

# --- Simple API Key middleware (applies to all routes except health/docs) ---
API_KEY = os.getenv("API_KEY", "")
from starlette.responses import JSONResponse

@app.middleware("http")
async def _require_api_key(request, call_next):
    # allow unauthenticated access to health and docs
    if request.url.path in {"/health", "/docs", "/openapi.json"} or request.url.path.startswith("/docs"):
        return await call_next(request)
    key = request.headers.get("X-API-Key")
    if not API_KEY or key != API_KEY:
        return JSONResponse(status_code=403, content={"detail": "Forbidden"})
    return await call_next(request)




@app.get("/services", summary="List services")
def list_services():
    return {
        prefix: resolve_service(prefix)
        for prefix in SERVICE_PREFIXES
    }


@app.api_route("/{service}/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(service: str, path: str, request: Request):
    base_url = resolve_service(service)
    if not base_url:
        raise HTTPException(status_code=404, detail=f"Unknown service '{service}'")
    # Build target URL
    url = f"{base_url}/{path}"
    method = request.method
    # Preserve the incoming headers except host and content-length
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length"}
    }
    body = await request.body()
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.request(
                method, url, content=body or None, headers=headers, params=request.query_params
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}