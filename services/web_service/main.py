"""
Web Service for the Full AI Platform
------------------------------------

This service provides the AI agent with the ability to interact with the internet.

Features:
- Web Search: Uses the `duckduckgo-search` library to perform web searches.
- Web Fetching: Uses `httpx` and `BeautifulSoup` to fetch and parse the content of a URL.
- Enhanced Error Handling: More robust error handling and logging.
"""

import os
import logging
from typing import List

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Configuration & Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("web_service")

# --- FastAPI App ---
app = FastAPI(title="Web Service", version="1.0.0")

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



# --- Pydantic Models ---
class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query.")
    max_results: int = Field(5, ge=1, le=20, description="The maximum number of search results to return.")

class FetchRequest(BaseModel):
    url: str = Field(..., description="The URL of the web page to fetch.")

# --- Core Service Logic ---

@app.post("/search")
def web_search(request: SearchRequest):
    """Performs a web search using DuckDuckGo."""
    log.info(f"Performing web search for: {request.query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(request.query, max_results=request.max_results))
        log.info(f"Web search for '{request.query}' completed with {len(results)} results.")
        return {"results": results}
    except Exception as e:
        log.error(f"Error during web search for '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to perform web search: {e}")

@app.post("/fetch")
async def fetch_url(request: FetchRequest):
    """Fetches and parses the content of a URL."""
    log.info(f"Attempting to fetch URL: {request.url}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(request.url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            text = soup.get_text(separator="\n", strip=True)
            log.info(f"Successfully fetched and parsed URL: {request.url}")
            return {"url": request.url, "content": text}
        except httpx.RequestError as e:
            log.error(f"Network error fetching URL {request.url}: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Network error fetching URL: {e}")
        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error fetching URL {request.url}: {e.response.status_code} - {e.response.text}", exc_info=True)
            raise HTTPException(status_code=e.response.status_code, detail=f"HTTP error fetching URL: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            log.error(f"Unexpected error fetching URL {request.url}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "message": "Web service is operational."}

