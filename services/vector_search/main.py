"""
Enhanced Vector Search Microservice
-----------------------------------

This service provides semantic search capabilities, upgraded with a more 
powerful model and persistent storage for the index.

**Upgraded Features:**

- **Powerful Embedding Model:** Uses the `all-mpnet-base-v2` model for 
  state-of-the-art sentence embeddings.
- **Persistent Index:** The FAISS index and document mappings are saved to disk, 
  ensuring that the search index is not lost on service restart.
- **Configurable:** The model and persistence path can be configured via 
  environment variables.
- **Enhanced Error Handling & Observability:** More robust error handling and 
  detailed logging for better debugging and monitoring.

**Environment Variables:**

- `MODEL_NAME`: The name of the sentence-transformer model to use (default: 
  `all-mpnet-base-v2`).
- `INDEX_PATH`: The path to the directory where the index is persisted 
  (default: `/data/vector_index`).
- `LOG_LEVEL`: The logging level (e.g., "INFO", "DEBUG").
"""

import os
import json
import logging
from typing import List, Optional

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# --- Configuration & Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MODEL_NAME = os.getenv("MODEL_NAME", "all-mpnet-base-v2")
INDEX_PATH = os.getenv("INDEX_PATH", "/data/vector_index")
FAISS_INDEX_FILE = os.path.join(INDEX_PATH, "index.faiss")
DOC_IDS_FILE = os.path.join(INDEX_PATH, "doc_ids.json")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("vector_search_service")

# --- FastAPI App & Global State ---
app = FastAPI(title="Enhanced Vector Search Service", version="2.0.0")

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



_model: Optional[SentenceTransformer] = None
_faiss_index: Optional[faiss.Index] = None
_doc_ids: List[str] = []

# --- Pydantic Models ---
class Document(BaseModel):
    id: str = Field(..., description="Unique identifier for the document.")
    text: str = Field(..., description="The content of the document.")

class SearchRequest(BaseModel):
    query: str = Field(..., description="The text query to search for.")
    top_k: int = Field(5, ge=1, le=100, description="The number of results to return.")

# --- Core Service Logic ---
def load_model_and_index():
    """Loads the embedding model and the persisted FAISS index from disk."""
    global _model, _faiss_index, _doc_ids

    try:
        log.info(f"Loading sentence-transformer model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        log.info("Model loaded successfully.")
    except Exception as e:
        log.critical(f"Failed to load SentenceTransformer model {MODEL_NAME}: {e}", exc_info=True)
        raise RuntimeError(f"Embedding model failed to load: {e}")

    os.makedirs(INDEX_PATH, exist_ok=True)

    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOC_IDS_FILE):
        try:
            log.info(f"Loading FAISS index from: {FAISS_INDEX_FILE}")
            _faiss_index = faiss.read_index(FAISS_INDEX_FILE)
            with open(DOC_IDS_FILE, "r") as f:
                _doc_ids = json.load(f)
            log.info(f"Index loaded with {_faiss_index.ntotal} vectors.")
        except Exception as e:
            log.error(f"Failed to load existing FAISS index or doc IDs: {e}. Starting with empty index.", exc_info=True)
            # If loading fails, start with an empty index
            embedding_dim = _model.get_sentence_embedding_dimension()
            _faiss_index = faiss.IndexFlatL2(embedding_dim)
            _doc_ids = []
    else:
        log.info("No existing index found. A new one will be created.")
        embedding_dim = _model.get_sentence_embedding_dimension()
        _faiss_index = faiss.IndexFlatL2(embedding_dim)
        _doc_ids = []

def save_index():
    """Saves the FAISS index and document IDs to disk."""
    if _faiss_index is not None:
        try:
            log.info(f"Saving FAISS index to: {FAISS_INDEX_FILE}")
            faiss.write_index(_faiss_index, FAISS_INDEX_FILE)
            with open(DOC_IDS_FILE, "w") as f:
                json.dump(_doc_ids, f)
            log.info("Index saved successfully.")
        except Exception as e:
            log.error(f"Failed to save FAISS index or doc IDs: {e}", exc_info=True)

# --- FastAPI Lifecycle & Endpoints ---
@app.on_event("startup")
def startup_event():
    load_model_and_index()

@app.post("/index")
def index_documents(docs: List[Document]):
    if not docs:
        log.warning("Index request received with no documents.")
        raise HTTPException(status_code=400, detail="No documents provided.")
    if _model is None or _faiss_index is None:
        log.error("Attempted to index documents before model or index was ready.")
        raise HTTPException(status_code=503, detail="Index not ready. Model or FAISS index not loaded.")

    texts = [d.text for d in docs]
    ids = [d.id for d in docs]
    
    try:
        log.info(f"Indexing {len(texts)} new documents.")
        embeddings = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False) # show_progress_bar=False for production
        _faiss_index.add(embeddings.astype("float32"))
        _doc_ids.extend(ids)
        
        save_index()
        log.info(f"Successfully indexed {len(docs)} documents. Total indexed: {_faiss_index.ntotal}")
        return {"indexed_count": len(docs), "total_count": _faiss_index.ntotal}
    except Exception as e:
        log.error(f"Error during document indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to index documents: {e}")

@app.post("/search")
def semantic_search(request: SearchRequest):
    if _faiss_index is None or _faiss_index.ntotal == 0:
        log.warning("Search request received but no documents have been indexed.")
        raise HTTPException(status_code=404, detail="No documents have been indexed yet. Please index documents before searching.")
    if _model is None:
        log.error("Attempted to search before model was ready.")
        raise HTTPException(status_code=503, detail="Model not ready. Embedding model not loaded.")

    try:
        log.info(f"Performing semantic search for query: '{request.query}' (top_k={request.top_k})")
        query_vector = _model.encode([request.query], convert_to_numpy=True).astype("float32")
        distances, indices = _faiss_index.search(query_vector, request.top_k)

        results = [
            {"id": _doc_ids[idx], "score": float(1 - dist)} # Ensure score is float for JSON serialization
            for dist, idx in zip(distances[0], indices[0])
            if idx != -1
        ]
        log.info(f"Search completed. Found {len(results)} results.")
        return {"results": results}
    except Exception as e:
        log.error(f"Error during semantic search for query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to perform semantic search: {e}")

@app.get("/health")
def health():
    model_status = "loaded" if _model else "not loaded"
    index_status = "loaded" if _faiss_index else "not loaded"
    indexed_count = len(_doc_ids)
    return {"status": "ok", "model_status": model_status, "index_status": index_status, "indexed_documents": indexed_count}