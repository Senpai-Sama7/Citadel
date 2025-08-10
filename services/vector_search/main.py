#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

"""
Vector Search Service
---------------------

This service provides vector search capabilities using Redis as a vector database.
"""

import os
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

# --- Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vector-index")
VECTOR_DIMENSION = 384 # Based on the chosen model

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("vector_search_service")

# --- Models ---
class Document(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = {}

class Query(BaseModel):
    query: str
    top_k: int = 3

# --- Service Initialization ---
app = FastAPI(
    title="Vector Search Service",
    description="Provides vector embedding and search functionality.",
    version="1.0.0"
)

try:
    log.info(f"Loading sentence transformer model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    log.info("Model loaded successfully.")
except Exception as e:
    log.critical(f"Failed to load embedding model: {e}", exc_info=True)
    raise RuntimeError(f"Model loading failed: {e}")

try:
    log.info(f"Connecting to Redis at {REDIS_URL} and initializing index '{INDEX_NAME}'")
    index = SearchIndex.from_yaml("services/vector_search/schema.yaml")
    index.set_client_from_url(REDIS_URL)
    index.create(overwrite=True)
    log.info("Redis index created/connected successfully.")
except Exception as e:
    log.critical(f"Failed to connect to Redis or create index: {e}", exc_info=True)
    raise RuntimeError(f"Redis connection failed: {e}")

# --- API Endpoints ---
@app.post("/index", summary="Index a list of documents")
async def index_documents(documents: List[Document]):
    """
    Generates embeddings for a list of documents and indexes them in Redis.
    """
    try:
        data_to_load = []
        for doc in documents:
            vector = model.encode(doc.text).tolist()
            data_to_load.append({
                "id": doc.id,
                "text": doc.text,
                "vector": vector,
                **doc.metadata
            })
        if data_to_load:
            index.load(data_to_load, id_field="id")
            log.info(f"Successfully indexed {len(data_to_load)} documents.")
        return {"message": f"Successfully indexed {len(data_to_load)} documents."}
    except Exception as e:
        log.error(f"Error indexing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", summary="Search for similar documents")
async def search(query: Query):
    """
    Searches for documents similar to the query text.
    """
    try:
        query_embedding = model.encode(query.query).tolist()
        vector_query = VectorQuery(
            vector=query_embedding,
            vector_field_name="vector",
            num_results=query.top_k,
            return_fields=["id", "text", "vector_score"]
        )
        results = index.query(vector_query)
        log.info(f"Search for '{query.query}' returned {len(results)} results.")
        return {"results": results}
    except Exception as e:
        log.error(f"Error during search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Provides a basic health check of the service."""
    try:
        index.client.ping()
        return {"status": "ok", "redis_connection": "ok"}
    except Exception as e:
        log.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service Unavailable: Cannot connect to Redis.")
