"""
Multi‑Modal Embedding Service
----------------------------

This service demonstrates how to work with multiple data modalities
using a unified embedding space. For the sake of offline deployment and
lightweight dependencies, the embedding functions here are deterministic
and based on hashing the input rather than neural networks. However,
the same API can be used with real models (e.g., CLIP) by replacing
`_text_to_vec` and `_image_to_vec` with calls to those models.
"""

from __future__ import annotations

import os
import base64
import hashlib
import io
from typing import List, Dict

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image


app = FastAPI(title="Multi‑Modal Service", version="1.0.0")

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




def _deterministic_vector(seed: bytes, dim: int = 512) -> np.ndarray:
    """Generate a deterministic pseudo‑random vector from a seed."""
    # Use a stable hash (MD5) of the seed as the random seed
    digest = hashlib.md5(seed).digest()
    # Convert first 8 bytes to an integer seed for reproducibility
    seed_int = int.from_bytes(digest[:8], byteorder='big', signed=False)
    rng = np.random.default_rng(seed_int)
    vec = rng.standard_normal(dim)
    # Normalize vector to unit length for cosine similarity
    vec /= np.linalg.norm(vec) + 1e-9
    return vec


def _text_to_vec(text: str) -> np.ndarray:
    return _deterministic_vector(text.encode('utf-8'))


def _image_to_vec(image_bytes: bytes) -> np.ndarray:
    return _deterministic_vector(image_bytes)


class SearchItem(BaseModel):
    id: str
    vector: List[float]


class SearchRequest(BaseModel):
    query_type: str = Field(..., description="Type of query: 'text' or 'image'", pattern="^(text|image)$")
    query: str = Field(..., description="The text string or base64‑encoded image")
    dataset: List[SearchItem] = Field(..., description="List of items with precomputed vectors")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=50)


@app.post("/embed/text", summary="Embed text")
def embed_text(payload: Dict[str, str]):
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' field")
    vec = _text_to_vec(text)
    return {"vector": vec.tolist()}


@app.post("/embed/image", summary="Embed image")
async def embed_image(file: UploadFile = File(...)):
    contents = await file.read()
    # Validate file is an image by trying to open it
    try:
        Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    vec = _image_to_vec(contents)
    return {"vector": vec.tolist()}


@app.post("/search", summary="Search dataset")
def search(req: SearchRequest):
    # Compute embedding for the query
    if req.query_type == 'text':
        query_vec = _text_to_vec(req.query)
    else:
        # query is expected to be base64 encoded image string
        try:
            img_bytes = base64.b64decode(req.query)
        except Exception:
            raise HTTPException(status_code=400, detail="Query must be base64 encoded when query_type is 'image'")
        query_vec = _image_to_vec(img_bytes)
    # Build matrix from dataset vectors
    dataset_vectors = []
    ids = []
    for item in req.dataset:
        vec = np.asarray(item.vector, dtype=np.float32)
        if vec.ndim != 1:
            raise HTTPException(status_code=400, detail=f"Vector for item {item.id} must be 1‑D")
        ids.append(item.id)
        # Normalize vectors to unit length
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        dataset_vectors.append(vec)
    if not dataset_vectors:
        raise HTTPException(status_code=400, detail="Dataset cannot be empty")
    data_matrix = np.vstack(dataset_vectors)
    # Compute cosine similarities
    sims = data_matrix @ query_vec
    # Get top_k
    top_indices = sims.argsort()[-req.top_k:][::-1]
    results = [
        {"id": ids[i], "score": float(sims[i])}
        for i in top_indices
    ]
    return {"results": results}


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}