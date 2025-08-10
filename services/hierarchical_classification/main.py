"""
Hierarchical Classification Micro‑service
---------------------------------------

This service trains and serves hierarchical classifiers using the
HiClass library. A model token is returned after training and must be
supplied when making predictions. The models are kept in memory for
the lifetime of the container; for persistence you would need to
implement model serialization and storage.
"""

from __future__ import annotations

import os
import uuid
from typing import Dict, List
import joblib

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from hiclass import Classifier

MODELS_PATH = "/data/models"

app = FastAPI(title="Hierarchical Classification Service", version="1.0.0")

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


class TrainRequest(BaseModel):
    features: List[Dict[str, float]] = Field(..., description="Feature vectors")
    labels: List[Dict[str, str]] = Field(
        ..., description="Label dictionaries; keys represent taxonomy levels"
    )


class PredictRequest(BaseModel):
    model_id: str = Field(..., description="Identifier returned by /train")
    features: Dict[str, float] = Field(..., description="Feature vector to classify")


_models: Dict[str, Classifier] = {}

@app.on_event("startup")
def startup_event():
    os.makedirs(MODELS_PATH, exist_ok=True)
    for filename in os.listdir(MODELS_PATH):
        if filename.endswith(".joblib"):
            model_id = filename.split(".")[0]
            _models[model_id] = joblib.load(os.path.join(MODELS_PATH, filename))

@app.post("/train", summary="Train a hierarchical classifier")
def train_model(req: TrainRequest):
    if not req.features or not req.labels:
        raise HTTPException(status_code=400, detail="Features and labels must be provided")
    try:
        X = pd.DataFrame(req.features)
        y = pd.DataFrame(req.labels)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # Define a base classifier. RandomForest generally works well for small examples.
    base_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    clf = Classifier(classifier=base_estimator)
    try:
        clf.fit(X, y)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")
    model_id = uuid.uuid4().hex
    _models[model_id] = clf
    joblib.dump(clf, os.path.join(MODELS_PATH, f"{model_id}.joblib"))
    return {"model_id": model_id}


@app.post("/predict", summary="Predict hierarchical labels")
def predict(req: PredictRequest):
    clf = _models.get(req.model_id)
    if clf is None:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        X = pd.DataFrame([req.features])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        preds = clf.predict(X)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    # preds is a DataFrame; convert first row to list
    path = [str(preds.iloc[0, i]) for i in range(preds.shape[1])]
    return {"path": path}


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}