"""
Causal Inference Micro‑service
-----------------------------

This service exposes a simple interface for estimating causal effects
using the DoWhy and EconML libraries. Given a dataset, a treatment
variable and an outcome variable, it identifies a causal model,
estimates the effect of the treatment on the outcome using backdoor
adjustment and returns the point estimate.

The API accepts data as a list of dictionaries (each row), but you can
also upload CSV files via the API gateway. The implementation is
intentionally basic to keep the service lightweight; more advanced
estimators (e.g. Double Machine Learning) can be plugged in with
minimal modifications.
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dowhy import CausalModel


app = FastAPI(title="Causal Inference Service", version="1.0.0")

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




class EffectRequest(BaseModel):
    data: List[Dict[str, float]] = Field(..., description="Rows of the dataset")
    treatment: str = Field(..., description="Name of the treatment column")
    outcome: str = Field(..., description="Name of the outcome column")
    confounders: Optional[List[str]] = Field(
        default=None, description="Optional list of confounder column names"
    )


@app.post("/effect", summary="Estimate treatment effect")
def estimate_effect(req: EffectRequest):
    """Estimate the causal effect of the specified treatment on the outcome.

    If confounders are not provided, all other columns are assumed
    potential confounders. The backdoor criterion is used and the
    propensity score matching estimator is applied.
    """
    if not req.data:
        raise HTTPException(status_code=400, detail="Dataset cannot be empty")
    df = pd.DataFrame(req.data)
    if req.treatment not in df.columns or req.outcome not in df.columns:
        raise HTTPException(status_code=400, detail="Treatment or outcome column missing")
    confounders = req.confounders
    if not confounders:
        confounders = [c for c in df.columns if c not in (req.treatment, req.outcome)]
    try:
        model = CausalModel(
            data=df,
            treatment=req.treatment,
            outcome=req.outcome,
            common_causes=confounders,
        )
        identified = model.identify_effect()
        estimate = model.estimate_effect(
            identified, method_name="backdoor.propensity_score_matching"
        )
        effect_value = float(estimate.value)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"effect": effect_value}


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}