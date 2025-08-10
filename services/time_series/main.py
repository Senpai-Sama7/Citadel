"""
Time‑Series Analytics Micro‑service
----------------------------------

This service provides endpoints for forecasting and anomaly detection
over univariate time‑series data using Facebook Prophet. Input data
should be supplied as a list of objects with `ds` (timestamp in
ISO‑format) and `y` (numeric value) keys. Forecasting returns the
predicted mean and confidence intervals for both historical and future
periods. Anomaly detection flags points where the observed value falls
outside the prediction interval.

Note: Prophet can take a few seconds to initialise on the first call
because it compiles a Stan model. In a long‑running service this cost
is amortised.
"""

from __future__ import annotations

import os
from typing import List, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prophet import Prophet


app = FastAPI(title="Time Series Service", version="1.0.0")

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




class DataPoint(BaseModel):
    ds: str = Field(..., description="Timestamp in YYYY-MM-DD or ISO format")
    y: float = Field(..., description="Observed value at the timestamp")


class ForecastRequest(BaseModel):
    data: List[DataPoint] = Field(..., description="Historical time-series observations")
    horizon: int = Field(..., description="Number of future steps to forecast", ge=1, le=365)


def _prepare_dataframe(points: List[DataPoint]) -> pd.DataFrame:
    df = pd.DataFrame([p.model_dump() for p in points])
    # Prophet expects columns named 'ds' and 'y' with ds as datetime
    try:
        df['ds'] = pd.to_datetime(df['ds'])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {exc}")
    return df


@app.post("/forecast", summary="Forecast time series")
def forecast(req: ForecastRequest):
    if not req.data:
        raise HTTPException(status_code=400, detail="Input data is empty")
    df = _prepare_dataframe(req.data)
    # Initialise and fit the model. Using daily seasonality by default.
    m = Prophet(interval_width=0.95)
    try:
        m.fit(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model fitting failed: {exc}")
    future = m.make_future_dataframe(periods=req.horizon)
    forecast = m.predict(future)
    result = [
        {
            "ds": row['ds'].strftime("%Y-%m-%d"),
            "yhat": float(row['yhat']),
            "yhat_lower": float(row['yhat_lower']),
            "yhat_upper": float(row['yhat_upper']),
        }
        for _, row in forecast.iterrows()
    ]
    return {"forecast": result}


@app.post("/anomaly", summary="Detect anomalies")
def anomaly(req: ForecastRequest):
    if not req.data:
        raise HTTPException(status_code=400, detail="Input data is empty")
    df = _prepare_dataframe(req.data)
    m = Prophet(interval_width=0.95)
    try:
        m.fit(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model fitting failed: {exc}")
    forecast = m.predict(df[['ds']])
    merged = df.copy()
    merged['yhat_lower'] = forecast['yhat_lower']
    merged['yhat_upper'] = forecast['yhat_upper']
    merged['anomaly'] = (merged['y'] < merged['yhat_lower']) | (merged['y'] > merged['yhat_upper'])
    results = [
        {
            "ds": row['ds'].strftime("%Y-%m-%d"),
            "y": float(row['y']),
            "yhat_lower": float(row['yhat_lower']),
            "yhat_upper": float(row['yhat_upper']),
            "anomaly": bool(row['anomaly']),
        }
        for _, row in merged.iterrows()
    ]
    return {"anomalies": results}


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}