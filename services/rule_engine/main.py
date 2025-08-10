"""
Rule Engine Micro‑service
-------------------------

This service wraps a simple rule engine based on the Experta library.
Clients send event data (arbitrary key/value pairs) and receive a list
of actions produced by the rule engine. The example rules provided here
illustrate how to encode business logic with thresholds. You can extend
the rules by modifying the `RuleEngine` class.
"""

from __future__ import annotations

import os
from typing import Dict, List

from experta import Fact, Field, KnowledgeEngine, Rule, P
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Rule Engine Service", version="1.0.0")

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




class Event(BaseModel):
    """An arbitrary event with numeric attributes."""

    # Use dynamic fields: additional numeric properties allowed
    __root__: Dict[str, float]


class SensorReading(Fact):
    temperature: float = Field()
    humidity: float = Field()


class RuleEngine(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.actions: List[str] = []

    @Rule(SensorReading(temperature=P(lambda t: t > 30), humidity=P(lambda h: h > 70)))
    def high_heat_and_humidity(self):
        self.actions.append("alert_high_heat_and_humidity")

    @Rule(SensorReading(temperature=P(lambda t: t < 0)))
    def freezing(self):
        self.actions.append("alert_freezing")


@app.post("/evaluate", summary="Evaluate event through rules")
def evaluate(event: Event):
    data = event.__root__
    engine = RuleEngine()
    engine.reset()
    # We only handle known keys (temperature & humidity) in this example
    kwargs = {}
    if 'temperature' in data:
        kwargs['temperature'] = data['temperature']
    if 'humidity' in data:
        kwargs['humidity'] = data['humidity']
    if kwargs:
        engine.declare(SensorReading(**kwargs))
    engine.run()
    return {"actions": engine.actions}


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}