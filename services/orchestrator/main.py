"""
Orchestrator Micro-service
-------------------------

Glues the AI platform together. Listens on a Redis stream for incoming events,
dispatches to micro-services via HTTP, and persists to Neo4j and TimescaleDB.
Exposes /publish to push events for testing.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import logging
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import httpx
import psycopg2
import redis.asyncio as redis  # <-- modern async Redis client
from fastapi import FastAPI, BackgroundTasks, HTTPException
from neo4j import GraphDatabase, basic_auth
from pydantic import BaseModel

# --- Environment (defaults set for docker-compose network) ---
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

NEO4J_URL = os.getenv("NEO4J_URL", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test")

TS_HOST = os.getenv("TS_HOST", "timescaledb")
TS_PORT = int(os.getenv("TS_PORT", "5432"))
TS_USER = os.getenv("TS_USER", "tsuser")
TS_PASSWORD = os.getenv("TS_PASSWORD", "tspassword")
TS_DB = os.getenv("TS_DB", "tsdb")

RULE_ENGINE_URL = os.getenv("RULE_ENGINE_URL", "http://rule_engine:8000")
VECTOR_SEARCH_URL = os.getenv("VECTOR_SEARCH_URL", "http://vector_search:8000")

app = FastAPI(title="Orchestrator Service", version="1.0.0")

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
    """A generic event payload."""
    type: str
    data: Dict[str, Any]


async def ensure_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URL, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))


def insert_timeseries(measurement: str, value: float, ts_seconds: float):
    """Insert a value into TimescaleDB. Synchronous for simplicity."""
    try:
        conn = psycopg2.connect(
            host=TS_HOST, port=TS_PORT, user=TS_USER, password=TS_PASSWORD, dbname=TS_DB
        )
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS metrics (time TIMESTAMPTZ, measurement TEXT, value DOUBLE PRECISION);"
        )
        cur.execute(
            "INSERT INTO metrics (time, measurement, value) VALUES (to_timestamp(%s), %s, %s);",
            (ts_seconds, measurement, value),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to insert timeseries data: {e}", exc_info=True)
        # Fail silently in demo mode (but now with logging)
        pass


async def process_event(event: Dict[bytes, bytes]):
    """Process a single event from the Redis stream."""
    try:
        event_data = {k.decode(): json.loads(v.decode()) for k, v in event.items()}
    except Exception as e:
        logger.error(f"Failed to decode event data: {e}", exc_info=True)
        return

    event_type = event_data.get("type")
    payload = event_data.get("data", {})

    if event_type == "sensor":
        # Example dispatch to rule engine
        actions = []
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{RULE_ENGINE_URL}/evaluate", json=payload, timeout=20.0)
                if resp.status_code == 200:
                    actions = resp.json().get("actions", [])
        except Exception as e:
            logger.error(f"Failed to call rule engine: {e}", exc_info=True)
            actions = []

        # Persist to Neo4j
        try:
            driver = await ensure_neo4j_driver()
            with driver.session() as session:
                for action in actions:
                    session.run(
                        "CREATE (e:Event {type: $type, action: $action})",
                        type=event_type,
                        action=action,
                    )
        except Exception as e:
            logger.error(f"Failed to persist to Neo4j: {e}", exc_info=True)
            pass

        # Write numeric values to timeseries DB
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                insert_timeseries(key, float(value), time.time())


async def ensure_consumer_group(r: redis.Redis, stream: str, group: str):
    try:
        await r.xgroup_create(stream, group, id="$", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" in str(e):
            pass
        else:
            raise

async def event_listener(r: redis.Redis):
    stream = REDIS_STREAM
    group = REDIS_GROUP
    consumer = CONSUMER_NAME
    await ensure_consumer_group(r, stream, group)
    while True:
        try:
            results = await r.xreadgroup(group, consumer, {stream: ">"}, count=50, block=5000)
            if results:
                for _stream, messages in results:
                    for msg_id, fields in messages:
                        try:
                            await process_event(fields)
                            await r.xack(stream, group, msg_id)
                        except Exception as e:
                            logger.error(f"Error processing event or acknowledging: {e}", exc_info=True)
                            # leave in PEL for retry; optionally log
                            pass
            # light trim to keep stream bounded
            try:
                await r.xtrim(stream, maxlen=100000, approximate=True)
            except Exception as e:
                logger.warning(f"Error trimming Redis stream: {e}", exc_info=True)
                pass
        except Exception as e:
            logger.error(f"Unhandled exception in event listener loop: {e}", exc_info=True)
            await asyncio.sleep(1)


@app.on_event("startup")
async def startup_event():
    app.state.redis = redis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", decode_responses=False)
    app.state.task = asyncio.create_task(event_listener(app.state.redis))


@app.on_event("shutdown")
async def shutdown_event():
    global _pgpool
    if _pgpool:
        _pgpool.closeall()
    task: asyncio.Task = app.state.task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    try:
        await app.state.redis.aclose()
    except Exception:
        pass


@app.post("/publish", summary="Publish an event")
async def publish_event(event: Event):
    try:
        data = {"type": event.type, "data": event.data}
        encoded = {k: json.dumps(v) for k, v in data.items()}
        msg_id = await app.state.redis.xadd("events", encoded)
        if isinstance(msg_id, (bytes, bytearray)):
            msg_id = msg_id.decode()
        return {"message_id": msg_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok"}

