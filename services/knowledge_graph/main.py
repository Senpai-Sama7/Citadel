#!/usr/-bin/env python
# -*- coding: utf-8 -*-

"""
Knowledge Graph Micro‑service
----------------------------

This service provides a thin REST API around a Neo4j graph database. It
allows clients to create nodes, establish relationships and execute
arbitrary read queries using Cypher. The service expects the Neo4j
connection details to be supplied via environment variables.

Environment variables:

* `NEO4J_URL` – Bolt URL to connect to (e.g., bolt://neo4j:7687)
* `NEO4J_USER` – Username for authentication (defaults to 'neo4j')
* `NEO4J_PASSWORD` – Password for authentication

This service provides an interface to a Neo4j graph database.

Knowledge Graph Service (Refactored)
------------------------------------

This service provides a secure and robust interface to a Neo4j graph database,
configured via environment variables and protected by API key authentication.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase, exceptions

# --- Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
API_KEY = os.getenv("API_KEY")

# --- Logging ---
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Pre-flight Checks ---
if not all([NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD]):
    raise ValueError("NEO4J_URL, NEO4J_USER, and NEO4J_PASSWORD environment variables must be set.")
if not API_KEY:
    raise ValueError("API_KEY environment variable is not set.")

# --- Neo4j Driver Management ---
driver: AsyncGraphDatabase = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the Neo4j driver lifecycle."""
    global driver
    logger.info(f"Initializing Neo4j driver for {NEO4J_URL}")
    try:
        driver = AsyncGraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
        await driver.verify_connectivity()
        logger.info("Neo4j driver initialized successfully.")
    except exceptions.AuthError as e:
        logger.critical(f"Neo4j authentication failed: {e}")
        raise RuntimeError("Neo4j authentication failed.")
    except Exception as e:
        logger.critical(f"Failed to initialize Neo4j driver: {e}", exc_info=True)
        raise RuntimeError(f"Neo4j driver initialization failed: {e}")
    
    yield
    
    if driver:
        logger.info("Closing Neo4j driver.")
        await driver.close()

# --- Models ---
class CypherQuery(BaseModel):
    query: str
    parameters: dict = {}

# --- Service Initialization ---
app = FastAPI(
    title="Knowledge Graph Service",
    description="Provides an interface for querying a Neo4j graph database.",
    version="2.0.0",
    lifespan=lifespan
)

# --- Security ---
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def get_api_key(key: str = Security(api_key_header)):
    if key == API_KEY:
        return key
    else:
        raise HTTPException(status_code=403, detail="Invalid API Key")

# --- API Endpoints ---
@app.post("/query", summary="Execute a Cypher query")
async def execute_query(request: CypherQuery, api_key: str = Security(get_api_key)):
    """
    Executes a Cypher query against the Neo4j database.
    For security, only read-only queries (starting with MATCH) are allowed.
    """
    query = request.query.strip()
    if not query.lower().startswith("match"):
        raise HTTPException(status_code=400, detail="Query is not read-only. Only queries beginning with 'MATCH' are allowed.")

    try:
        async with driver.session() as session:
            result = await session.run(query, request.parameters)
            records = [record.data() async for record in result]
            logger.info(f"Query executed successfully, returned {len(records)} records.")
            return {"result": records}
    except exceptions.CypherSyntaxError as e:
        logger.error(f"Cypher syntax error in query '{query}': {e}")
        raise HTTPException(status_code=400, detail=f"Cypher Syntax Error: {e.message}")
    except Exception as e:
        logger.error(f"Error executing Cypher query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Provides a basic health check of the service."""
    if not driver:
        return {"status": "error", "neo4j_connection": "uninitialized"}
    try:
        await driver.verify_connectivity()
        return {"status": "ok", "neo4j_connection": "ok"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "neo4j_connection": "failed"}
