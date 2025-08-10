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
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase, exceptions

# --- Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("knowledge_graph_service")

# --- Neo4j Driver Management ---
driver = None

def get_driver():
    global driver
    if driver is None:
        log.info(f"Initializing Neo4j driver for {NEO4J_URL}")
        try:
            driver = AsyncGraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
            log.info("Neo4j driver initialized successfully.")
        except exceptions.AuthError as e:
            log.critical(f"Neo4j authentication failed: {e}")
            raise RuntimeError("Neo4j authentication failed.")
        except Exception as e:
            log.critical(f"Failed to initialize Neo4j driver: {e}", exc_info=True)
            raise RuntimeError(f"Neo4j driver initialization failed: {e}")
    return driver

async def close_driver():
    global driver
    if driver:
        log.info("Closing Neo4j driver.")
        await driver.close()
        driver = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_driver()
    yield
    await close_driver()

# --- Models ---
class CypherQuery(BaseModel):
    query: str
    parameters: dict = {}

# --- Service Initialization ---
app = FastAPI(
    title="Knowledge Graph Service",
    description="Provides an interface for querying a Neo4j graph database.",
    version="1.0.0",
    lifespan=lifespan
)

# --- API Endpoints ---
@app.post("/query", summary="Execute a Cypher query")
async def execute_query(request: CypherQuery):
    """
    Executes a Cypher query against the Neo4j database.
    """
    driver = get_driver()
    try:
        async with driver.session() as session:
            result = await session.run(request.query, request.parameters)
            records = [record.data() async for record in result]
            log.info(f"Query executed successfully, returned {len(records)} records.")
            return {"result": records}
    except exceptions.CypherSyntaxError as e:
        log.error(f"Cypher syntax error in query '{request.query}': {e}")
        raise HTTPException(status_code=400, detail=f"Cypher Syntax Error: {e.message}")
    except Exception as e:
        log.error(f"Error executing Cypher query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Provides a basic health check of the service."""
    driver = get_driver()
    try:
        await driver.verify_connectivity()
        return {"status": "ok", "neo4j_connection": "ok"}
    except Exception as e:
        log.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service Unavailable: Cannot connect to Neo4j. {e}")
