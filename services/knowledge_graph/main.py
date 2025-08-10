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
"""

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from neo4j import GraphDatabase, basic_auth
from pydantic import BaseModel, Field


def get_driver():
    url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "test")
    return GraphDatabase.driver(url, auth=basic_auth(user, password))


driver = get_driver()

app = FastAPI(title="Knowledge Graph Service", version="1.0.0")

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




class Node(BaseModel):
    label: str = Field(..., description="The label of the node (e.g., Sensor)")
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Key/value properties to attach to the node"
    )


class Relationship(BaseModel):
    start_id: int = Field(..., description="Internal ID of the start node")
    end_id: int = Field(..., description="Internal ID of the end node")
    rel_type: str = Field(..., description="Type of the relationship (e.g., INSTALLED_IN)")
    properties: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional properties on the relationship"
    )


class QueryRequest(BaseModel):
    cypher: str = Field(..., description="Cypher query to run")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional parameters for the query"
    )


@app.post("/nodes", summary="Create a node")
def create_node(node: Node):
    """Create a single node in the graph and return its internal ID."""
    label = node.label.strip()
    if not label.isidentifier():
        raise HTTPException(status_code=400, detail="Label must be a valid identifier")
    props = node.properties
    # Use a dynamic Cypher query to inject the label. Parameterised queries
    # cannot substitute labels, but we still parameterise the properties.
    query = f"CREATE (n:{label} $props) RETURN id(n) AS id"
    try:
        with driver.session() as session:
            result = session.run(query, props=props)
            record = result.single()
            return {"id": record["id"]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/relationships", summary="Create a relationship")
def create_relationship(rel: Relationship):
    """Create a relationship between two nodes by internal IDs."""
    rel_type = rel.rel_type.strip().upper()
    if not rel_type.isidentifier():
        raise HTTPException(status_code=400, detail="Relationship type must be a valid identifier")
    props = rel.properties or {}
    query = (
        f"MATCH (a), (b) WHERE id(a) = $start AND id(b) = $end "
        f"CREATE (a)-[r:{rel_type} $props]->(b) RETURN id(r) AS id"
    )
    try:
        with driver.session() as session:
            result = session.run(query, start=rel.start_id, end=rel.end_id, props=props)
            record = result.single()
            return {"id": record["id"]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query", summary="Run an arbitrary Cypher query")
def run_query(req: QueryRequest):
    """Execute a Cypher query and return the raw result list.

    For safety, only queries starting with MATCH/RETURN are allowed. This
    restriction prevents accidental writes or destructive operations via
    the API. More sophisticated access control could be added here.
    """
    cypher = req.cypher.strip()
    if not cypher.lower().startswith("match"):
        raise HTTPException(status_code=400, detail="Only read queries beginning with MATCH are allowed")
    params = req.params or {}
    try:
        with driver.session() as session:
            result = session.run(cypher, **params)
            records = [record.data() for record in result]
            return {"results": records}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}