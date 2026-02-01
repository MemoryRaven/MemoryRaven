from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from memory_empire.knowledge_graph.neo4j_store import Neo4jConfig, Neo4jGraphStore
from memory_empire.knowledge_graph.pipeline import GraphIngestor

from .auth import require_api_key
from .settings import settings


class GraphIngestTextIn(BaseModel):
    text: str
    source_id: str | None = None


class CypherQueryIn(BaseModel):
    cypher: str
    params: dict[str, Any] = Field(default_factory=dict)
    limit: int = 200


class NeighborsIn(BaseModel):
    node_id: str
    depth: int = 1
    limit: int = 200


class PathIn(BaseModel):
    src_id: str
    dst_id: str
    max_hops: int = 6


@lru_cache(maxsize=1)
def _graph_store() -> Neo4jGraphStore:
    # Prefer settings (pydantic) but allow env vars for quick deploy.
    uri = getattr(settings, "neo4j_uri", None) or os.getenv("NEO4J_URI")
    user = getattr(settings, "neo4j_user", None) or os.getenv("NEO4J_USER")
    password = getattr(settings, "neo4j_password", None) or os.getenv("NEO4J_PASSWORD")
    database = getattr(settings, "neo4j_database", None) or os.getenv("NEO4J_DATABASE") or "neo4j"

    if not (uri and user and password):
        raise RuntimeError(
            "Neo4j not configured. Set MEMORY_SERVICE_NEO4J_URI/USER/PASSWORD (or NEO4J_* env vars)."
        )

    store = Neo4jGraphStore(Neo4jConfig(uri=uri, user=user, password=password, database=database))
    store.ensure_schema()
    return store


def build_graph_router() -> APIRouter:
    r = APIRouter(prefix="/v1/graph", tags=["graph"])

    @r.post("/ingest_text")
    async def ingest_text(payload: GraphIngestTextIn, _auth: None = Depends(require_api_key)):
        try:
            store = _graph_store()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        ingestor = GraphIngestor(store)
        stats = ingestor.ingest_text(payload.text, source_id=payload.source_id)
        return {
            "ok": True,
            "entities": stats.entities,
            "relations": stats.relations,
            "timing_ms": {"extract": stats.extract_ms, "upsert": stats.upsert_ms},
        }

    @r.post("/cypher")
    async def cypher(payload: CypherQueryIn, _auth: None = Depends(require_api_key)):
        try:
            store = _graph_store()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        # Guardrail: cap results.
        limited_cypher = payload.cypher
        # Do not attempt to parse Cypher; just apply a hard cap if missing.
        if " limit " not in payload.cypher.lower():
            limited_cypher = f"{payload.cypher}\nLIMIT {max(1, min(payload.limit, 1000))}"

        rows = store.query(limited_cypher, payload.params)
        return {"count": len(rows), "rows": rows}

    @r.post("/neighbors")
    async def neighbors(payload: NeighborsIn, _auth: None = Depends(require_api_key)):
        try:
            store = _graph_store()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        return store.neighbors(payload.node_id, depth=payload.depth, limit=payload.limit)

    @r.post("/shortest_path")
    async def shortest_path(payload: PathIn, _auth: None = Depends(require_api_key)):
        try:
            store = _graph_store()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        out = store.shortest_path(payload.src_id, payload.dst_id, max_hops=payload.max_hops)
        return {"found": out is not None, "path": out}

    return r
