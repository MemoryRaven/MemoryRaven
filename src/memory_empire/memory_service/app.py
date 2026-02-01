from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .db import DB
from .embedder import build_embedder
from .qdrant_ops import QdrantOpsConfig, build_client
from .settings import settings
from .auth import require_api_key

# Optional graph API (requires neo4j optional deps + config)
try:
    from .graph_api import build_graph_router
except Exception:  # pragma: no cover
    build_graph_router = None


class EventIn(BaseModel):
    source: str
    event_type: str = Field(default="note")
    actor_id: str | None = None
    thread_id: str | None = None
    content_text: str | None = None
    content_json: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    sensitivity: str = "internal"
    created_at: str | None = None
    observed_at: str | None = None


class SearchOut(BaseModel):
    id: str
    score: float
    kind: str
    source: str | None = None
    event_type: str | None = None
    actor_id: str | None = None
    thread_id: str | None = None
    created_at: str | None = None
    content_text: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


# moved to memory_empire.memory_service.auth.require_api_key

def create_app(db: DB):
    app = FastAPI(title="Memory Empire - Memory Service", version="0.1.0")

    # shared clients
    embedder = build_embedder(st_model=settings.st_model, dim=settings.embedding_dim)
    qdrant = build_client(
        QdrantOpsConfig(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection=settings.qdrant_collection,
            dim=embedder.dim,
        )
    )

    @app.get("/health")
    async def health():
        return {"ok": True, "host": os.uname().nodename}

    # Graph endpoints (Neo4j-backed). If optional deps/config are missing we keep service up.
    if build_graph_router is not None:
        try:
            app.include_router(build_graph_router())
        except Exception:
            pass

    @app.post("/v1/events")
    async def capture_event(payload: EventIn, _auth: None = Depends(require_api_key)):
        event = payload.model_dump()
        event_id = await db.insert_event(event=event)

        # enqueue for embedding (best effort)
        try:
            import redis.asyncio as redis

            r = redis.from_url(settings.redis_url)
            await r.lpush(settings.queue_name, event_id)
            await r.close()
        except Exception:
            pass

        return {"id": event_id, "queued": True}

    @app.get("/v1/events/{event_id}")
    async def get_event(event_id: str, _auth: None = Depends(require_api_key)):
        ev = await db.get_event(event_id=event_id)
        if not ev:
            raise HTTPException(status_code=404, detail="not found")
        return ev

    @app.get("/v1/search")
    async def search(
        q: str,
        limit: int = 10,
        _auth: None = Depends(require_api_key),
    ):
        # Lexical (fast, reliable)
        fts = await db.search_fts(query=q, limit=max(limit, 10))

        # Vector search (optional; stub embedder still works but is lower quality)
        vec_hits: list[dict[str, Any]] = []
        try:
            from .qdrant_ops import search as q_search

            qv = embedder.embed([q])[0]
            vec_hits = q_search(qdrant, collection=settings.qdrant_collection, vector=qv, limit=max(limit, 10))
        except Exception:
            vec_hits = []

        # Merge + score
        merged: dict[str, SearchOut] = {}

        for r in fts:
            merged[r["id"]] = SearchOut(
                id=r["id"],
                score=float(r["score"]),
                kind=r["kind"],
                source=r.get("source"),
                event_type=r.get("event_type"),
                actor_id=r.get("actor_id"),
                thread_id=r.get("thread_id"),
                created_at=r.get("created_at"),
                content_text=r.get("content_text"),
            )

        for h in vec_hits:
            _id = h["id"]
            payload = h.get("payload") or {}
            if _id in merged:
                merged[_id].score = max(merged[_id].score, float(h["score"]))
                merged[_id].payload.update(payload)
                merged[_id].kind = "hybrid"
            else:
                merged[_id] = SearchOut(
                    id=_id,
                    score=float(h["score"]),
                    kind=h.get("kind") or "vector",
                    source=payload.get("source"),
                    event_type=payload.get("event_type"),
                    actor_id=payload.get("actor_id"),
                    thread_id=payload.get("thread_id"),
                    created_at=payload.get("created_at"),
                    content_text=payload.get("content_text"),
                    payload=payload,
                )

        out = sorted(merged.values(), key=lambda x: x.score, reverse=True)[:limit]
        return {"query": q, "count": len(out), "results": [o.model_dump() for o in out]}

    return app
