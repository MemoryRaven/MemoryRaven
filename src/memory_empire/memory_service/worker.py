from __future__ import annotations

import asyncio
import logging

import redis.asyncio as redis

from .db import DB
from .embedder import build_embedder
from .qdrant_ops import QdrantOpsConfig, build_client, upsert_event_embedding
from .settings import settings

logger = logging.getLogger(__name__)


async def run_worker() -> None:
    logging.basicConfig(level=(settings.log_level or "INFO"))

    db = await DB.connect(settings.postgres_dsn)
    r = redis.from_url(settings.redis_url)

    embedder = build_embedder(st_model=settings.st_model, dim=settings.embedding_dim)
    qdrant = build_client(
        QdrantOpsConfig(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection=settings.qdrant_collection,
            dim=embedder.dim,
        )
    )

    try:
        while True:
            # BLPOP returns (key, value)
            item = await r.brpop(settings.queue_name, timeout=5)
            if not item:
                continue
            _key, event_id_b = item
            event_id = event_id_b.decode("utf-8") if isinstance(event_id_b, (bytes, bytearray)) else str(event_id_b)

            ev = await db.get_event(event_id=event_id)
            if not ev:
                continue

            text = ev.get("content_text") or ""
            if not text.strip():
                continue

            vec = embedder.embed([text])[0]

            # 1) store in pgvector for local ANN
            try:
                await db.set_embedding(event_id=event_id, embedding=vec)
            except Exception as e:
                logger.warning("Failed to set pgvector embedding for %s: %s", event_id, e)

            # 2) upsert to Qdrant for fast vector retrieval
            try:
                payload = {
                    "source": ev.get("source"),
                    "event_type": ev.get("event_type"),
                    "actor_id": ev.get("actor_id"),
                    "thread_id": ev.get("thread_id"),
                    "created_at": ev.get("created_at"),
                    "content_text": (text[:2000] if text else ""),
                    "tags": ev.get("tags") or [],
                }
                upsert_event_embedding(
                    qdrant,
                    collection=settings.qdrant_collection,
                    event_id=event_id,
                    vector=vec,
                    payload=payload,
                )
            except Exception as e:
                logger.warning("Failed to upsert qdrant embedding for %s: %s", event_id, e)

    finally:
        await r.close()
        await db.close()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
