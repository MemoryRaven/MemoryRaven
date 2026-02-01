from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QdrantOpsConfig:
    url: str
    api_key: str | None
    collection: str
    dim: int


def build_client(cfg: QdrantOpsConfig):
    from qdrant_client import QdrantClient

    # Our stack pins server close to LTS; client may be slightly newer.
    # We intentionally disable strict version check to avoid boot loops.
    return QdrantClient(
        url=cfg.url,
        api_key=cfg.api_key,
        timeout=10,
        check_compatibility=False,
    )


def ensure_collection(client, *, name: str, dim: int) -> None:
    from qdrant_client.http.models import Distance, VectorParams

    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def upsert_event_embedding(
    client,
    *,
    collection: str,
    event_id: str,
    vector: list[float],
    payload: dict[str, Any],
) -> None:
    ensure_collection(client, name=collection, dim=len(vector))

    # Qdrant IDs must be int or UUID. Our event_id is UUID string.
    client.upsert(
        collection_name=collection,
        points=[{"id": event_id, "vector": vector, "payload": payload}],
    )


def search(client, *, collection: str, vector: list[float], limit: int = 10) -> list[dict[str, Any]]:
    ensure_collection(client, name=collection, dim=len(vector))
    hits = client.search(collection_name=collection, query_vector=vector, limit=limit)
    out: list[dict[str, Any]] = []
    for h in hits:
        out.append(
            {
                "id": str(h.id),
                "score": float(h.score),
                "payload": h.payload or {},
                "kind": "vector",
            }
        )
    return out
