from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional


class VectorStore:
    def upsert(
        self,
        *,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
        namespace: str,
        model_name: str,
    ) -> None:
        raise NotImplementedError


@dataclass
class JSONLVectorStore(VectorStore):
    base_dir: str

    def upsert(
        self,
        *,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
        namespace: str,
        model_name: str,
    ) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        path = os.path.join(self.base_dir, f"{namespace}__{model_name}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for i, _id in enumerate(ids):
                f.write(
                    json.dumps(
                        {
                            "id": _id,
                            "vector": vectors[i],
                            "metadata": metadatas[i],
                            "document": documents[i],
                            "namespace": namespace,
                            "model": model_name,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: str):
        try:
            import chromadb
        except Exception as e:
            raise RuntimeError("chromadb not installed; pip install chromadb") from e
        self._client = chromadb.PersistentClient(path=persist_dir)

    def _collection_name(self, namespace: str, model_name: str) -> str:
        # Chroma collection naming restrictions: keep simple.
        return f"{namespace}__{model_name}".replace(":", "_")

    def upsert(
        self,
        *,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
        namespace: str,
        model_name: str,
    ) -> None:
        col = self._client.get_or_create_collection(self._collection_name(namespace, model_name))
        col.upsert(ids=ids, embeddings=vectors, metadatas=metadatas, documents=documents)


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store.

    Collection naming: {namespace}__{model_name}
    """

    def __init__(self, *, url: str, api_key: str | None = None):
        from memory_empire.qdrant.client import QdrantConfig, build_qdrant_client

        self._client = build_qdrant_client(QdrantConfig(url=url, api_key=api_key))

    def _collection_name(self, namespace: str, model_name: str) -> str:
        return f"{namespace}__{model_name}".replace(":", "_")

    def _ensure_collection(self, name: str, dim: int) -> None:
        try:
            from qdrant_client.http.models import Distance, VectorParams
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "qdrant-client is not installed. Install with: pip install qdrant-client"
            ) from e

        if self._client.collection_exists(name):
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def upsert(
        self,
        *,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
        namespace: str,
        model_name: str,
    ) -> None:
        if not vectors:
            return

        col = self._collection_name(namespace, model_name)
        self._ensure_collection(col, dim=len(vectors[0]))

        payloads: list[dict[str, Any]] = []
        for i in range(len(ids)):
            payload = dict(metadatas[i] or {})
            payload["document"] = documents[i]
            payloads.append(payload)

        # qdrant-client expects a list of PointStruct-like objects (dicts are fine).
        # Using tuples (e.g. zip(ids, vectors, payloads)) fails validation on newer clients.
        import uuid

        points: list[dict[str, Any]] = []
        for i in range(len(ids)):
            # Qdrant point IDs must be either an unsigned integer or a UUID.
            # Our internal ids are stable chunk keys like "<sha256>:0000", so we map
            # them to a deterministic UUIDv5 and keep the original id in payload.
            stable_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, ids[i]))

            payload = dict(payloads[i] or {})
            payload["chunk_pk"] = ids[i]

            points.append(
                {
                    "id": stable_uuid,
                    "vector": vectors[i],
                    "payload": payload,
                }
            )

        self._client.upsert(
            collection_name=col,
            points=points,
        )


def build_vector_store(vector_dir: Optional[str]) -> VectorStore:
    """Factory for the configured vector store.

    Selection order:
    1) MEMORY_EMPIRE_VECTOR_STORE=qdrant|chroma|jsonl
    2) If vector_dir provided: try chroma; else jsonl
    """

    choice = (os.getenv("MEMORY_EMPIRE_VECTOR_STORE") or "").strip().lower()
    if choice == "qdrant":
        url = os.getenv("MEMORY_EMPIRE_QDRANT_URL") or "http://localhost:6333"
        api_key = os.getenv("MEMORY_EMPIRE_QDRANT_API_KEY")
        return QdrantVectorStore(url=url, api_key=api_key)

    if choice == "chroma":
        if not vector_dir:
            raise ValueError("vector_dir is required for chroma vector store")
        return ChromaVectorStore(persist_dir=vector_dir)

    if choice == "jsonl":
        return JSONLVectorStore(base_dir=vector_dir or "vector_indexing/data/jsonl_vectors")

    # Default behavior.
    if not vector_dir:
        return JSONLVectorStore(base_dir="vector_indexing/data/jsonl_vectors")

    try:
        return ChromaVectorStore(persist_dir=vector_dir)
    except Exception:
        return JSONLVectorStore(base_dir=vector_dir)
