from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg


def compute_content_hash(event: dict[str, Any]) -> str:
    payload = {
        "source": event.get("source"),
        "event_type": event.get("event_type"),
        "actor_id": event.get("actor_id"),
        "thread_id": event.get("thread_id"),
        "content_text": event.get("content_text"),
        "content_json": event.get("content_json") or {},
        "observed_at": event.get("observed_at"),
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class DB:
    pool: asyncpg.Pool

    @classmethod
    async def connect(cls, dsn: str) -> "DB":
        pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=10)
        return cls(pool=pool)

    async def close(self) -> None:
        await self.pool.close()

    def _parse_ts(self, v: Any) -> datetime | None:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            s = v.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(s)
            except Exception:
                return None
        return None

    async def insert_event(self, *, event: dict[str, Any]) -> str:
        content_hash = compute_content_hash(event)
        created_at = self._parse_ts(event.get("created_at")) or datetime.utcnow()
        observed_at = self._parse_ts(event.get("observed_at"))

        q = """
        INSERT INTO memory_events(
            source, event_type, actor_id, thread_id,
            created_at, observed_at,
            content_text, content_json, tags, sensitivity, content_hash
        )
        VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
        ON CONFLICT(content_hash) DO UPDATE SET content_hash=excluded.content_hash
        RETURNING id::text
        """

        tags = event.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]

        async with self.pool.acquire() as con:
            try:
                _id = await con.fetchval(
                    q,
                    event.get("source"),
                    event.get("event_type"),
                    event.get("actor_id"),
                    event.get("thread_id"),
                    created_at,
                    observed_at,
                    event.get("content_text"),
                    json.dumps(event.get("content_json") or {}, ensure_ascii=False),
                    tags,
                    event.get("sensitivity") or "internal",
                    content_hash,
                )
                return str(_id)
            except asyncpg.exceptions.UniqueViolationError:
                # best effort fallback (should be handled by ON CONFLICT)
                existing = await con.fetchval(
                    "SELECT id::text FROM memory_events WHERE content_hash=$1",
                    content_hash,
                )
                return str(existing)

    async def set_embedding(self, *, event_id: str, embedding: list[float]) -> None:
        q = "UPDATE memory_events SET embedding=$2::vector, indexed_at=now() WHERE id=$1::uuid"
        async with self.pool.acquire() as con:
            await con.execute(q, event_id, embedding)

    async def get_event(self, *, event_id: str) -> dict[str, Any] | None:
        q = """
        SELECT id::text, source, event_type, actor_id, thread_id,
               created_at::text, observed_at::text, content_text,
               content_json::text, tags, sensitivity
        FROM memory_events WHERE id=$1::uuid
        """
        async with self.pool.acquire() as con:
            row = await con.fetchrow(q, event_id)
            if not row:
                return None
            return {
                "id": row["id"],
                "source": row["source"],
                "event_type": row["event_type"],
                "actor_id": row["actor_id"],
                "thread_id": row["thread_id"],
                "created_at": row["created_at"],
                "observed_at": row["observed_at"],
                "content_text": row["content_text"],
                "content_json": json.loads(row["content_json"] or "{}"),
                "tags": list(row["tags"] or []),
                "sensitivity": row["sensitivity"],
            }

    async def search_fts(self, *, query: str, limit: int = 20) -> list[dict[str, Any]]:
        q = """
        SELECT id::text, source, event_type, actor_id, thread_id,
               created_at::text, content_text,
               ts_rank_cd(content_tsv, plainto_tsquery('english', $1)) AS rank
        FROM memory_events
        WHERE content_tsv @@ plainto_tsquery('english', $1)
        ORDER BY rank DESC, created_at DESC
        LIMIT $2
        """
        async with self.pool.acquire() as con:
            rows = await con.fetch(q, query, limit)
            return [
                {
                    "id": r["id"],
                    "score": float(r["rank"] or 0.0),
                    "source": r["source"],
                    "event_type": r["event_type"],
                    "actor_id": r["actor_id"],
                    "thread_id": r["thread_id"],
                    "created_at": r["created_at"],
                    "content_text": r["content_text"],
                    "kind": "fts",
                }
                for r in rows
            ]
