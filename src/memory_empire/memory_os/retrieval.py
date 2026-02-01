"""
Memory retrieval and search functionality
Smart retrieval across hot/warm tiers with reranking
"""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A memory search result"""

    event_id: str
    score: float
    source: str
    event_type: str
    content_text: str
    content_json: dict
    created_at: str
    metadata: dict | None = None

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "score": self.score,
            "source": self.source,
            "event_type": self.event_type,
            "content_text": self.content_text,
            "content_json": self.content_json,
            "created_at": self.created_at,
            "metadata": self.metadata or {},
        }


class MemoryRetrieval:
    """Advanced retrieval capabilities for the memory system"""

    def __init__(self, memory_bridge):
        self.bridge = memory_bridge
        self.conn = memory_bridge.conn

    def search(
        self,
        query: str,
        limit: int = 20,
        time_range: tuple[str, str] = None,
        sources: list[str] = None,
        event_types: list[str] = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Unified search across all memory tiers
        Combines semantic, lexical, and temporal search
        """
        results = []

        # 1. Semantic search (if embedder available)
        if self.bridge.embedder:
            semantic_results = self._semantic_search(query, limit * 2)
            results.extend(semantic_results)

        # 2. Lexical search (FTS)
        lexical_results = self._lexical_search(query, limit * 2)
        results.extend(lexical_results)

        # 3. Merge and deduplicate
        seen_ids = set()
        deduped_results = []
        for result in results:
            if result.event_id not in seen_ids:
                seen_ids.add(result.event_id)
                deduped_results.append(result)

        # 4. Apply filters
        filtered = self._apply_filters(
            deduped_results,
            time_range=time_range,
            sources=sources,
            event_types=event_types,
            min_score=min_score,
        )

        # 5. Rerank and limit
        reranked = self._rerank_results(filtered, query)

        return reranked[:limit]

    def _semantic_search(self, query: str, limit: int) -> list[SearchResult]:
        """Semantic search using embeddings"""
        results = []

        # Encode query
        query_embedding_raw = self.bridge.embedder.encode(query)
        if hasattr(query_embedding_raw, "tolist"):
            query_embedding = query_embedding_raw.tolist()
        else:
            query_embedding = list(query_embedding_raw)

        # Search in local SQLite first (hot tier)
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, source, event_type, content_text, content_json, 
                   created_at, embeddings
            FROM events 
            WHERE embeddings IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1000
        """)

        local_results = []
        for row in cursor.fetchall():
            (
                event_id,
                source,
                event_type,
                content_text,
                content_json,
                created_at,
                embeddings_blob,
            ) = row

            # Calculate cosine similarity (no numpy required)
            from array import array

            a = array("f")
            a.frombytes(embeddings_blob)
            embeddings = a.tolist()

            # dot + norms
            dot = 0.0
            q_norm = 0.0
            e_norm = 0.0
            for qv, ev in zip(query_embedding, embeddings):
                dot += float(qv) * float(ev)
                q_norm += float(qv) * float(qv)
                e_norm += float(ev) * float(ev)
            denom = (q_norm**0.5) * (e_norm**0.5)
            similarity = (dot / denom) if denom else 0.0

            if similarity > 0.3:  # Threshold
                local_results.append(
                    SearchResult(
                        event_id=event_id,
                        score=float(similarity),
                        source=source,
                        event_type=event_type,
                        content_text=content_text,
                        content_json=json.loads(content_json),
                        created_at=created_at,
                        metadata={"search_type": "semantic_local"},
                    )
                )

        # Sort by score
        local_results.sort(key=lambda x: x.score, reverse=True)
        results.extend(local_results[:limit])

        # Search in Pinecone (warm tier) if available
        if self.bridge.pinecone_enabled:
            try:
                pinecone_results = self.bridge.index.query(
                    query_embedding, top_k=limit, include_metadata=True
                )

                for match in pinecone_results["matches"]:
                    # Fetch full event from SQLite
                    cursor.execute(
                        "SELECT source, event_type, content_text, content_json, created_at "
                        "FROM events WHERE id = ?",
                        (match["id"],),
                    )
                    row = cursor.fetchone()
                    if row:
                        source, event_type, content_text, content_json, created_at = row
                        results.append(
                            SearchResult(
                                event_id=match["id"],
                                score=match["score"],
                                source=source,
                                event_type=event_type,
                                content_text=content_text,
                                content_json=json.loads(content_json),
                                created_at=created_at,
                                metadata={"search_type": "semantic_pinecone"},
                            )
                        )
            except Exception as e:
                logger.error(f"Pinecone search failed: {e}")

        return results

    def _lexical_search(self, query: str, limit: int) -> list[SearchResult]:
        """Full-text search using SQLite FTS.

        Notes:
        - SQLite FTS5 has its own query syntax; raw user text may contain punctuation
          (e.g. '?', ':', quotes) that can cause syntax errors.
        - We sanitize into a token-only query and strip common "memory trigger" words
          like "remember" that are usually not present in stored content.
        """
        cursor = self.conn.cursor()

        import re

        stop = {
            "remember",
            "remind",
            "please",
            "could",
            "would",
            "what",
            "did",
            "we",
            "i",
            "me",
            "my",
            "about",
            "the",
            "a",
            "an",
        }

        tokens = re.findall(r"[a-z0-9]+", (query or "").lower())
        tokens = [t for t in tokens if t not in stop]
        q = " ".join(tokens)
        if not q:
            return []

        # FTS search
        cursor.execute(
            """
            SELECT e.id, e.source, e.event_type, e.content_text, 
                   e.content_json, e.created_at,
                   rank
            FROM events e
            JOIN events_fts ON e.rowid = events_fts.rowid
            WHERE events_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """,
            (q, limit),
        )

        results = []
        for row in cursor.fetchall():
            event_id, source, event_type, content_text, content_json, created_at, rank = row

            # Convert rank to normalized score
            score = 1.0 / (1.0 + abs(rank))

            results.append(
                SearchResult(
                    event_id=event_id,
                    score=score,
                    source=source,
                    event_type=event_type,
                    content_text=content_text,
                    content_json=json.loads(content_json),
                    created_at=created_at,
                    metadata={"search_type": "lexical"},
                )
            )

        return results

    def _apply_filters(self, results: list[SearchResult], **filters) -> list[SearchResult]:
        """Apply filters to search results"""
        filtered = results

        # Time range filter
        if filters.get("time_range"):
            start, end = filters["time_range"]
            filtered = [r for r in filtered if start <= r.created_at <= end]

        # Source filter
        if filters.get("sources"):
            filtered = [r for r in filtered if r.source in filters["sources"]]

        # Event type filter
        if filters.get("event_types"):
            filtered = [r for r in filtered if r.event_type in filters["event_types"]]

        # Min score filter
        if filters.get("min_score"):
            filtered = [r for r in filtered if r.score >= filters["min_score"]]

        return filtered

    def _rerank_results(self, results: list[SearchResult], query: str) -> list[SearchResult]:
        """Rerank results using multiple signals"""
        # Simple reranking based on:
        # - Original score
        # - Recency boost
        # - Source/type relevance

        now = datetime.now(UTC)

        for result in results:
            # Recency boost
            created = datetime.fromisoformat(result.created_at.replace("Z", "+00:00"))
            age_days = (now - created).days
            recency_boost = 1.0 / (1.0 + age_days / 30.0)  # Decay over 30 days

            # Source boost (configurable)
            source_boost = {"decision": 1.2, "code": 1.1, "telegram": 1.0, "web": 0.9}.get(
                result.source, 1.0
            )

            # Combine scores
            result.score = result.score * recency_boost * source_boost

        # Sort by final score
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def get_timeline(
        self, start: str, end: str = None, sources: list[str] = None, limit: int = 100
    ) -> list[dict]:
        """Get events in chronological order"""
        if not end:
            end = datetime.now(UTC).isoformat()

        query = """
            SELECT id, source, event_type, content_text, content_json, created_at
            FROM events
            WHERE created_at BETWEEN ? AND ?
        """
        params: list[object] = [start, end]

        if sources:
            placeholders = ",".join("?" * len(sources))
            query += f" AND source IN ({placeholders})"
            params.extend(sources)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.cursor()
        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "event_id": row[0],
                    "source": row[1],
                    "event_type": row[2],
                    "content_text": row[3],
                    "content_json": json.loads(row[4]),
                    "created_at": row[5],
                }
            )

        return results

    def get_context(self, event_id: str, window_size: int = 10) -> dict:
        """Get surrounding context for an event"""
        cursor = self.conn.cursor()

        # Get the target event
        cursor.execute("SELECT thread_id, created_at FROM events WHERE id = ?", (event_id,))
        row = cursor.fetchone()
        if not row:
            return {"error": "Event not found"}

        thread_id, created_at = row

        # Get events before
        cursor.execute(
            """
            SELECT id, source, event_type, content_text, created_at
            FROM events
            WHERE thread_id = ? AND created_at < ?
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (thread_id, created_at, window_size // 2),
        )

        before = list(reversed(cursor.fetchall()))

        # Get events after
        cursor.execute(
            """
            SELECT id, source, event_type, content_text, created_at
            FROM events
            WHERE thread_id = ? AND created_at > ?
            ORDER BY created_at ASC
            LIMIT ?
        """,
            (thread_id, created_at, window_size // 2),
        )

        after = cursor.fetchall()

        # Combine
        context_events = []
        for row in before + after:
            context_events.append(
                {
                    "event_id": row[0],
                    "source": row[1],
                    "event_type": row[2],
                    "content_text": row[3],
                    "created_at": row[4],
                }
            )

        return {"event_id": event_id, "thread_id": thread_id, "context": context_events}

    def find_related_entities(self, entity_name: str) -> list[dict]:
        """Find events related to an entity"""
        cursor = self.conn.cursor()

        # Find entity
        cursor.execute(
            "SELECT entity_id FROM entities WHERE name = ? OR aliases LIKE ?",
            (entity_name, f"%{entity_name}%"),
        )
        row = cursor.fetchone()
        if not row:
            return []

        entity_id = row[0]

        # Find mentions
        cursor.execute(
            """
            SELECT e.id, e.source, e.event_type, e.content_text, 
                   e.created_at, em.confidence, em.context
            FROM events e
            JOIN entity_mentions em ON e.id = em.event_id
            WHERE em.entity_id = ?
            ORDER BY e.created_at DESC
        """,
            (entity_id,),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "event_id": row[0],
                    "source": row[1],
                    "event_type": row[2],
                    "content_text": row[3],
                    "created_at": row[4],
                    "confidence": row[5],
                    "mention_context": row[6],
                }
            )

        return results
