"""claude_mem.memory_os

Memory-OS Layer for Clawdbot (MemGPT-inspired)

Goal: transparent memory virtualization that:
- decides when to retrieve
- routes to namespaces
- performs hybrid retrieval (vector + keyword + lightweight reranking)
- compresses retrieved memories into an injection block
- optionally captures/consolidates memories in background

This module is designed to be *invisible* to the user:
call `MemoryOS.virtualize(messages, ...)` before the LLM call.

No hard dependency on an LLM is assumed here; compression is heuristic.
"""

from __future__ import annotations

import glob
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from .core import Event, MemoryBridge
from .retrieval import MemoryRetrieval, SearchResult

Role = Literal["system", "developer", "user", "assistant", "tool"]


# --------------------------
# Policies / configuration
# --------------------------


@dataclass
class MemoryPolicy:
    """Controls how much and what gets stored / retrieved / injected."""

    # Retrieval
    enabled: bool = True
    min_user_chars_for_retrieval: int = 12
    inject_max_chars: int = 3500
    inject_max_items: int = 12
    min_score: float = 0.15

    # Capture
    autocapture_user_messages: bool = False
    autocapture_assistant_messages: bool = False
    autocapture_min_chars: int = 80

    # Consolidation
    consolidate_enabled: bool = True
    consolidate_older_than_days: int = 14
    consolidate_max_events_per_thread: int = 80
    consolidation_summary_max_chars: int = 1200


@dataclass
class TriggerDecision:
    should_retrieve: bool
    reasons: list[str] = field(default_factory=list)
    query: str = ""
    namespaces: list[str] = field(default_factory=list)
    limit: int = 8


# --------------------------
# Namespaces / routing
# --------------------------


DEFAULT_NAMESPACES = [
    "personal",
    "web",
    "academic",
    "code",
    "ops",
]


class MemoryRouter:
    """Routes a query to relevant namespaces."""

    def route(self, user_text: str, messages: Sequence[dict[str, Any]] | None = None) -> list[str]:
        t = (user_text or "").lower()

        # Strong hints
        if any(
            k in t for k in ["paper", "citation", "arxiv", "journal", "theorem", "proof", "derive"]
        ):
            return ["academic", "web"]
        if any(k in t for k in ["docs", "documentation", "api", "reference", "snippet"]):
            return ["web", "code"]
        if any(
            k in t
            for k in [
                "deploy",
                "prod",
                "incident",
                "error",
                "stack trace",
                "logs",
                "gateway",
                "node",
            ]
        ):
            return ["ops", "code"]
        if any(
            k in t
            for k in [
                "remember",
                "last time",
                "as we discussed",
                "you said",
                "my",
                "i prefer",
                "about me",
            ]
        ):
            return ["personal", "ops", "code"]

        # Default: broad
        return ["personal", "code", "ops"]


# --------------------------
# Retrieval triggers
# --------------------------


class RetrievalTrigger:
    """Heuristic trigger for when to retrieve memories.

    MemGPT-style: retrieve when the user likely refers to prior context,
    preferences, decisions, or when the query is underspecified.
    """

    _ref_patterns = [
        r"\bremember\b",
        r"\bremind me\b",
        r"\bwhat did (we|i)\b",
        r"\b(last|previous) (time|week|month|year)\b",
        r"\bas we discussed\b",
        r"\byou said\b",
        r"\bwe decided\b",
        r"\bmy preference\b",
        r"\bmy favorite\b",
        r"\bmy (name|email|phone|address)\b",
        r"\bwhere did we\b",
        r"\bwhat was the plan\b",
    ]

    def decide(
        self,
        user_text: str,
        messages: Sequence[dict[str, Any]] | None,
        policy: MemoryPolicy,
        router: MemoryRouter,
    ) -> TriggerDecision:
        user_text = (user_text or "").strip()
        reasons: list[str] = []

        if not policy.enabled:
            return TriggerDecision(False, ["policy.disabled"], user_text)

        if len(user_text) < policy.min_user_chars_for_retrieval:
            return TriggerDecision(False, ["query.too_short"], user_text)

        t = user_text.lower()

        if any(re.search(p, t) for p in self._ref_patterns):
            reasons.append("explicit_reference")

        # Underspecified tasks often benefit from memory (e.g. "do it again")
        if any(
            k in t for k in ["again", "same as before", "like last time", "continue", "pick up"]
        ):
            reasons.append("underspecified")

        # If conversation contains open loops / references to earlier things
        if messages:
            # Look for mentions of files, ids, or prior decisions in the last assistant turn
            last_assistant = next(
                (m for m in reversed(messages) if m.get("role") == "assistant"), None
            )
            if last_assistant:
                lt = (last_assistant.get("content") or "").lower()
                if any(
                    k in lt
                    for k in ["as discussed", "earlier", "previously", "we decided", "recall"]
                ):
                    reasons.append("conversation_reference")

        # If asking about preferences / identity
        if any(
            k in t
            for k in ["my", "me", "i like", "i prefer", "my setup", "my machine", "my account"]
        ):
            reasons.append("personal_context")

        # Default if query is long / complex: try retrieval for grounding.
        if len(user_text) > 220:
            reasons.append("long_query")

        should = len(reasons) > 0
        namespaces = router.route(user_text, messages)

        limit = 10 if "long_query" in reasons else 8
        return TriggerDecision(should, reasons, user_text, namespaces, limit)


# --------------------------
# Hybrid retrieval + rerank
# --------------------------


class HybridRetriever:
    """Vector + keyword retrieval with lightweight fusion and reranking."""

    def __init__(self, bridge: MemoryBridge):
        self.bridge = bridge
        self.retrieval = MemoryRetrieval(bridge)

    def retrieve(
        self,
        query: str,
        namespaces: Sequence[str] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        # Route namespaces via tags filter: namespace stored as tag "ns:<name>".
        # If namespaces omitted, search across everything.
        results = self.retrieval.search(query=query, limit=max(limit * 3, 30), min_score=min_score)

        if namespaces:
            allowed = set(namespaces)
            filtered: list[SearchResult] = []
            for r in results:
                tags = (r.content_json or {}).get("tags")
                # tags may exist in content_json, but in our DB, tags are column too.
                # We use heuristic: check content_text prefix or metadata.
                # Best effort: allow if tag present in content_json.tags or metadata.
                ns = _infer_namespace(r)
                if ns in allowed:
                    filtered.append(r)
            results = filtered

        # Rerank with query-term overlap + recency + original score.
        return self._rerank_fusion(query, results)[:limit]

    def _rerank_fusion(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        q_terms = _tokenize(query)
        now = datetime.now(UTC)

        for r in results:
            doc = r.content_text or ""
            d_terms = _tokenize(doc)

            overlap = 0.0
            if q_terms:
                overlap = len(q_terms.intersection(d_terms)) / max(1, len(q_terms))

            # recency decay ~ 90 days
            try:
                created = datetime.fromisoformat((r.created_at or "").replace("Z", "+00:00"))
                age_days = max(0.0, (now - created).total_seconds() / 86400.0)
            except Exception:
                age_days = 9999.0
            recency = 1.0 / (1.0 + age_days / 90.0)

            # namespace boost
            ns = _infer_namespace(r)
            ns_boost = {
                "personal": 1.15,
                "code": 1.05,
                "ops": 1.05,
                "academic": 1.0,
                "web": 0.95,
            }.get(ns, 1.0)

            r.score = float(r.score) * 0.65 + overlap * 0.25 + recency * 0.10
            r.score *= ns_boost

        results.sort(key=lambda x: x.score, reverse=True)
        return results


# --------------------------
# Compression / injection
# --------------------------


@dataclass
class InjectedMemory:
    header: str
    items: list[dict[str, Any]]
    text: str


class ContextualCompressor:
    """Compress retrieved memories to fit into prompt budget.

    Heuristic (non-LLM) compression:
    - keep top-k
    - trim content
    - format into a compact reference block
    """

    def compress(
        self,
        query: str,
        results: Sequence[SearchResult],
        max_items: int,
        max_chars: int,
    ) -> InjectedMemory:
        items: list[dict[str, Any]] = []

        for r in results[:max_items]:
            ns = _infer_namespace(r)
            preview = _squash_ws(r.content_text or "")
            preview = _truncate(preview, 420)
            items.append(
                {
                    "id": r.event_id,
                    "score": round(float(r.score), 4),
                    "ns": ns,
                    "source": r.source,
                    "type": r.event_type,
                    "created_at": r.created_at,
                    "text": preview,
                }
            )

        header = "MEMORY_CONTEXT (retrieved; may be partial; use as hints, not ground truth)"

        # Build injection block; cut to max_chars.
        lines = [header, f"Query: {query}", ""]
        for i, it in enumerate(items, 1):
            lines.append(
                f"[{i}] ({it['ns']}/{it['source']}/{it['type']}; {it['created_at']}; score={it['score']}) {it['text']}"
            )

        text = "\n".join(lines)
        if len(text) > max_chars:
            # Aggressive truncation: reduce items then trim overall
            while len(items) > 1 and len(text) > max_chars:
                items.pop()
                lines = [header, f"Query: {query}", ""]
                for i, it in enumerate(items, 1):
                    lines.append(
                        f"[{i}] ({it['ns']}/{it['source']}/{it['type']}; {it['created_at']}; score={it['score']}) {it['text']}"
                    )
                text = "\n".join(lines)
            text = text[:max_chars]

        return InjectedMemory(header=header, items=items, text=text)


# --------------------------
# Memory virtualization layer
# --------------------------


class MemoryOS:
    """High-level interface.

    Usage:
        bridge = MemoryBridge(...)
        memos = MemoryOS(bridge)
        augmented_messages, meta = memos.virtualize(messages)

    `messages` is expected to be OpenAI/Anthropic style list of dicts:
      {"role": "user"|"assistant"|..., "content": "..."}

    The returned messages include an extra system message containing the
    compressed memory context.
    """

    def __init__(
        self,
        bridge: MemoryBridge,
        policy: MemoryPolicy | None = None,
        router: MemoryRouter | None = None,
        trigger: RetrievalTrigger | None = None,
        retriever: HybridRetriever | None = None,
        compressor: ContextualCompressor | None = None,
        workspace_root: str | None = None,
    ):
        self.bridge = bridge
        self.policy = policy or MemoryPolicy()
        self.router = router or MemoryRouter()
        self.trigger = trigger or RetrievalTrigger()
        self.retriever = retriever or HybridRetriever(bridge)
        self.compressor = compressor or ContextualCompressor()
        self.workspace_root = Path(workspace_root or os.getcwd())

    def virtualize(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        user_text: str | None = None,
        inject_role: Role = "system",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Intercept context before an LLM call.

        Returns (augmented_messages, metadata)
        """

        msgs = list(messages or [])

        if user_text is None:
            user_text = self._last_user_text(msgs)

        decision = self.trigger.decide(user_text, msgs, self.policy, self.router)
        meta: dict[str, Any] = {
            "trigger": {
                "should_retrieve": decision.should_retrieve,
                "reasons": decision.reasons,
                "namespaces": decision.namespaces,
                "limit": decision.limit,
            }
        }

        if not decision.should_retrieve:
            return msgs, meta

        results = self.retriever.retrieve(
            query=decision.query,
            namespaces=decision.namespaces,
            limit=decision.limit,
            min_score=self.policy.min_score,
        )

        injected = self.compressor.compress(
            query=decision.query,
            results=results,
            max_items=self.policy.inject_max_items,
            max_chars=self.policy.inject_max_chars,
        )

        meta["retrieval"] = {
            "count": len(results),
            "injected_count": len(injected.items),
            "top_ids": [r.event_id for r in results[:5]],
        }

        # Insert injection near the top (after any existing system messages)
        inject_msg = {"role": inject_role, "content": injected.text}
        insert_at = 0
        while insert_at < len(msgs) and msgs[insert_at].get("role") == "system":
            insert_at += 1

        augmented = msgs[:insert_at] + [inject_msg] + msgs[insert_at:]
        return augmented, meta

    def autocapture_turn(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        channel: str = "clawdbot",
        thread_id: str | None = None,
        actor_id: str | None = None,
        namespace: str = "personal",
    ) -> list[str]:
        """Optional: capture long user/assistant messages into the memory DB."""

        ids: list[str] = []
        if not messages:
            return ids

        for m in messages:
            role = m.get("role")
            content = m.get("content") or ""
            if len(content) < self.policy.autocapture_min_chars:
                continue

            if role == "user" and not self.policy.autocapture_user_messages:
                continue
            if role == "assistant" and not self.policy.autocapture_assistant_messages:
                continue

            eid = self.bridge.capture(
                Event(
                    source=channel,
                    event_type=f"chat_{role}",
                    thread_id=thread_id,
                    actor_id=actor_id,
                    content_text=content,
                    content_json={
                        "role": role,
                        "content": content,
                        "ns": namespace,
                        "tags": [f"ns:{namespace}", "autocapture"],
                    },
                    tags=[f"ns:{namespace}", "autocapture"],
                )
            )
            if eid:
                ids.append(eid)

        return ids

    def ingest_clawdbot_memory_files(
        self,
        *,
        memory_dir: str | None = None,
        long_term_file: str | None = None,
        namespace: str = "personal",
    ) -> dict[str, Any]:
        """Integration point: ingest existing Clawdbot memory markdown files.

        - memory/YYYY-MM-DD.md (daily raw logs)
        - MEMORY.md (curated long-term)

        Stores them as events with tags ns:<namespace> and source='clawdbot_file'.
        """

        root = self.workspace_root
        memory_dir = memory_dir or str(root / "memory")
        long_term_file = long_term_file or str(root / "MEMORY.md")

        created: list[str] = []
        skipped: int = 0

        # Daily files
        for p in sorted(glob.glob(os.path.join(memory_dir, "*.md"))):
            text = Path(p).read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue
            eid = self.bridge.capture(
                Event(
                    source="clawdbot_file",
                    event_type="daily_memory_md",
                    content_text=text[:2000],
                    content_json={
                        "path": p,
                        "content": text,
                        "ns": namespace,
                        "tags": [f"ns:{namespace}", "ingest:clawdbot"],
                    },
                    tags=[f"ns:{namespace}", "ingest:clawdbot"],
                )
            )
            if eid:
                created.append(eid)
            else:
                skipped += 1

        # Long-term file
        lf = Path(long_term_file)
        if lf.exists():
            text = lf.read_text(encoding="utf-8", errors="ignore")
            eid = self.bridge.capture(
                Event(
                    source="clawdbot_file",
                    event_type="long_term_memory_md",
                    content_text=text[:2000],
                    content_json={
                        "path": str(lf),
                        "content": text,
                        "ns": namespace,
                        "tags": [f"ns:{namespace}", "ingest:clawdbot"],
                    },
                    tags=[f"ns:{namespace}", "ingest:clawdbot"],
                )
            )
            if eid:
                created.append(eid)
            else:
                skipped += 1

        return {"created": created, "skipped": skipped}

    def consolidate(self) -> dict[str, Any]:
        """Background consolidation: summarize older, long threads.

        Heuristic implementation: for each thread with many events older than
        `consolidate_older_than_days`, write a summary event and tag it.
        """

        if not self.policy.consolidate_enabled:
            return {"enabled": False, "created": 0}

        cutoff = datetime.now(UTC) - timedelta(days=self.policy.consolidate_older_than_days)
        cutoff_iso = cutoff.isoformat()

        cur = self.bridge.conn.cursor()
        cur.execute(
            """
            SELECT thread_id, COUNT(*) as n
            FROM events
            WHERE thread_id IS NOT NULL
              AND created_at < ?
              AND event_type LIKE 'chat_%'
            GROUP BY thread_id
            HAVING n >= 12
            ORDER BY n DESC
            LIMIT 50
            """,
            (cutoff_iso,),
        )

        threads = cur.fetchall()
        created = 0
        summaries: list[str] = []

        for thread_id, n in threads:
            # Skip if already summarized
            cur.execute(
                "SELECT 1 FROM events WHERE thread_id = ? AND event_type = 'thread_summary' LIMIT 1",
                (thread_id,),
            )
            if cur.fetchone():
                continue

            cur.execute(
                """
                SELECT created_at, source, event_type, content_text
                FROM events
                WHERE thread_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (thread_id, self.policy.consolidate_max_events_per_thread),
            )
            rows = cur.fetchall()
            if not rows:
                continue

            summary_text = self._heuristic_thread_summary(rows)
            summary_text = _truncate(summary_text, self.policy.consolidation_summary_max_chars)

            eid = self.bridge.capture(
                Event(
                    source="memory_os",
                    event_type="thread_summary",
                    thread_id=thread_id,
                    content_text=summary_text,
                    content_json={
                        "thread_id": thread_id,
                        "n_events": int(n),
                        "cutoff": cutoff_iso,
                        "summary": summary_text,
                        "tags": ["consolidated", "summary"],
                    },
                    tags=["consolidated", "summary"],
                )
            )
            if eid:
                created += 1
                summaries.append(eid)

        return {"enabled": True, "created": created, "summary_event_ids": summaries}

    # --------------------------
    # helpers
    # --------------------------

    def _last_user_text(self, messages: Sequence[dict[str, Any]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return (m.get("content") or "").strip()
        return ""

    def _heuristic_thread_summary(self, rows: Sequence[tuple[str, str, str, str]]) -> str:
        # rows: (created_at, source, event_type, content_text)
        # Simple compression: keep salient sentences / decisions.
        buf: list[str] = []
        buf.append("Thread summary (heuristic):")
        for created_at, source, event_type, text in rows:
            if not text:
                continue
            t = _squash_ws(text)
            # Prefer lines with decisions, TODOs, errors
            if re.search(r"\b(decision|decided|todo|next|fix|bug|error|plan)\b", t, re.I):
                buf.append(f"- {created_at}: {event_type}: {_truncate(t, 200)}")

        # fallback: include first/last
        if len(buf) <= 1:
            first = _truncate(_squash_ws(rows[0][3] or ""), 220)
            last = _truncate(_squash_ws(rows[-1][3] or ""), 220)
            buf.append(f"- start: {first}")
            if last != first:
                buf.append(f"- end: {last}")
        return "\n".join(buf)


# --------------------------
# Utility
# --------------------------


def _truncate(s: str, n: int) -> str:
    if s is None:
        return ""
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"


def _squash_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _tokenize(s: str) -> set[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9_\-\s]", " ", s)
    terms = [t for t in s.split() if len(t) >= 3]
    return set(terms)


def _infer_namespace(r: SearchResult) -> str:
    # First look for explicit content_json ns
    try:
        ns = (r.content_json or {}).get("ns")
        if ns:
            return str(ns)
        tags = (r.content_json or {}).get("tags") or []
        for t in tags:
            if isinstance(t, str) and t.startswith("ns:"):
                return t.split(":", 1)[1]
    except Exception:
        pass

    # Fallback by source
    src = (r.source or "").lower()
    if src in ("web", "web_fetch"):
        return "web"
    if src in ("git", "code"):
        return "code"
    if src in ("decision",):
        return "personal"
    if src in ("clawdbot_file",):
        return "personal"
    return "ops"
