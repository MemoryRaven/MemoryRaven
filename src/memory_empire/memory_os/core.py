"""
Claude Memory Bridge - Core implementation
Never forget anything, ever.
"""

import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Immutable event - the atomic unit of memory"""

    source: str  # telegram, code, web, decision, etc.
    event_type: str  # message, commit, web_fetch, decision, etc.
    content_text: str  # Searchable text
    content_json: dict[str, Any]  # Full fidelity data
    id: str | None = None
    thread_id: str | None = None
    parent_event_id: str | None = None
    actor_id: str | None = None
    created_at: str | None = None
    observed_at: str | None = None
    tags: list[str] | None = None
    sensitivity: str = "internal"
    embeddings: list[float] | None = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()
        if not self.tags:
            self.tags = []

    def to_dict(self) -> dict:
        return asdict(self)

    def content_hash(self) -> str:
        """Deterministic hash for deduplication"""
        content = json.dumps(
            {
                "source": self.source,
                "event_type": self.event_type,
                "content_text": self.content_text,
                "content_json": self.content_json,
                "thread_id": self.thread_id,
                "observed_at": self.observed_at,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()


class MemoryBridge:
    """
    Main memory interface - captures everything, forgets nothing
    """

    def __init__(
        self,
        db_path: str = "~/.claude_memory/memory.db",
        pinecone_api_key: str | None = None,
        pinecone_environment: str | None = None,
        pinecone_index_name: str = "claude-memory",
    ):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite (hot tier)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

        # Initialize Pinecone (warm tier) if configured
        self.pinecone_enabled = False
        if pinecone_api_key:
            try:
                import pinecone

                pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
                self.index = pinecone.Index(pinecone_index_name)
                self.pinecone_enabled = True
                logger.info(f"Pinecone initialized: {pinecone_index_name}")
            except Exception as e:
                logger.warning(f"Pinecone init failed: {e}. Using local only.")

        # Initialize local embedder
        self._init_embedder()

    def _init_schema(self) -> None:
        """Initialize SQLite schema"""
        cursor = self.conn.cursor()

        # Events table (source of truth)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                event_type TEXT NOT NULL,
                thread_id TEXT,
                parent_event_id TEXT,
                actor_id TEXT,
                created_at TEXT NOT NULL,
                observed_at TEXT,
                content_text TEXT,
                content_json TEXT NOT NULL,
                content_hash TEXT,
                tags TEXT,
                sensitivity TEXT DEFAULT 'internal',
                embeddings BLOB,
                indexed_at TEXT,
                UNIQUE(content_hash)
            )
        """)

        # Indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON events(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON events(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_thread_id ON events(thread_id)")

        # Full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
                content_text, 
                content='events',
                content_rowid='rowid'
            )
        """)

        # Entity extraction table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT,
                aliases TEXT,
                first_seen TEXT,
                last_seen TEXT,
                UNIQUE(name, entity_type)
            )
        """)

        # Entity mentions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_mentions (
                entity_id TEXT,
                event_id TEXT,
                confidence REAL,
                context TEXT,
                FOREIGN KEY(entity_id) REFERENCES entities(entity_id),
                FOREIGN KEY(event_id) REFERENCES events(id)
            )
        """)

        self.conn.commit()

    def _init_embedder(self):
        """Initialize local embedding model"""
        try:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_dim = 384
            logger.info("Local embedder initialized")
        except:
            logger.warning(
                "Sentence transformers not available. Install with: pip install sentence-transformers"
            )
            self.embedder = None

    def capture(self, event: Event) -> str | None:
        """
        Capture an event - this is the primary write interface
        Returns the event ID
        """
        # Add content hash
        event_dict = event.to_dict()
        event_dict["content_hash"] = event.content_hash()

        # Generate embeddings if possible
        if self.embedder and event.content_text:
            embeddings = self.embedder.encode(event.content_text)
            event_dict["embeddings"] = embeddings.tobytes()

        # Store in SQLite (hot tier)
        cursor = self.conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO events (
                    id, source, event_type, thread_id, parent_event_id,
                    actor_id, created_at, observed_at, content_text,
                    content_json, content_hash, tags, sensitivity, embeddings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event_dict["id"],
                    event_dict["source"],
                    event_dict["event_type"],
                    event_dict["thread_id"],
                    event_dict["parent_event_id"],
                    event_dict["actor_id"],
                    event_dict["created_at"],
                    event_dict["observed_at"],
                    event_dict["content_text"],
                    json.dumps(event_dict["content_json"]),
                    event_dict["content_hash"],
                    json.dumps(event_dict["tags"]),
                    event_dict["sensitivity"],
                    event_dict.get("embeddings"),
                ),
            )

            # Update FTS
            cursor.execute(
                "INSERT INTO events_fts(rowid, content_text) VALUES (last_insert_rowid(), ?)",
                (event_dict["content_text"],),
            )

            self.conn.commit()

            # Async index to Pinecone (if enabled)
            if self.pinecone_enabled:
                self._index_to_pinecone(event_dict)

            logger.info(f"Captured event: {event_dict['id']} ({event_dict['event_type']})")
            return str(event_dict["id"])

        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                logger.info(f"Duplicate event ignored: {event_dict['content_hash'][:8]}")
                return None
            raise

    def _index_to_pinecone(self, event_dict: dict):
        """Index event to Pinecone (warm tier)"""
        try:
            if not event_dict.get("embeddings"):
                return

            # Convert embeddings back from bytes (float32)
            from array import array

            a = array("f")
            a.frombytes(event_dict["embeddings"])
            embeddings = a.tolist()

            # Prepare metadata
            metadata = {
                "source": event_dict["source"],
                "event_type": event_dict["event_type"],
                "created_at": event_dict["created_at"],
                "thread_id": event_dict.get("thread_id", ""),
                "actor_id": event_dict.get("actor_id", ""),
                "content_preview": event_dict["content_text"][:200]
                if event_dict["content_text"]
                else "",
            }

            # Upsert to Pinecone
            self.index.upsert([(event_dict["id"], embeddings, metadata)])

            # Mark as indexed
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE events SET indexed_at = ? WHERE id = ?",
                (datetime.now(UTC).isoformat(), event_dict["id"]),
            )
            self.conn.commit()

        except Exception as e:
            logger.error(f"Pinecone indexing failed: {e}")

    # Convenience capture methods
    def capture_message(self, channel: str, text: str, **kwargs) -> str | None:
        """Capture a message event"""
        return self.capture(
            Event(
                source=channel,
                event_type="message",
                content_text=text,
                content_json={"text": text, **kwargs},
                **kwargs,
            )
        )

    def capture_code_change(self, file: str, diff: str, commit: str = None, **kwargs) -> str | None:
        """Capture a code change event"""
        return self.capture(
            Event(
                source="git",
                event_type="code_change",
                content_text=f"Changed {file}: {diff[:200]}...",
                content_json={"file": file, "diff": diff, "commit": commit, **kwargs},
                **kwargs,
            )
        )

    def capture_decision(self, decision: str, rationale: str = None, **kwargs) -> str | None:
        """Capture a decision event"""
        return self.capture(
            Event(
                source="decision",
                event_type="decision",
                content_text=f"Decision: {decision}. {rationale or ''}",
                content_json={"decision": decision, "rationale": rationale, **kwargs},
                **kwargs,
            )
        )

    def capture_web_content(self, url: str, content: str, **kwargs) -> str | None:
        """Capture web content"""
        return self.capture(
            Event(
                source="web",
                event_type="web_fetch",
                content_text=content[:1000],
                content_json={"url": url, "content": content, **kwargs},
                **kwargs,
            )
        )
