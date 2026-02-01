from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sources (
  source TEXT NOT NULL,
  uri TEXT NOT NULL,
  namespace TEXT NOT NULL,
  last_seen REAL NOT NULL,
  last_revision TEXT,
  PRIMARY KEY (source, uri, namespace)
);

CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  uri TEXT NOT NULL,
  namespace TEXT NOT NULL,
  revision TEXT,
  content_type TEXT,
  title TEXT,
  fetched_at REAL,
  created_at REAL,
  metadata_json TEXT,
  content_sha256 TEXT,
  inserted_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_pk TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  chunk_id TEXT NOT NULL,
  namespace TEXT NOT NULL,
  content_type TEXT,
  text TEXT,
  sha256 TEXT,
  simhash64 TEXT,
  quality REAL,
  metadata_json TEXT,
  inserted_at REAL NOT NULL,
  UNIQUE(doc_id, chunk_id, namespace)
);

CREATE TABLE IF NOT EXISTS embeddings (
  chunk_pk TEXT NOT NULL,
  model_name TEXT NOT NULL,
  vector_json TEXT NOT NULL,
  inserted_at REAL NOT NULL,
  PRIMARY KEY (chunk_pk, model_name)
);

-- helpful indexes for dedup / filtering
CREATE INDEX IF NOT EXISTS idx_chunks_namespace_sha256 ON chunks(namespace, sha256);
CREATE INDEX IF NOT EXISTS idx_chunks_namespace_simhash ON chunks(namespace, simhash64);
CREATE INDEX IF NOT EXISTS idx_docs_namespace_uri ON documents(namespace, uri);
"""


@dataclass
class SQLiteIndexDB:
    path: str

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def init(self) -> None:
        con = self.connect()
        try:
            con.executescript(SCHEMA)
            con.commit()
        finally:
            con.close()

    def upsert_source_state(
        self, *, source: str, uri: str, namespace: str, revision: str | None
    ) -> None:
        con = self.connect()
        try:
            con.execute(
                """
                INSERT INTO sources(source, uri, namespace, last_seen, last_revision)
                VALUES(?,?,?,?,?)
                ON CONFLICT(source, uri, namespace)
                DO UPDATE SET last_seen=excluded.last_seen, last_revision=excluded.last_revision
                """,
                (source, uri, namespace, time.time(), revision),
            )
            con.commit()
        finally:
            con.close()

    def get_last_revision(self, *, source: str, uri: str, namespace: str) -> str | None:
        con = self.connect()
        try:
            row = con.execute(
                "SELECT last_revision FROM sources WHERE source=? AND uri=? AND namespace=?",
                (source, uri, namespace),
            ).fetchone()
            return row[0] if row else None
        finally:
            con.close()

    def put_document(
        self,
        *,
        doc_id: str,
        source: str,
        uri: str,
        namespace: str,
        revision: str | None,
        content_type: str,
        title: str | None,
        fetched_at: float | None,
        created_at: float | None,
        metadata_json: str,
        content_sha256: str,
    ) -> None:
        con = self.connect()
        try:
            con.execute(
                """
                INSERT OR REPLACE INTO documents(
                  doc_id, source, uri, namespace, revision, content_type, title,
                  fetched_at, created_at, metadata_json, content_sha256, inserted_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    doc_id,
                    source,
                    uri,
                    namespace,
                    revision,
                    content_type,
                    title,
                    fetched_at,
                    created_at,
                    metadata_json,
                    content_sha256,
                    time.time(),
                ),
            )
            con.commit()
        finally:
            con.close()

    def chunk_exists(self, *, chunk_pk: str) -> bool:
        con = self.connect()
        try:
            row = con.execute("SELECT 1 FROM chunks WHERE chunk_pk=?", (chunk_pk,)).fetchone()
            return bool(row)
        finally:
            con.close()

    def chunk_sha_exists(self, *, namespace: str, sha256: str) -> bool:
        con = self.connect()
        try:
            row = con.execute(
                "SELECT 1 FROM chunks WHERE namespace=? AND sha256=? LIMIT 1",
                (namespace, sha256),
            ).fetchone()
            return bool(row)
        finally:
            con.close()

    def iter_simhash_candidates(
        self, *, namespace: str, simhash64_prefix: str
    ) -> Iterable[tuple[str, str]]:
        """Return (chunk_pk, simhash64) candidates sharing a prefix.

        Prefix filtering keeps sqlite scans small; final hamming check happens in code.
        """
        con = self.connect()
        try:
            cur = con.execute(
                "SELECT chunk_pk, simhash64 FROM chunks WHERE namespace=? AND simhash64 LIKE ?",
                (namespace, simhash64_prefix + "%"),
            )
            for row in cur:
                yield row[0], row[1]
        finally:
            con.close()

    def put_chunk(
        self,
        *,
        chunk_pk: str,
        doc_id: str,
        chunk_id: str,
        namespace: str,
        content_type: str,
        text: str,
        sha256: str,
        simhash64: str,
        quality: float,
        metadata_json: str,
    ) -> None:
        con = self.connect()
        try:
            con.execute(
                """
                INSERT OR REPLACE INTO chunks(
                  chunk_pk, doc_id, chunk_id, namespace, content_type, text,
                  sha256, simhash64, quality, metadata_json, inserted_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    chunk_pk,
                    doc_id,
                    chunk_id,
                    namespace,
                    content_type,
                    text,
                    sha256,
                    simhash64,
                    quality,
                    metadata_json,
                    time.time(),
                ),
            )
            con.commit()
        finally:
            con.close()

    def put_embeddings(self, *, chunk_pk: str, vectors: dict[str, str]) -> None:
        con = self.connect()
        try:
            for model_name, vector_json in vectors.items():
                con.execute(
                    """
                    INSERT OR REPLACE INTO embeddings(chunk_pk, model_name, vector_json, inserted_at)
                    VALUES (?,?,?,?)
                    """,
                    (chunk_pk, model_name, vector_json, time.time()),
                )
            con.commit()
        finally:
            con.close()

    def iter_chunks_for_doc(self, *, doc_id: str) -> Iterable[tuple[str, str]]:
        """Yields (chunk_pk, sha256)."""
        con = self.connect()
        try:
            cur = con.execute("SELECT chunk_pk, sha256 FROM chunks WHERE doc_id=?", (doc_id,))
            for row in cur:
                yield row[0], row[1]
        finally:
            con.close()
