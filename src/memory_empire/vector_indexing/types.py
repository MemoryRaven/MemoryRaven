from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ContentType = Literal[
    "web",
    "paper",
    "pdf",
    "conversation",
    "tool_output",
    "code",
    "log",
    "generic",
]


@dataclass(frozen=True)
class SourceRef:
    """Identifies the upstream origin and revision for incremental indexing."""

    source: str  # e.g. "web", "conversations", "tools"
    uri: str  # URL, file path, or opaque id
    revision: str | None = None  # etag, last-modified, mtime hash, git sha, etc.


@dataclass
class Document:
    """A unit of ingestion before chunking."""

    ref: SourceRef
    content_type: ContentType
    text: str
    media_type: str | None = None  # e.g. text/html, application/pdf
    title: str | None = None
    created_at: float | None = None
    fetched_at: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A chunk ready for embedding and storage."""

    doc_ref: SourceRef
    chunk_id: str  # deterministic id under doc
    content_type: ContentType
    text: str
    start: int | None = None
    end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    vectors: dict[str, list[float]]  # model_name -> vector
    fingerprints: dict[str, str]  # e.g. sha256, simhash
    quality: float
