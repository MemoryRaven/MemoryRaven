from __future__ import annotations

import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

from .types import Chunk, Document
from .util_text import guess_language

_TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


@dataclass
class MetadataConfig:
    enrich_keywords: bool = False


def _keywords_simple(text: str, *, max_k: int = 12) -> list[str]:
    # Extremely simple TF-ish keyword extraction.
    tokens = [t.strip(".,:;!?()[]{}\"'`).").lower() for t in text.split()]
    tokens = [t for t in tokens if len(t) >= 4]
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return [k for k, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:max_k]]


def extract_doc_metadata(doc: Document) -> dict[str, Any]:
    md: dict[str, Any] = {}
    md["source"] = doc.ref.source
    md["uri"] = doc.ref.uri
    md["revision"] = doc.ref.revision
    md["content_type"] = doc.content_type
    md["media_type"] = doc.media_type
    md["title"] = doc.title
    md["fetched_at"] = doc.fetched_at
    md["created_at"] = doc.created_at

    # Web
    if doc.content_type == "web":
        try:
            u = urllib.parse.urlparse(doc.ref.uri)
            md["domain"] = u.netloc
            md["path"] = u.path
        except Exception:
            pass

    # Files
    if os.path.exists(doc.ref.uri):
        try:
            st = os.stat(doc.ref.uri)
            md["file_mtime"] = st.st_mtime
            md["file_size"] = st.st_size
        except Exception:
            pass

    # Heuristic title from markdown
    if not md.get("title"):
        m = _TITLE_RE.search(doc.text or "")
        if m:
            md["title"] = m.group(1).strip()[:200]

    md["lang"] = guess_language(doc.text)
    md["ingested_at"] = time.time()
    return md


def enrich_chunk(chunk: Chunk, doc_md: dict[str, Any], cfg: MetadataConfig | None = None) -> Chunk:
    cfg = cfg or MetadataConfig()
    md = dict(doc_md)
    md.update(chunk.metadata or {})
    md["chunk_id"] = chunk.chunk_id

    if cfg.enrich_keywords:
        md["keywords"] = _keywords_simple(chunk.text)

    chunk.metadata = md
    return chunk
