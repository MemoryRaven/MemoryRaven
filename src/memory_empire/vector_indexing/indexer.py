from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

from .chunking import ChunkingConfig, chunk_document
from .embeddings import Embedder
from .fingerprints import fingerprint
from .metadata import MetadataConfig, enrich_chunk, extract_doc_metadata
from .quality import quality_score, should_index
from .router import Router
from .storage_sqlite import SQLiteIndexDB
from .types import Document
from .util_text import normalize_text
from .vector_store import VectorStore


@dataclass
class IndexerConfig:
    # Default lowered so short-but-useful notes still get indexed.
    min_quality: float = 0.1
    # near-duplicate filtering threshold (Hamming distance on simhash64)
    near_dup_hamming: int = 3


@dataclass
class Indexer:
    db: SQLiteIndexDB
    vector_store: VectorStore
    embedders: dict[str, Embedder]
    router: Router
    cfg: IndexerConfig = field(default_factory=IndexerConfig)
    chunk_cfg: ChunkingConfig = field(default_factory=ChunkingConfig)
    md_cfg: MetadataConfig = field(default_factory=MetadataConfig)

    def init(self) -> None:
        self.db.init()

    def _doc_id(self, doc: Document, *, namespace: str) -> str:
        # stable id per (namespace, source, uri, revision)
        base = f"{namespace}|{doc.ref.source}|{doc.ref.uri}|{doc.ref.revision or ''}"
        from .fingerprints import sha256_text

        return sha256_text(base)

    def index_documents(self, docs: Iterable[Document], *, namespace: str) -> dict[str, int]:
        self.init()
        stats = {
            "docs_seen": 0,
            "docs_indexed": 0,
            "chunks_seen": 0,
            "chunks_indexed": 0,
            "chunks_skipped_low_quality": 0,
        }

        for doc in docs:
            stats["docs_seen"] += 1

            # Incremental change detection.
            last_rev = self.db.get_last_revision(
                source=doc.ref.source, uri=doc.ref.uri, namespace=namespace
            )
            if last_rev and doc.ref.revision and last_rev == doc.ref.revision:
                # unchanged
                continue

            doc_md = extract_doc_metadata(doc)
            doc_id = self._doc_id(doc, namespace=namespace)

            # Store document record
            from .fingerprints import sha256_text

            content_sha = sha256_text(normalize_text(doc.text))
            self.db.put_document(
                doc_id=doc_id,
                source=doc.ref.source,
                uri=doc.ref.uri,
                namespace=namespace,
                revision=doc.ref.revision,
                content_type=str(doc.content_type),
                title=doc.title,
                fetched_at=doc.fetched_at,
                created_at=doc.created_at,
                metadata_json=json.dumps(doc_md, ensure_ascii=False),
                content_sha256=content_sha,
            )
            self.db.upsert_source_state(
                source=doc.ref.source,
                uri=doc.ref.uri,
                namespace=namespace,
                revision=doc.ref.revision,
            )
            stats["docs_indexed"] += 1

            # Chunk + index
            chunks = list(chunk_document(doc, self.chunk_cfg))
            stats["chunks_seen"] += len(chunks)

            # Batch embed per model per doc
            for ch in chunks:
                ch = enrich_chunk(ch, doc_md, self.md_cfg)
                q = quality_score(ch.text, content_type=str(ch.content_type))
                if q < self.cfg.min_quality or not should_index(
                    ch.text, min_quality=self.cfg.min_quality
                ):
                    stats["chunks_skipped_low_quality"] += 1
                    continue

                fps = fingerprint(ch.text)

                # Exact dedup (global within namespace)
                if self.db.chunk_sha_exists(namespace=namespace, sha256=fps.sha256):
                    continue

                # Near-dup dedup (simhash prefix + hamming)
                from .fingerprints import hamming_distance_hex

                prefix = fps.simhash64[:6]
                is_near_dup = False
                for _pk, sh in self.db.iter_simhash_candidates(
                    namespace=namespace, simhash64_prefix=prefix
                ):
                    if hamming_distance_hex(fps.simhash64, sh) <= self.cfg.near_dup_hamming:
                        is_near_dup = True
                        break
                if is_near_dup:
                    continue

                chunk_pk = f"{doc_id}:{ch.chunk_id}"  # stable id

                self.db.put_chunk(
                    chunk_pk=chunk_pk,
                    doc_id=doc_id,
                    chunk_id=ch.chunk_id,
                    namespace=namespace,
                    content_type=str(ch.content_type),
                    text=ch.text,
                    sha256=fps.sha256,
                    simhash64=fps.simhash64,
                    quality=q,
                    metadata_json=json.dumps(ch.metadata or {}, ensure_ascii=False),
                )

                # Embeddings: compute all models for this chunk (small batch = 1; can batch later)
                vectors_by_model: dict[str, list[float]] = {}
                for model_alias, embedder in self.embedders.items():
                    vectors_by_model[model_alias] = embedder.embed([ch.text])[0]

                # Write to vector store grouped by model
                for model_alias, vec in vectors_by_model.items():
                    decision = self.router.route(ch, namespace_override=namespace)
                    self.vector_store.upsert(
                        ids=[chunk_pk],
                        vectors=[vec],
                        metadatas=[ch.metadata or {}],
                        documents=[ch.text],
                        namespace=decision.namespace,
                        model_name=model_alias,
                    )

                self.db.put_embeddings(
                    chunk_pk=chunk_pk,
                    vectors={k: json.dumps(v) for k, v in vectors_by_model.items()},
                )

                stats["chunks_indexed"] += 1

        return stats


@dataclass
class StreamWatcher:
    """Simple file tailing for real-time indexing."""

    path: str
    source_name: str = "stream"
    content_type: str = "log"

    def iter_documents(self) -> Iterable[Document]:
        import os
        import time

        # Tail file and emit each new line as its own document.
        # For production: batch lines and create larger docs.
        with open(self.path, encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.25)
                    continue
                uri = f"{self.path}::offset:{f.tell()}"
                rev = f"pos:{f.tell()}"
                from .types import SourceRef

                yield Document(
                    ref=SourceRef(source=self.source_name, uri=uri, revision=rev),
                    content_type=self.content_type,  # type: ignore
                    text=line,
                    media_type="text/plain",
                    fetched_at=time.time(),
                    title=os.path.basename(self.path),
                )


# Utility


def ensure_dirs(*paths: str | None) -> None:
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)
