from __future__ import annotations

import time
from dataclasses import dataclass

from .extractors import EntityRelationExtractor, RegexExtractor
from .models import Entity, Relation
from .store import GraphStore


@dataclass(slots=True)
class IngestStats:
    entities: int
    relations: int
    extract_ms: float
    upsert_ms: float


class GraphIngestor:
    def __init__(self, store: GraphStore, extractor: EntityRelationExtractor | None = None):
        self.store = store
        self.extractor = extractor or RegexExtractor()

    def ingest_text(self, text: str, *, source_id: str | None = None) -> IngestStats:
        t0 = time.perf_counter()
        entities, relations = self.extractor.extract(text, source_id=source_id)
        t1 = time.perf_counter()
        self.store.upsert(entities=entities, relations=relations)
        t2 = time.perf_counter()
        return IngestStats(
            entities=len(entities),
            relations=len(relations),
            extract_ms=(t1 - t0) * 1000.0,
            upsert_ms=(t2 - t1) * 1000.0,
        )

    def ingest(self, *, entities: list[Entity], relations: list[Relation]) -> None:
        self.store.upsert(entities=entities, relations=relations)
