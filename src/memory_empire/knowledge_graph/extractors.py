from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Protocol

from .models import Entity, Relation


class EntityRelationExtractor(Protocol):
    def extract(self, text: str, *, source_id: str | None = None) -> tuple[list[Entity], list[Relation]]: ...


_WORD = r"[A-Za-z][A-Za-z0-9_\-']*"


def _slug(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"[^A-Za-z0-9 _\-]", "", s)
    s = s.strip().lower().replace(" ", "-")
    s = re.sub(r"\-+", "-", s)
    return s or "unknown"


@dataclass(slots=True)
class RegexExtractor:
    """Fast, dependency-free extractor.

    This is *not* meant to be perfect; it's meant to be cheap and deterministic.
    Replace with spaCy/LLM-based extractor if needed.

    Patterns:
    - "X is Y" => (X)-[:IS_A]->(Y)
    - "X works at Y" => (X)-[:WORKS_AT]->(Y)
    - "X founded Y" / "X created Y" => (X)-[:FOUNDED]->(Y)
    - "X met Y" => (X)-[:MET]->(Y)

    Entities are derived from simple proper-noun heuristics.
    """

    min_entity_len: int = 2

    def extract(self, text: str, *, source_id: str | None = None) -> tuple[list[Entity], list[Relation]]:
        if not text:
            return [], []

        entities: dict[str, Entity] = {}
        rels: list[Relation] = []

        # Candidate entity spans: sequences of CapitalizedWords (very cheap heuristic)
        # Example: "John Doe" "Acme Corp" "New York"
        for m in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b", text):
            name = m.group(0).strip()
            if len(name) < self.min_entity_len:
                continue
            eid = f"ent:{_slug(name)}"
            entities.setdefault(eid, Entity(id=eid, label="Entity", name=name, props={"source_id": source_id}))

        # Also include @handles and #tags
        for m in re.finditer(r"[@#]" + _WORD, text):
            token = m.group(0)
            label = "Handle" if token.startswith("@") else "Tag"
            eid = f"{label.lower()}:{_slug(token)}"
            entities.setdefault(eid, Entity(id=eid, label=label, name=token, props={"source_id": source_id}))

        # relation patterns
        patterns: list[tuple[str, str, float]] = [
            (rf"(?P<x>.+?)\s+is\s+(?P<y>.+)", "IS_A", 0.6),
            (rf"(?P<x>.+?)\s+works\s+at\s+(?P<y>.+)", "WORKS_AT", 0.7),
            (rf"(?P<x>.+?)\s+(?:founded|created|started)\s+(?P<y>.+)", "FOUNDED", 0.7),
            (rf"(?P<x>.+?)\s+met\s+(?P<y>.+)", "MET", 0.55),
        ]

        # Apply patterns line-by-line to reduce catastrophic backtracking.
        for line in (t.strip() for t in re.split(r"[\n\r]+", text) if t.strip()):
            for pat, rel_type, conf in patterns:
                mm = re.fullmatch(pat, line, flags=re.IGNORECASE)
                if not mm:
                    continue
                x = mm.group("x").strip(" .,:;\t\"")
                y = mm.group("y").strip(" .,:;\t\"")
                if not x or not y:
                    continue

                x_id = f"ent:{_slug(x)}"
                y_id = f"ent:{_slug(y)}"
                entities.setdefault(x_id, Entity(id=x_id, label="Entity", name=x, props={"source_id": source_id}))
                entities.setdefault(y_id, Entity(id=y_id, label="Entity", name=y, props={"source_id": source_id}))

                rels.append(
                    Relation(
                        src_id=x_id,
                        dst_id=y_id,
                        rel_type=rel_type,
                        props={"source_id": source_id, "pattern": pat},
                        confidence=conf,
                    )
                )

        # De-dup relations (cheap)
        seen: set[tuple[str, str, str]] = set()
        deduped: list[Relation] = []
        for r in rels:
            key = (r.src_id, r.dst_id, r.rel_type)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)

        return list(entities.values()), deduped


def batched(it: Iterable, batch_size: int) -> Iterable[list]:
    batch: list = []
    for x in it:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
