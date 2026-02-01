from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Entity:
    """A canonical entity node.

    `id` is a stable identifier suitable for graph DB node keys.
    """

    id: str
    label: str
    name: str
    props: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Relation:
    """A directed edge between two entities."""

    src_id: str
    dst_id: str
    rel_type: str
    props: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
