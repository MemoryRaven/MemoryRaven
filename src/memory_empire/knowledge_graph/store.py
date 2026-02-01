from __future__ import annotations

from typing import Any, Protocol

from .models import Entity, Relation


class GraphStore(Protocol):
    """Abstraction for the backing graph database."""

    def ensure_schema(self) -> None: ...

    def upsert(self, *, entities: list[Entity], relations: list[Relation]) -> None: ...

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...

    def neighbors(self, node_id: str, *, depth: int = 1, limit: int = 200) -> dict[str, Any]: ...

    def shortest_path(
        self, src_id: str, dst_id: str, *, max_hops: int = 6
    ) -> dict[str, Any] | None: ...
