from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .store import GraphStore


@dataclass(slots=True)
class GraphQueryEngine:
    """Convenience layer for common graph queries.

    The goal is low-latency traversal using DB-native execution (Cypher),
    while keeping a stable Python API.
    """

    store: GraphStore

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        q = "MATCH (n {id: $id}) RETURN n LIMIT 1"
        rows = self.store.query(q, {"id": node_id})
        return rows[0] if rows else None

    def expand(self, node_id: str, *, depth: int = 1, limit: int = 200) -> dict[str, Any]:
        return self.store.neighbors(node_id, depth=depth, limit=limit)

    def shortest_path(self, src_id: str, dst_id: str, *, max_hops: int = 6) -> dict[str, Any] | None:
        return self.store.shortest_path(src_id, dst_id, max_hops=max_hops)
