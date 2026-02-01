from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .extractors import batched
from .models import Entity, Relation


@dataclass(slots=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"
    # ingestion performance
    batch_size: int = 500


class Neo4jGraphStore:
    """Neo4j-backed graph store.

    Uses UNWIND for bulk upserts and maintains node key constraints.

    Dependency: neo4j>=5 (optional extra).
    """

    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        from neo4j import GraphDatabase  # type: ignore

        # Driver is thread-safe; sessions are lightweight.
        self._driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))

    def close(self) -> None:
        self._driver.close()

    def ensure_schema(self) -> None:
        stmts = [
            # Stable node key
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT handle_id IF NOT EXISTS FOR (n:Handle) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT tag_id IF NOT EXISTS FOR (n:Tag) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
        ]
        with self._driver.session(database=self.cfg.database) as s:
            for q in stmts:
                s.run(q)

    def upsert(self, *, entities: list[Entity], relations: list[Relation]) -> None:
        if not entities and not relations:
            return

        with self._driver.session(database=self.cfg.database) as s:
            if entities:
                for batch in batched(entities, self.cfg.batch_size):
                    s.execute_write(self._upsert_nodes_tx, batch)
            if relations:
                for batch in batched(relations, self.cfg.batch_size):
                    s.execute_write(self._upsert_rels_tx, batch)

    @staticmethod
    def _upsert_nodes_tx(tx, batch: list[Entity]):
        rows = [
            {"id": e.id, "label": e.label, "name": e.name, "props": e.props or {}}
            for e in batch
        ]
        # Dynamic labels aren't allowed in MERGE; we merge on :_Base then set label.
        # We keep small label set (Entity/Handle/Tag).
        q = """
        UNWIND $rows as row
        CALL {
          WITH row
          WITH row WHERE row.label = 'Entity'
          MERGE (n:Entity {id: row.id})
          SET n.name = row.name
          SET n += row.props
          RETURN 1 as _
          UNION
          WITH row
          WITH row WHERE row.label = 'Handle'
          MERGE (n:Handle {id: row.id})
          SET n.name = row.name
          SET n += row.props
          RETURN 1 as _
          UNION
          WITH row
          WITH row WHERE row.label = 'Tag'
          MERGE (n:Tag {id: row.id})
          SET n.name = row.name
          SET n += row.props
          RETURN 1 as _
        }
        RETURN count(*) as n
        """
        tx.run(q, rows=rows)

    @staticmethod
    def _upsert_rels_tx(tx, batch: list[Relation]):
        rows = [
            {
                "src": r.src_id,
                "dst": r.dst_id,
                "type": r.rel_type,
                "props": r.props or {},
                "confidence": float(r.confidence),
            }
            for r in batch
        ]
        # Relationship types cannot be parameterized; whitelist a small set.
        q = """
        UNWIND $rows as row
        MATCH (a {id: row.src})
        MATCH (b {id: row.dst})
        CALL {
          WITH a,b,row
          WITH a,b,row WHERE row.type = 'IS_A'
          MERGE (a)-[r:IS_A]->(b)
          SET r += row.props
          SET r.confidence = row.confidence
          RETURN 1 as _
          UNION
          WITH a,b,row
          WITH a,b,row WHERE row.type = 'WORKS_AT'
          MERGE (a)-[r:WORKS_AT]->(b)
          SET r += row.props
          SET r.confidence = row.confidence
          RETURN 1 as _
          UNION
          WITH a,b,row
          WITH a,b,row WHERE row.type = 'FOUNDED'
          MERGE (a)-[r:FOUNDED]->(b)
          SET r += row.props
          SET r.confidence = row.confidence
          RETURN 1 as _
          UNION
          WITH a,b,row
          WITH a,b,row WHERE row.type = 'MET'
          MERGE (a)-[r:MET]->(b)
          SET r += row.props
          SET r.confidence = row.confidence
          RETURN 1 as _
        }
        RETURN count(*) as n
        """
        tx.run(q, rows=rows)

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._driver.session(database=self.cfg.database) as s:
            res = s.run(cypher, **(params or {}))
            return [dict(r) for r in res]

    def neighbors(self, node_id: str, *, depth: int = 1, limit: int = 200) -> dict[str, Any]:
        depth = max(1, min(int(depth), 5))
        limit = max(1, min(int(limit), 1000))
        q = """
        MATCH (start {id: $id})
        CALL {
          WITH start
          MATCH p=(start)-[r*1..$depth]-(n)
          RETURN p LIMIT $limit
        }
        RETURN p
        """
        paths = self.query(q, {"id": node_id, "depth": depth, "limit": limit})
        # Return raw paths; API layer can normalize.
        return {"id": node_id, "depth": depth, "paths": paths}

    def shortest_path(self, src_id: str, dst_id: str, *, max_hops: int = 6) -> dict[str, Any] | None:
        max_hops = max(1, min(int(max_hops), 12))
        q = """
        MATCH (a {id: $a}), (b {id: $b})
        MATCH p = shortestPath((a)-[*..$max_hops]-(b))
        RETURN p
        """
        rows = self.query(q, {"a": src_id, "b": dst_id, "max_hops": max_hops})
        if not rows:
            return None
        return rows[0]
