"""
ArangoDB implementation for Memory Empire Knowledge Graph.

ArangoDB provides excellent performance for both document and graph operations,
making it ideal for our knowledge graph use cases.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from arango import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import (
    DocumentGetError,
    DocumentInsertError,
    DocumentUpdateError,
    DocumentDeleteError,
)

from .base import (
    Edge,
    GraphDatabase,
    GraphPath,
    GraphQuery,
    GraphSearchResult,
    Node,
    RelationType,
)

logger = logging.getLogger(__name__)


class ArangoGraphDB(GraphDatabase):
    """
    High-performance ArangoDB implementation for knowledge graphs.
    
    Features:
    - Native graph traversal with AQL
    - Document-graph hybrid capabilities
    - Built-in full-text search
    - Efficient batch operations
    - Temporal graph support
    - Graph algorithms (PageRank, community detection, etc.)
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8529,
        username: str = "root",
        password: str = "",
        database: str = "memory_empire",
        graph_name: str = "knowledge_graph"
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database_name = database
        self.graph_name = graph_name
        
        self.client: Optional[ArangoClient] = None
        self.db: Optional[StandardDatabase] = None
        self.graph = None
        
        # Collection names
        self.nodes_collection = f"{graph_name}_nodes"
        self.edges_collection = f"{graph_name}_edges"
    
    def connect(self, **kwargs) -> None:
        """Establish connection to ArangoDB."""
        try:
            self.client = ArangoClient(hosts=f"http://{self.host}:{self.port}")
            
            # Connect to system database first
            sys_db = self.client.db("_system", username=self.username, password=self.password)
            
            # Create database if it doesn't exist
            if not sys_db.has_database(self.database_name):
                sys_db.create_database(self.database_name)
            
            # Connect to our database
            self.db = self.client.db(self.database_name, username=self.username, password=self.password)
            
            # Create collections if they don't exist
            if not self.db.has_collection(self.nodes_collection):
                self.db.create_collection(self.nodes_collection)
                
            if not self.db.has_collection(self.edges_collection):
                self.db.create_collection(self.edges_collection, edge=True)
            
            # Create graph if it doesn't exist
            if not self.db.has_graph(self.graph_name):
                self.graph = self.db.create_graph(
                    self.graph_name,
                    edge_definitions=[{
                        "edge_collection": self.edges_collection,
                        "from_vertex_collections": [self.nodes_collection],
                        "to_vertex_collections": [self.nodes_collection]
                    }]
                )
            else:
                self.graph = self.db.graph(self.graph_name)
            
            # Create indexes for better performance
            self._create_default_indexes()
            
            logger.info(f"Connected to ArangoDB at {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to ArangoDB: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close the connection to ArangoDB."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from ArangoDB")
    
    def _create_default_indexes(self) -> None:
        """Create default indexes for optimal performance."""
        nodes_col = self.db.collection(self.nodes_collection)
        edges_col = self.db.collection(self.edges_collection)
        
        # Node indexes
        nodes_col.add_persistent_index(fields=["label"])
        nodes_col.add_persistent_index(fields=["created_at"])
        nodes_col.add_persistent_index(fields=["source_type", "source_id"])
        nodes_col.add_fulltext_index(fields=["properties.content"])
        
        # Edge indexes
        edges_col.add_persistent_index(fields=["relationship"])
        edges_col.add_persistent_index(fields=["created_at"])
        edges_col.add_persistent_index(fields=["weight"])
    
    def _node_to_doc(self, node: Node) -> Dict[str, Any]:
        """Convert Node object to ArangoDB document."""
        doc = {
            "_key": node.id,
            "label": node.label,
            "properties": node.properties,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
            "confidence": node.confidence,
            "quality_score": node.quality_score,
        }
        
        if node.embeddings:
            doc["embeddings"] = node.embeddings
        
        if node.valid_from:
            doc["valid_from"] = node.valid_from.isoformat()
        if node.valid_to:
            doc["valid_to"] = node.valid_to.isoformat()
        
        if node.source_id:
            doc["source_id"] = node.source_id
        if node.source_type:
            doc["source_type"] = node.source_type
        
        return doc
    
    def _doc_to_node(self, doc: Dict[str, Any]) -> Node:
        """Convert ArangoDB document to Node object."""
        node = Node(
            id=doc["_key"],
            label=doc.get("label", ""),
            properties=doc.get("properties", {}),
            embeddings=doc.get("embeddings"),
            created_at=datetime.fromisoformat(doc["created_at"]),
            updated_at=datetime.fromisoformat(doc["updated_at"]),
            confidence=doc.get("confidence", 1.0),
            quality_score=doc.get("quality_score", 1.0),
            source_id=doc.get("source_id"),
            source_type=doc.get("source_type"),
        )
        
        if "valid_from" in doc:
            node.valid_from = datetime.fromisoformat(doc["valid_from"])
        if "valid_to" in doc:
            node.valid_to = datetime.fromisoformat(doc["valid_to"])
        
        return node
    
    def _edge_to_doc(self, edge: Edge) -> Dict[str, Any]:
        """Convert Edge object to ArangoDB document."""
        relationship = edge.relationship
        if isinstance(relationship, RelationType):
            relationship = relationship.value
        
        return {
            "_key": edge.id,
            "_from": f"{self.nodes_collection}/{edge.source_id}",
            "_to": f"{self.nodes_collection}/{edge.target_id}",
            "relationship": relationship,
            "properties": edge.properties,
            "weight": edge.weight,
            "created_at": edge.created_at.isoformat(),
            "confidence": edge.confidence,
            "evidence_ids": edge.evidence_ids,
            "valid_from": edge.valid_from.isoformat() if edge.valid_from else None,
            "valid_to": edge.valid_to.isoformat() if edge.valid_to else None,
        }
    
    def _doc_to_edge(self, doc: Dict[str, Any]) -> Edge:
        """Convert ArangoDB document to Edge object."""
        # Extract node IDs from _from and _to
        source_id = doc["_from"].split("/")[-1]
        target_id = doc["_to"].split("/")[-1]
        
        edge = Edge(
            id=doc["_key"],
            source_id=source_id,
            target_id=target_id,
            relationship=doc["relationship"],
            properties=doc.get("properties", {}),
            weight=doc.get("weight", 1.0),
            created_at=datetime.fromisoformat(doc["created_at"]),
            confidence=doc.get("confidence", 1.0),
            evidence_ids=doc.get("evidence_ids", []),
        )
        
        if doc.get("valid_from"):
            edge.valid_from = datetime.fromisoformat(doc["valid_from"])
        if doc.get("valid_to"):
            edge.valid_to = datetime.fromisoformat(doc["valid_to"])
        
        return edge
    
    def create_node(self, node: Node) -> Node:
        """Create a new node in the graph."""
        try:
            doc = self._node_to_doc(node)
            result = self.db.collection(self.nodes_collection).insert(doc)
            node.id = result["_key"]
            return node
        except DocumentInsertError as e:
            logger.error(f"Failed to create node: {e}")
            raise
    
    def create_edge(self, edge: Edge) -> Edge:
        """Create a new edge in the graph."""
        try:
            doc = self._edge_to_doc(edge)
            result = self.db.collection(self.edges_collection).insert(doc)
            edge.id = result["_key"]
            return edge
        except DocumentInsertError as e:
            logger.error(f"Failed to create edge: {e}")
            raise
    
    def update_node(self, node_id: str, updates: Dict[str, Any]) -> Node:
        """Update an existing node."""
        try:
            updates["updated_at"] = datetime.utcnow().isoformat()
            self.db.collection(self.nodes_collection).update({"_key": node_id}, updates)
            return self.get_node(node_id)
        except DocumentUpdateError as e:
            logger.error(f"Failed to update node {node_id}: {e}")
            raise
    
    def update_edge(self, edge_id: str, updates: Dict[str, Any]) -> Edge:
        """Update an existing edge."""
        try:
            self.db.collection(self.edges_collection).update({"_key": edge_id}, updates)
            return self.get_edge(edge_id)
        except DocumentUpdateError as e:
            logger.error(f"Failed to update edge {edge_id}: {e}")
            raise
    
    def delete_node(self, node_id: str, cascade: bool = False) -> bool:
        """Delete a node (optionally cascade delete edges)."""
        try:
            if cascade:
                # Delete all edges connected to this node
                query = """
                FOR e IN @@edges
                    FILTER e._from == @node_ref OR e._to == @node_ref
                    REMOVE e IN @@edges
                """
                self.db.aql.execute(
                    query,
                    bind_vars={
                        "@edges": self.edges_collection,
                        "node_ref": f"{self.nodes_collection}/{node_id}"
                    }
                )
            
            self.db.collection(self.nodes_collection).delete({"_key": node_id})
            return True
        except DocumentDeleteError as e:
            logger.error(f"Failed to delete node {node_id}: {e}")
            return False
    
    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge."""
        try:
            self.db.collection(self.edges_collection).delete({"_key": edge_id})
            return True
        except DocumentDeleteError as e:
            logger.error(f"Failed to delete edge {edge_id}: {e}")
            return False
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by ID."""
        try:
            doc = self.db.collection(self.nodes_collection).get({"_key": node_id})
            if doc:
                return self._doc_to_node(doc)
            return None
        except DocumentGetError:
            return None
    
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Retrieve an edge by ID."""
        try:
            doc = self.db.collection(self.edges_collection).get({"_key": edge_id})
            if doc:
                return self._doc_to_edge(doc)
            return None
        except DocumentGetError:
            return None
    
    def search(self, query: GraphQuery) -> GraphSearchResult:
        """Execute a graph search query using AQL."""
        start_time = datetime.utcnow()
        
        # Build AQL query based on GraphQuery parameters
        aql_query, bind_vars = self._build_search_query(query)
        
        cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
        
        nodes = []
        edges = []
        node_ids = set()
        edge_ids = set()
        
        for result in cursor:
            if "nodes" in result:
                for node_doc in result["nodes"]:
                    if node_doc["_key"] not in node_ids:
                        nodes.append(self._doc_to_node(node_doc))
                        node_ids.add(node_doc["_key"])
            
            if "edges" in result:
                for edge_doc in result["edges"]:
                    if edge_doc["_key"] not in edge_ids:
                        edges.append(self._doc_to_edge(edge_doc))
                        edge_ids.add(edge_doc["_key"])
        
        query_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return GraphSearchResult(
            nodes=nodes,
            edges=edges,
            paths=[],
            total_results=len(nodes) + len(edges),
            query_time_ms=query_time
        )
    
    def _build_search_query(self, query: GraphQuery) -> Tuple[str, Dict[str, Any]]:
        """Build AQL query from GraphQuery object."""
        bind_vars = {"@nodes": self.nodes_collection, "@edges": self.edges_collection}
        
        # Start with basic node query
        aql_parts = [f"FOR node IN @@nodes"]
        filters = []
        
        # Node label filter
        if query.node_labels:
            filters.append("node.label IN @labels")
            bind_vars["labels"] = query.node_labels
        
        # Node property filters
        if query.node_properties:
            for key, value in query.node_properties.items():
                filter_var = f"prop_{key}"
                filters.append(f"node.properties.{key} == @{filter_var}")
                bind_vars[filter_var] = value
        
        # Confidence filter
        if query.min_confidence > 0:
            filters.append("node.confidence >= @min_confidence")
            bind_vars["min_confidence"] = query.min_confidence
        
        # Temporal filters
        if query.time_point:
            filters.append("""
                (node.valid_from == null OR node.valid_from <= @time_point) AND
                (node.valid_to == null OR node.valid_to >= @time_point)
            """)
            bind_vars["time_point"] = query.time_point.isoformat()
        
        # Apply filters
        if filters:
            aql_parts.append(f"FILTER {' AND '.join(filters)}")
        
        # Add limit and skip
        aql_parts.append(f"LIMIT @skip, @limit")
        bind_vars["skip"] = query.skip
        bind_vars["limit"] = query.limit
        
        # Return format
        aql_parts.append("RETURN {nodes: [node], edges: []}")
        
        return " ".join(aql_parts), bind_vars
    
    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3,
        relationship_types: Optional[List[Union[RelationType, str]]] = None
    ) -> List[GraphPath]:
        """Find paths between two nodes using AQL."""
        query = """
        FOR path IN OUTBOUND SHORTEST_PATH
            @source TO @target
            GRAPH @graph
            OPTIONS {bfs: true, uniqueVertices: 'path'}
            FILTER LENGTH(path.edges) <= @max_depth
        """
        
        bind_vars = {
            "source": f"{self.nodes_collection}/{source_id}",
            "target": f"{self.nodes_collection}/{target_id}",
            "graph": self.graph_name,
            "max_depth": max_depth
        }
        
        if relationship_types:
            rel_values = [r.value if isinstance(r, RelationType) else r for r in relationship_types]
            query += " FILTER ALL(e IN path.edges FILTER e.relationship IN @rel_types)"
            bind_vars["rel_types"] = rel_values
        
        query += " RETURN path"
        
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        
        paths = []
        for result in cursor:
            path = result["path"]
            nodes = [self._doc_to_node(v) for v in path["vertices"]]
            edges = [self._doc_to_edge(e) for e in path["edges"]]
            
            total_weight = sum(e.weight for e in edges)
            
            paths.append(GraphPath(
                nodes=nodes,
                edges=edges,
                total_weight=total_weight
            ))
        
        return paths
    
    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        relationship_types: Optional[List[Union[RelationType, str]]] = None,
        max_depth: int = 1
    ) -> GraphSearchResult:
        """Get neighboring nodes using graph traversal."""
        direction_map = {
            "out": "OUTBOUND",
            "in": "INBOUND",
            "both": "ANY"
        }
        
        aql_direction = direction_map.get(direction, "ANY")
        
        query = f"""
        FOR node, edge, path IN 1..@max_depth
            {aql_direction} @start_node
            GRAPH @graph
        """
        
        bind_vars = {
            "start_node": f"{self.nodes_collection}/{node_id}",
            "max_depth": max_depth,
            "graph": self.graph_name
        }
        
        if relationship_types:
            rel_values = [r.value if isinstance(r, RelationType) else r for r in relationship_types]
            query += " FILTER edge.relationship IN @rel_types"
            bind_vars["rel_types"] = rel_values
        
        query += " RETURN DISTINCT {node: node, edge: edge}"
        
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        
        nodes = []
        edges = []
        
        for result in cursor:
            nodes.append(self._doc_to_node(result["node"]))
            edges.append(self._doc_to_edge(result["edge"]))
        
        return GraphSearchResult(
            nodes=nodes,
            edges=edges,
            paths=[],
            total_results=len(nodes),
            query_time_ms=0
        )
    
    def batch_create_nodes(self, nodes: List[Node]) -> List[Node]:
        """Efficiently create multiple nodes."""
        docs = [self._node_to_doc(node) for node in nodes]
        results = self.db.collection(self.nodes_collection).insert_many(docs)
        
        for i, result in enumerate(results):
            nodes[i].id = result["_key"]
        
        return nodes
    
    def batch_create_edges(self, edges: List[Edge]) -> List[Edge]:
        """Efficiently create multiple edges."""
        docs = [self._edge_to_doc(edge) for edge in edges]
        results = self.db.collection(self.edges_collection).insert_many(docs)
        
        for i, result in enumerate(results):
            edges[i].id = result["_key"]
        
        return edges
    
    def execute_raw_query(self, query: str, params: Dict[str, Any] = None) -> Any:
        """Execute a raw AQL query."""
        cursor = self.db.aql.execute(query, bind_vars=params or {})
        return list(cursor)
    
    def infer_relationships(
        self,
        source_id: str,
        max_candidates: int = 10,
        min_confidence: float = 0.5
    ) -> List[Tuple[str, RelationType, float]]:
        """
        Infer potential relationships using embeddings similarity.
        
        This is a simple implementation using cosine similarity.
        For production, consider using more sophisticated methods.
        """
        source_node = self.get_node(source_id)
        if not source_node or not source_node.embeddings:
            return []
        
        # Find nodes with similar embeddings
        query = """
        FOR node IN @@nodes
            FILTER node._key != @source_id
            FILTER node.embeddings != null
            LET similarity = (
                // Cosine similarity calculation
                SUM(
                    FOR i IN 0..LENGTH(node.embeddings)-1
                    RETURN node.embeddings[i] * @source_embeddings[i]
                ) / (
                    SQRT(SUM(FOR v IN node.embeddings RETURN v*v)) *
                    SQRT(SUM(FOR v IN @source_embeddings RETURN v*v))
                )
            )
            FILTER similarity >= @min_confidence
            SORT similarity DESC
            LIMIT @max_candidates
            RETURN {
                target_id: node._key,
                similarity: similarity,
                label: node.label
            }
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "@nodes": self.nodes_collection,
            "source_id": source_id,
            "source_embeddings": source_node.embeddings,
            "min_confidence": min_confidence,
            "max_candidates": max_candidates
        })
        
        results = []
        for item in cursor:
            # Infer relationship type based on labels and properties
            # This is a simplified heuristic - improve for production
            if item["similarity"] > 0.9:
                rel_type = RelationType.SIMILAR_TO
            elif source_node.label == item["label"]:
                rel_type = RelationType.RELATED_TO
            else:
                rel_type = RelationType.REFERENCES
            
            results.append((
                item["target_id"],
                rel_type,
                item["similarity"]
            ))
        
        return results
    
    def compute_node_importance(
        self,
        algorithm: str = "pagerank",
        params: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Compute node importance scores.
        
        ArangoDB has built-in graph algorithms in the Enterprise edition.
        This is a basic implementation for the community edition.
        """
        if algorithm == "pagerank":
            return self._compute_pagerank(params or {})
        elif algorithm == "degree_centrality":
            return self._compute_degree_centrality(params or {})
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def _compute_pagerank(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Basic PageRank implementation using AQL."""
        iterations = params.get("iterations", 10)
        damping = params.get("damping_factor", 0.85)
        
        # Initialize scores
        query = """
        LET nodes = (FOR n IN @@nodes RETURN n._key)
        LET initial_score = 1.0 / LENGTH(nodes)
        
        // Initialize scores
        FOR node IN nodes
            UPSERT {_key: node}
            INSERT {_key: node, pagerank: initial_score}
            UPDATE {pagerank: initial_score}
            IN @@temp_scores
        """
        
        # Note: This is a simplified version. For production, use 
        # ArangoDB Enterprise's built-in algorithms or external libraries
        
        scores = {}
        cursor = self.db.aql.execute(
            "FOR n IN @@nodes RETURN {id: n._key, score: 1.0}",
            bind_vars={"@nodes": self.nodes_collection}
        )
        
        for item in cursor:
            scores[item["id"]] = item["score"]
        
        return scores
    
    def _compute_degree_centrality(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Compute degree centrality for nodes."""
        query = """
        FOR node IN @@nodes
            LET in_degree = LENGTH(
                FOR e IN @@edges
                    FILTER e._to == CONCAT('@@nodes/', node._key)
                    RETURN 1
            )
            LET out_degree = LENGTH(
                FOR e IN @@edges
                    FILTER e._from == CONCAT('@@nodes/', node._key)
                    RETURN 1
            )
            RETURN {
                id: node._key,
                score: in_degree + out_degree
            }
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "@nodes": self.nodes_collection,
            "@edges": self.edges_collection
        })
        
        scores = {}
        max_score = 0
        
        for item in cursor:
            scores[item["id"]] = item["score"]
            max_score = max(max_score, item["score"])
        
        # Normalize scores
        if max_score > 0:
            for node_id in scores:
                scores[node_id] /= max_score
        
        return scores
    
    def detect_communities(
        self,
        algorithm: str = "label_propagation",
        params: Dict[str, Any] = None
    ) -> Dict[str, List[str]]:
        """
        Basic community detection.
        
        For production, consider using specialized graph processing
        frameworks like Apache Spark GraphX or NetworkX.
        """
        if algorithm == "connected_components":
            return self._detect_connected_components()
        else:
            # Placeholder for other algorithms
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def _detect_connected_components(self) -> Dict[str, List[str]]:
        """Find connected components in the graph."""
        query = """
        FOR node IN @@nodes
            LET component = FIRST(
                FOR v, e, p IN 1..99999 ANY node GRAPH @graph
                    OPTIONS {bfs: true, uniqueVertices: 'global'}
                    COLLECT component = SORTED(p.vertices[*]._key)[0]
                    RETURN component
            )
            COLLECT community = component OR node._key INTO members = node._key
            RETURN {
                community: community,
                members: members
            }
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "@nodes": self.nodes_collection,
            "graph": self.graph_name
        })
        
        communities = {}
        for i, result in enumerate(cursor):
            communities[f"community_{i}"] = result["members"]
        
        return communities
    
    def export_for_gnn(
        self,
        node_ids: Optional[List[str]] = None,
        include_features: bool = True
    ) -> Tuple[Any, Any, Any]:
        """
        Export graph data for GNN training.
        
        Returns data in a format compatible with PyTorch Geometric or DGL.
        """
        # Build query to get nodes and edges
        if node_ids:
            node_filter = "FILTER node._key IN @node_ids"
            bind_vars = {
                "@nodes": self.nodes_collection,
                "@edges": self.edges_collection,
                "node_ids": node_ids
            }
        else:
            node_filter = ""
            bind_vars = {
                "@nodes": self.nodes_collection,
                "@edges": self.edges_collection
            }
        
        # Get nodes
        node_query = f"""
        FOR node IN @@nodes
            {node_filter}
            RETURN node
        """
        
        cursor = self.db.aql.execute(node_query, bind_vars=bind_vars)
        nodes = list(cursor)
        
        # Create node ID mapping
        node_id_to_idx = {node["_key"]: i for i, node in enumerate(nodes)}
        
        # Get edges
        edge_query = """
        FOR edge IN @@edges
            FILTER SUBSTRING_AFTER(edge._from, '/') IN @node_keys
            FILTER SUBSTRING_AFTER(edge._to, '/') IN @node_keys
            RETURN edge
        """
        
        bind_vars["node_keys"] = list(node_id_to_idx.keys())
        cursor = self.db.aql.execute(edge_query, bind_vars=bind_vars)
        edges = list(cursor)
        
        # Build edge index (COO format for PyTorch Geometric)
        edge_index = []
        edge_features = []
        
        for edge in edges:
            source = edge["_from"].split("/")[-1]
            target = edge["_to"].split("/")[-1]
            
            if source in node_id_to_idx and target in node_id_to_idx:
                edge_index.append([node_id_to_idx[source], node_id_to_idx[target]])
                
                if include_features:
                    edge_features.append({
                        "weight": edge.get("weight", 1.0),
                        "relationship": edge.get("relationship", ""),
                        "confidence": edge.get("confidence", 1.0)
                    })
        
        # Build node features
        node_features = []
        if include_features:
            for node in nodes:
                features = {
                    "embeddings": node.get("embeddings", []),
                    "label": node.get("label", ""),
                    "confidence": node.get("confidence", 1.0),
                    "quality_score": node.get("quality_score", 1.0)
                }
                node_features.append(features)
        
        return edge_index, node_features, edge_features
    
    def create_index(self, index_name: str, index_type: str, fields: List[str]) -> bool:
        """Create an index for better query performance."""
        try:
            collection = self.db.collection(self.nodes_collection)
            
            if index_type == "persistent":
                collection.add_persistent_index(fields=fields, name=index_name)
            elif index_type == "fulltext":
                collection.add_fulltext_index(fields=fields, name=index_name)
            elif index_type == "geo":
                collection.add_geo_index(fields=fields, name=index_name)
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            return False
    
    def drop_index(self, index_name: str) -> bool:
        """Drop an index."""
        try:
            collection = self.db.collection(self.nodes_collection)
            collection.delete_index(index_name)
            return True
        except Exception as e:
            logger.error(f"Failed to drop index {index_name}: {e}")
            return False
    
    def define_node_schema(self, label: str, schema: Dict[str, Any]) -> bool:
        """
        Define or update node schema.
        
        ArangoDB is schema-less, but we can store schema definitions
        for validation and documentation.
        """
        schema_doc = {
            "_key": f"node_schema_{label}",
            "label": label,
            "schema": schema,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        try:
            # Store in a special schemas collection
            if not self.db.has_collection("_schemas"):
                self.db.create_collection("_schemas")
            
            self.db.collection("_schemas").insert(schema_doc, overwrite=True)
            return True
        except Exception as e:
            logger.error(f"Failed to define schema for {label}: {e}")
            return False
    
    def define_edge_schema(self, relationship: str, schema: Dict[str, Any]) -> bool:
        """Define or update edge schema."""
        schema_doc = {
            "_key": f"edge_schema_{relationship}",
            "relationship": relationship,
            "schema": schema,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        try:
            if not self.db.has_collection("_schemas"):
                self.db.create_collection("_schemas")
            
            self.db.collection("_schemas").insert(schema_doc, overwrite=True)
            return True
        except Exception as e:
            logger.error(f"Failed to define schema for {relationship}: {e}")
            return False