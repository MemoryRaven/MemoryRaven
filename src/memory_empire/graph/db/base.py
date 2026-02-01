"""
Base graph database abstractions for Memory Empire.

This module provides a high-performance, feature-rich interface for graph operations
including temporal graphs, pattern matching, and graph neural network integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union
import uuid

# Type variable for generic graph elements
T = TypeVar("T")


class RelationType(Enum):
    """Standard relationship types for knowledge graphs."""
    MENTIONS = "mentions"
    REFERENCES = "references"
    AUTHORED_BY = "authored_by"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    FOLLOWS = "follows"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    DERIVED_FROM = "derived_from"
    SIMILAR_TO = "similar_to"
    TEMPORAL_BEFORE = "temporal_before"
    TEMPORAL_AFTER = "temporal_after"
    CUSTOM = "custom"


@dataclass
class Node:
    """Represents a node in the knowledge graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None  # For GNN integration
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Temporal properties
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    
    # Quality and confidence scores
    confidence: float = 1.0
    quality_score: float = 1.0
    
    # Source tracking
    source_id: Optional[str] = None
    source_type: Optional[str] = None  # e.g., "document", "memory", "web"


@dataclass
class Edge:
    """Represents an edge (relationship) in the knowledge graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relationship: Union[RelationType, str] = RelationType.RELATED_TO
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Temporal properties
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    
    # Confidence and provenance
    confidence: float = 1.0
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class GraphQuery:
    """Encapsulates a graph query with various options."""
    # Pattern matching
    pattern: Optional[str] = None  # Cypher-like pattern
    
    # Node filters
    node_labels: List[str] = field(default_factory=list)
    node_properties: Dict[str, Any] = field(default_factory=dict)
    
    # Edge filters
    edge_types: List[Union[RelationType, str]] = field(default_factory=list)
    edge_properties: Dict[str, Any] = field(default_factory=dict)
    
    # Traversal options
    max_depth: int = 3
    direction: str = "both"  # "in", "out", "both"
    
    # Temporal filters
    time_point: Optional[datetime] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    
    # Result options
    limit: int = 100
    skip: int = 0
    include_properties: bool = True
    include_embeddings: bool = False
    
    # Advanced options
    use_inference: bool = False  # Enable relationship inference
    min_confidence: float = 0.0
    aggregate_temporal: bool = False  # Aggregate temporal versions


@dataclass
class GraphPath:
    """Represents a path through the graph."""
    nodes: List[Node]
    edges: List[Edge]
    total_weight: float = 0.0
    
    @property
    def length(self) -> int:
        return len(self.edges)


@dataclass
class GraphSearchResult:
    """Results from a graph search query."""
    nodes: List[Node]
    edges: List[Edge]
    paths: List[GraphPath]
    total_results: int
    query_time_ms: float


class TemporalGraph:
    """Handles temporal graph operations."""
    
    def __init__(self, graph_db: 'GraphDatabase'):
        self.graph_db = graph_db
    
    def snapshot_at(self, time_point: datetime) -> GraphSearchResult:
        """Get graph snapshot at a specific time."""
        query = GraphQuery(time_point=time_point)
        return self.graph_db.search(query)
    
    def evolution_between(
        self, 
        start_time: datetime, 
        end_time: datetime,
        entity_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Track evolution of the graph or specific entities over time."""
        # Implementation depends on specific database
        raise NotImplementedError


class GraphDatabase(ABC):
    """Abstract base class for graph database implementations."""
    
    @abstractmethod
    def connect(self, **kwargs) -> None:
        """Establish connection to the graph database."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection to the graph database."""
        pass
    
    @abstractmethod
    def create_node(self, node: Node) -> Node:
        """Create a new node in the graph."""
        pass
    
    @abstractmethod
    def create_edge(self, edge: Edge) -> Edge:
        """Create a new edge in the graph."""
        pass
    
    @abstractmethod
    def update_node(self, node_id: str, updates: Dict[str, Any]) -> Node:
        """Update an existing node."""
        pass
    
    @abstractmethod
    def update_edge(self, edge_id: str, updates: Dict[str, Any]) -> Edge:
        """Update an existing edge."""
        pass
    
    @abstractmethod
    def delete_node(self, node_id: str, cascade: bool = False) -> bool:
        """Delete a node (optionally cascade delete edges)."""
        pass
    
    @abstractmethod
    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge."""
        pass
    
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by ID."""
        pass
    
    @abstractmethod
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Retrieve an edge by ID."""
        pass
    
    @abstractmethod
    def search(self, query: GraphQuery) -> GraphSearchResult:
        """Execute a graph search query."""
        pass
    
    @abstractmethod
    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3,
        relationship_types: Optional[List[Union[RelationType, str]]] = None
    ) -> List[GraphPath]:
        """Find paths between two nodes."""
        pass
    
    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        relationship_types: Optional[List[Union[RelationType, str]]] = None,
        max_depth: int = 1
    ) -> GraphSearchResult:
        """Get neighboring nodes."""
        pass
    
    @abstractmethod
    def batch_create_nodes(self, nodes: List[Node]) -> List[Node]:
        """Efficiently create multiple nodes."""
        pass
    
    @abstractmethod
    def batch_create_edges(self, edges: List[Edge]) -> List[Edge]:
        """Efficiently create multiple edges."""
        pass
    
    @abstractmethod
    def execute_raw_query(self, query: str, params: Dict[str, Any] = None) -> Any:
        """Execute a raw database-specific query (e.g., AQL, Cypher)."""
        pass
    
    # Advanced features
    
    @abstractmethod
    def infer_relationships(
        self,
        source_id: str,
        max_candidates: int = 10,
        min_confidence: float = 0.5
    ) -> List[Tuple[str, RelationType, float]]:
        """
        Infer potential relationships using embeddings or graph patterns.
        Returns list of (target_id, relationship_type, confidence).
        """
        pass
    
    @abstractmethod
    def compute_node_importance(
        self,
        algorithm: str = "pagerank",
        params: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """Compute node importance scores using graph algorithms."""
        pass
    
    @abstractmethod
    def detect_communities(
        self,
        algorithm: str = "louvain",
        params: Dict[str, Any] = None
    ) -> Dict[str, List[str]]:
        """Detect communities in the graph."""
        pass
    
    @abstractmethod
    def export_for_gnn(
        self,
        node_ids: Optional[List[str]] = None,
        include_features: bool = True
    ) -> Tuple[Any, Any, Any]:
        """
        Export graph data for GNN training.
        Returns (edge_index, node_features, edge_features).
        """
        pass
    
    def get_temporal_handler(self) -> TemporalGraph:
        """Get temporal graph handler."""
        return TemporalGraph(self)
    
    # Index management
    
    @abstractmethod
    def create_index(self, index_name: str, index_type: str, fields: List[str]) -> bool:
        """Create an index for better query performance."""
        pass
    
    @abstractmethod
    def drop_index(self, index_name: str) -> bool:
        """Drop an index."""
        pass
    
    # Schema management
    
    @abstractmethod
    def define_node_schema(self, label: str, schema: Dict[str, Any]) -> bool:
        """Define or update node schema."""
        pass
    
    @abstractmethod
    def define_edge_schema(self, relationship: str, schema: Dict[str, Any]) -> bool:
        """Define or update edge schema."""
        pass