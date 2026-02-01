"""
High-performance Graph API for Memory Empire Knowledge Graph.

Provides efficient query building, caching, and advanced graph operations.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
from functools import lru_cache
import hashlib

# Optional caching backend
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..db.base import (
    GraphDatabase,
    GraphQuery,
    GraphSearchResult,
    Node,
    Edge,
    RelationType,
    GraphPath,
)

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of graph queries."""
    SEARCH = "search"
    TRAVERSE = "traverse"
    PATH = "path"
    AGGREGATE = "aggregate"
    PATTERN = "pattern"
    SUBGRAPH = "subgraph"


@dataclass
class GraphResponse:
    """Standard response format for graph API."""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    query_time_ms: float = 0.0
    cached: bool = False


@dataclass
class QueryBuilder:
    """
    Fluent interface for building complex graph queries.
    
    Example:
        query = (QueryBuilder()
                .match_nodes(labels=["Person", "Organization"])
                .with_properties({"active": True})
                .connected_by(["WORKS_FOR", "COLLABORATES_WITH"])
                .within_hops(3)
                .having_confidence(0.8)
                .limit(50)
                .build())
    """
    
    _query: GraphQuery = field(default_factory=GraphQuery)
    
    def match_nodes(self, labels: List[str]) -> 'QueryBuilder':
        """Match nodes with specific labels."""
        self._query.node_labels = labels
        return self
    
    def with_properties(self, properties: Dict[str, Any]) -> 'QueryBuilder':
        """Filter nodes by properties."""
        self._query.node_properties.update(properties)
        return self
    
    def connected_by(self, relationships: List[Union[str, RelationType]]) -> 'QueryBuilder':
        """Filter by relationship types."""
        self._query.edge_types = relationships
        return self
    
    def within_hops(self, max_depth: int) -> 'QueryBuilder':
        """Set maximum traversal depth."""
        self._query.max_depth = max_depth
        return self
    
    def in_direction(self, direction: str) -> 'QueryBuilder':
        """Set traversal direction: 'in', 'out', or 'both'."""
        if direction not in ("in", "out", "both"):
            raise ValueError("Direction must be 'in', 'out', or 'both'")
        self._query.direction = direction
        return self
    
    def at_time(self, time_point: datetime) -> 'QueryBuilder':
        """Query at a specific point in time."""
        self._query.time_point = time_point
        return self
    
    def between_times(self, start: datetime, end: datetime) -> 'QueryBuilder':
        """Query within a time range."""
        self._query.time_range = (start, end)
        return self
    
    def having_confidence(self, min_confidence: float) -> 'QueryBuilder':
        """Set minimum confidence threshold."""
        self._query.min_confidence = min_confidence
        return self
    
    def with_pattern(self, pattern: str) -> 'QueryBuilder':
        """Set Cypher-like pattern matching."""
        self._query.pattern = pattern
        return self
    
    def limit(self, limit: int) -> 'QueryBuilder':
        """Limit number of results."""
        self._query.limit = limit
        return self
    
    def skip(self, skip: int) -> 'QueryBuilder':
        """Skip results for pagination."""
        self._query.skip = skip
        return self
    
    def include_embeddings(self, include: bool = True) -> 'QueryBuilder':
        """Include node embeddings in results."""
        self._query.include_embeddings = include
        return self
    
    def enable_inference(self, enable: bool = True) -> 'QueryBuilder':
        """Enable relationship inference."""
        self._query.use_inference = enable
        return self
    
    def build(self) -> GraphQuery:
        """Build and return the query."""
        return self._query
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary for serialization."""
        return {
            "node_labels": self._query.node_labels,
            "node_properties": self._query.node_properties,
            "edge_types": [
                e.value if isinstance(e, RelationType) else e
                for e in self._query.edge_types
            ],
            "edge_properties": self._query.edge_properties,
            "pattern": self._query.pattern,
            "max_depth": self._query.max_depth,
            "direction": self._query.direction,
            "time_point": self._query.time_point.isoformat() if self._query.time_point else None,
            "time_range": (
                [t.isoformat() for t in self._query.time_range]
                if self._query.time_range else None
            ),
            "limit": self._query.limit,
            "skip": self._query.skip,
            "include_properties": self._query.include_properties,
            "include_embeddings": self._query.include_embeddings,
            "use_inference": self._query.use_inference,
            "min_confidence": self._query.min_confidence,
            "aggregate_temporal": self._query.aggregate_temporal,
        }


class GraphAPI:
    """
    High-performance Graph API with caching and optimization.
    
    Features:
    - Query building and optimization
    - Result caching (in-memory and Redis)
    - Async operations
    - Batch operations
    - Query performance monitoring
    - Graph algorithms
    """
    
    def __init__(
        self,
        graph_db: GraphDatabase,
        cache_ttl: int = 3600,
        enable_redis_cache: bool = False,
        redis_url: str = "redis://localhost:6379",
        max_cache_size: int = 1000,
        enable_query_optimization: bool = True
    ):
        self.graph_db = graph_db
        self.cache_ttl = cache_ttl
        self.enable_redis_cache = enable_redis_cache and REDIS_AVAILABLE
        self.max_cache_size = max_cache_size
        self.enable_query_optimization = enable_query_optimization
        
        # Initialize caches
        self._init_caches(redis_url)
        
        # Query statistics
        self.query_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_query_time_ms": 0.0,
        }
    
    def _init_caches(self, redis_url: str) -> None:
        """Initialize caching backends."""
        # In-memory LRU cache
        self._query_cache = {}
        self._cache_order = []
        
        # Redis cache
        if self.enable_redis_cache:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using in-memory cache only.")
                self.enable_redis_cache = False
                self.redis_client = None
        else:
            self.redis_client = None
    
    def _get_cache_key(self, query_type: str, params: Dict[str, Any]) -> str:
        """Generate cache key for query."""
        # Create stable hash of query parameters
        param_str = json.dumps(params, sort_keys=True, default=str)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        return f"graph:{query_type}:{param_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get result from cache."""
        # Try in-memory cache first
        if cache_key in self._query_cache:
            self.query_stats["cache_hits"] += 1
            return self._query_cache[cache_key]
        
        # Try Redis cache
        if self.enable_redis_cache and self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    self.query_stats["cache_hits"] += 1
                    return json.loads(cached)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        self.query_stats["cache_misses"] += 1
        return None
    
    def _set_cache(self, cache_key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set result in cache."""
        ttl = ttl or self.cache_ttl
        
        # In-memory cache with LRU eviction
        self._query_cache[cache_key] = value
        self._cache_order.append(cache_key)
        
        # Evict old entries if cache is full
        if len(self._query_cache) > self.max_cache_size:
            oldest_key = self._cache_order.pop(0)
            self._query_cache.pop(oldest_key, None)
        
        # Redis cache
        if self.enable_redis_cache and self.redis_client:
            try:
                serialized = json.dumps(value, default=str)
                self.redis_client.setex(cache_key, ttl, serialized)
            except Exception as e:
                logger.error(f"Redis set error: {e}")
    
    def search(
        self,
        query: Union[GraphQuery, QueryBuilder],
        cache_ttl: Optional[int] = None,
        force_refresh: bool = False
    ) -> GraphResponse:
        """
        Execute a graph search query.
        
        Args:
            query: GraphQuery or QueryBuilder instance
            cache_ttl: Override default cache TTL
            force_refresh: Force bypass cache
            
        Returns:
            GraphResponse with search results
        """
        start_time = datetime.utcnow()
        self.query_stats["total_queries"] += 1
        
        # Convert QueryBuilder to GraphQuery if needed
        if isinstance(query, QueryBuilder):
            query = query.build()
        
        # Generate cache key
        cache_key = self._get_cache_key("search", query.__dict__)
        
        # Check cache unless force refresh
        if not force_refresh:
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return GraphResponse(
                    success=True,
                    data=cached_result,
                    cached=True,
                    query_time_ms=0.0
                )
        
        try:
            # Optimize query if enabled
            if self.enable_query_optimization:
                query = self._optimize_query(query)
            
            # Execute search
            result = self.graph_db.search(query)
            
            # Convert result to serializable format
            data = {
                "nodes": [self._node_to_dict(n) for n in result.nodes],
                "edges": [self._edge_to_dict(e) for e in result.edges],
                "paths": [self._path_to_dict(p) for p in result.paths],
                "total_results": result.total_results,
            }
            
            # Cache result
            self._set_cache(cache_key, data, cache_ttl)
            
            query_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.query_stats["total_query_time_ms"] += query_time
            
            return GraphResponse(
                success=True,
                data=data,
                query_time_ms=query_time
            )
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return GraphResponse(
                success=False,
                data=None,
                error=str(e),
                query_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3,
        relationship_types: Optional[List[Union[RelationType, str]]] = None,
        algorithm: str = "shortest",
        limit: int = 10
    ) -> GraphResponse:
        """
        Find paths between nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_depth: Maximum path length
            relationship_types: Filter by relationship types
            algorithm: Path algorithm ('shortest', 'all', 'weighted')
            limit: Maximum paths to return
            
        Returns:
            GraphResponse with paths
        """
        start_time = datetime.utcnow()
        
        try:
            paths = self.graph_db.find_paths(
                source_id=source_id,
                target_id=target_id,
                max_depth=max_depth,
                relationship_types=relationship_types
            )
            
            # Apply algorithm-specific filtering
            if algorithm == "shortest" and paths:
                min_length = min(p.length for p in paths)
                paths = [p for p in paths if p.length == min_length]
            elif algorithm == "weighted":
                paths.sort(key=lambda p: p.total_weight)
            
            # Apply limit
            paths = paths[:limit]
            
            data = {
                "paths": [self._path_to_dict(p) for p in paths],
                "source_id": source_id,
                "target_id": target_id,
            }
            
            query_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return GraphResponse(
                success=True,
                data=data,
                query_time_ms=query_time,
                metadata={"algorithm": algorithm, "max_depth": max_depth}
            )
            
        except Exception as e:
            logger.error(f"Path finding error: {e}")
            return GraphResponse(
                success=False,
                data=None,
                error=str(e),
                query_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    def traverse(
        self,
        start_nodes: List[str],
        direction: str = "out",
        max_depth: int = 2,
        relationship_types: Optional[List[Union[RelationType, str]]] = None,
        node_filter: Optional[Dict[str, Any]] = None,
        collect_paths: bool = False
    ) -> GraphResponse:
        """
        Traverse graph from starting nodes.
        
        Args:
            start_nodes: List of starting node IDs
            direction: Traversal direction ('in', 'out', 'both')
            max_depth: Maximum traversal depth
            relationship_types: Filter by relationship types
            node_filter: Filter nodes by properties
            collect_paths: Whether to collect full paths
            
        Returns:
            GraphResponse with traversal results
        """
        start_time = datetime.utcnow()
        
        try:
            visited_nodes = set()
            visited_edges = set()
            all_paths = []
            
            # BFS traversal from each start node
            for start_id in start_nodes:
                result = self.graph_db.get_neighbors(
                    node_id=start_id,
                    direction=direction,
                    relationship_types=relationship_types,
                    max_depth=max_depth
                )
                
                # Apply node filters
                if node_filter:
                    filtered_nodes = []
                    for node in result.nodes:
                        if self._matches_filter(node, node_filter):
                            filtered_nodes.append(node)
                    result.nodes = filtered_nodes
                
                # Collect unique nodes and edges
                for node in result.nodes:
                    visited_nodes.add(node.id)
                
                for edge in result.edges:
                    visited_edges.add(edge.id)
                
                # Optionally collect paths
                if collect_paths:
                    # This would require path tracking in the traversal
                    pass
            
            data = {
                "nodes": list(visited_nodes),
                "edges": list(visited_edges),
                "paths": all_paths if collect_paths else [],
                "start_nodes": start_nodes,
            }
            
            query_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return GraphResponse(
                success=True,
                data=data,
                query_time_ms=query_time,
                metadata={
                    "direction": direction,
                    "max_depth": max_depth,
                    "total_nodes": len(visited_nodes),
                    "total_edges": len(visited_edges),
                }
            )
            
        except Exception as e:
            logger.error(f"Traversal error: {e}")
            return GraphResponse(
                success=False,
                data=None,
                error=str(e),
                query_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    def aggregate(
        self,
        aggregation_type: str,
        group_by: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        time_window: Optional[str] = None
    ) -> GraphResponse:
        """
        Perform graph aggregations.
        
        Args:
            aggregation_type: Type of aggregation ('count', 'degree', 'centrality')
            group_by: Property to group by
            filters: Filter criteria
            time_window: Time window for temporal aggregations
            
        Returns:
            GraphResponse with aggregation results
        """
        start_time = datetime.utcnow()
        
        try:
            if aggregation_type == "count":
                data = self._aggregate_counts(group_by, filters)
            elif aggregation_type == "degree":
                data = self._aggregate_degrees(filters)
            elif aggregation_type == "centrality":
                data = self.graph_db.compute_node_importance()
            else:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
            
            query_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return GraphResponse(
                success=True,
                data=data,
                query_time_ms=query_time,
                metadata={"aggregation_type": aggregation_type}
            )
            
        except Exception as e:
            logger.error(f"Aggregation error: {e}")
            return GraphResponse(
                success=False,
                data=None,
                error=str(e),
                query_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    def pattern_match(
        self,
        pattern: str,
        params: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> GraphResponse:
        """
        Match graph patterns using Cypher-like syntax.
        
        Args:
            pattern: Pattern string (e.g., "(a:Person)-[:KNOWS]->(b:Person)")
            params: Pattern parameters
            limit: Result limit
            
        Returns:
            GraphResponse with pattern matches
        """
        query = QueryBuilder().with_pattern(pattern).limit(limit).build()
        return self.search(query)
    
    def subgraph(
        self,
        node_ids: List[str],
        include_edges: bool = True,
        expand_depth: int = 0
    ) -> GraphResponse:
        """
        Extract a subgraph containing specified nodes.
        
        Args:
            node_ids: List of node IDs to include
            include_edges: Include edges between nodes
            expand_depth: Expand subgraph by N hops
            
        Returns:
            GraphResponse with subgraph
        """
        start_time = datetime.utcnow()
        
        try:
            nodes = []
            edges = []
            
            # Get specified nodes
            for node_id in node_ids:
                node = self.graph_db.get_node(node_id)
                if node:
                    nodes.append(node)
            
            # Expand if requested
            if expand_depth > 0:
                expanded_ids = set(node_ids)
                for _ in range(expand_depth):
                    for node_id in list(expanded_ids):
                        neighbors = self.graph_db.get_neighbors(
                            node_id, direction="both", max_depth=1
                        )
                        for neighbor in neighbors.nodes:
                            if neighbor.id not in expanded_ids:
                                expanded_ids.add(neighbor.id)
                                nodes.append(neighbor)
            
            # Get edges if requested
            if include_edges:
                # This would require a query to find all edges between nodes
                # Implementation depends on the specific database
                pass
            
            data = {
                "nodes": [self._node_to_dict(n) for n in nodes],
                "edges": [self._edge_to_dict(e) for e in edges],
                "node_count": len(nodes),
                "edge_count": len(edges),
            }
            
            query_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return GraphResponse(
                success=True,
                data=data,
                query_time_ms=query_time
            )
            
        except Exception as e:
            logger.error(f"Subgraph extraction error: {e}")
            return GraphResponse(
                success=False,
                data=None,
                error=str(e),
                query_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    # Async methods for high-performance operations
    
    async def search_async(
        self,
        query: Union[GraphQuery, QueryBuilder],
        cache_ttl: Optional[int] = None,
        force_refresh: bool = False
    ) -> GraphResponse:
        """Async version of search."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.search,
            query,
            cache_ttl,
            force_refresh
        )
    
    async def batch_search_async(
        self,
        queries: List[Union[GraphQuery, QueryBuilder]],
        max_concurrent: int = 10
    ) -> List[GraphResponse]:
        """Execute multiple searches concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def search_with_limit(query):
            async with semaphore:
                return await self.search_async(query)
        
        tasks = [search_with_limit(query) for query in queries]
        return await asyncio.gather(*tasks)
    
    # Helper methods
    
    def _optimize_query(self, query: GraphQuery) -> GraphQuery:
        """
        Optimize query for better performance.
        
        Optimizations:
        - Add indexes hints
        - Reorder filters
        - Limit early termination
        """
        # This is a placeholder for query optimization logic
        # Actual optimizations depend on the specific database
        return query
    
    def _matches_filter(self, node: Node, filters: Dict[str, Any]) -> bool:
        """Check if node matches filter criteria."""
        for key, value in filters.items():
            if key in node.properties:
                if node.properties[key] != value:
                    return False
            elif hasattr(node, key):
                if getattr(node, key) != value:
                    return False
            else:
                return False
        return True
    
    def _node_to_dict(self, node: Node) -> Dict[str, Any]:
        """Convert Node to dictionary."""
        return {
            "id": node.id,
            "label": node.label,
            "properties": node.properties,
            "confidence": node.confidence,
            "quality_score": node.quality_score,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
            "source_type": node.source_type,
            "source_id": node.source_id,
        }
    
    def _edge_to_dict(self, edge: Edge) -> Dict[str, Any]:
        """Convert Edge to dictionary."""
        return {
            "id": edge.id,
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "relationship": edge.relationship.value if isinstance(edge.relationship, RelationType) else edge.relationship,
            "properties": edge.properties,
            "weight": edge.weight,
            "confidence": edge.confidence,
            "created_at": edge.created_at.isoformat(),
        }
    
    def _path_to_dict(self, path: GraphPath) -> Dict[str, Any]:
        """Convert GraphPath to dictionary."""
        return {
            "nodes": [self._node_to_dict(n) for n in path.nodes],
            "edges": [self._edge_to_dict(e) for e in path.edges],
            "length": path.length,
            "total_weight": path.total_weight,
        }
    
    def _aggregate_counts(
        self,
        group_by: Optional[str],
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate node/edge counts."""
        # This would be implemented with database-specific aggregation queries
        return {
            "total_nodes": 0,
            "total_edges": 0,
            "groups": {}
        }
    
    def _aggregate_degrees(
        self,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate node degrees."""
        # This would be implemented with database-specific queries
        return {
            "degree_distribution": {},
            "avg_degree": 0.0,
            "max_degree": 0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        cache_hit_rate = (
            self.query_stats["cache_hits"] / 
            max(1, self.query_stats["total_queries"])
        ) * 100
        
        avg_query_time = (
            self.query_stats["total_query_time_ms"] /
            max(1, self.query_stats["total_queries"])
        )
        
        return {
            "total_queries": self.query_stats["total_queries"],
            "cache_hits": self.query_stats["cache_hits"],
            "cache_misses": self.query_stats["cache_misses"],
            "cache_hit_rate": f"{cache_hit_rate:.2f}%",
            "avg_query_time_ms": f"{avg_query_time:.2f}",
            "cache_size": len(self._query_cache),
        }
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._query_cache.clear()
        self._cache_order.clear()
        
        if self.enable_redis_cache and self.redis_client:
            try:
                # Clear graph-related keys
                for key in self.redis_client.scan_iter("graph:*"):
                    self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis clear error: {e}")
        
        logger.info("Graph API cache cleared")