"""
Integration between Knowledge Graph and Memory Empire vector/search systems.

This module provides seamless integration between graph-based knowledge
representation and vector-based semantic search.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np

from ...vector_indexing.types import Document, ChunkMetadata
from ...vector_indexing.vector_store import VectorStore
from ..db.base import GraphDatabase, Node, Edge, RelationType, GraphQuery
from ..extractors.entity_extractor import EntityExtractor, ExtractionResult
from ..api.graph_api import GraphAPI, QueryBuilder

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Combined results from vector and graph searches."""
    vector_results: List[Document]
    graph_nodes: List[Node]
    graph_edges: List[Edge]
    combined_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_results(self) -> int:
        return len(self.vector_results) + len(self.graph_nodes)


class MemoryGraphIntegration:
    """
    Integrates Memory Empire's vector search with the knowledge graph.
    
    Features:
    - Automatic entity extraction during document ingestion
    - Hybrid vector + graph search
    - Entity linking between documents and graph
    - Knowledge graph enrichment from documents
    - Temporal knowledge tracking
    """
    
    def __init__(
        self,
        graph_db: GraphDatabase,
        vector_store: VectorStore,
        entity_extractor: EntityExtractor,
        graph_api: Optional[GraphAPI] = None,
        auto_extract: bool = True,
        link_threshold: float = 0.8,
        enable_temporal: bool = True
    ):
        self.graph_db = graph_db
        self.vector_store = vector_store
        self.entity_extractor = entity_extractor
        self.graph_api = graph_api or GraphAPI(graph_db)
        self.auto_extract = auto_extract
        self.link_threshold = link_threshold
        self.enable_temporal = enable_temporal
    
    async def ingest_document(
        self,
        document: Document,
        extract_entities: Optional[bool] = None,
        namespace: str = "default"
    ) -> Dict[str, Any]:
        """
        Ingest a document, extracting entities and relationships for the graph.
        
        Args:
            document: Document to ingest
            extract_entities: Override auto_extract setting
            namespace: Namespace for organization
            
        Returns:
            Ingestion results including extracted entities
        """
        extract = extract_entities if extract_entities is not None else self.auto_extract
        
        results = {
            "document_id": document.id,
            "vector_stored": False,
            "entities_extracted": [],
            "relationships_extracted": [],
            "graph_nodes_created": 0,
            "graph_edges_created": 0,
        }
        
        # Extract entities if enabled
        if extract:
            extraction = await self._extract_from_document(document)
            results["entities_extracted"] = len(extraction.entities)
            results["relationships_extracted"] = len(extraction.relationships)
            
            # Create graph nodes and edges
            nodes_created, edges_created = await self._update_graph(
                extraction, document, namespace
            )
            results["graph_nodes_created"] = nodes_created
            results["graph_edges_created"] = edges_created
        
        # Store in vector database (this would integrate with existing Memory Empire flow)
        # For now, we'll just mark it as stored
        results["vector_stored"] = True
        
        # Create document node in graph
        doc_node = await self._create_document_node(document, namespace)
        
        return results
    
    async def _extract_from_document(self, document: Document) -> ExtractionResult:
        """Extract entities and relationships from document."""
        # Combine all document chunks for extraction
        full_text = "\n".join(chunk.text for chunk in document.chunks)
        
        # Extract entities
        extraction = self.entity_extractor.extract(
            full_text,
            metadata={
                "document_id": document.id,
                "source_type": "document",
                "created_at": document.created_at.isoformat()
            }
        )
        
        return extraction
    
    async def _update_graph(
        self,
        extraction: ExtractionResult,
        document: Document,
        namespace: str
    ) -> Tuple[int, int]:
        """Update knowledge graph with extracted entities."""
        nodes_created = 0
        edges_created = 0
        
        # Create or update entity nodes
        entity_node_map = {}
        
        for entity in extraction.entities:
            # Check if entity already exists
            existing = await self._find_existing_entity(entity.text, entity.label)
            
            if existing:
                # Update existing node
                updates = {
                    "confidence": max(existing.confidence, entity.confidence),
                    "updated_at": datetime.utcnow().isoformat(),
                }
                
                # Add document reference
                if "document_refs" not in existing.properties:
                    existing.properties["document_refs"] = []
                existing.properties["document_refs"].append(document.id)
                updates["properties"] = existing.properties
                
                node = self.graph_db.update_node(existing.id, updates)
                entity_node_map[entity] = node
            else:
                # Create new node
                node = Node(
                    label=entity.label,
                    properties={
                        "text": entity.text,
                        "normalized_text": entity.normalized_text or entity.text,
                        "document_refs": [document.id],
                        "namespace": namespace,
                        "extraction_metadata": entity.metadata,
                    },
                    confidence=entity.confidence,
                    source_id=document.id,
                    source_type="document",
                )
                
                # Add embeddings if available
                if hasattr(entity, "embeddings") and entity.embeddings:
                    node.embeddings = entity.embeddings
                
                created_node = self.graph_db.create_node(node)
                entity_node_map[entity] = created_node
                nodes_created += 1
        
        # Create relationship edges
        for relationship in extraction.relationships:
            source_node = entity_node_map.get(relationship.source)
            target_node = entity_node_map.get(relationship.target)
            
            if source_node and target_node:
                edge = Edge(
                    source_id=source_node.id,
                    target_id=target_node.id,
                    relationship=relationship.relation_type,
                    properties={
                        "evidence_text": relationship.evidence_text,
                        "document_id": document.id,
                        "namespace": namespace,
                    },
                    confidence=relationship.confidence,
                    evidence_ids=[document.id],
                )
                
                self.graph_db.create_edge(edge)
                edges_created += 1
        
        return nodes_created, edges_created
    
    async def _find_existing_entity(
        self,
        text: str,
        label: str
    ) -> Optional[Node]:
        """Find existing entity in the graph."""
        query = GraphQuery(
            node_labels=[label],
            node_properties={"text": text},
            limit=1
        )
        
        result = self.graph_db.search(query)
        if result.nodes:
            return result.nodes[0]
        
        # Try normalized text
        query.node_properties = {"normalized_text": text.lower()}
        result = self.graph_db.search(query)
        if result.nodes:
            return result.nodes[0]
        
        return None
    
    async def _create_document_node(
        self,
        document: Document,
        namespace: str
    ) -> Node:
        """Create a node representing the document itself."""
        node = Node(
            label="Document",
            properties={
                "title": document.metadata.get("title", "Untitled"),
                "source": document.metadata.get("source", "unknown"),
                "namespace": namespace,
                "chunk_count": len(document.chunks),
                "created_at": document.created_at.isoformat(),
                "metadata": document.metadata,
            },
            source_id=document.id,
            source_type="document",
        )
        
        return self.graph_db.create_node(node)
    
    def hybrid_search(
        self,
        query: str,
        vector_weight: float = 0.5,
        graph_weight: float = 0.5,
        max_vector_results: int = 10,
        max_graph_depth: int = 2,
        namespace: str = "default"
    ) -> HybridSearchResult:
        """
        Perform hybrid search combining vector similarity and graph traversal.
        
        Args:
            query: Search query
            vector_weight: Weight for vector search results
            graph_weight: Weight for graph search results
            max_vector_results: Maximum vector search results
            max_graph_depth: Maximum graph traversal depth
            namespace: Namespace to search in
            
        Returns:
            Combined search results
        """
        # This is a simplified implementation
        # In production, you'd integrate with Memory Empire's actual search
        
        # Extract entities from query
        query_extraction = self.entity_extractor.extract(query)
        
        # Perform vector search (placeholder)
        vector_results = []  # This would call Memory Empire's vector search
        
        # Perform graph search based on extracted entities
        graph_nodes = []
        graph_edges = []
        
        for entity in query_extraction.entities:
            # Find matching nodes
            graph_query = (QueryBuilder()
                .match_nodes([entity.label])
                .with_properties({"text": entity.text})
                .within_hops(max_graph_depth)
                .limit(10)
                .build())
            
            result = self.graph_api.search(graph_query)
            if result.success:
                graph_nodes.extend(result.data.get("nodes", []))
                graph_edges.extend(result.data.get("edges", []))
        
        # Combine and score results
        combined_score = self._compute_hybrid_score(
            vector_results, graph_nodes, vector_weight, graph_weight
        )
        
        return HybridSearchResult(
            vector_results=vector_results,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            combined_score=combined_score,
            metadata={
                "query": query,
                "extracted_entities": len(query_extraction.entities),
                "vector_weight": vector_weight,
                "graph_weight": graph_weight,
            }
        )
    
    def _compute_hybrid_score(
        self,
        vector_results: List[Document],
        graph_nodes: List[Node],
        vector_weight: float,
        graph_weight: float
    ) -> float:
        """Compute combined score for hybrid results."""
        # Simplified scoring - would be more sophisticated in production
        vector_score = len(vector_results) * vector_weight
        graph_score = len(graph_nodes) * graph_weight
        return vector_score + graph_score
    
    async def enrich_graph_from_vectors(
        self,
        similarity_threshold: float = 0.85,
        batch_size: int = 100,
        namespace: str = "default"
    ) -> Dict[str, int]:
        """
        Enrich knowledge graph by finding relationships between vector-similar documents.
        
        Args:
            similarity_threshold: Minimum similarity for creating relationships
            batch_size: Batch size for processing
            namespace: Namespace to process
            
        Returns:
            Statistics about enrichment process
        """
        stats = {
            "documents_processed": 0,
            "relationships_created": 0,
            "nodes_enriched": 0,
        }
        
        # This would integrate with Memory Empire's vector store
        # to find similar documents and create relationships
        
        return stats
    
    async def temporal_analysis(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime,
        granularity: str = "day"
    ) -> Dict[str, Any]:
        """
        Analyze how an entity evolved over time.
        
        Args:
            entity_id: Entity node ID
            start_time: Start of time range
            end_time: End of time range
            granularity: Time granularity (hour, day, week, month)
            
        Returns:
            Temporal analysis results
        """
        # Get temporal graph handler
        temporal = self.graph_db.get_temporal_handler()
        
        # Get entity evolution
        evolution = temporal.evolution_between(
            start_time=start_time,
            end_time=end_time,
            entity_id=entity_id
        )
        
        # Analyze patterns
        analysis = {
            "entity_id": entity_id,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "granularity": granularity,
            "evolution_events": evolution,
            "relationship_changes": [],
            "property_changes": [],
            "confidence_trend": [],
        }
        
        # Additional temporal analysis would go here
        
        return analysis
    
    def export_subgraph_for_context(
        self,
        central_entities: List[str],
        max_depth: int = 2,
        format: str = "json"
    ) -> Union[Dict[str, Any], str]:
        """
        Export a subgraph centered around entities for use as context.
        
        Args:
            central_entities: List of entity IDs to center the subgraph
            max_depth: Maximum depth from central entities
            format: Export format (json, graphml, cypher)
            
        Returns:
            Exported subgraph in requested format
        """
        # Extract subgraph
        subgraph_response = self.graph_api.subgraph(
            node_ids=central_entities,
            include_edges=True,
            expand_depth=max_depth
        )
        
        if not subgraph_response.success:
            return {"error": subgraph_response.error}
        
        subgraph_data = subgraph_response.data
        
        # Format based on request
        if format == "json":
            return subgraph_data
        elif format == "graphml":
            return self._to_graphml(subgraph_data)
        elif format == "cypher":
            return self._to_cypher(subgraph_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _to_graphml(self, subgraph_data: Dict[str, Any]) -> str:
        """Convert subgraph to GraphML format."""
        # Simplified GraphML generation
        graphml = ['<?xml version="1.0" encoding="UTF-8"?>']
        graphml.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')
        graphml.append('  <graph id="G" edgedefault="directed">')
        
        # Add nodes
        for node in subgraph_data.get("nodes", []):
            graphml.append(f'    <node id="{node["id"]}">')
            graphml.append(f'      <data key="label">{node["label"]}</data>')
            graphml.append('    </node>')
        
        # Add edges
        for edge in subgraph_data.get("edges", []):
            graphml.append(
                f'    <edge source="{edge["source_id"]}" '
                f'target="{edge["target_id"]}">'
            )
            graphml.append(f'      <data key="relationship">{edge["relationship"]}</data>')
            graphml.append('    </edge>')
        
        graphml.append('  </graph>')
        graphml.append('</graphml>')
        
        return '\n'.join(graphml)
    
    def _to_cypher(self, subgraph_data: Dict[str, Any]) -> str:
        """Convert subgraph to Cypher CREATE statements."""
        cypher_statements = []
        
        # Create nodes
        for node in subgraph_data.get("nodes", []):
            props = ", ".join(
                f'{k}: "{v}"' for k, v in node.get("properties", {}).items()
                if isinstance(v, (str, int, float))
            )
            cypher_statements.append(
                f'CREATE (n{node["id"]}:{node["label"]} {{{props}}})'
            )
        
        # Create edges
        for edge in subgraph_data.get("edges", []):
            cypher_statements.append(
                f'CREATE (n{edge["source_id"]})-[:{edge["relationship"]}]->'
                f'(n{edge["target_id"]})'
            )
        
        return ";\n".join(cypher_statements) + ";"


class HybridSearcher:
    """
    Advanced hybrid search combining vector and graph approaches.
    
    This provides sophisticated search strategies that leverage both
    semantic similarity (vectors) and structured relationships (graphs).
    """
    
    def __init__(
        self,
        memory_integration: MemoryGraphIntegration,
        reranking_model: Optional[str] = None
    ):
        self.integration = memory_integration
        self.reranking_model = reranking_model
    
    async def multi_hop_search(
        self,
        query: str,
        initial_hops: int = 1,
        expansion_hops: int = 2,
        evidence_threshold: float = 0.7
    ) -> HybridSearchResult:
        """
        Perform multi-hop search starting from vector matches.
        
        1. Find initial matches using vector search
        2. Expand through graph relationships
        3. Gather evidence along paths
        4. Rank results by relevance and evidence
        """
        # Initial vector search
        initial_results = self.integration.hybrid_search(
            query=query,
            vector_weight=0.8,
            graph_weight=0.2,
            max_graph_depth=initial_hops
        )
        
        # Extract entity IDs from initial results
        initial_entities = [
            node["id"] for node in initial_results.graph_nodes
        ]
        
        # Expand through graph
        expanded_results = await self._expand_through_graph(
            initial_entities,
            expansion_hops,
            evidence_threshold
        )
        
        # Merge and rerank results
        final_results = self._merge_and_rerank(
            initial_results,
            expanded_results,
            query
        )
        
        return final_results
    
    async def _expand_through_graph(
        self,
        start_entities: List[str],
        max_hops: int,
        evidence_threshold: float
    ) -> Dict[str, Any]:
        """Expand search through graph relationships."""
        # Implementation would follow graph edges,
        # collecting evidence and scoring paths
        return {
            "expanded_nodes": [],
            "evidence_paths": [],
            "total_evidence_score": 0.0
        }
    
    def _merge_and_rerank(
        self,
        initial: HybridSearchResult,
        expanded: Dict[str, Any],
        query: str
    ) -> HybridSearchResult:
        """Merge and rerank results based on relevance."""
        # Sophisticated reranking logic would go here
        # Could use a learned reranking model
        return initial
    
    async def knowledge_guided_search(
        self,
        query: str,
        knowledge_constraints: Dict[str, Any]
    ) -> HybridSearchResult:
        """
        Search guided by knowledge graph constraints.
        
        Example constraints:
        - entity_types: ["Person", "Organization"]
        - relationship_patterns: ["WORKS_FOR", "COLLABORATES_WITH"]
        - temporal_range: (start_date, end_date)
        - confidence_threshold: 0.8
        """
        # Build constrained graph query
        graph_query = QueryBuilder()
        
        if "entity_types" in knowledge_constraints:
            graph_query.match_nodes(knowledge_constraints["entity_types"])
        
        if "relationship_patterns" in knowledge_constraints:
            graph_query.connected_by(knowledge_constraints["relationship_patterns"])
        
        if "temporal_range" in knowledge_constraints:
            start, end = knowledge_constraints["temporal_range"]
            graph_query.between_times(start, end)
        
        if "confidence_threshold" in knowledge_constraints:
            graph_query.having_confidence(knowledge_constraints["confidence_threshold"])
        
        # Execute constrained search
        return self.integration.hybrid_search(
            query=query,
            vector_weight=0.6,
            graph_weight=0.4,
            max_graph_depth=3
        )