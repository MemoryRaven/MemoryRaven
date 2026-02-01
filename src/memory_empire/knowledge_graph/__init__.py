"""Knowledge graph subsystem.

This module provides:
- A pluggable entity+relation extraction pipeline
- A graph store abstraction + Neo4j implementation
- A small query engine for common traversals

The default extractor is lightweight and has no heavy NLP dependencies.
"""

from .models import Entity, Relation
from .pipeline import GraphIngestor
from .store import GraphStore

__all__ = ["Entity", "Relation", "GraphIngestor", "GraphStore"]
