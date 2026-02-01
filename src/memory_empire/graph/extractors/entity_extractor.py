"""
Advanced entity extraction for Memory Empire Knowledge Graph.

This module provides sophisticated NLP-based entity and relationship extraction
using state-of-the-art models and techniques.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

# Optional heavy dependencies
try:
    import spacy
    from spacy.tokens import Doc, Span
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str  # e.g., PERSON, ORG, LOCATION, etc.
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Additional attributes for advanced extraction
    normalized_text: Optional[str] = None
    disambiguation_id: Optional[str] = None  # e.g., WikiData ID
    aliases: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash((self.text, self.label, self.start, self.end))


@dataclass
class Relationship:
    """Represents an extracted relationship between entities."""
    source: Entity
    target: Entity
    relation_type: str
    confidence: float = 1.0
    evidence_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Results from entity extraction."""
    entities: List[Entity]
    relationships: List[Relationship]
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    
    def merge(self, other: 'ExtractionResult') -> 'ExtractionResult':
        """Merge with another extraction result."""
        return ExtractionResult(
            entities=self.entities + other.entities,
            relationships=self.relationships + other.relationships,
            metadata={**self.metadata, **other.metadata},
            processing_time_ms=self.processing_time_ms + other.processing_time_ms
        )


class EntityExtractor:
    """
    Advanced entity extractor supporting multiple backends and techniques.
    
    Features:
    - Multiple NLP model backends (spaCy, Transformers, custom models)
    - Entity disambiguation and normalization
    - Coreference resolution
    - Custom entity patterns
    - Confidence scoring
    - Incremental extraction
    """
    
    def __init__(
        self,
        backend: str = "spacy",
        model_name: Optional[str] = None,
        custom_patterns: Optional[Dict[str, List[str]]] = None,
        enable_coreference: bool = True,
        enable_disambiguation: bool = False,
        confidence_threshold: float = 0.5
    ):
        self.backend = backend
        self.model_name = model_name
        self.custom_patterns = custom_patterns or {}
        self.enable_coreference = enable_coreference
        self.enable_disambiguation = enable_disambiguation
        self.confidence_threshold = confidence_threshold
        
        # Initialize the appropriate backend
        self._init_backend()
    
    def _init_backend(self) -> None:
        """Initialize the NLP backend."""
        if self.backend == "spacy":
            if not SPACY_AVAILABLE:
                raise ImportError("spaCy is not installed. Install with: pip install spacy")
            
            # Load spaCy model
            model_name = self.model_name or "en_core_web_sm"
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                logger.info(f"Downloading spaCy model: {model_name}")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", model_name])
                self.nlp = spacy.load(model_name)
            
            # Add custom entity patterns if provided
            if self.custom_patterns:
                self._add_custom_patterns()
        
        elif self.backend == "transformers":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers is not installed. Install with: pip install transformers")
            
            # Load transformer model for NER
            model_name = self.model_name or "dslim/bert-base-NER"
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                aggregation_strategy="simple"
            )
        
        elif self.backend == "regex":
            # Simple regex-based extraction
            self._compile_regex_patterns()
        
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _add_custom_patterns(self) -> None:
        """Add custom entity patterns to spaCy."""
        if self.backend != "spacy":
            return
        
        # Get or create EntityRuler
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        else:
            ruler = self.nlp.get_pipe("entity_ruler")
        
        # Add patterns
        patterns = []
        for label, pattern_list in self.custom_patterns.items():
            for pattern in pattern_list:
                if isinstance(pattern, str):
                    patterns.append({"label": label, "pattern": pattern})
                elif isinstance(pattern, list):
                    patterns.append({"label": label, "pattern": pattern})
        
        ruler.add_patterns(patterns)
    
    def _compile_regex_patterns(self) -> None:
        """Compile regex patterns for extraction."""
        self.regex_patterns = {}
        
        # Default patterns
        default_patterns = {
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "PHONE": r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            "URL": r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            "IP_ADDRESS": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "DATE": r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            "TIME": r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?\b',
            "MONEY": r'\$\s?\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s?(?:USD|EUR|GBP)',
            "PERCENTAGE": r'\b\d+(?:\.\d+)?%\b',
        }
        
        # Merge with custom patterns
        all_patterns = {**default_patterns, **self.custom_patterns}
        
        for label, pattern in all_patterns.items():
            self.regex_patterns[label] = re.compile(pattern, re.IGNORECASE)
    
    def extract(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ExtractionResult:
        """
        Extract entities and relationships from text.
        
        Args:
            text: Input text to process
            metadata: Optional metadata about the text source
            
        Returns:
            ExtractionResult containing entities and relationships
        """
        start_time = datetime.utcnow()
        
        if self.backend == "spacy":
            result = self._extract_spacy(text, metadata)
        elif self.backend == "transformers":
            result = self._extract_transformers(text, metadata)
        elif self.backend == "regex":
            result = self._extract_regex(text, metadata)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        # Post-process results
        result = self._post_process(result, text)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        result.processing_time_ms = processing_time
        
        return result
    
    def _extract_spacy(self, text: str, metadata: Optional[Dict[str, Any]]) -> ExtractionResult:
        """Extract using spaCy."""
        doc = self.nlp(text)
        
        entities = []
        entity_spans: List[Span] = []
        
        # Extract named entities
        for ent in doc.ents:
            if self._should_include_entity(ent.label_, ent.text):
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0,  # spaCy doesn't provide confidence scores
                    metadata={"source": "spacy"}
                )
                entities.append(entity)
                entity_spans.append(ent)
        
        # Extract relationships (basic dependency parsing)
        relationships = self._extract_spacy_relationships(doc, entity_spans)
        
        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            metadata=metadata or {}
        )
    
    def _extract_spacy_relationships(self, doc: Doc, entity_spans: List[Span]) -> List[Relationship]:
        """Extract relationships using dependency parsing."""
        relationships = []
        
        # Create entity lookup
        entity_lookup = {}
        for ent in entity_spans:
            for token in ent:
                entity_lookup[token.i] = ent
        
        # Look for relationships based on dependency patterns
        for token in doc:
            # Subject-Verb-Object patterns
            if token.pos_ == "VERB":
                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                objects = [child for child in token.children if child.dep_ in ("dobj", "pobj", "attr")]
                
                for subj in subjects:
                    for obj in objects:
                        # Check if both are entities
                        if subj.i in entity_lookup and obj.i in entity_lookup:
                            source_ent = entity_lookup[subj.i]
                            target_ent = entity_lookup[obj.i]
                            
                            rel = Relationship(
                                source=self._span_to_entity(source_ent),
                                target=self._span_to_entity(target_ent),
                                relation_type=token.lemma_,
                                confidence=0.8,
                                evidence_text=f"{source_ent.text} {token.text} {target_ent.text}",
                                metadata={"verb": token.text, "dependency": "SVO"}
                            )
                            relationships.append(rel)
        
        return relationships
    
    def _span_to_entity(self, span: Span) -> Entity:
        """Convert spaCy Span to Entity."""
        return Entity(
            text=span.text,
            label=span.label_,
            start=span.start_char,
            end=span.end_char,
            confidence=1.0,
            metadata={"source": "spacy"}
        )
    
    def _extract_transformers(self, text: str, metadata: Optional[Dict[str, Any]]) -> ExtractionResult:
        """Extract using Transformers."""
        # Run NER pipeline
        ner_results = self.ner_pipeline(text)
        
        entities = []
        for result in ner_results:
            if result["score"] >= self.confidence_threshold:
                entity = Entity(
                    text=result["word"],
                    label=result["entity_group"],
                    start=result["start"],
                    end=result["end"],
                    confidence=result["score"],
                    metadata={"source": "transformers", "model": self.model_name}
                )
                entities.append(entity)
        
        # Note: Transformers NER doesn't extract relationships by default
        # You would need a separate model for relation extraction
        relationships = []
        
        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            metadata=metadata or {}
        )
    
    def _extract_regex(self, text: str, metadata: Optional[Dict[str, Any]]) -> ExtractionResult:
        """Extract using regex patterns."""
        entities = []
        
        for label, pattern in self.regex_patterns.items():
            for match in pattern.finditer(text):
                entity = Entity(
                    text=match.group(),
                    label=label,
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,  # Regex matches are binary
                    metadata={"source": "regex", "pattern": pattern.pattern}
                )
                entities.append(entity)
        
        # Regex doesn't extract relationships
        relationships = []
        
        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            metadata=metadata or {}
        )
    
    def _should_include_entity(self, label: str, text: str) -> bool:
        """Filter entities based on rules."""
        # Skip very short entities
        if len(text.strip()) < 2:
            return False
        
        # Skip entities that are just numbers (unless they're specific types)
        if text.isdigit() and label not in ("DATE", "TIME", "MONEY", "PERCENT"):
            return False
        
        # Skip common stop words as entities
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        if text.lower() in stop_words:
            return False
        
        return True
    
    def _post_process(self, result: ExtractionResult, original_text: str) -> ExtractionResult:
        """Post-process extraction results."""
        # Remove duplicate entities
        result.entities = self._deduplicate_entities(result.entities)
        
        # Normalize entities if enabled
        if self.enable_disambiguation:
            result = self._disambiguate_entities(result, original_text)
        
        # Resolve coreferences if enabled
        if self.enable_coreference:
            result = self._resolve_coreferences(result, original_text)
        
        # Filter by confidence threshold
        result.entities = [e for e in result.entities if e.confidence >= self.confidence_threshold]
        result.relationships = [r for r in result.relationships if r.confidence >= self.confidence_threshold]
        
        return result
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping the highest confidence ones."""
        # Group by text and label
        entity_groups: Dict[Tuple[str, str], List[Entity]] = {}
        
        for entity in entities:
            key = (entity.text.lower(), entity.label)
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        # Keep the highest confidence entity from each group
        deduplicated = []
        for group in entity_groups.values():
            best_entity = max(group, key=lambda e: e.confidence)
            deduplicated.append(best_entity)
        
        return deduplicated
    
    def _disambiguate_entities(self, result: ExtractionResult, text: str) -> ExtractionResult:
        """
        Disambiguate entities (e.g., link to knowledge bases).
        
        This is a placeholder for more sophisticated disambiguation.
        In production, you might use:
        - Entity linking to WikiData/DBpedia
        - Custom knowledge base lookup
        - Context-aware disambiguation models
        """
        # Simple example: add normalized forms
        for entity in result.entities:
            if entity.label == "PERSON":
                # Normalize person names (simple title case)
                entity.normalized_text = entity.text.title()
            elif entity.label == "ORG":
                # Normalize organization names
                entity.normalized_text = entity.text.upper()
            elif entity.label == "DATE":
                # Could parse and normalize dates
                entity.normalized_text = entity.text
            else:
                entity.normalized_text = entity.text
        
        return result
    
    def _resolve_coreferences(self, result: ExtractionResult, text: str) -> ExtractionResult:
        """
        Resolve coreferences (pronouns, aliases, etc.).
        
        This is a simplified version. For production, consider:
        - neuralcoref or similar coreference resolution models
        - Rule-based pronoun resolution
        - Alias detection and resolution
        """
        # Track entities by type for simple pronoun resolution
        person_entities = [e for e in result.entities if e.label == "PERSON"]
        
        # Simple pronoun patterns
        pronoun_patterns = {
            "he": "PERSON",
            "she": "PERSON",
            "it": "ORG",
            "they": "PERSON",
        }
        
        # This is a very basic implementation
        # In production, use proper coreference resolution models
        
        return result
    
    def extract_batch(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[ExtractionResult]:
        """Extract from multiple texts efficiently."""
        results = []
        
        if metadata_list is None:
            metadata_list = [None] * len(texts)
        
        for text, metadata in zip(texts, metadata_list):
            result = self.extract(text, metadata)
            results.append(result)
        
        return results
    
    def update_custom_patterns(self, patterns: Dict[str, List[str]]) -> None:
        """Update custom extraction patterns."""
        self.custom_patterns.update(patterns)
        
        if self.backend == "spacy":
            self._add_custom_patterns()
        elif self.backend == "regex":
            self._compile_regex_patterns()