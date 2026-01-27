"""
Phoenix AI GraphRAG Module

Graph-based Retrieval Augmented Generation that combines knowledge graphs
with vector search for enhanced multi-hop reasoning and reduced hallucinations.
"""

from .entity_extractor import (
    EntityExtractor,
    LLMEntityExtractor,
    RegexEntityExtractor,
)
from .graph_builder import GraphBuilder
from .graph_retriever import GraphRetriever
from .graphrag_inferencer import GraphRAGInferencer
from .knowledge_graph import Entity, KnowledgeGraph, Relationship
from .relationship_extractor import LLMRelationshipExtractor, RelationshipExtractor

__all__ = [
    # Core data structures
    "KnowledgeGraph",
    "Entity",
    "Relationship",
    # Extractors
    "EntityExtractor",
    "LLMEntityExtractor",
    "RegexEntityExtractor",
    "RelationshipExtractor",
    "LLMRelationshipExtractor",
    # Builders and retrievers
    "GraphBuilder",
    "GraphRetriever",
    # Main inferencer
    "GraphRAGInferencer",
]
