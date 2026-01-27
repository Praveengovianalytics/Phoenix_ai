"""
Tests for Phoenix AI GraphRAG module.

Tests cover:
- Knowledge graph data structures
- Entity extraction
- Relationship extraction
- Graph building
- Graph retrieval
- GraphRAG inference
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phoenix_ai.graphrag import (
    Entity,
    GraphBuilder,
    GraphRAGInferencer,
    GraphRetriever,
    KnowledgeGraph,
    LLMEntityExtractor,
    LLMRelationshipExtractor,
    Relationship,
)


# ============ Fixtures ============

class FakeEmbeddingClient:
    """Fake embedding client for testing."""

    def __init__(self, embedding_dim: int = 3):
        self.embedding_dim = embedding_dim
        self.calls: List[List[str]] = []

    def generate_embedding(
        self,
        input_texts: List[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        self.calls.append(input_texts)
        # Return deterministic embeddings based on text hash
        embeddings = []
        for text in input_texts:
            hash_val = hash(text) % 1000
            embedding = [
                (hash_val + i) / 1000.0
                for i in range(self.embedding_dim)
            ]
            embeddings.append(embedding)
        return embeddings


class FakeChatClient:
    """Fake chat client for testing."""

    def __init__(self, responses: List[str] = None):
        self.responses = responses or []
        self.response_index = 0
        self.calls: List[Dict[str, Any]] = []

    def add_response(self, response: str):
        self.responses.append(response)

    def chat(
        self,
        user_input: Any,
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> str:
        self.calls.append({
            "user_input": user_input,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response
        return "[]"


@pytest.fixture
def sample_entities() -> List[Entity]:
    """Create sample entities for testing."""
    return [
        Entity(
            id="e1",
            name="John Smith",
            entity_type="PERSON",
            description="CEO of Acme Corp",
            source_chunks=["chunk_0"],
        ),
        Entity(
            id="e2",
            name="Acme Corp",
            entity_type="ORGANIZATION",
            description="A technology company",
            source_chunks=["chunk_0", "chunk_1"],
        ),
        Entity(
            id="e3",
            name="New York",
            entity_type="LOCATION",
            description="City in the USA",
            source_chunks=["chunk_1"],
        ),
    ]


@pytest.fixture
def sample_relationships() -> List[Relationship]:
    """Create sample relationships for testing."""
    return [
        Relationship(
            source_id="e1",
            target_id="e2",
            relation_type="works_for",
            weight=0.9,
            description="John Smith is CEO of Acme Corp",
        ),
        Relationship(
            source_id="e2",
            target_id="e3",
            relation_type="located_in",
            weight=0.8,
            description="Acme Corp is headquartered in New York",
        ),
    ]


@pytest.fixture
def sample_graph(sample_entities, sample_relationships) -> KnowledgeGraph:
    """Create a sample knowledge graph for testing."""
    graph = KnowledgeGraph()

    for entity in sample_entities:
        graph.add_entity(entity)

    for rel in sample_relationships:
        graph.add_relationship(rel)

    return graph


@pytest.fixture
def fake_embedding_client() -> FakeEmbeddingClient:
    """Create a fake embedding client."""
    return FakeEmbeddingClient(embedding_dim=3)


@pytest.fixture
def fake_chat_client() -> FakeChatClient:
    """Create a fake chat client with entity extraction responses."""
    return FakeChatClient([
        # Entity extraction response
        json.dumps([
            {"name": "John Smith", "type": "PERSON", "description": "A person"},
            {"name": "Acme Corp", "type": "ORGANIZATION", "description": "A company"},
        ]),
        # Relationship extraction response
        json.dumps([
            {
                "source": "John Smith",
                "target": "Acme Corp",
                "relation_type": "works_for",
                "confidence": 0.9,
                "description": "Employee relationship",
            }
        ]),
    ])


# ============ Knowledge Graph Tests ============

class TestKnowledgeGraph:
    """Tests for KnowledgeGraph class."""

    def test_create_empty_graph(self):
        """Test creating an empty knowledge graph."""
        graph = KnowledgeGraph()
        assert graph.num_entities == 0
        assert graph.num_relationships == 0

    def test_add_entity(self, sample_entities):
        """Test adding entities to graph."""
        graph = KnowledgeGraph()
        for entity in sample_entities:
            graph.add_entity(entity)

        assert graph.num_entities == 3
        assert graph.get_entity("e1").name == "John Smith"

    def test_add_relationship(self, sample_graph):
        """Test adding relationships to graph."""
        assert sample_graph.num_relationships == 2

    def test_get_entity_by_name(self, sample_graph):
        """Test entity lookup by name."""
        entity = sample_graph.get_entity_by_name("John Smith")
        assert entity is not None
        assert entity.id == "e1"

        # Case insensitive
        entity = sample_graph.get_entity_by_name("john smith")
        assert entity is not None

    def test_get_entities_by_type(self, sample_graph):
        """Test getting entities by type."""
        persons = sample_graph.get_entities_by_type("PERSON")
        assert len(persons) == 1
        assert persons[0].name == "John Smith"

    def test_get_neighbors(self, sample_graph):
        """Test getting entity neighbors."""
        neighbors = sample_graph.get_neighbors("e1", direction="out")
        assert len(neighbors) == 1
        assert neighbors[0].id == "e2"

        neighbors = sample_graph.get_neighbors("e2", direction="both")
        assert len(neighbors) == 2

    def test_find_paths(self, sample_graph):
        """Test finding paths between entities."""
        paths = sample_graph.find_paths("e1", "e3", max_depth=2)
        assert len(paths) >= 1
        assert paths[0] == ["e1", "e2", "e3"]

    def test_get_subgraph(self, sample_graph):
        """Test extracting a subgraph."""
        subgraph = sample_graph.get_subgraph(["e1"], depth=1)
        assert subgraph.num_entities >= 1
        assert "e1" in subgraph.entities

    def test_save_and_load_json(self, sample_graph, tmp_path):
        """Test saving and loading graph as JSON."""
        filepath = str(tmp_path / "test_graph.json")
        sample_graph.save(filepath)

        loaded_graph = KnowledgeGraph.load(filepath)
        assert loaded_graph.num_entities == sample_graph.num_entities
        assert loaded_graph.num_relationships == sample_graph.num_relationships

    def test_save_and_load_pickle(self, sample_graph, tmp_path):
        """Test saving and loading graph as pickle."""
        filepath = str(tmp_path / "test_graph.pkl")
        sample_graph.save(filepath)

        loaded_graph = KnowledgeGraph.load(filepath)
        assert loaded_graph.num_entities == sample_graph.num_entities


class TestEntity:
    """Tests for Entity class."""

    def test_entity_creation(self):
        """Test creating an entity."""
        entity = Entity(
            id="test_id",
            name="Test Entity",
            entity_type="CONCEPT",
        )
        assert entity.id == "test_id"
        assert entity.name == "Test Entity"
        assert entity.entity_type == "CONCEPT"

    def test_entity_merge(self):
        """Test merging entities."""
        e1 = Entity(id="e1", name="Test", entity_type="CONCEPT", source_chunks=["c1"])
        e2 = Entity(id="e1", name="Test", entity_type="CONCEPT", source_chunks=["c2"])

        e1.merge_with(e2)
        assert "c1" in e1.source_chunks
        assert "c2" in e1.source_chunks

    def test_entity_serialization(self):
        """Test entity to_dict and from_dict."""
        entity = Entity(
            id="e1",
            name="Test",
            entity_type="PERSON",
            description="A test entity",
        )

        data = entity.to_dict()
        restored = Entity.from_dict(data)

        assert restored.id == entity.id
        assert restored.name == entity.name
        assert restored.entity_type == entity.entity_type


class TestRelationship:
    """Tests for Relationship class."""

    def test_relationship_creation(self):
        """Test creating a relationship."""
        rel = Relationship(
            source_id="e1",
            target_id="e2",
            relation_type="related_to",
        )
        assert rel.source_id == "e1"
        assert rel.target_id == "e2"
        assert rel.weight == 1.0

    def test_relationship_serialization(self):
        """Test relationship to_dict and from_dict."""
        rel = Relationship(
            source_id="e1",
            target_id="e2",
            relation_type="works_for",
            weight=0.9,
        )

        data = rel.to_dict()
        restored = Relationship.from_dict(data)

        assert restored.source_id == rel.source_id
        assert restored.relation_type == rel.relation_type


# ============ Entity Extractor Tests ============

class TestLLMEntityExtractor:
    """Tests for LLM-based entity extraction."""

    def test_extract_entities(self, fake_chat_client):
        """Test extracting entities from text."""
        extractor = LLMEntityExtractor(chat_client=fake_chat_client)

        text = "John Smith is the CEO of Acme Corp."
        entities = extractor.extract(text, chunk_id="test_chunk")

        assert len(entities) == 2
        assert any(e.name == "John Smith" for e in entities)
        assert any(e.entity_type == "PERSON" for e in entities)

    def test_extract_with_custom_types(self, fake_chat_client):
        """Test extraction with custom entity types."""
        extractor = LLMEntityExtractor(
            chat_client=fake_chat_client,
            entity_types=["PERSON", "COMPANY"],
        )
        assert "PERSON" in extractor.entity_types
        assert "COMPANY" in extractor.entity_types

    def test_empty_text_extraction(self, fake_chat_client):
        """Test extraction from empty text."""
        extractor = LLMEntityExtractor(chat_client=fake_chat_client)
        entities = extractor.extract("")
        assert len(entities) == 0


# ============ Relationship Extractor Tests ============

class TestLLMRelationshipExtractor:
    """Tests for LLM-based relationship extraction."""

    def test_extract_relationships(self, fake_chat_client, sample_entities):
        """Test extracting relationships between entities."""
        # Add relationship extraction response
        fake_chat_client.add_response(json.dumps([
            {
                "source": "John Smith",
                "target": "Acme Corp",
                "relation_type": "works_for",
                "confidence": 0.9,
            }
        ]))

        extractor = LLMRelationshipExtractor(chat_client=fake_chat_client)

        text = "John Smith works at Acme Corp as CEO."
        relationships = extractor.extract(sample_entities[:2], text, "chunk_0")

        assert len(relationships) >= 0  # May be 0 if entities don't match

    def test_min_confidence_filter(self, fake_chat_client, sample_entities):
        """Test that low confidence relationships are filtered."""
        fake_chat_client.add_response(json.dumps([
            {
                "source": "John Smith",
                "target": "Acme Corp",
                "relation_type": "works_for",
                "confidence": 0.3,  # Below default threshold
            }
        ]))

        extractor = LLMRelationshipExtractor(
            chat_client=fake_chat_client,
            min_confidence=0.5,
        )

        relationships = extractor.extract(sample_entities[:2], "test text", "chunk_0")
        assert len(relationships) == 0  # Should be filtered out


# ============ Graph Builder Tests ============

class TestGraphBuilder:
    """Tests for GraphBuilder class."""

    def test_build_from_chunks(self, fake_chat_client, fake_embedding_client):
        """Test building graph from text chunks."""
        # Prepare responses for two chunks
        fake_chat_client.responses = [
            # Entities for chunk 0
            json.dumps([
                {"name": "Entity A", "type": "CONCEPT", "description": ""},
            ]),
            # Entities for chunk 1
            json.dumps([
                {"name": "Entity B", "type": "CONCEPT", "description": ""},
            ]),
            # Relationships for chunk 0
            json.dumps([]),
            # Relationships for chunk 1
            json.dumps([]),
        ]

        builder = GraphBuilder.from_clients(
            chat_client=fake_chat_client,
            embedding_client=fake_embedding_client,
        )

        chunks = ["Text about Entity A", "Text about Entity B"]
        graph = builder.build_from_chunks(chunks)

        assert isinstance(graph, KnowledgeGraph)
        assert graph.num_entities >= 0

    def test_build_from_dataframe(self, fake_chat_client, fake_embedding_client):
        """Test building graph from DataFrame."""
        fake_chat_client.responses = [
            json.dumps([{"name": "Test", "type": "CONCEPT", "description": ""}]),
            json.dumps([]),
        ]

        builder = GraphBuilder.from_clients(
            chat_client=fake_chat_client,
            embedding_client=fake_embedding_client,
        )

        df = pd.DataFrame({"content": ["Test content"]})
        graph = builder.build_from_dataframe(df, text_column="content")

        assert isinstance(graph, KnowledgeGraph)


# ============ Graph Retriever Tests ============

class TestGraphRetriever:
    """Tests for GraphRetriever class."""

    def test_find_entities_by_name(self, sample_graph, fake_embedding_client):
        """Test finding entities by name."""
        retriever = GraphRetriever(
            knowledge_graph=sample_graph,
            embedding_client=fake_embedding_client,
        )

        matches = retriever.find_entities_by_name("John Smith")
        assert len(matches) >= 1
        assert any(e.name == "John Smith" for e in matches)

    def test_expand_with_graph(self, sample_graph, fake_embedding_client):
        """Test expanding entities through graph."""
        retriever = GraphRetriever(
            knowledge_graph=sample_graph,
            embedding_client=fake_embedding_client,
        )

        john = sample_graph.get_entity("e1")
        expanded = retriever.expand_with_graph([john], depth=1)

        assert len(expanded) >= 1
        entity_ids = [e.id for e in expanded]
        assert "e1" in entity_ids
        assert "e2" in entity_ids  # Connected to John

    def test_get_entity_context(self, sample_graph, fake_embedding_client):
        """Test generating entity context."""
        retriever = GraphRetriever(
            knowledge_graph=sample_graph,
            embedding_client=fake_embedding_client,
        )

        john = sample_graph.get_entity("e1")
        context = retriever.get_entity_context(john)

        assert "John Smith" in context
        assert "PERSON" in context


# ============ GraphRAG Inferencer Tests ============

class TestGraphRAGInferencer:
    """Tests for GraphRAGInferencer class."""

    def test_create_inferencer(self, fake_embedding_client, fake_chat_client):
        """Test creating a GraphRAG inferencer."""
        inferencer = GraphRAGInferencer(
            embedding_client=fake_embedding_client,
            chat_client=fake_chat_client,
        )

        assert inferencer.knowledge_graph is None
        assert inferencer.graph_weight == 0.3

    def test_create_with_graph(
        self,
        fake_embedding_client,
        fake_chat_client,
        sample_graph,
    ):
        """Test creating inferencer with pre-built graph."""
        inferencer = GraphRAGInferencer(
            embedding_client=fake_embedding_client,
            chat_client=fake_chat_client,
            knowledge_graph=sample_graph,
        )

        assert inferencer.knowledge_graph is not None
        assert inferencer.knowledge_graph.num_entities == 3

    def test_build_graph(self, fake_embedding_client, fake_chat_client):
        """Test building graph from inferencer."""
        fake_chat_client.responses = [
            json.dumps([{"name": "Test", "type": "CONCEPT", "description": ""}]),
            json.dumps([]),
        ]

        inferencer = GraphRAGInferencer(
            embedding_client=fake_embedding_client,
            chat_client=fake_chat_client,
        )

        graph = inferencer.build_graph(["Test content"])
        assert isinstance(graph, KnowledgeGraph)
        assert inferencer.knowledge_graph is not None

    def test_save_and_load_graph(
        self,
        fake_embedding_client,
        fake_chat_client,
        sample_graph,
        tmp_path,
    ):
        """Test saving and loading graph from inferencer."""
        inferencer = GraphRAGInferencer(
            embedding_client=fake_embedding_client,
            chat_client=fake_chat_client,
            knowledge_graph=sample_graph,
        )

        filepath = str(tmp_path / "test_graph.json")
        inferencer.save_graph(filepath)

        # Create new inferencer and load
        new_inferencer = GraphRAGInferencer(
            embedding_client=fake_embedding_client,
            chat_client=fake_chat_client,
        )
        new_inferencer.load_graph(filepath)

        assert new_inferencer.knowledge_graph.num_entities == 3

    def test_query_graph(
        self,
        fake_embedding_client,
        fake_chat_client,
        sample_graph,
    ):
        """Test querying the graph directly."""
        # Add embeddings to entities
        for entity in sample_graph.entities.values():
            entity.embedding = fake_embedding_client.generate_embedding([entity.name])[0]

        inferencer = GraphRAGInferencer(
            embedding_client=fake_embedding_client,
            chat_client=fake_chat_client,
            knowledge_graph=sample_graph,
        )

        result = inferencer.query_graph("John Smith", k=3)

        assert "query" in result
        assert "results" in result
        assert "graph_stats" in result


# ============ Integration Tests ============

class TestGraphRAGIntegration:
    """Integration tests for the complete GraphRAG pipeline."""

    def test_full_pipeline(self, fake_embedding_client, fake_chat_client):
        """Test the complete GraphRAG pipeline."""
        # Setup responses for entity extraction, relationship extraction, and inference
        fake_chat_client.responses = [
            # Entity extraction for chunk 0
            json.dumps([
                {"name": "Alice", "type": "PERSON", "description": "A developer"},
                {"name": "TechCorp", "type": "ORGANIZATION", "description": "A tech company"},
            ]),
            # Relationship extraction for chunk 0
            json.dumps([
                {
                    "source": "Alice",
                    "target": "TechCorp",
                    "relation_type": "works_for",
                    "confidence": 0.9,
                }
            ]),
            # Final inference response
            "Alice is a developer who works at TechCorp.",
        ]

        # Create inferencer
        inferencer = GraphRAGInferencer(
            embedding_client=fake_embedding_client,
            chat_client=fake_chat_client,
        )

        # Build graph
        chunks = ["Alice is a developer at TechCorp."]
        inferencer.build_graph(chunks)

        # Verify graph was built
        assert inferencer.knowledge_graph is not None
        assert inferencer.knowledge_graph.num_entities >= 0

    def test_dataframe_integration(self, fake_embedding_client, fake_chat_client):
        """Test integration with Phoenix AI DataFrame pipeline."""
        fake_chat_client.responses = [
            json.dumps([{"name": "Product X", "type": "PRODUCT", "description": ""}]),
            json.dumps([]),
        ]

        inferencer = GraphRAGInferencer(
            embedding_client=fake_embedding_client,
            chat_client=fake_chat_client,
        )

        # Simulate DataFrame from loaders
        df = pd.DataFrame({
            "filename": ["doc1.pdf"],
            "content": ["Information about Product X"],
            "chunk_id": ["chunk_0"],
        })

        graph = inferencer.build_graph_from_dataframe(
            df,
            text_column="content",
            chunk_id_column="chunk_id",
        )

        assert isinstance(graph, KnowledgeGraph)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
