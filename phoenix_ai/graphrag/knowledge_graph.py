"""
Knowledge Graph data structures for GraphRAG.

This module provides the core data structures for representing
entities, relationships, and the knowledge graph itself.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Entity:
    """
    Represents an entity extracted from documents.

    Attributes:
        id: Unique identifier for the entity
        name: Display name of the entity
        entity_type: Type classification (PERSON, ORG, LOCATION, CONCEPT, etc.)
        description: Optional description of the entity
        source_chunks: List of chunk IDs where this entity appears
        embedding: Cached vector embedding for similarity search
        metadata: Additional metadata about the entity
    """
    id: str
    name: str
    entity_type: str
    description: str = ""
    source_chunks: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_source_chunk(self, chunk_id: str) -> None:
        """Add a source chunk reference if not already present."""
        if chunk_id not in self.source_chunks:
            self.source_chunks.append(chunk_id)

    def merge_with(self, other: Entity) -> None:
        """Merge another entity instance into this one (for deduplication)."""
        for chunk_id in other.source_chunks:
            self.add_source_chunk(chunk_id)
        if other.description and not self.description:
            self.description = other.description
        self.metadata.update(other.metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "source_chunks": self.source_chunks,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Entity:
        """Create entity from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=data["entity_type"],
            description=data.get("description", ""),
            source_chunks=data.get("source_chunks", []),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Relationship:
    """
    Represents a relationship between two entities.

    Attributes:
        source_id: ID of the source entity
        target_id: ID of the target entity
        relation_type: Type of relationship (e.g., "works_for", "located_in")
        weight: Confidence/strength of the relationship (0.0 to 1.0)
        description: Optional description of the relationship
        evidence_chunks: Chunks that support this relationship
        metadata: Additional metadata
    """
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    description: str = ""
    evidence_chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_evidence(self, chunk_id: str) -> None:
        """Add evidence chunk for this relationship."""
        if chunk_id not in self.evidence_chunks:
            self.evidence_chunks.append(chunk_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "description": self.description,
            "evidence_chunks": self.evidence_chunks,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Relationship:
        """Create relationship from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            weight=data.get("weight", 1.0),
            description=data.get("description", ""),
            evidence_chunks=data.get("evidence_chunks", []),
            metadata=data.get("metadata", {}),
        )


class KnowledgeGraph:
    """
    Knowledge Graph for storing entities and their relationships.

    Provides efficient lookup, traversal, and persistence capabilities
    for graph-based retrieval augmented generation.
    """

    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self._adjacency_out: Dict[str, List[str]] = defaultdict(list)
        self._adjacency_in: Dict[str, List[str]] = defaultdict(list)
        self._name_to_id: Dict[str, str] = {}
        self._type_index: Dict[str, Set[str]] = defaultdict(set)
        self._chunk_to_entities: Dict[str, Set[str]] = defaultdict(set)

    @property
    def num_entities(self) -> int:
        """Return the number of entities in the graph."""
        return len(self.entities)

    @property
    def num_relationships(self) -> int:
        """Return the number of relationships in the graph."""
        return len(self.relationships)

    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the knowledge graph.

        If an entity with the same ID exists, merge them.
        """
        if entity.id in self.entities:
            self.entities[entity.id].merge_with(entity)
        else:
            self.entities[entity.id] = entity
            self._name_to_id[entity.name.lower()] = entity.id
            self._type_index[entity.entity_type].add(entity.id)

        for chunk_id in entity.source_chunks:
            self._chunk_to_entities[chunk_id].add(entity.id)

    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship between two entities.

        Both source and target entities must exist in the graph.
        """
        if relationship.source_id not in self.entities:
            raise ValueError(f"Source entity {relationship.source_id} not found")
        if relationship.target_id not in self.entities:
            raise ValueError(f"Target entity {relationship.target_id} not found")

        self.relationships.append(relationship)
        self._adjacency_out[relationship.source_id].append(relationship.target_id)
        self._adjacency_in[relationship.target_id].append(relationship.source_id)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get an entity by name (case-insensitive)."""
        entity_id = self._name_to_id.get(name.lower())
        if entity_id:
            return self.entities.get(entity_id)
        return None

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        entity_ids = self._type_index.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def get_entities_in_chunk(self, chunk_id: str) -> List[Entity]:
        """Get all entities that appear in a specific chunk."""
        entity_ids = self._chunk_to_entities.get(chunk_id, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def get_neighbors(
        self,
        entity_id: str,
        direction: str = "both",
        relation_types: Optional[List[str]] = None,
    ) -> List[Entity]:
        """
        Get neighboring entities connected to the given entity.

        Args:
            entity_id: ID of the entity to find neighbors for
            direction: "out" (outgoing), "in" (incoming), or "both"
            relation_types: Optional filter for specific relationship types

        Returns:
            List of neighboring Entity objects
        """
        neighbor_ids: Set[str] = set()

        if direction in ("out", "both"):
            neighbor_ids.update(self._adjacency_out.get(entity_id, []))
        if direction in ("in", "both"):
            neighbor_ids.update(self._adjacency_in.get(entity_id, []))

        if relation_types:
            filtered_ids = set()
            for rel in self.relationships:
                if rel.relation_type in relation_types:
                    if rel.source_id == entity_id and rel.target_id in neighbor_ids:
                        filtered_ids.add(rel.target_id)
                    if rel.target_id == entity_id and rel.source_id in neighbor_ids:
                        filtered_ids.add(rel.source_id)
            neighbor_ids = filtered_ids

        return [self.entities[nid] for nid in neighbor_ids if nid in self.entities]

    def get_relationships_for_entity(
        self,
        entity_id: str,
        direction: str = "both",
    ) -> List[Relationship]:
        """Get all relationships involving an entity."""
        results = []
        for rel in self.relationships:
            if direction in ("out", "both") and rel.source_id == entity_id:
                results.append(rel)
            elif direction in ("in", "both") and rel.target_id == entity_id:
                results.append(rel)
        return results

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3,
    ) -> List[List[str]]:
        """
        Find all paths between two entities up to a maximum depth.

        Uses BFS to find paths, returning entity IDs in path order.
        """
        if source_id not in self.entities or target_id not in self.entities:
            return []

        paths = []
        queue = [(source_id, [source_id])]
        visited_paths: Set[Tuple[str, ...]] = set()

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_depth + 1:
                continue

            if current == target_id and len(path) > 1:
                path_tuple = tuple(path)
                if path_tuple not in visited_paths:
                    visited_paths.add(path_tuple)
                    paths.append(path)
                continue

            for neighbor_id in self._adjacency_out.get(current, []):
                if neighbor_id not in path:
                    queue.append((neighbor_id, path + [neighbor_id]))

            for neighbor_id in self._adjacency_in.get(current, []):
                if neighbor_id not in path:
                    queue.append((neighbor_id, path + [neighbor_id]))

        return paths

    def get_subgraph(
        self,
        entity_ids: List[str],
        depth: int = 1,
    ) -> "KnowledgeGraph":
        """
        Extract a subgraph containing specified entities and their neighbors.

        Args:
            entity_ids: Starting entity IDs
            depth: How many hops to include from starting entities

        Returns:
            New KnowledgeGraph containing the subgraph
        """
        subgraph = KnowledgeGraph()
        included_ids: Set[str] = set()

        # BFS to find all entities within depth
        current_level = set(entity_ids)
        for _ in range(depth + 1):
            included_ids.update(current_level)
            next_level: Set[str] = set()
            for eid in current_level:
                next_level.update(self._adjacency_out.get(eid, []))
                next_level.update(self._adjacency_in.get(eid, []))
            current_level = next_level - included_ids

        # Add entities to subgraph
        for eid in included_ids:
            if eid in self.entities:
                subgraph.add_entity(self.entities[eid])

        # Add relationships where both endpoints are in subgraph
        for rel in self.relationships:
            if rel.source_id in included_ids and rel.target_id in included_ids:
                subgraph.add_relationship(rel)

        return subgraph

    def get_community_chunks(
        self,
        entity_ids: List[str],
        max_chunks: int = 10,
    ) -> List[str]:
        """
        Get the most relevant chunks for a set of entities.

        Ranks chunks by how many of the specified entities they contain.
        """
        chunk_scores: Dict[str, int] = defaultdict(int)

        for eid in entity_ids:
            entity = self.entities.get(eid)
            if entity:
                for chunk_id in entity.source_chunks:
                    chunk_scores[chunk_id] += 1

        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [chunk_id for chunk_id, _ in sorted_chunks[:max_chunks]]

    def save(self, filepath: str) -> None:
        """Save the knowledge graph to a file."""
        data = {
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "relationships": [r.to_dict() for r in self.relationships],
        }

        if filepath.endswith(".json"):
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "KnowledgeGraph":
        """Load a knowledge graph from a file."""
        if filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

        graph = cls()

        for entity_data in data["entities"].values():
            graph.add_entity(Entity.from_dict(entity_data))

        for rel_data in data["relationships"]:
            graph.add_relationship(Relationship.from_dict(rel_data))

        return graph

    def to_networkx(self):
        """
        Convert to NetworkX graph for advanced algorithms.

        Requires networkx to be installed.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX is required for this feature. "
                "Install it with: pip install networkx"
            )

        G = nx.DiGraph()

        for eid, entity in self.entities.items():
            G.add_node(eid, **entity.to_dict())

        for rel in self.relationships:
            G.add_edge(
                rel.source_id,
                rel.target_id,
                relation_type=rel.relation_type,
                weight=rel.weight,
                **rel.metadata,
            )

        return G

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraph(entities={self.num_entities}, "
            f"relationships={self.num_relationships})"
        )
