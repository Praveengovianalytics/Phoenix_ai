"""
Graph-aware retrieval for GraphRAG.

Combines vector similarity search with knowledge graph traversal
for enhanced multi-hop retrieval.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .knowledge_graph import Entity, KnowledgeGraph


class GraphRetriever:
    """
    Hybrid retriever combining vector search and graph traversal.

    Provides enhanced retrieval by:
    1. Finding semantically similar entities via embeddings
    2. Expanding context through graph relationships
    3. Fusing vector and graph scores for final ranking
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        embedding_client=None,
        graph_weight: float = 0.3,
        max_graph_depth: int = 2,
    ):
        """
        Initialize the graph retriever.

        Args:
            knowledge_graph: The knowledge graph to use for retrieval
            embedding_client: GenAIEmbeddingClient for query embeddings
            graph_weight: Weight for graph-based scores (0.0 to 1.0)
            max_graph_depth: Maximum depth for graph traversal
        """
        self.graph = knowledge_graph
        self.embedding_client = embedding_client
        self.graph_weight = graph_weight
        self.max_graph_depth = max_graph_depth

        # Cache entity embeddings as matrix for efficient search
        self._entity_embeddings: Optional[np.ndarray] = None
        self._entity_ids: List[str] = []
        self._build_embedding_index()

    def _build_embedding_index(self) -> None:
        """Build the entity embedding index for similarity search."""
        entities_with_embeddings = [
            (eid, e) for eid, e in self.graph.entities.items()
            if e.embedding is not None
        ]

        if not entities_with_embeddings:
            return

        self._entity_ids = [eid for eid, _ in entities_with_embeddings]
        embeddings = [e.embedding for _, e in entities_with_embeddings]
        self._entity_embeddings = np.array(embeddings, dtype="float32")

        # Normalize for cosine similarity
        norms = np.linalg.norm(self._entity_embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        self._entity_embeddings = self._entity_embeddings / norms

    def refresh_index(self) -> None:
        """Rebuild the embedding index after graph updates."""
        self._build_embedding_index()

    def _compute_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Compute embedding for query text."""
        if not self.embedding_client:
            return None

        try:
            embedding = self.embedding_client.generate_embedding([query])[0]
            embedding = np.array(embedding, dtype="float32")
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            print(f"Error computing query embedding: {e}")
            return None

    def find_similar_entities(
        self,
        query: str,
        k: int = 5,
        entity_types: Optional[List[str]] = None,
    ) -> List[Tuple[Entity, float]]:
        """
        Find entities similar to the query via embedding similarity.

        Args:
            query: Query text to match against entities
            k: Number of entities to return
            entity_types: Optional filter for entity types

        Returns:
            List of (Entity, similarity_score) tuples
        """
        if self._entity_embeddings is None or len(self._entity_ids) == 0:
            return []

        query_embedding = self._compute_query_embedding(query)
        if query_embedding is None:
            return []

        # Compute cosine similarities
        similarities = np.dot(self._entity_embeddings, query_embedding)

        # Get top-k indices
        if entity_types:
            # Filter by entity type
            mask = np.array([
                self.graph.entities[eid].entity_type in entity_types
                for eid in self._entity_ids
            ])
            masked_similarities = np.where(mask, similarities, -1)
            top_indices = np.argsort(masked_similarities)[::-1][:k]
        else:
            top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                entity_id = self._entity_ids[idx]
                entity = self.graph.entities.get(entity_id)
                if entity:
                    results.append((entity, float(similarities[idx])))

        return results

    def find_entities_by_name(
        self,
        query: str,
        fuzzy: bool = True,
    ) -> List[Entity]:
        """
        Find entities by name matching.

        Args:
            query: Query text to search for entity names
            fuzzy: Whether to use fuzzy matching

        Returns:
            List of matching entities
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        matches = []
        for entity in self.graph.entities.values():
            name_lower = entity.name.lower()

            # Exact match
            if name_lower in query_lower or query_lower in name_lower:
                matches.append(entity)
                continue

            # Word overlap match (fuzzy)
            if fuzzy:
                name_words = set(name_lower.split())
                overlap = query_words & name_words
                if overlap and len(overlap) / len(name_words) > 0.5:
                    matches.append(entity)

        return matches

    def expand_with_graph(
        self,
        entities: List[Entity],
        depth: int = 1,
        relation_types: Optional[List[str]] = None,
    ) -> List[Entity]:
        """
        Expand a set of entities through graph relationships.

        Args:
            entities: Starting entities to expand from
            depth: Number of hops to expand
            relation_types: Optional filter for relationship types

        Returns:
            Expanded list of entities including original and connected
        """
        expanded = {e.id: e for e in entities}
        current_level = [e.id for e in entities]

        for _ in range(depth):
            next_level = []
            for entity_id in current_level:
                neighbors = self.graph.get_neighbors(
                    entity_id,
                    direction="both",
                    relation_types=relation_types,
                )
                for neighbor in neighbors:
                    if neighbor.id not in expanded:
                        expanded[neighbor.id] = neighbor
                        next_level.append(neighbor.id)
            current_level = next_level

        return list(expanded.values())

    def get_entity_context(
        self,
        entity: Entity,
        max_relationships: int = 5,
    ) -> str:
        """
        Generate a text context for an entity from its relationships.

        Args:
            entity: Entity to generate context for
            max_relationships: Maximum relationships to include

        Returns:
            Text description of entity and its relationships
        """
        parts = [f"{entity.name} ({entity.entity_type})"]

        if entity.description:
            parts.append(f": {entity.description}")

        relationships = self.graph.get_relationships_for_entity(entity.id)[:max_relationships]

        if relationships:
            parts.append("\nRelationships:")
            for rel in relationships:
                if rel.source_id == entity.id:
                    target = self.graph.get_entity(rel.target_id)
                    if target:
                        parts.append(f"  - {rel.relation_type} -> {target.name}")
                else:
                    source = self.graph.get_entity(rel.source_id)
                    if source:
                        parts.append(f"  - {source.name} -> {rel.relation_type}")

        return "".join(parts)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        use_graph_expansion: bool = True,
        include_context: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval method combining semantic search and graph traversal.

        Args:
            query: Query text
            k: Number of results to return
            use_graph_expansion: Whether to expand via graph relationships
            include_context: Whether to include relationship context

        Returns:
            List of retrieval results with entities and scores
        """
        results = []

        # Step 1: Find similar entities via embeddings
        similar_entities = self.find_similar_entities(query, k=k * 2)

        # Step 2: Find entities by name matching
        name_matches = self.find_entities_by_name(query)

        # Combine and deduplicate
        entity_scores: Dict[str, Tuple[Entity, float]] = {}

        for entity, score in similar_entities:
            entity_scores[entity.id] = (entity, score)

        for entity in name_matches:
            if entity.id in entity_scores:
                # Boost score for entities matching both methods
                existing_entity, existing_score = entity_scores[entity.id]
                entity_scores[entity.id] = (existing_entity, min(1.0, existing_score + 0.2))
            else:
                entity_scores[entity.id] = (entity, 0.5)

        # Step 3: Expand through graph if enabled
        if use_graph_expansion and entity_scores:
            top_entities = [e for e, _ in sorted(
                entity_scores.values(),
                key=lambda x: x[1],
                reverse=True
            )[:k]]

            expanded = self.expand_with_graph(top_entities, depth=self.max_graph_depth)

            # Add expanded entities with decayed scores
            for entity in expanded:
                if entity.id not in entity_scores:
                    # Calculate score based on connection to top entities
                    connection_score = 0.0
                    for top_entity in top_entities:
                        paths = self.graph.find_paths(
                            top_entity.id,
                            entity.id,
                            max_depth=self.max_graph_depth
                        )
                        if paths:
                            # Score decays with path length
                            min_path_len = min(len(p) for p in paths)
                            connection_score = max(
                                connection_score,
                                1.0 / (min_path_len + 1)
                            )

                    if connection_score > 0:
                        entity_scores[entity.id] = (
                            entity,
                            connection_score * self.graph_weight
                        )

        # Step 4: Get relevant chunks for entities
        sorted_entities = sorted(
            entity_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        for entity, score in sorted_entities:
            result = {
                "entity": entity,
                "score": score,
                "chunks": entity.source_chunks,
            }

            if include_context:
                result["context"] = self.get_entity_context(entity)

            results.append(result)

        return results

    def get_chunks_for_query(
        self,
        query: str,
        k: int = 5,
        chunk_store: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Get the most relevant chunk IDs for a query.

        Combines entity-based retrieval with chunk scoring.

        Args:
            query: Query text
            k: Number of chunks to return
            chunk_store: Optional mapping of chunk_id -> chunk_text

        Returns:
            List of (chunk_id, score) tuples or (chunk_text, score) if store provided
        """
        retrieval_results = self.retrieve(query, k=k * 2)

        # Aggregate chunk scores from entities
        chunk_scores: Dict[str, float] = defaultdict(float)

        for result in retrieval_results:
            entity_score = result["score"]
            for chunk_id in result["chunks"]:
                chunk_scores[chunk_id] += entity_score

        # Normalize and sort
        if chunk_scores:
            max_score = max(chunk_scores.values())
            chunk_scores = {
                cid: score / max_score
                for cid, score in chunk_scores.items()
            }

        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        if chunk_store:
            return [
                (chunk_store.get(cid, cid), score)
                for cid, score in sorted_chunks
            ]

        return sorted_chunks


class HybridRetriever:
    """
    Combines traditional vector retrieval with graph-based retrieval.

    Uses a weighted fusion of vector similarity and graph-based scores.
    """

    def __init__(
        self,
        graph_retriever: GraphRetriever,
        vector_weight: float = 0.7,
    ):
        """
        Initialize hybrid retriever.

        Args:
            graph_retriever: GraphRetriever instance
            vector_weight: Weight for vector similarity (graph weight = 1 - vector_weight)
        """
        self.graph_retriever = graph_retriever
        self.vector_weight = vector_weight

    def retrieve(
        self,
        query: str,
        vector_results: List[Tuple[str, float]],
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Combine vector search results with graph-based retrieval.

        Args:
            query: Query text
            vector_results: Results from vector search as (chunk_id, score) tuples
            k: Number of results to return

        Returns:
            Fused results as (chunk_id, combined_score) tuples
        """
        # Get graph-based chunk scores
        graph_results = self.graph_retriever.get_chunks_for_query(query, k=k * 2)
        graph_scores = dict(graph_results)

        # Normalize vector scores
        vector_scores = dict(vector_results)
        if vector_scores:
            max_vec = max(vector_scores.values())
            if max_vec > 0:
                vector_scores = {k: v / max_vec for k, v in vector_scores.items()}

        # Fuse scores
        all_chunks = set(vector_scores.keys()) | set(graph_scores.keys())
        fused_scores = {}

        for chunk_id in all_chunks:
            vec_score = vector_scores.get(chunk_id, 0.0)
            graph_score = graph_scores.get(chunk_id, 0.0)

            fused_score = (
                self.vector_weight * vec_score +
                (1 - self.vector_weight) * graph_score
            )
            fused_scores[chunk_id] = fused_score

        # Sort and return top-k
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        return sorted_results
