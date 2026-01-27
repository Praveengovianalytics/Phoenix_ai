"""
Graph Builder for constructing knowledge graphs from documents.

Orchestrates entity extraction, relationship extraction, and graph assembly.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from .entity_extractor import EntityExtractor, LLMEntityExtractor
from .knowledge_graph import Entity, KnowledgeGraph, Relationship
from .relationship_extractor import (
    LLMRelationshipExtractor,
    RelationshipExtractor,
)


class GraphBuilder:
    """
    Builds knowledge graphs from document chunks.

    Orchestrates the extraction of entities and relationships,
    entity deduplication, and graph assembly.
    """

    def __init__(
        self,
        entity_extractor: EntityExtractor,
        relationship_extractor: RelationshipExtractor,
        embedding_client=None,
        deduplicate_entities: bool = True,
        similarity_threshold: float = 0.85,
    ):
        """
        Initialize the graph builder.

        Args:
            entity_extractor: Extractor for entities
            relationship_extractor: Extractor for relationships
            embedding_client: Optional GenAIEmbeddingClient for entity embeddings
            deduplicate_entities: Whether to merge similar entities
            similarity_threshold: Threshold for entity deduplication
        """
        self.entity_extractor = entity_extractor
        self.relationship_extractor = relationship_extractor
        self.embedding_client = embedding_client
        self.deduplicate_entities = deduplicate_entities
        self.similarity_threshold = similarity_threshold

    @classmethod
    def from_clients(
        cls,
        chat_client,
        embedding_client=None,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> "GraphBuilder":
        """
        Create a GraphBuilder from GenAI clients.

        Convenience factory method for quick setup.

        Args:
            chat_client: GenAIChatClient for LLM operations
            embedding_client: Optional GenAIEmbeddingClient
            entity_types: Custom entity types to extract
            relation_types: Custom relationship types to extract

        Returns:
            Configured GraphBuilder instance
        """
        entity_extractor = LLMEntityExtractor(
            chat_client=chat_client,
            entity_types=entity_types,
        )

        relationship_extractor = LLMRelationshipExtractor(
            chat_client=chat_client,
            relation_types=relation_types,
        )

        return cls(
            entity_extractor=entity_extractor,
            relationship_extractor=relationship_extractor,
            embedding_client=embedding_client,
        )

    def _compute_entity_embeddings(
        self,
        entities: List[Entity],
        batch_size: int = 16,
    ) -> None:
        """Compute embeddings for entities that don't have them."""
        if not self.embedding_client:
            return

        entities_needing_embedding = [
            e for e in entities if e.embedding is None
        ]

        if not entities_needing_embedding:
            return

        # Create text representations for embedding
        texts = [
            f"{e.name}: {e.description}" if e.description else e.name
            for e in entities_needing_embedding
        ]

        try:
            embeddings = self.embedding_client.generate_embedding(
                texts,
                batch_size=batch_size,
            )

            for entity, embedding in zip(entities_needing_embedding, embeddings):
                entity.embedding = embedding

        except Exception as e:
            print(f"Error computing entity embeddings: {e}")

    def _deduplicate_entities(
        self,
        entities: List[Entity],
    ) -> Tuple[List[Entity], Dict[str, str]]:
        """
        Deduplicate entities based on name similarity.

        Returns:
            Tuple of (deduplicated entities, old_id -> new_id mapping)
        """
        if not entities:
            return [], {}

        # Group by normalized name
        name_groups: Dict[str, List[Entity]] = {}
        for entity in entities:
            key = entity.name.lower().strip()
            if key not in name_groups:
                name_groups[key] = []
            name_groups[key].append(entity)

        # Merge entities with same normalized name
        merged_entities = []
        id_mapping: Dict[str, str] = {}

        for name_key, group in name_groups.items():
            if not group:
                continue

            # Use first entity as base, merge others into it
            base_entity = group[0]
            for other in group[1:]:
                base_entity.merge_with(other)
                id_mapping[other.id] = base_entity.id

            id_mapping[base_entity.id] = base_entity.id
            merged_entities.append(base_entity)

        return merged_entities, id_mapping

    def _update_relationship_ids(
        self,
        relationships: List[Relationship],
        id_mapping: Dict[str, str],
    ) -> List[Relationship]:
        """Update relationship entity IDs after deduplication."""
        updated = []
        seen = set()

        for rel in relationships:
            source_id = id_mapping.get(rel.source_id, rel.source_id)
            target_id = id_mapping.get(rel.target_id, rel.target_id)

            # Skip self-relationships after dedup
            if source_id == target_id:
                continue

            # Skip duplicates
            key = (source_id, target_id, rel.relation_type)
            if key in seen:
                continue
            seen.add(key)

            rel.source_id = source_id
            rel.target_id = target_id
            updated.append(rel)

        return updated

    def build_from_chunks(
        self,
        chunks: List[str],
        chunk_ids: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> KnowledgeGraph:
        """
        Build a knowledge graph from text chunks.

        Args:
            chunks: List of text chunks
            chunk_ids: Optional list of chunk identifiers
            progress_callback: Optional callback(current, total, stage)

        Returns:
            Constructed KnowledgeGraph
        """
        if chunk_ids is None:
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

        all_entities: List[Entity] = []
        all_relationships: List[Relationship] = []

        total_chunks = len(chunks)

        # Stage 1: Extract entities from each chunk
        for i, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
            if progress_callback:
                progress_callback(i + 1, total_chunks, "Extracting entities")

            entities = self.entity_extractor.extract(chunk, chunk_id)
            all_entities.extend(entities)

        # Stage 2: Deduplicate entities
        if self.deduplicate_entities:
            all_entities, id_mapping = self._deduplicate_entities(all_entities)
        else:
            id_mapping = {e.id: e.id for e in all_entities}

        # Stage 3: Compute entity embeddings
        if self.embedding_client:
            if progress_callback:
                progress_callback(0, 1, "Computing entity embeddings")
            self._compute_entity_embeddings(all_entities)

        # Stage 4: Extract relationships from each chunk
        entity_by_chunk: Dict[str, List[Entity]] = {}
        for entity in all_entities:
            for chunk_id in entity.source_chunks:
                if chunk_id not in entity_by_chunk:
                    entity_by_chunk[chunk_id] = []
                entity_by_chunk[chunk_id].append(entity)

        for i, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
            if progress_callback:
                progress_callback(i + 1, total_chunks, "Extracting relationships")

            chunk_entities = entity_by_chunk.get(chunk_id, [])
            if len(chunk_entities) >= 2:
                relationships = self.relationship_extractor.extract(
                    chunk_entities, chunk, chunk_id
                )
                all_relationships.extend(relationships)

        # Stage 5: Update relationship IDs after deduplication
        all_relationships = self._update_relationship_ids(
            all_relationships, id_mapping
        )

        # Stage 6: Build the graph
        graph = KnowledgeGraph()

        for entity in all_entities:
            graph.add_entity(entity)

        for relationship in all_relationships:
            try:
                graph.add_relationship(relationship)
            except ValueError:
                # Skip relationships with missing entities
                pass

        if progress_callback:
            progress_callback(1, 1, "Complete")

        return graph

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "content",
        chunk_id_column: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> KnowledgeGraph:
        """
        Build a knowledge graph from a DataFrame.

        Compatible with Phoenix AI document loading pipeline.

        Args:
            df: DataFrame with text content
            text_column: Column containing text chunks
            chunk_id_column: Optional column for chunk IDs
            progress_callback: Optional progress callback

        Returns:
            Constructed KnowledgeGraph
        """
        chunks = df[text_column].dropna().astype(str).tolist()

        if chunk_id_column and chunk_id_column in df.columns:
            chunk_ids = df[chunk_id_column].astype(str).tolist()
        else:
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

        return self.build_from_chunks(
            chunks=chunks,
            chunk_ids=chunk_ids,
            progress_callback=progress_callback,
        )

    def add_documents_to_graph(
        self,
        graph: KnowledgeGraph,
        chunks: List[str],
        chunk_ids: Optional[List[str]] = None,
    ) -> KnowledgeGraph:
        """
        Add new documents to an existing knowledge graph.

        Incrementally updates the graph with new entities and relationships.

        Args:
            graph: Existing KnowledgeGraph to update
            chunks: New text chunks to add
            chunk_ids: Optional chunk identifiers

        Returns:
            Updated KnowledgeGraph
        """
        new_graph = self.build_from_chunks(chunks, chunk_ids)

        # Merge new entities
        for entity in new_graph.entities.values():
            existing = graph.get_entity_by_name(entity.name)
            if existing:
                existing.merge_with(entity)
            else:
                graph.add_entity(entity)

        # Add new relationships
        for rel in new_graph.relationships:
            # Check if both entities exist in main graph
            source = graph.get_entity(rel.source_id)
            target = graph.get_entity(rel.target_id)

            if not source:
                source = graph.get_entity_by_name(
                    new_graph.entities[rel.source_id].name
                )
            if not target:
                target = graph.get_entity_by_name(
                    new_graph.entities[rel.target_id].name
                )

            if source and target:
                rel.source_id = source.id
                rel.target_id = target.id
                try:
                    graph.add_relationship(rel)
                except ValueError:
                    pass

        return graph


class IncrementalGraphBuilder:
    """
    Builds knowledge graphs incrementally for large document sets.

    Processes documents in batches to manage memory and LLM costs.
    """

    def __init__(
        self,
        graph_builder: GraphBuilder,
        batch_size: int = 10,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize incremental builder.

        Args:
            graph_builder: Base GraphBuilder instance
            batch_size: Number of chunks to process per batch
            checkpoint_path: Optional path to save checkpoints
        """
        self.graph_builder = graph_builder
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path

    def build(
        self,
        chunks: List[str],
        chunk_ids: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> KnowledgeGraph:
        """
        Build graph incrementally in batches.

        Args:
            chunks: List of text chunks
            chunk_ids: Optional chunk identifiers
            progress_callback: Optional progress callback

        Returns:
            Complete KnowledgeGraph
        """
        if chunk_ids is None:
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

        graph = KnowledgeGraph()
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(chunks))

            batch_chunks = chunks[start:end]
            batch_ids = chunk_ids[start:end]

            if progress_callback:
                progress_callback(
                    batch_idx + 1,
                    total_batches,
                    f"Processing batch {batch_idx + 1}/{total_batches}"
                )

            # Build graph for this batch
            batch_graph = self.graph_builder.build_from_chunks(
                batch_chunks, batch_ids
            )

            # Merge into main graph
            for entity in batch_graph.entities.values():
                existing = graph.get_entity_by_name(entity.name)
                if existing:
                    existing.merge_with(entity)
                else:
                    graph.add_entity(entity)

            for rel in batch_graph.relationships:
                source = graph.get_entity(rel.source_id)
                target = graph.get_entity(rel.target_id)

                if not source:
                    source = graph.get_entity_by_name(
                        batch_graph.entities.get(rel.source_id, Entity("", "", "")).name
                    )
                if not target:
                    target = graph.get_entity_by_name(
                        batch_graph.entities.get(rel.target_id, Entity("", "", "")).name
                    )

                if source and target:
                    rel.source_id = source.id
                    rel.target_id = target.id
                    try:
                        graph.add_relationship(rel)
                    except ValueError:
                        pass

            # Save checkpoint
            if self.checkpoint_path:
                graph.save(self.checkpoint_path)

        return graph
