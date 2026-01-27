"""
Relationship extraction for GraphRAG.

Provides methods to extract relationships between entities from text.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .knowledge_graph import Entity, Relationship


class RelationshipExtractor(ABC):
    """Abstract base class for relationship extraction."""

    @abstractmethod
    def extract(
        self,
        entities: List[Entity],
        text: str,
        chunk_id: str = "",
    ) -> List[Relationship]:
        """
        Extract relationships between entities from text.

        Args:
            entities: List of entities to find relationships between
            text: The text containing the entities
            chunk_id: Optional identifier for the source chunk

        Returns:
            List of extracted Relationship objects
        """
        pass


class LLMRelationshipExtractor(RelationshipExtractor):
    """
    LLM-based relationship extractor.

    Uses the GenAIChatClient to extract relationships between entities
    using natural language understanding.
    """

    DEFAULT_RELATION_TYPES = [
        "works_for",
        "manages",
        "created_by",
        "located_in",
        "part_of",
        "related_to",
        "owns",
        "uses",
        "depends_on",
        "competes_with",
        "partners_with",
        "produces",
        "founded",
        "acquired",
        "invested_in",
    ]

    DEFAULT_EXTRACTION_PROMPT = """Analyze the relationships between the given entities based on the text.

Entities to analyze:
{entities}

Text:
\"\"\"
{text}
\"\"\"

For each pair of entities that have a relationship in the text, provide:
- source: The source entity name
- target: The target entity name
- relation_type: One of {relation_types}
- confidence: How confident you are (0.0 to 1.0)
- description: Brief description of the relationship

Return ONLY a valid JSON array with no additional text:
[{{"source": "...", "target": "...", "relation_type": "...", "confidence": 0.9, "description": "..."}}]

If no relationships are found, return an empty array: []

JSON array of relationships:"""

    def __init__(
        self,
        chat_client,
        relation_types: Optional[List[str]] = None,
        extraction_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        min_confidence: float = 0.5,
    ):
        """
        Initialize the LLM relationship extractor.

        Args:
            chat_client: GenAIChatClient instance for LLM calls
            relation_types: List of relationship types to extract
            extraction_prompt: Custom prompt template
            max_tokens: Maximum tokens for LLM response
            temperature: LLM temperature
            min_confidence: Minimum confidence threshold for relationships
        """
        self.chat_client = chat_client
        self.relation_types = relation_types or self.DEFAULT_RELATION_TYPES
        self.extraction_prompt = extraction_prompt or self.DEFAULT_EXTRACTION_PROMPT
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.min_confidence = min_confidence

    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract relationship data."""
        response = response.strip()

        # Handle markdown code blocks
        if "```json" in response:
            match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                response = match.group(1)

        # Try to find JSON array
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            response = match.group(0)

        try:
            data = json.loads(response)
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            response = re.sub(r",\s*([}\]])", r"\1", response)
            try:
                data = json.loads(response)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
            return []

    def extract(
        self,
        entities: List[Entity],
        text: str,
        chunk_id: str = "",
    ) -> List[Relationship]:
        """
        Extract relationships between entities from text.

        Args:
            entities: List of entities to find relationships between
            text: The text containing the entities
            chunk_id: Optional identifier for the source chunk

        Returns:
            List of extracted Relationship objects
        """
        if len(entities) < 2 or not text.strip():
            return []

        # Build entity name to ID mapping
        name_to_entity = {e.name.lower(): e for e in entities}

        # Format entities for prompt
        entity_list = "\n".join(
            f"- {e.name} ({e.entity_type}): {e.description}"
            for e in entities
        )

        # Prepare the prompt
        prompt = self.extraction_prompt.format(
            entities=entity_list,
            text=text[:4000],
            relation_types=", ".join(self.relation_types),
        )

        try:
            response = self.chat_client.chat(
                user_input=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            rel_data = self._parse_llm_response(response)
            relationships = []

            for item in rel_data:
                if not isinstance(item, dict):
                    continue

                source_name = item.get("source", "").strip().lower()
                target_name = item.get("target", "").strip().lower()
                relation_type = item.get("relation_type", "related_to").lower()
                confidence = float(item.get("confidence", 0.5))
                description = item.get("description", "")

                # Skip if below confidence threshold
                if confidence < self.min_confidence:
                    continue

                # Find matching entities
                source_entity = name_to_entity.get(source_name)
                target_entity = name_to_entity.get(target_name)

                if not source_entity or not target_entity:
                    # Try partial matching
                    for name, entity in name_to_entity.items():
                        if source_name in name or name in source_name:
                            source_entity = source_entity or entity
                        if target_name in name or name in target_name:
                            target_entity = target_entity or entity

                if not source_entity or not target_entity:
                    continue

                if source_entity.id == target_entity.id:
                    continue

                # Validate relation type
                if relation_type not in self.relation_types:
                    relation_type = "related_to"

                relationship = Relationship(
                    source_id=source_entity.id,
                    target_id=target_entity.id,
                    relation_type=relation_type,
                    weight=confidence,
                    description=description,
                    evidence_chunks=[chunk_id] if chunk_id else [],
                )
                relationships.append(relationship)

            return relationships

        except Exception as e:
            print(f"Relationship extraction error: {e}")
            return []

    def extract_pairwise(
        self,
        entity1: Entity,
        entity2: Entity,
        text: str,
        chunk_id: str = "",
    ) -> Optional[Relationship]:
        """
        Extract relationship between a specific pair of entities.

        More focused extraction for known entity pairs.
        """
        relationships = self.extract([entity1, entity2], text, chunk_id)
        return relationships[0] if relationships else None


class CooccurrenceRelationshipExtractor(RelationshipExtractor):
    """
    Simple co-occurrence based relationship extractor.

    Creates relationships based on entities appearing in the same context window.
    Fast but less precise than LLM-based extraction.
    """

    def __init__(
        self,
        window_size: int = 100,
        min_cooccurrences: int = 1,
    ):
        """
        Initialize co-occurrence extractor.

        Args:
            window_size: Character window for co-occurrence detection
            min_cooccurrences: Minimum co-occurrences to create relationship
        """
        self.window_size = window_size
        self.min_cooccurrences = min_cooccurrences

    def _find_entity_positions(
        self,
        entity: Entity,
        text: str,
    ) -> List[Tuple[int, int]]:
        """Find all positions of an entity in text."""
        positions = []
        text_lower = text.lower()
        name_lower = entity.name.lower()
        start = 0

        while True:
            pos = text_lower.find(name_lower, start)
            if pos == -1:
                break
            positions.append((pos, pos + len(name_lower)))
            start = pos + 1

        return positions

    def extract(
        self,
        entities: List[Entity],
        text: str,
        chunk_id: str = "",
    ) -> List[Relationship]:
        """
        Extract relationships based on co-occurrence.

        Creates bidirectional "related_to" relationships for entities
        appearing within the window size of each other.
        """
        if len(entities) < 2:
            return []

        # Find positions of all entities
        entity_positions = {
            e.id: self._find_entity_positions(e, text)
            for e in entities
        }

        # Track co-occurrences
        cooccurrences: Dict[Tuple[str, str], int] = {}

        for e1 in entities:
            for e2 in entities:
                if e1.id >= e2.id:
                    continue

                count = 0
                for pos1_start, pos1_end in entity_positions.get(e1.id, []):
                    for pos2_start, pos2_end in entity_positions.get(e2.id, []):
                        distance = min(
                            abs(pos1_end - pos2_start),
                            abs(pos2_end - pos1_start)
                        )
                        if distance <= self.window_size:
                            count += 1

                if count >= self.min_cooccurrences:
                    cooccurrences[(e1.id, e2.id)] = count

        # Create relationships
        relationships = []
        max_count = max(cooccurrences.values()) if cooccurrences else 1

        for (source_id, target_id), count in cooccurrences.items():
            weight = count / max_count
            relationship = Relationship(
                source_id=source_id,
                target_id=target_id,
                relation_type="related_to",
                weight=weight,
                description=f"Co-occurred {count} times within {self.window_size} chars",
                evidence_chunks=[chunk_id] if chunk_id else [],
            )
            relationships.append(relationship)

        return relationships


class HybridRelationshipExtractor(RelationshipExtractor):
    """
    Combines LLM and co-occurrence extraction for better coverage.
    """

    def __init__(
        self,
        llm_extractor: LLMRelationshipExtractor,
        cooccurrence_extractor: Optional[CooccurrenceRelationshipExtractor] = None,
        llm_weight: float = 0.7,
    ):
        """
        Initialize hybrid extractor.

        Args:
            llm_extractor: LLM-based extractor
            cooccurrence_extractor: Co-occurrence extractor
            llm_weight: Weight for LLM relationships vs co-occurrence
        """
        self.llm_extractor = llm_extractor
        self.cooccurrence_extractor = (
            cooccurrence_extractor or CooccurrenceRelationshipExtractor()
        )
        self.llm_weight = llm_weight

    def extract(
        self,
        entities: List[Entity],
        text: str,
        chunk_id: str = "",
    ) -> List[Relationship]:
        """
        Extract relationships using both methods.

        LLM relationships override co-occurrence relationships when
        the same entity pair is detected by both methods.
        """
        rel_dict: Dict[Tuple[str, str], Relationship] = {}

        # First, co-occurrence (will be overridden by LLM if found)
        cooc_rels = self.cooccurrence_extractor.extract(entities, text, chunk_id)
        for rel in cooc_rels:
            key = (rel.source_id, rel.target_id)
            rel.weight *= (1 - self.llm_weight)
            rel_dict[key] = rel

        # Then, LLM extraction (higher priority)
        llm_rels = self.llm_extractor.extract(entities, text, chunk_id)
        for rel in llm_rels:
            key = (rel.source_id, rel.target_id)
            rel.weight *= self.llm_weight
            if key in rel_dict:
                # Combine weights if both found the relationship
                existing = rel_dict[key]
                rel.weight = min(1.0, rel.weight + existing.weight)
            rel_dict[key] = rel

        return list(rel_dict.values())
