"""
Entity extraction for GraphRAG.

Provides multiple strategies for extracting entities from text:
- LLM-based extraction using chat models
- Regex-based extraction for known patterns
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .knowledge_graph import Entity


class EntityExtractor(ABC):
    """Abstract base class for entity extraction."""

    @abstractmethod
    def extract(self, text: str, chunk_id: str = "") -> List[Entity]:
        """
        Extract entities from text.

        Args:
            text: The text to extract entities from
            chunk_id: Optional identifier for the source chunk

        Returns:
            List of extracted Entity objects
        """
        pass


class LLMEntityExtractor(EntityExtractor):
    """
    LLM-based entity extractor using chat models.

    Uses the GenAIChatClient to extract structured entity information
    from text using natural language understanding.
    """

    DEFAULT_ENTITY_TYPES = [
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "PRODUCT",
        "TECHNOLOGY",
        "CONCEPT",
        "EVENT",
        "METRIC",
        "DATE",
    ]

    DEFAULT_EXTRACTION_PROMPT = """Extract all important entities from the following text.
For each entity, provide:
- name: The entity name as it appears in the text
- type: One of {entity_types}
- description: A brief description based on the context (1 sentence max)

Return ONLY a valid JSON array with no additional text:
[{{"name": "...", "type": "...", "description": "..."}}]

If no entities are found, return an empty array: []

Text:
\"\"\"
{text}
\"\"\"

JSON array of entities:"""

    def __init__(
        self,
        chat_client,
        entity_types: Optional[List[str]] = None,
        extraction_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        """
        Initialize the LLM entity extractor.

        Args:
            chat_client: GenAIChatClient instance for LLM calls
            entity_types: List of entity types to extract
            extraction_prompt: Custom prompt template (must include {text} and {entity_types})
            max_tokens: Maximum tokens for LLM response
            temperature: LLM temperature (lower = more deterministic)
        """
        self.chat_client = chat_client
        self.entity_types = entity_types or self.DEFAULT_ENTITY_TYPES
        self.extraction_prompt = extraction_prompt or self.DEFAULT_EXTRACTION_PROMPT
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate a unique ID for an entity based on name and type."""
        key = f"{entity_type}:{name.lower().strip()}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract entity data."""
        # Try to find JSON array in response
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
            # Try to fix common JSON issues
            response = re.sub(r",\s*([}\]])", r"\1", response)
            try:
                data = json.loads(response)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
            return []

    def extract(self, text: str, chunk_id: str = "") -> List[Entity]:
        """
        Extract entities from text using LLM.

        Args:
            text: The text to extract entities from
            chunk_id: Optional identifier for the source chunk

        Returns:
            List of extracted Entity objects
        """
        if not text.strip():
            return []

        # Prepare the prompt
        prompt = self.extraction_prompt.format(
            text=text[:4000],  # Limit text length
            entity_types=", ".join(self.entity_types),
        )

        try:
            response = self.chat_client.chat(
                user_input=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            entity_data = self._parse_llm_response(response)
            entities = []

            for item in entity_data:
                if not isinstance(item, dict):
                    continue

                name = item.get("name", "").strip()
                entity_type = item.get("type", "CONCEPT").upper()
                description = item.get("description", "")

                if not name:
                    continue

                # Validate entity type
                if entity_type not in self.entity_types:
                    entity_type = "CONCEPT"

                entity_id = self._generate_entity_id(name, entity_type)

                entity = Entity(
                    id=entity_id,
                    name=name,
                    entity_type=entity_type,
                    description=description,
                    source_chunks=[chunk_id] if chunk_id else [],
                )
                entities.append(entity)

            return entities

        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []

    def extract_batch(
        self,
        texts: List[str],
        chunk_ids: Optional[List[str]] = None,
    ) -> List[List[Entity]]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of texts to extract entities from
            chunk_ids: Optional list of chunk identifiers

        Returns:
            List of entity lists, one per input text
        """
        if chunk_ids is None:
            chunk_ids = [f"chunk_{i}" for i in range(len(texts))]

        results = []
        for text, chunk_id in zip(texts, chunk_ids):
            entities = self.extract(text, chunk_id)
            results.append(entities)

        return results


class RegexEntityExtractor(EntityExtractor):
    """
    Regex-based entity extractor for known patterns.

    Fast extraction for common entity patterns like emails,
    URLs, dates, and custom patterns.
    """

    DEFAULT_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "URL": r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*",
        "PHONE": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
        "DATE": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
        "MONEY": r"\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY)\b",
        "PERCENTAGE": r"\b\d+(?:\.\d+)?%\b",
    }

    def __init__(
        self,
        patterns: Optional[Dict[str, str]] = None,
        include_defaults: bool = True,
    ):
        """
        Initialize the regex entity extractor.

        Args:
            patterns: Custom patterns as {entity_type: regex_pattern}
            include_defaults: Whether to include default patterns
        """
        self.patterns = {}
        if include_defaults:
            self.patterns.update(self.DEFAULT_PATTERNS)
        if patterns:
            self.patterns.update(patterns)

        # Compile patterns
        self._compiled = {
            entity_type: re.compile(pattern, re.IGNORECASE)
            for entity_type, pattern in self.patterns.items()
        }

    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate a unique ID for an entity."""
        key = f"{entity_type}:{name.lower().strip()}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def extract(self, text: str, chunk_id: str = "") -> List[Entity]:
        """
        Extract entities from text using regex patterns.

        Args:
            text: The text to extract entities from
            chunk_id: Optional identifier for the source chunk

        Returns:
            List of extracted Entity objects
        """
        entities = []
        seen = set()

        for entity_type, pattern in self._compiled.items():
            matches = pattern.findall(text)
            for match in matches:
                name = match if isinstance(match, str) else match[0]
                name = name.strip()

                if not name or name.lower() in seen:
                    continue
                seen.add(name.lower())

                entity_id = self._generate_entity_id(name, entity_type)
                entity = Entity(
                    id=entity_id,
                    name=name,
                    entity_type=entity_type,
                    description=f"Extracted {entity_type.lower()} from text",
                    source_chunks=[chunk_id] if chunk_id else [],
                )
                entities.append(entity)

        return entities


class HybridEntityExtractor(EntityExtractor):
    """
    Combines LLM and regex extraction for comprehensive entity coverage.

    Uses regex for known patterns (fast) and LLM for semantic entities.
    """

    def __init__(
        self,
        llm_extractor: LLMEntityExtractor,
        regex_extractor: Optional[RegexEntityExtractor] = None,
    ):
        """
        Initialize hybrid extractor.

        Args:
            llm_extractor: LLM-based extractor for semantic entities
            regex_extractor: Regex extractor for pattern-based entities
        """
        self.llm_extractor = llm_extractor
        self.regex_extractor = regex_extractor or RegexEntityExtractor()

    def extract(self, text: str, chunk_id: str = "") -> List[Entity]:
        """
        Extract entities using both regex and LLM.

        Deduplicates based on entity ID.
        """
        entities_dict: Dict[str, Entity] = {}

        # First, fast regex extraction
        regex_entities = self.regex_extractor.extract(text, chunk_id)
        for entity in regex_entities:
            entities_dict[entity.id] = entity

        # Then, LLM extraction
        llm_entities = self.llm_extractor.extract(text, chunk_id)
        for entity in llm_entities:
            if entity.id in entities_dict:
                entities_dict[entity.id].merge_with(entity)
            else:
                entities_dict[entity.id] = entity

        return list(entities_dict.values())
