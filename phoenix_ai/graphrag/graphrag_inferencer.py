"""
GraphRAG Inferencer for Phoenix AI.

Extends the standard RAGInferencer with knowledge graph capabilities
for enhanced multi-hop reasoning and context retrieval.
"""

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config_param import Param
from ..rag_inference import RAGInferencer
from .entity_extractor import LLMEntityExtractor
from .graph_builder import GraphBuilder
from .graph_retriever import GraphRetriever, HybridRetriever
from .knowledge_graph import Entity, KnowledgeGraph


class GraphRAGInferencer(RAGInferencer):
    """
    Graph-enhanced RAG Inferencer.

    Extends RAGInferencer with knowledge graph capabilities:
    - Entity-aware retrieval
    - Multi-hop graph traversal
    - Hybrid vector + graph scoring
    - Enhanced context building with relationship information
    """

    def __init__(
        self,
        embedding_client,
        chat_client,
        keyword_search_client=None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        graph_weight: float = 0.3,
        max_graph_depth: int = 2,
        include_graph_context: bool = True,
    ):
        """
        Initialize the GraphRAG Inferencer.

        Args:
            embedding_client: GenAIEmbeddingClient for embeddings
            chat_client: GenAIChatClient for generation
            keyword_search_client: Optional keyword search client
            knowledge_graph: Pre-built knowledge graph (or build with build_graph)
            graph_weight: Weight for graph-based retrieval (0.0 to 1.0)
            max_graph_depth: Maximum depth for graph traversal
            include_graph_context: Whether to include graph context in prompts
        """
        super().__init__(embedding_client, chat_client, keyword_search_client)

        self.knowledge_graph = knowledge_graph
        self.graph_weight = graph_weight
        self.max_graph_depth = max_graph_depth
        self.include_graph_context = include_graph_context

        # Initialize components
        self._graph_retriever: Optional[GraphRetriever] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self._entity_extractor: Optional[LLMEntityExtractor] = None
        self._graph_builder: Optional[GraphBuilder] = None
        self._chunk_store: Dict[str, str] = {}

        if knowledge_graph:
            self._initialize_retriever()

    def _initialize_retriever(self) -> None:
        """Initialize the graph retriever with the current knowledge graph."""
        if self.knowledge_graph:
            self._graph_retriever = GraphRetriever(
                knowledge_graph=self.knowledge_graph,
                embedding_client=self.embedding_client,
                graph_weight=self.graph_weight,
                max_graph_depth=self.max_graph_depth,
            )
            self._hybrid_retriever = HybridRetriever(
                graph_retriever=self._graph_retriever,
                vector_weight=1 - self.graph_weight,
            )

    def _get_entity_extractor(self) -> LLMEntityExtractor:
        """Get or create the entity extractor."""
        if self._entity_extractor is None:
            self._entity_extractor = LLMEntityExtractor(
                chat_client=self.chat_client,
            )
        return self._entity_extractor

    def _get_graph_builder(self) -> GraphBuilder:
        """Get or create the graph builder."""
        if self._graph_builder is None:
            self._graph_builder = GraphBuilder.from_clients(
                chat_client=self.chat_client,
                embedding_client=self.embedding_client,
            )
        return self._graph_builder

    def build_graph(
        self,
        chunks: List[str],
        chunk_ids: Optional[List[str]] = None,
        progress_callback=None,
    ) -> KnowledgeGraph:
        """
        Build a knowledge graph from document chunks.

        Args:
            chunks: List of text chunks
            chunk_ids: Optional chunk identifiers
            progress_callback: Optional callback(current, total, stage)

        Returns:
            Built KnowledgeGraph
        """
        if chunk_ids is None:
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

        # Store chunks for later retrieval
        for chunk_id, chunk in zip(chunk_ids, chunks):
            self._chunk_store[chunk_id] = chunk

        builder = self._get_graph_builder()
        self.knowledge_graph = builder.build_from_chunks(
            chunks=chunks,
            chunk_ids=chunk_ids,
            progress_callback=progress_callback,
        )

        self._initialize_retriever()
        return self.knowledge_graph

    def build_graph_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "content",
        chunk_id_column: Optional[str] = None,
        progress_callback=None,
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
            Built KnowledgeGraph
        """
        chunks = df[text_column].dropna().astype(str).tolist()

        if chunk_id_column and chunk_id_column in df.columns:
            chunk_ids = df[chunk_id_column].astype(str).tolist()
        else:
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

        return self.build_graph(
            chunks=chunks,
            chunk_ids=chunk_ids,
            progress_callback=progress_callback,
        )

    def load_graph(self, filepath: str) -> KnowledgeGraph:
        """
        Load a knowledge graph from file.

        Args:
            filepath: Path to the saved knowledge graph

        Returns:
            Loaded KnowledgeGraph
        """
        self.knowledge_graph = KnowledgeGraph.load(filepath)
        self._initialize_retriever()
        return self.knowledge_graph

    def save_graph(self, filepath: str) -> None:
        """
        Save the knowledge graph to file.

        Args:
            filepath: Path to save the knowledge graph
        """
        if self.knowledge_graph:
            self.knowledge_graph.save(filepath)

    def load_chunks(self, chunks: Dict[str, str]) -> None:
        """
        Load chunk store for retrieval.

        Args:
            chunks: Mapping of chunk_id -> chunk_text
        """
        self._chunk_store.update(chunks)

    def _extract_query_entities(self, query: str) -> List[Entity]:
        """Extract entities from the query for graph-based retrieval."""
        extractor = self._get_entity_extractor()
        return extractor.extract(query, chunk_id="query")

    def _get_graph_context(self, entities: List[Entity]) -> str:
        """Build context from graph entities and their relationships."""
        if not self._graph_retriever or not entities:
            return ""

        context_parts = []
        for entity in entities[:5]:  # Limit to top 5 entities
            entity_context = self._graph_retriever.get_entity_context(entity)
            context_parts.append(entity_context)

        return "\n".join(context_parts)

    def _retrieve_with_graph(
        self,
        query: str,
        query_embedding: np.ndarray,
        index,
        index_type: str,
        top_k: int,
        chunks: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[Entity]]:
        """
        Perform hybrid retrieval using both vector search and graph.

        Returns:
            Tuple of (retrieved_docs, relevant_entities)
        """
        # Step 1: Standard vector retrieval
        if index_type == "local_index":
            distances, indices = self._search_faiss_index(index, query_embedding, k=top_k * 2)
            vector_results = [
                (f"chunk_{i}", 1.0 / (1.0 + float(distances[rank])))
                for rank, i in enumerate(indices)
            ]
        elif index_type == "databricks_vector_index":
            docs = self._search_databricks_index(index, query_embedding, k=top_k * 2)
            vector_results = [(f"doc_{i}", 1.0 - i * 0.1) for i, _ in enumerate(docs)]
        elif index_type == "azure_ai_search_vector_index":
            docs = self._search_azure_ai_search_index(index, query_embedding, k=top_k * 2)
            vector_results = [(f"doc_{i}", 1.0 - i * 0.1) for i, _ in enumerate(docs)]
        elif index_type == "milvus_vector_index":
            docs = self._search_milvus_index(index, query_embedding, k=top_k * 2)
            vector_results = [(f"doc_{i}", 1.0 - i * 0.1) for i, _ in enumerate(docs)]
        else:
            vector_results = []

        # Step 2: Graph-based retrieval
        relevant_entities = []
        if self._hybrid_retriever and self.knowledge_graph:
            # Get graph-enhanced results
            hybrid_results = self._hybrid_retriever.retrieve(
                query=query,
                vector_results=vector_results,
                k=top_k,
            )

            # Get relevant entities
            graph_results = self._graph_retriever.retrieve(
                query=query,
                k=top_k,
                use_graph_expansion=True,
            )
            relevant_entities = [r["entity"] for r in graph_results]

            # Retrieve chunks based on hybrid scores
            if chunks:
                retrieved_chunk_ids = [cid for cid, _ in hybrid_results]
                retrieved_docs = []
                for cid in retrieved_chunk_ids:
                    # Try to get from chunk store or use index
                    if cid in self._chunk_store:
                        retrieved_docs.append(self._chunk_store[cid])
                    elif cid.startswith("chunk_"):
                        try:
                            idx = int(cid.replace("chunk_", ""))
                            if idx < len(chunks):
                                retrieved_docs.append(chunks[idx])
                        except (ValueError, IndexError):
                            pass

                if retrieved_docs:
                    return retrieved_docs[:top_k], relevant_entities

        # Fallback to standard vector retrieval
        if index_type == "local_index" and chunks:
            return [chunks[i] for i in indices[:top_k]], relevant_entities

        return [], relevant_entities

    def _build_enhanced_context(
        self,
        documents: List[str],
        entities: List[Entity],
    ) -> str:
        """Build context including both document chunks and graph information."""
        context_parts = []

        # Add document chunks
        doc_context = "\n\n".join([
            textwrap.shorten(doc, width=800)
            for doc in documents
        ])
        context_parts.append("Document Context:\n" + doc_context)

        # Add graph context if enabled
        if self.include_graph_context and entities:
            graph_context = self._get_graph_context(entities)
            if graph_context:
                context_parts.append("\nKnowledge Graph Context:\n" + graph_context)

        return "\n\n".join(context_parts)

    def infer(
        self,
        system_prompt: str = None,
        question: str = "",
        top_k: int = 5,
        max_tokens: int = 256,
        mode: str = "graphrag",
        index_type: str = "local_index",
        index=None,
        index_path: Optional[str] = None,
        use_graph: bool = True,
    ) -> pd.DataFrame:
        """
        Run GraphRAG inference.

        Args:
            system_prompt: System prompt for the LLM
            question: User question
            top_k: Number of documents to retrieve
            max_tokens: Maximum tokens for response
            mode: Retrieval mode ("graphrag", "standard", "hybrid", "hyde")
            index_type: Type of vector index
            index: Vector index object (for non-local indexes)
            index_path: Path to local FAISS index
            use_graph: Whether to use graph-enhanced retrieval

        Returns:
            DataFrame with retrieved_docs, question, answer, and entities
        """
        system_prompt = system_prompt or Param.get_rag_prompt()

        # Get query embedding
        query_embedding = self._get_query_embedding(question)

        # Load index and chunks for local index
        chunks = None
        if index_type == "local_index":
            if index_path is None or not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index file not found: {index_path}")

            import faiss
            index = faiss.read_index(index_path)
            chunks = self._load_chunks(index_path)

            # Update chunk store
            for i, chunk in enumerate(chunks):
                self._chunk_store[f"chunk_{i}"] = chunk

        # Perform retrieval based on mode
        relevant_entities = []

        if mode == "graphrag" and use_graph and self.knowledge_graph:
            # GraphRAG mode: hybrid vector + graph retrieval
            retrieved_docs, relevant_entities = self._retrieve_with_graph(
                query=question,
                query_embedding=query_embedding,
                index=index,
                index_type=index_type,
                top_k=top_k,
                chunks=chunks,
            )

            if not retrieved_docs and chunks:
                # Fallback to standard retrieval
                distances, indices = self._search_faiss_index(index, query_embedding, k=top_k)
                retrieved_docs = [chunks[i] for i in indices]

        elif mode == "hyde":
            # HyDE mode (from parent class)
            hypothetical_answer = self.chat_client.chat(
                system_prompt="Generate a detailed answer to the following question:",
                user_input=question,
                max_tokens=max_tokens,
            )
            hyde_embedding = self._get_query_embedding(hypothetical_answer)

            if index_type == "local_index":
                distances, indices = self._search_faiss_index(index, hyde_embedding, k=top_k)
                retrieved_docs = [chunks[i] for i in indices]
            elif index_type == "databricks_vector_index":
                retrieved_docs = self._search_databricks_index(index, hyde_embedding, k=top_k)
            elif index_type == "azure_ai_search_vector_index":
                retrieved_docs = self._search_azure_ai_search_index(index, hyde_embedding, k=top_k)
            elif index_type == "milvus_vector_index":
                retrieved_docs = self._search_milvus_index(index, hyde_embedding, k=top_k)
            else:
                retrieved_docs = []

        else:
            # Standard mode (from parent class)
            if index_type == "local_index":
                distances, indices = self._search_faiss_index(index, query_embedding, k=top_k)
                retrieved_docs = [chunks[i] for i in indices]
            elif index_type == "databricks_vector_index":
                retrieved_docs = self._search_databricks_index(index, query_embedding, k=top_k)
            elif index_type == "azure_ai_search_vector_index":
                retrieved_docs = self._search_azure_ai_search_index(index, query_embedding, k=top_k)
            elif index_type == "milvus_vector_index":
                retrieved_docs = self._search_milvus_index(index, query_embedding, k=top_k)
            else:
                retrieved_docs = []

        # Build context
        if use_graph and relevant_entities and self.include_graph_context:
            context = self._build_enhanced_context(retrieved_docs, relevant_entities)
        else:
            context = self._build_context(retrieved_docs)

        # Generate response
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        result = self.chat_client.chat(
            system_prompt=system_prompt,
            user_input=prompt,
            max_tokens=max_tokens,
        )

        print("GraphRAG Answer:\n", result)

        # Build response DataFrame
        response_data = {
            "retrieved_docs": retrieved_docs,
            "question": question,
            "answer": result,
        }

        if relevant_entities:
            response_data["entities"] = [
                {"name": e.name, "type": e.entity_type}
                for e in relevant_entities
            ]

        if self.knowledge_graph:
            response_data["graph_stats"] = {
                "num_entities": self.knowledge_graph.num_entities,
                "num_relationships": self.knowledge_graph.num_relationships,
            }

        return pd.DataFrame([response_data])

    def query_graph(
        self,
        query: str,
        k: int = 5,
    ) -> Dict[str, Any]:
        """
        Query the knowledge graph directly without LLM generation.

        Useful for debugging and understanding graph structure.

        Args:
            query: Query text
            k: Number of results

        Returns:
            Dictionary with entities and their relationships
        """
        if not self._graph_retriever:
            return {"error": "No knowledge graph loaded"}

        results = self._graph_retriever.retrieve(
            query=query,
            k=k,
            use_graph_expansion=True,
            include_context=True,
        )

        return {
            "query": query,
            "results": [
                {
                    "entity": r["entity"].to_dict(),
                    "score": r["score"],
                    "context": r.get("context", ""),
                    "chunks": r["chunks"],
                }
                for r in results
            ],
            "graph_stats": {
                "num_entities": self.knowledge_graph.num_entities,
                "num_relationships": self.knowledge_graph.num_relationships,
            } if self.knowledge_graph else None,
        }
