import os
import numpy as np
import faiss
import textwrap
import pickle
import pandas as pd
from openai import OpenAI
from typing import List, Tuple, Optional

class RAGInferencer:
    def __init__(self, embedding_client, chat_client, keyword_search_client=None):
        self.embedding_client = embedding_client  # For generating embeddings
        self.chat_client = chat_client            # For generating responses
        self.keyword_search_client = keyword_search_client  # For keyword-based search (e.g., BM25)

    def _get_query_embedding(self, text: str) -> np.ndarray:
        embedding = self.embedding_client.generate_embedding([text])[0]
        return np.array([embedding], dtype="float32")

    def _search_faiss_index(self, index, query_embedding: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        distances, indices = index.search(query_embedding, k)
        return distances[0], indices[0]

    def _search_databricks_index(self, index, query_embedding: np.ndarray, k: int = 3) -> List[str]:
        response = index.similarity_search(
            query_vector=query_embedding.tolist()[0],  # convert np.ndarray to list
            columns=["content"],
            num_results=k
        )
        return [str(row[1]) for row in response["result"]["data_array"]]  # ensure it's a string
        # return [row[1] for row in response["result"]["data_array"]]  # row = [id, content, score]

    def _search_keyword(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        if self.keyword_search_client is None:
            return []
        return self.keyword_search_client.search(query, top_k=k)

    def _fuse_results(self, semantic_results: List[Tuple[str, float]], keyword_results: List[Tuple[str, float]], alpha: float = 0.5) -> List[Tuple[str, float]]:
        # Combine semantic and keyword results using weighted scoring
        combined = {}
        for doc, score in semantic_results:
            combined[doc] = alpha * score
        for doc, score in keyword_results:
            combined[doc] = combined.get(doc, 0) + (1 - alpha) * score
        # Sort combined results by score
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)
    
    def _build_context(self, documents: List[str]) -> str:
        return "\n\n".join([textwrap.shorten(doc, width=800) for doc in documents])

    def _load_chunks(self, index_path: str) -> List[str]:
        chunk_path = os.path.splitext(index_path)[0] + "_chunks.pkl"
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
        with open(chunk_path, "rb") as f:
            return pickle.load(f)

    def infer(self, system_prompt: str, question: str, top_k: int = 5, max_tokens: int = 256, mode: str = "standard", index_type: str = "local_index", index=None, index_path: Optional[str] = None) -> pd.DataFrame:

        query_embedding = self._get_query_embedding(question)

        # Step 1: Retrieve documents
        if index_type == "local_index":
            if index_path is None or not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index file not found: {index_path}")

            # Load FAISS index and chunks
            index = faiss.read_index(index_path)
            chunks = self._load_chunks(index_path)

        elif index_type == "databricks_vector_index":
            if index is None:
                raise ValueError("Databricks vector search index must be provided")

        else:
            raise ValueError(f"Unsupported index_type: {index_type}")

        # Step 2: Perform search
        if mode == "standard":
            # Standard RAG: Semantic search using question embedding
            if index_type == "local_index":
                distances, indices = self._search_faiss_index(index, query_embedding, k=top_k)
                retrieved_docs = [chunks[i] for i in indices]
            elif index_type == "databricks_vector_index":
                retrieved_docs = self._search_databricks_index(index, query_embedding, k=top_k)

        elif mode == "hybrid":
            # Hybrid RAG: Combine semantic and keyword search
            # Semantic search
            if index_type != "local_index":
                raise NotImplementedError("Hybrid mode is only supported for FAISS/local index")
            distances, indices = self._search_faiss_index(index, query_embedding, k=top_k)
            semantic_results = [(chunks[i], float(distances[rank])) for rank, i in enumerate(indices)]
            # Keyword search
            keyword_results = self._search_keyword(question, k=top_k)
            # Fuse results
            fused_results = self._fuse_results(semantic_results, keyword_results)
            retrieved_docs = [doc for doc, _ in fused_results[:top_k]]

        elif mode == "hyde":
            # HyDE RAG: Generate hypothetical answer and use its embedding for search
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
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Step 2: Build prompt and run generation
        context = self._build_context(retrieved_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        result = self.chat_client.chat(
            system_prompt=system_prompt,
            user_input=prompt,
            max_tokens=max_tokens,
        )

        print("RAG Answer:\n", result)
        return pd.DataFrame([{
            "retrieved_docs": retrieved_docs,
            "question": question,
            "answer": result
        }])
