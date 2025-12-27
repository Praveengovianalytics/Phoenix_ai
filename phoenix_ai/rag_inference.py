import json
import os
import pickle
import textwrap
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from openai import OpenAI

from .config_param import Param


class RAGInferencer:
    def __init__(self, embedding_client, chat_client, keyword_search_client=None):
        self.embedding_client = embedding_client  # For generating embeddings
        self.chat_client = chat_client  # For generating responses
        self.keyword_search_client = (
            keyword_search_client  # For keyword-based search (e.g., BM25)
        )

    def _get_query_embedding(self, text: str) -> np.ndarray:
        embedding = self.embedding_client.generate_embedding([text])[0]
        return np.array([embedding], dtype="float32")

    def _search_faiss_index(
        self, index, query_embedding: np.ndarray, k: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        distances, indices = index.search(query_embedding, k)
        return distances[0], indices[0]

    def _search_databricks_index(
        self, index, query_embedding: np.ndarray, k: int = 3
    ) -> List[str]:
        response = index.similarity_search(
            query_vector=query_embedding.tolist()[0],  # convert np.ndarray to list
            columns=["content"],
            num_results=k,
        )
        return [
            str(row[1]) for row in response["result"]["data_array"]
        ]  # ensure it's a string
        # return [row[1] for row in response["result"]["data_array"]]  # row = [id, content, score]

    def _search_azure_ai_search_index(
        self, index, query_embedding: np.ndarray, k: int = 3
    ) -> List[str]:
        # index is expected to be an AzureAISearchVectorStore with vector_search method
        return index.vector_search(query_embedding.tolist()[0], k)

    def _search_keyword(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        if self.keyword_search_client is None:
            return []
        return self.keyword_search_client.search(query, top_k=k)

    def _fuse_results(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]],
        alpha: float = 0.5,
    ) -> List[Tuple[str, float]]:
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

    def infer(
        self,
        system_prompt: str,
        question: str,
        top_k: int = 5,
        max_tokens: int = 256,
        mode: str = "standard",
        index_type: str = "local_index",
        index=None,
        index_path: Optional[str] = None,
    ) -> pd.DataFrame:

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
        elif index_type == "azure_ai_search_vector_index":
            if index is None:
                raise ValueError("Azure AI Search vector index must be provided")

        else:
            raise ValueError(f"Unsupported index_type: {index_type}")

        # Step 2: Perform search
        if mode == "standard":
            # Standard RAG: Semantic search using question embedding
            if index_type == "local_index":
                distances, indices = self._search_faiss_index(
                    index, query_embedding, k=top_k
                )
                retrieved_docs = [chunks[i] for i in indices]
            elif index_type == "databricks_vector_index":
                retrieved_docs = self._search_databricks_index(
                    index, query_embedding, k=top_k
                )
            elif index_type == "azure_ai_search_vector_index":
                retrieved_docs = self._search_azure_ai_search_index(
                    index, query_embedding, k=top_k
                )

        elif mode == "hybrid":
            # Hybrid RAG: Combine semantic and keyword search
            # Semantic search
            if index_type != "local_index":
                raise NotImplementedError(
                    "Hybrid mode is only supported for FAISS/local index"
                )
            distances, indices = self._search_faiss_index(
                index, query_embedding, k=top_k
            )
            semantic_results = [
                (chunks[i], float(distances[rank])) for rank, i in enumerate(indices)
            ]
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
                distances, indices = self._search_faiss_index(
                    index, hyde_embedding, k=top_k
                )
                retrieved_docs = [chunks[i] for i in indices]
            elif index_type == "databricks_vector_index":
                retrieved_docs = self._search_databricks_index(
                    index, hyde_embedding, k=top_k
                )
            elif index_type == "azure_ai_search_vector_index":
                retrieved_docs = self._search_azure_ai_search_index(
                    index, hyde_embedding, k=top_k
                )
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
        return pd.DataFrame(
            [{"retrieved_docs": retrieved_docs, "question": question, "answer": result}]
        )


class SelfRAGInferencer:
    """
    Self-RAG adds a lightweight self-critique loop on top of retrieval.
    It first drafts an answer from retrieved context, then asks the model to
    verify grounding and, if needed, rewrite the answer to better align with the evidence.
    """

    def __init__(self, embedding_client, chat_client, keyword_search_client=None):
        self.embedding_client = embedding_client
        self.chat_client = chat_client
        self.keyword_search_client = keyword_search_client

    def _get_query_embedding(self, text: str) -> np.ndarray:
        embedding = self.embedding_client.generate_embedding([text])[0]
        return np.array([embedding], dtype="float32")

    def _search_faiss_index(
        self, index, query_embedding: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        distances, indices = index.search(query_embedding, k)
        return distances[0], indices[0]

    def _search_databricks_index(
        self, index, query_embedding: np.ndarray, k: int
    ) -> List[str]:
        response = index.similarity_search(
            query_vector=query_embedding.tolist()[0], columns=["content"], num_results=k
        )
        return [str(row[1]) for row in response["result"]["data_array"]]

    def _load_chunks(self, index_path: str) -> List[str]:
        chunk_path = os.path.splitext(index_path)[0] + "_chunks.pkl"
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
        with open(chunk_path, "rb") as f:
            return pickle.load(f)

    def _retrieve_documents(
        self,
        question: str,
        top_k: int,
        index_type: str,
        index,
        index_path: Optional[str],
    ) -> List[str]:
        query_embedding = self._get_query_embedding(question)

        if index_type == "local_index":
            if index_path is None or not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index file not found: {index_path}")
            index = faiss.read_index(index_path)
            chunks = self._load_chunks(index_path)
            _, indices = self._search_faiss_index(index, query_embedding, k=top_k)
            return [chunks[i] for i in indices]

        if index_type == "databricks_vector_index":
            if index is None:
                raise ValueError("Databricks vector search index must be provided")
            return self._search_databricks_index(index, query_embedding, k=top_k)

        raise ValueError(f"Unsupported index_type: {index_type}")

    def _build_context(self, documents: List[str]) -> str:
        return "\n\n".join([textwrap.shorten(doc, width=800) for doc in documents])

    def _run_draft(
        self,
        system_prompt: str,
        question: str,
        documents: List[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        context = self._build_context(documents)
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Draft your best answer using only the context. Provide a concise, factual response."
        )
        return self.chat_client.chat(
            system_prompt=system_prompt,
            user_input=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _run_critique(
        self,
        critique_prompt: str,
        question: str,
        documents: List[str],
        draft_answer: str,
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, Dict[str, str]]:
        critique_message = (
            "You are a fact-checking assistant. Given the context and draft answer,\n"
            "decide if the answer is grounded in the provided context. Reply with JSON:\n"
            '{"verdict": "approve" | "revise", "rationale": "...", "final_answer": "..."}.\n'
            "If revise, rewrite final_answer to align with the evidence only."
        )
        context = self._build_context(documents)
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Draft Answer: {draft_answer}\n\n"
            "Return the JSON object only."
        )
        response = self.chat_client.chat(
            system_prompt=critique_prompt or critique_message,
            user_input=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        parsed: Dict[str, str] = {
            "verdict": "approve",
            "rationale": "",
            "final_answer": draft_answer,
        }
        try:
            loaded = json.loads(response)
            if isinstance(loaded, dict):
                parsed.update({k: str(v) for k, v in loaded.items() if k in parsed})
        except Exception:
            parsed["rationale"] = response

        final_answer = parsed.get("final_answer") or draft_answer
        return final_answer, parsed

    def infer(
        self,
        question: str,
        index_type: str = "local_index",
        index=None,
        index_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        critique_prompt: Optional[str] = None,
        top_k: int = 5,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> pd.DataFrame:
        system_prompt = system_prompt or Param.get_rag_prompt()
        critique_prompt = critique_prompt or Param.get_self_rag_critique_prompt()

        retrieved_docs = self._retrieve_documents(
            question=question,
            top_k=top_k,
            index_type=index_type,
            index=index,
            index_path=index_path,
        )

        draft_answer = self._run_draft(
            system_prompt=system_prompt,
            question=question,
            documents=retrieved_docs,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        final_answer, critique = self._run_critique(
            critique_prompt=critique_prompt,
            question=question,
            documents=retrieved_docs,
            draft_answer=draft_answer,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return pd.DataFrame(
            [
                {
                    "question": question,
                    "retrieved_docs": retrieved_docs,
                    "draft_answer": draft_answer,
                    "critique": critique,
                    "final_answer": final_answer,
                }
            ]
        )
