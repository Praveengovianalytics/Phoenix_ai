"""
Azure AI Search vector index helper with MSI (DefaultAzureCredential) support.

This module keeps the dependency optional at import time; if the Azure SDK
packages are missing, import will raise a clear error telling the user what to install.
"""

from __future__ import annotations

from typing import Iterable, List, Optional


class AzureAISearchVectorStore:
    """
    Minimal helper to create/update a vector index and run vector searches
    against Azure AI Search using the HNSW algorithm.
    """

    def __init__(
        self,
        search_service_endpoint: str,
        index_name: str,
        embedding_dim: int,
        credential: Optional[object] = None,
    ) -> None:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.search.documents.indexes import SearchIndexClient
            from azure.search.documents.indexes.models import (
                HnswAlgorithmConfiguration, SearchField, SearchFieldDataType,
                SearchIndex, SimpleField, VectorSearch,
                VectorSearchAlgorithmConfiguration, VectorSearchProfile)
        except Exception as import_error:  # pragma: no cover - optional dep
            raise ImportError(
                "Install azure-identity and azure-search-documents to use Azure AI Search: "
                "pip install azure-identity azure-search-documents"
            ) from import_error

        self._SearchIndexClient = SearchIndexClient
        self._SearchIndex = SearchIndex
        self._SimpleField = SimpleField
        self._SearchField = SearchField
        self._SearchFieldDataType = SearchFieldDataType
        self._HnswAlgorithmConfiguration = HnswAlgorithmConfiguration
        self._VectorSearch = VectorSearch
        self._VectorSearchAlgorithmConfiguration = VectorSearchAlgorithmConfiguration
        self._VectorSearchProfile = VectorSearchProfile

        self.endpoint = search_service_endpoint.rstrip("/")
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self.credential = credential or DefaultAzureCredential()

        self._index_client = self._SearchIndexClient(
            endpoint=self.endpoint, credential=self.credential
        )
        self._ensure_index()

        try:
            from azure.search.documents import SearchClient
        except Exception as import_error:  # pragma: no cover - optional dep
            raise ImportError(
                "Install azure-search-documents to use Azure AI Search: "
                "pip install azure-search-documents"
            ) from import_error

        self._search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential,
        )

    def _ensure_index(self) -> None:
        fields = [
            self._SimpleField(
                name="id", type=self._SearchFieldDataType.String, key=True
            ),
            self._SearchField(
                name="content",
                type=self._SearchFieldDataType.String,
                searchable=True,
                filterable=False,
                facetable=False,
                sortable=False,
            ),
            self._SearchField(
                name="embedding",
                type=self._SearchFieldDataType.Collection(
                    self._SearchFieldDataType.Single
                ),
                searchable=True,
                vector_search_dimensions=self.embedding_dim,
                vector_search_profile_name="default-hnsw",
            ),
        ]

        vector_search = self._VectorSearch(
            algorithms=[
                self._VectorSearchAlgorithmConfiguration(
                    name="hnsw-config",
                    kind="hnsw",
                )
            ],
            profiles=[
                self._VectorSearchProfile(
                    name="default-hnsw",
                    algorithm_configuration_name="hnsw-config",
                )
            ],
        )

        index_def = self._SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
        )

        existing = {idx.name for idx in self._index_client.list_indexes()}
        if self.index_name not in existing:
            self._index_client.create_index(index_def)

    def upsert_documents(self, documents: Iterable[dict]) -> None:
        """Upsert documents with precomputed embeddings."""
        results = self._search_client.merge_or_upload_documents(
            documents=list(documents)
        )
        failed = [r for r in results if not r.succeeded]
        if failed:
            raise RuntimeError(f"Azure AI Search upsert failures: {failed}")

    def vector_search(self, query_vector: List[float], k: int) -> List[str]:
        """Return top-k content strings for the given query vector."""
        results = self._search_client.search(
            search_text="",
            vector={
                "value": query_vector,
                "fields": "embedding",
                "k": k,
                "kind": "vector",
                "exhaustive": False,
            },
            select=["content"],
        )
        return [hit["content"] for hit in results]
