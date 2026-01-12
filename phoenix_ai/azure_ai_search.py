"""
Azure AI Search vector index helper with MSI (DefaultAzureCredential) or API key support.

This module keeps the dependency optional at import time; if the Azure SDK
packages are missing, import will raise a clear error telling the user what to install.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence


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
        search_api_key: Optional[str] = None,
        id_field_name: str = "id",
        content_field_name: str = "content",
        vector_field_name: str = "embedding",
        title_field_name: Optional[str] = None,
        metadata_fields: Optional[Sequence[str]] = None,
        vector_search_profile_name: str = "default-hnsw",
        hnsw_algorithm_configuration_name: str = "hnsw-config",
        hnsw_metric: str = "cosine",
        hnsw_m: int = 4,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 300,
        update_index: bool = False,
    ) -> None:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.search.documents.indexes import SearchIndexClient
            from azure.search.documents.indexes.models import (
                HnswAlgorithmConfiguration,
                HnswParameters,
                SearchField,
                SearchFieldDataType,
                SearchIndex,
                SearchableField,
                SimpleField,
                VectorSearch,
                VectorSearchProfile,
            )
            from azure.core.credentials import AzureKeyCredential
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
        self._HnswParameters = HnswParameters
        self._VectorSearch = VectorSearch
        self._VectorSearchProfile = VectorSearchProfile
        self._SearchableField = SearchableField
        self._AzureKeyCredential = AzureKeyCredential

        self.endpoint = search_service_endpoint.rstrip("/")
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self.id_field_name = id_field_name
        self.content_field_name = content_field_name
        self.vector_field_name = vector_field_name
        self.title_field_name = title_field_name
        self.metadata_fields = list(metadata_fields) if metadata_fields else []
        self.vector_search_profile_name = vector_search_profile_name
        self.hnsw_algorithm_configuration_name = hnsw_algorithm_configuration_name
        self.hnsw_metric = hnsw_metric
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.update_index = update_index

        if search_api_key:
            self.credential = self._AzureKeyCredential(search_api_key)
        else:
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
                name=self.id_field_name,
                type=self._SearchFieldDataType.String,
                key=True,
            ),
        ]

        if self.title_field_name:
            fields.append(
                self._SearchableField(
                    name=self.title_field_name,
                    type=self._SearchFieldDataType.String,
                )
            )

        fields.append(
            self._SearchableField(
                name=self.content_field_name,
                type=self._SearchFieldDataType.String,
            )
        )

        for field_name in self.metadata_fields:
            fields.append(
                self._SearchField(
                    name=field_name,
                    type=self._SearchFieldDataType.String,
                    searchable=False,
                    filterable=True,
                    facetable=True,
                    sortable=True,
                )
            )

        fields.append(
            self._SearchField(
                name=self.vector_field_name,
                type=self._SearchFieldDataType.Collection(
                    self._SearchFieldDataType.Single
                ),
                searchable=True,
                vector_search_dimensions=self.embedding_dim,
                vector_search_profile_name=self.vector_search_profile_name,
            )
        )

        vector_search = self._VectorSearch(
            algorithms=[
                self._HnswAlgorithmConfiguration(
                    name=self.hnsw_algorithm_configuration_name,
                    parameters=self._HnswParameters(
                        metric=self.hnsw_metric,
                        m=self.hnsw_m,
                        ef_construction=self.hnsw_ef_construction,
                        ef_search=self.hnsw_ef_search,
                    ),
                )
            ],
            profiles=[
                self._VectorSearchProfile(
                    name=self.vector_search_profile_name,
                    algorithm_configuration_name=self.hnsw_algorithm_configuration_name,
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
        elif self.update_index:
            self._index_client.create_or_update_index(index_def)

    def upsert_documents(self, documents: Iterable[dict]) -> None:
        """Upsert documents with precomputed embeddings."""
        results = self._search_client.merge_or_upload_documents(
            documents=list(documents)
        )
        failed = [r for r in results if not r.succeeded]
        if failed:
            raise RuntimeError(f"Azure AI Search upsert failures: {failed}")

    def vector_search(
        self,
        query_vector: List[float],
        k: int,
        select_fields: Optional[Sequence[str]] = None,
        return_documents: bool = False,
    ) -> List[str] | List[dict]:
        """Return top-k content strings (or documents when requested) for the vector."""
        select_fields = (
            list(select_fields) if select_fields else [self.content_field_name]
        )
        results = self._search_client.search(
            search_text="",
            vector={
                "value": query_vector,
                "fields": self.vector_field_name,
                "k": k,
                "kind": "vector",
                "exhaustive": False,
            },
            select=select_fields,
        )
        if return_documents or len(select_fields) > 1:
            return [dict(hit) for hit in results]
        return [hit[select_fields[0]] for hit in results]
