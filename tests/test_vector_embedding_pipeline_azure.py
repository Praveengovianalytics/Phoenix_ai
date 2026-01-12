import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phoenix_ai.vector_embedding_pipeline import VectorEmbedding


class DummyEmbeddingClient:
    def __init__(self, model="dummy-embedding"):
        self.model = model

    def generate_embedding(self, texts):
        return [[float(i), float(i) + 0.1] for i, _ in enumerate(texts)]


def test_generate_index_azure_ai_search_maps_fields(monkeypatch):
    captured = {}

    class DummyAzureStore:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def upsert_documents(self, documents):
            captured["documents"] = documents

    monkeypatch.setattr(
        "phoenix_ai.vector_embedding_pipeline.AzureAISearchVectorStore",
        DummyAzureStore,
    )

    df = pd.DataFrame(
        {
            "content": ["alpha", "beta"],
            "title": ["Doc A", "Doc B"],
            "category": ["one", "two"],
        }
    )
    embedding_client = DummyEmbeddingClient()
    vector = VectorEmbedding(embedding_client, chunk_size=10, overlap=0)

    store = vector.generate_index(
        df=df,
        text_column="content",
        index_path="",
        vector_index_type="azure_ai_search_vector_index",
        search_service_endpoint="https://example.search.windows.net",
        index_name="sample-index",
        embedding_dim=2,
        search_api_key="api-key",
        content_field_name="content",
        vector_field_name="contentVector",
        title_field_name="title",
        metadata_fields=["category"],
        vector_search_profile_name="vector-profile",
        hnsw_algorithm_configuration_name="hnsw-config",
        hnsw_metric="cosine",
        hnsw_m=4,
        hnsw_ef_construction=200,
        hnsw_ef_search=300,
        update_index=True,
    )

    assert isinstance(store, DummyAzureStore)
    assert captured["init_kwargs"]["search_service_endpoint"] == (
        "https://example.search.windows.net"
    )
    assert captured["init_kwargs"]["vector_field_name"] == "contentVector"
    assert captured["init_kwargs"]["search_api_key"] == "api-key"
    assert captured["init_kwargs"]["update_index"] is True

    documents = captured["documents"]
    assert documents == [
        {
            "id": "0",
            "content": "alpha",
            "contentVector": [0.0, 0.1],
            "title": "Doc A",
            "category": "one",
        },
        {
            "id": "1",
            "content": "beta",
            "contentVector": [1.0, 1.1],
            "title": "Doc B",
            "category": "two",
        },
    ]
