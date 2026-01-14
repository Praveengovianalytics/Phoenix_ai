import os

import pandas as pd
from azure.identity import ClientSecretCredential, get_bearer_token_provider

from phoenix_ai.config_param import Param
from phoenix_ai.rag_inference import RAGInferencer
from phoenix_ai.utils import GenAIChatClient, GenAIEmbeddingClient
from phoenix_ai.vector_embedding_pipeline import VectorEmbedding


def main() -> None:
    credential = ClientSecretCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
    )
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )

    embedding_client = GenAIEmbeddingClient(
        provider="azure-openai",
        model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
    )
    chat_client = GenAIChatClient(
        provider="azure-openai",
        model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
    )

    df = pd.DataFrame(
        [
            {
                "title": "Azure AI Search Overview",
                "content": "Azure AI Search provides full text and vector search.",
            },
            {
                "title": "Vector Search",
                "content": "Vector search enables semantic similarity with embeddings.",
            },
            {
                "title": "RAG Architecture",
                "content": "RAG combines retrieval with LLMs for grounded answers.",
            },
        ]
    )

    vector = VectorEmbedding(embedding_client, chunk_size=500, overlap=50)
    azure_store = vector.generate_index(
        df=df,
        text_column="content",
        index_path="",
        vector_index_type="azure_ai_search_vector_index",
        search_service_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        search_api_key=os.environ["AZURE_SEARCH_API_KEY"],
        index_name=os.environ.get("AZURE_SEARCH_INDEX", "sample-text-index"),
        embedding_dim=1536,
        content_field_name="content",
        vector_field_name="contentVector",
        title_field_name="title",
        vector_search_profile_name="vector-profile",
        hnsw_algorithm_configuration_name="hnsw-config",
        update_index=True,
    )

    rag_inferencer = RAGInferencer(embedding_client, chat_client)
    result_df = rag_inferencer.infer(
        system_prompt=Param.get_rag_prompt(),
        question="What is RAG and how does Azure AI Search help?",
        top_k=3,
        mode="standard",
        index_type="azure_ai_search_vector_index",
        index=azure_store,
    )
    print(result_df[["question", "answer"]])


if __name__ == "__main__":
    main()
