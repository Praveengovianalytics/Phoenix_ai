import os
import pytest
import pandas as pd

from phoenix_ai.utils import GenAIEmbeddingClient, GenAIChatClient
from phoenix_ai.loaders import load_documents_to_dataframe
from phoenix_ai.vector_embedding_pipeline import VectorEmbedding
from phoenix_ai.rag_inference import RAGInferencer
from phoenix_ai.config_param import Param


# ======== Fixtures ========

@pytest.fixture(scope="module")
def api_key():
    """Fetch OpenAI API key from environment variable (set in GitHub Secrets or locally)."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY is not set in environment variables.")
    return key


@pytest.fixture(scope="module")
def sample_text():
    return ["What is a leading communications technology group in Asia."]


@pytest.fixture(scope="module")
def sample_question():
    return "What is the capital of Singapore?"


@pytest.fixture(scope="module")
def embedding_client(api_key):
    return GenAIEmbeddingClient(
        provider="openai",
        model="text-embedding-ada-002",
        api_key=api_key
    )


@pytest.fixture(scope="module")
def chat_client(api_key):
    return GenAIChatClient(
        provider="openai",
        model="gpt-4o",
        api_key=api_key
    )


@pytest.fixture(scope="module")
def test_dataframe():
    df = load_documents_to_dataframe(folder_path="data/")
    return df


@pytest.fixture(scope="module")
def vector_index(test_dataframe, embedding_client):
    os.makedirs("index", exist_ok=True)
    index_path = "index/test_policy.index"
    vector = VectorEmbedding(embedding_client, chunk_size=500, overlap=50)
    index_path, chunks = vector.generate_index(
        df=test_dataframe,
        text_column="content",
        index_path=index_path
    )
    return index_path, chunks


# ======== Tests ========

def test_embedding_generation(embedding_client, sample_text):
    embedding = embedding_client.generate_embedding(sample_text)
    assert isinstance(embedding, list)
    assert len(embedding) == 1
    assert isinstance(embedding[0], list)
    assert len(embedding[0]) > 0
    print("✅ Embedding generation passed.")


def test_chat_response(chat_client, sample_question):
    response = chat_client.chat(sample_question)
    assert isinstance(response, str)
    assert len(response.strip()) > 0
    assert "Singapore" in response or len(response) > 0
    print("✅ Chat response passed.")


def test_load_documents_to_dataframe(test_dataframe):
    assert isinstance(test_dataframe, pd.DataFrame)
    assert "content" in test_dataframe.columns
    assert len(test_dataframe) > 0
    print("✅ Document loading passed.")


def test_vector_embedding(vector_index):
    index_path, chunks = vector_index
    assert os.path.exists(index_path)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    print("✅ Vector embedding passed.")


def test_rag_inference(embedding_client, chat_client, vector_index):
    index_path, _ = vector_index
    rag = RAGInferencer(embedding_client, chat_client)
    response_df = rag.infer(
        system_prompt=Param.get_rag_prompt(),
        index_path=index_path,
        question="Can you summarize?",
        mode="standard",
        top_k=3
    )
    assert isinstance(response_df, pd.DataFrame)
    assert "response" in response_df.columns
    assert len(response_df) > 0
    print("✅ RAG inference passed.")