from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from phoenix_ai.utils import GenAIChatClient, GenAIEmbeddingClient


class _DummyChoice:
    def __init__(self, content: str):
        self.message = type("Msg", (), {"content": content})


class _DummyChatResponse:
    def __init__(self, content: str):
        self.choices = [_DummyChoice(content)]


class _DummyChatCompletions:
    def __init__(self, holder: Dict[str, Any]):
        self.holder = holder

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        **_: Any,
    ) -> _DummyChatResponse:
        self.holder["chat_args"] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return _DummyChatResponse("stubbed response")


class _DummyEmbeddings:
    def __init__(self, holder: Dict[str, Any]):
        self.holder = holder

    def create(self, input: List[str], model: str, encoding_format: str = "float"):
        self.holder["embedding_args"] = {
            "input": input,
            "model": model,
            "encoding_format": encoding_format,
        }
        return type(
            "Resp",
            (),
            {"data": [type("Item", (), {"embedding": [0.1, 0.2, 0.3]}) for _ in input]},
        )


class _DummyClient:
    def __init__(self, holder: Dict[str, Any], base_url: str, api_key: str):
        self.holder = holder
        self.base_url = base_url
        self.api_key = api_key
        self.chat = type(
            "Chat", (), {"completions": _DummyChatCompletions(self.holder)}
        )
        self.embeddings = _DummyEmbeddings(self.holder)


def _make_openai_stub(holder: Dict[str, Any]):
    def _factory(api_key: str | None = None, base_url: str | None = None, **_: Any):
        holder["init"] = {"api_key": api_key, "base_url": base_url}
        return _DummyClient(holder, base_url=base_url, api_key=api_key)  # type: ignore[arg-type]

    return _factory


def test_huggingface_chat_uses_router_and_formats_messages(
    monkeypatch: pytest.MonkeyPatch,
):
    holder: Dict[str, Any] = {}
    monkeypatch.setattr("phoenix_ai.utils.OpenAI", _make_openai_stub(holder))

    client = GenAIChatClient(
        provider="huggingface",
        model="aisingapore/Qwen-SEA-LION-v4-32B-IT:featherless-ai",
        api_key="hf_token",
    )
    response = client.chat("What is the capital of France?", max_tokens=32)

    assert response == "stubbed response"
    assert holder["init"]["base_url"] == "https://router.huggingface.co/v1"
    assert (
        holder["chat_args"]["model"]
        == "aisingapore/Qwen-SEA-LION-v4-32B-IT:featherless-ai"
    )
    assert holder["chat_args"]["messages"][0]["role"] == "system"
    assert (
        holder["chat_args"]["messages"][1]["content"]
        == "What is the capital of France?"
    )


def test_huggingface_chat_custom_base_url(monkeypatch: pytest.MonkeyPatch):
    holder: Dict[str, Any] = {}
    monkeypatch.setattr("phoenix_ai.utils.OpenAI", _make_openai_stub(holder))

    client = GenAIChatClient(
        provider="huggingface",
        model="custom-model",
        api_key="hf_token",
        base_url="https://example/v1",
    )
    _ = client.chat("ping")

    assert holder["init"]["base_url"] == "https://example/v1"


def test_huggingface_embedding_uses_router(monkeypatch: pytest.MonkeyPatch):
    holder: Dict[str, Any] = {}
    monkeypatch.setattr("phoenix_ai.utils.OpenAI", _make_openai_stub(holder))

    client = GenAIEmbeddingClient(
        provider="huggingface", model="text-embedding-model", api_key="hf_token"
    )
    vectors = client.generate_embedding(["hello world"])

    assert holder["init"]["base_url"] == "https://router.huggingface.co/v1"
    assert holder["embedding_args"]["model"] == "text-embedding-model"
    assert vectors == [[0.1, 0.2, 0.3]]


def test_huggingface_requires_api_key_chat():
    with pytest.raises(ValueError):
        GenAIChatClient(provider="huggingface", model="any-model", api_key=None)


def test_huggingface_requires_api_key_embedding():
    with pytest.raises(ValueError):
        GenAIEmbeddingClient(provider="huggingface", model="any-model", api_key=None)
