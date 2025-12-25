from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phoenix_ai.self_rag import SelfRAGInferencer


class FakeEmbeddingClient:
    def __init__(self):
        self.calls: List[List[str]] = []

    def generate_embedding(self, input_texts: List[str], **_: Any) -> List[List[float]]:
        self.calls.append(input_texts)
        # return deterministic embeddings
        return [[0.1, 0.2, 0.3] for _ in input_texts]


class FakeChatClient:
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.calls: List[Dict[str, Any]] = []

    def chat(
        self,
        user_input: Any,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_k: float = 1.0,
    ) -> str:
        self.calls.append(
            {
                "user_input": user_input,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
            }
        )
        return self.responses.pop(0)


@pytest.fixture
def fake_inferencer(monkeypatch: pytest.MonkeyPatch) -> SelfRAGInferencer:
    embedding = FakeEmbeddingClient()
    chat = FakeChatClient(
        responses=[
            "Draft answer from context.",
            '{"verdict": "revise", "rationale": "Add details", "final_answer": "Rewritten answer."}',
        ]
    )
    inferencer = SelfRAGInferencer(embedding, chat)

    def _stub_retrieve_documents(*_: Any, **__: Any) -> List[str]:
        return ["Doc A content", "Doc B content"]

    monkeypatch.setattr(inferencer, "_retrieve_documents", _stub_retrieve_documents)
    return inferencer


def test_self_rag_returns_dataframe(fake_inferencer: SelfRAGInferencer):
    df = fake_inferencer.infer(
        question="Test question?",
        index_type="local_index",
        index_path="unused.index",
        top_k=2,
        max_tokens=128,
    )

    assert isinstance(df, pd.DataFrame)
    row = df.iloc[0]
    assert row["draft_answer"] == "Draft answer from context."
    assert row["final_answer"] == "Rewritten answer."
    assert row["critique"]["verdict"] == "revise"


def test_self_rag_uses_default_prompts(fake_inferencer: SelfRAGInferencer):
    df = fake_inferencer.infer(
        question="Another question?",
        index_type="local_index",
        index_path="unused.index",
        top_k=1,
    )

    # First call is drafting, second is critique
    assert "Draft" in fake_inferencer.chat_client.calls[0]["user_input"]
    assert "JSON object" in fake_inferencer.chat_client.calls[1]["user_input"]
    assert isinstance(df, pd.DataFrame)
