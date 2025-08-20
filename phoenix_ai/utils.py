import os
import time
from typing import Dict, List, Union

from openai import AzureOpenAI, OpenAI


class GenAIEmbeddingClient:
    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str = None,
        api_key: str = None,
        api_version: str = None,
        azure_endpoint: str = None,
        device: str = "cpu",
    ):
        """
        Initializes the embedding client for OpenAI (public), Databricks, Azure, Ollama, or Sentence Transformers.
        """
        self.provider = provider.lower()
        self.model = model
        self.client = None
        self.api_key = api_key

        if self.provider == "databricks":
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        elif self.provider == "azure-openai":
            if not all([api_key, api_version, azure_endpoint]):
                raise ValueError(
                    "Azure requires api_key, api_version, and azure_endpoint."
                )
            self.client = AzureOpenAI(
                api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
            )
        elif self.provider == "openai":
            if not api_key:
                raise ValueError("OpenAI provider requires api_key.")
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "ollama":
            # Ollama exposes an OpenAI-compatible API at /v1 by default on localhost:11434
            self.client = OpenAI(
                api_key=api_key or "ollama",
                base_url=base_url or "http://localhost:11434/v1",
            )
        elif self.provider == "sentence-transformer":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception as import_error:  # pragma: no cover - optional dependency
                raise ImportError(
                    "Install sentence-transformers to use the 'sentence-transformer' provider: pip install sentence-transformers"
                ) from import_error

            # Initialize local sentence-transformers model (CPU or CUDA)
            self._st_model = SentenceTransformer(self.model, device=device)
        else:
            raise ValueError(
                "Provider must be 'databricks', 'azure-openai', 'openai', 'ollama', or 'sentence-transformer'."
            )

    def generate_embedding(
        self,
        input_texts: List[str],
        batch_size: int = 16,
        max_retries: int = 5,
        backoff_factor: float = 5.0,
    ) -> List[List[float]]:
        # Local provider path for Sentence Transformers
        if self.provider == "sentence-transformer":
            return self._sentence_transformer_embedding(input_texts, batch_size=batch_size)

        all_embeddings = []
        for i in range(0, len(input_texts), batch_size):
            batch = input_texts[i : i + batch_size]
            retries = 0
            while retries <= max_retries:
                try:
                    response = self.client.embeddings.create(
                        input=batch, model=self.model, encoding_format="float"
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break  # success
                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in err_str or "rate limit" in err_str:
                        wait_time = backoff_factor * (2**retries)
                        print(
                            f"[Batch {i // batch_size + 1}] Rate limited. Retrying in {wait_time:.1f}s (attempt {retries + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        retries += 1
                    else:
                        raise e
            else:
                raise RuntimeError(
                    f"Failed to get embeddings for batch after {max_retries} retries."
                )
        return all_embeddings

    def _sentence_transformer_embedding(
        self,
        input_texts: List[str],
        batch_size: int = 32,
        **_: Dict,
    ) -> List[List[float]]:
        """Generate embeddings locally using Sentence Transformers."""
        # convert_to_numpy yields a numpy array; tolist() returns List[List[float]]
        return (
            self._st_model.encode(
                input_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).tolist()
        )


class GenAIChatClient:
    def __init__(
        self,
        provider: str,
        model: str,
        system_prompt: str = "You are a helpful assistant.",
        base_url: str = None,
        api_key: str = None,
        api_version: str = None,
        azure_endpoint: str = None,
    ):
        """
        Initializes the chat client for OpenAI (public), Azure, Databricks, or Ollama.
        """
        self.provider = provider.lower()
        self.model = model
        self.system_prompt = system_prompt
        self.client = None
        self.api_key = api_key

        if self.provider == "azure-openai":
            if not all([api_key, api_version, azure_endpoint]):
                raise ValueError(
                    "Azure requires api_key, api_version, and azure_endpoint."
                )
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
            )
        elif self.provider == "databricks":
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        elif self.provider == "openai":
            if not api_key:
                raise ValueError("OpenAI provider requires api_key.")
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "ollama":
            # Ollama exposes an OpenAI-compatible API at /v1 by default on localhost:11434
            self.client = OpenAI(
                api_key=api_key or "ollama",
                base_url=base_url or "http://localhost:11434/v1",
            )
        else:
            raise ValueError(
                "Provider must be 'azure-openai', 'databricks', 'openai', or 'ollama'."
            )

    def chat(
        self,
        user_input: Union[str, List[Dict[str, str]]],
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_k: float = 1.0,
    ) -> str:
        system_prompt = system_prompt or self.system_prompt

        if isinstance(user_input, str):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
        else:
            messages = user_input

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
