
import os
import time
from typing import Union, List, Dict
from openai import OpenAI, AzureOpenAI

class GenAIEmbeddingClient:
    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str = None,
        api_key: str = None,
        api_version: str = None,
        azure_endpoint: str = None
    ):
        """
        Initializes the embedding client for OpenAI (public), Databricks, or Azure.
        """
        self.provider = provider.lower()
        self.model = model
        self.client = None
        self.api_key = api_key

        if self.provider == "databricks":
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=base_url
            )
        elif self.provider == "azure-openai":
            if not all([api_key, api_version, azure_endpoint]):
                raise ValueError("Azure requires api_key, api_version, and azure_endpoint.")
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint
            )
        elif self.provider == "openai":
            if not api_key:
                raise ValueError("OpenAI provider requires api_key.")
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError("Provider must be 'databricks', 'azure-openai', or 'openai'.")

    def generate_embedding(
        self,
        input_texts: List[str],
        batch_size: int = 16,
        max_retries: int = 5,
        backoff_factor: float = 5.0
    ) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(input_texts), batch_size):
            batch = input_texts[i:i+batch_size]
            retries = 0
            while retries <= max_retries:
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model,
                        encoding_format="float"
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break  # success
                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in err_str or "rate limit" in err_str:
                        wait_time = backoff_factor * (2 ** retries)
                        print(f"[Batch {i // batch_size + 1}] Rate limited. Retrying in {wait_time:.1f}s (attempt {retries + 1}/{max_retries})")
                        time.sleep(wait_time)
                        retries += 1
                    else:
                        raise e
            else:
                raise RuntimeError(f"Failed to get embeddings for batch after {max_retries} retries.")
        return all_embeddings


class GenAIChatClient:
    def __init__(
        self,
        provider: str,
        model: str,
        system_prompt: str = "You are a helpful assistant.",
        base_url: str = None,
        api_key: str = None,
        api_version: str = None,
        azure_endpoint: str = None
    ):
        """
        Initializes the chat client for OpenAI (public), Azure, or Databricks.
        """
        self.provider = provider.lower()
        self.model = model
        self.system_prompt = system_prompt
        self.client = None
        self.api_key = api_key

        if self.provider == "azure-openai":
            if not all([api_key, api_version, azure_endpoint]):
                raise ValueError("Azure requires api_key, api_version, and azure_endpoint.")
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
            )
        elif self.provider == "databricks":
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=base_url
            )
        elif self.provider == "openai":
            if not api_key:
                raise ValueError("OpenAI provider requires api_key.")
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError("Provider must be 'azure-openai', 'databricks', or 'openai'.")

    def chat(
        self,
        user_input: Union[str, List[Dict[str, str]]],
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_k: float = 1.0
    ) -> str:
        system_prompt = system_prompt or self.system_prompt

        if isinstance(user_input, str):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        else:
            messages = user_input

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
