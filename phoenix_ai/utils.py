import os
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

    def generate_embedding(self, input_text: List[str]) -> List[float]:
        response = self.client.embeddings.create(
            input=input_text,
            model=self.model,
            encoding_format="float"
        )
        return [item.embedding for item in response.data]


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