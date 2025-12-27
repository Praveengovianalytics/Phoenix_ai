from __future__ import annotations

from typing import Dict, List, Optional


class HuggingFaceTextGenerationClient:
    """
    Local/hosted Hugging Face inference via transformers.pipeline.
    Supports GPU/CPU selection with the `device` argument.
    """

    def __init__(
        self,
        model: str,
        device: str | int = "cpu",
        trust_remote_code: bool = True,
        **pipeline_kwargs,
    ) -> None:
        try:  # pragma: no cover - optional dependency
            from transformers import AutoTokenizer, pipeline
        except Exception as import_error:
            raise ImportError(
                "Install transformers to use local Hugging Face models: pip install transformers"
            ) from import_error

        self.model = model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=trust_remote_code
        )

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        model_kwargs = pipeline_kwargs.pop("model_kwargs", {})
        model_kwargs.setdefault("trust_remote_code", trust_remote_code)

        self._generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=self._tokenizer,
            device=device,
            model_kwargs=model_kwargs,
            **pipeline_kwargs,
        )

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []
        for message in messages:
            role = message.get("role", "user").capitalize()
            content = message.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
    ) -> str:
        prompt = self._format_prompt(messages)
        outputs = self._generator(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=int(top_k) if top_k else None,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        generated_text = outputs[0]["generated_text"]
        if generated_text.startswith(prompt):
            return generated_text[len(prompt) :].strip()
        return generated_text.strip()
