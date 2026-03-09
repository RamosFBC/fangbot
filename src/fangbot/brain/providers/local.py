"""Local LLM provider — OpenAI-compatible servers (Ollama, LM Studio, vLLM)."""

from __future__ import annotations

import openai

from fangbot.brain.providers.openai import OpenAIProvider


class LocalProvider(OpenAIProvider):
    """Provider for local LLM servers exposing an OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.2",
        api_key: str = "not-needed",
    ):
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._use_responses_api = False

    def _supports_temperature(self) -> bool:
        return True
