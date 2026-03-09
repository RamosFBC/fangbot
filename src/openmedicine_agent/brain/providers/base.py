"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from openmedicine_agent.models import Message, ProviderResponse, ToolDefinition, ToolResult


class LLMProvider(ABC):
    """Base class that all LLM providers must implement."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""
        ...

    @abstractmethod
    async def call(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> ProviderResponse:
        """Send messages to the LLM and return a normalized response."""
        ...

    @abstractmethod
    def format_tool_result(self, result: ToolResult) -> Message:
        """Convert a ToolResult into a Message the provider can consume."""
        ...
