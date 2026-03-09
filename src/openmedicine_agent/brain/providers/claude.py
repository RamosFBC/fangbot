"""Anthropic Claude LLM provider."""

from __future__ import annotations

import anthropic

from openmedicine_agent.brain.providers.base import LLMProvider
from openmedicine_agent.models import (
    Message,
    ProviderResponse,
    Role,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider with tool-use support."""

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def call(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> ProviderResponse:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "temperature": 0.0,
            "messages": self._format_messages(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [self._format_tool(t) for t in tools]

        response = await self._client.messages.create(**kwargs)
        return self._parse_response(response)

    def format_tool_result(self, result: ToolResult) -> Message:
        return Message(
            role=Role.USER,
            content=result.content,
            tool_call_id=result.tool_call_id,
        )

    def _format_messages(self, messages: list[Message]) -> list[dict]:
        """Convert normalized messages to Anthropic format."""
        anthropic_messages: list[dict] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue

            if msg.tool_calls:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                anthropic_messages.append({"role": "assistant", "content": content})
            elif msg.tool_call_id:
                # Tool result — Anthropic expects these in a user message with tool_result blocks
                # Check if last message is already a user message with tool results
                if (
                    anthropic_messages
                    and anthropic_messages[-1]["role"] == "user"
                    and isinstance(anthropic_messages[-1]["content"], list)
                    and any(
                        b.get("type") == "tool_result"
                        for b in anthropic_messages[-1]["content"]
                    )
                ):
                    anthropic_messages[-1]["content"].append({
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    })
                else:
                    anthropic_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }],
                    })
            else:
                anthropic_messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        return anthropic_messages

    def _format_tool(self, tool: ToolDefinition) -> dict:
        """Convert a ToolDefinition to Anthropic tool format."""
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }

    def _parse_response(self, response: anthropic.types.Message) -> ProviderResponse:
        """Parse Anthropic response into normalized ProviderResponse."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return ProviderResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            model=response.model,
        )
