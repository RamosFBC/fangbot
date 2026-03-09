"""OpenAI GPT provider."""

from __future__ import annotations

import json
import re
import uuid

import openai

from fangbot.brain.providers.base import LLMProvider
from fangbot.models import (
    Message,
    ProviderResponse,
    Role,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider with tool-use support."""

    # Models that don't support custom temperature (reasoning + newer GPT-5 variants)
    _NO_TEMPERATURE_PREFIXES = ("o1", "o3", "o4", "gpt-5")

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        self._use_completions = False

    @property
    def model_name(self) -> str:
        return self._model

    def _supports_temperature(self) -> bool:
        return not any(self._model.startswith(p) for p in self._NO_TEMPERATURE_PREFIXES)

    async def call(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> ProviderResponse:
        if self._use_completions:
            return await self._call_completions(messages, tools, system)

        oai_messages = self._format_messages(messages, system)

        kwargs: dict = {
            "model": self._model,
            "messages": oai_messages,
        }
        if self._supports_temperature():
            kwargs["temperature"] = 0.0
        if tools:
            kwargs["tools"] = [self._format_tool(t) for t in tools]

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except openai.NotFoundError as e:
            if "not a chat model" in str(e):
                self._use_completions = True
                return await self._call_completions(messages, tools, system)
            raise

        return self._parse_response(response)

    def format_tool_result(self, result: ToolResult) -> Message:
        return Message(
            role=Role.TOOL,
            content=result.content,
            tool_call_id=result.tool_call_id,
        )

    def _format_messages(self, messages: list[Message], system: str | None) -> list[dict]:
        """Convert normalized messages to OpenAI chat format."""
        oai_messages: list[dict] = []

        if system:
            oai_messages.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == Role.SYSTEM:
                oai_messages.append({"role": "system", "content": msg.content})
            elif msg.tool_calls:
                # Assistant message with tool calls
                oai_msg: dict = {"role": "assistant"}
                if msg.content:
                    oai_msg["content"] = msg.content
                else:
                    oai_msg["content"] = None
                oai_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                oai_messages.append(oai_msg)
            elif msg.tool_call_id:
                # Tool result message
                oai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
            else:
                oai_messages.append(
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                    }
                )

        return oai_messages

    def _format_tool(self, tool: ToolDefinition) -> dict:
        """Convert a ToolDefinition to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    def _parse_response(self, response: openai.types.chat.ChatCompletion) -> ProviderResponse:
        """Parse OpenAI response into normalized ProviderResponse."""
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return ProviderResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "",
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            model=response.model,
        )

    # ------------------------------------------------------------------
    # Completions API fallback for non-chat models
    # ------------------------------------------------------------------

    async def _call_completions(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> ProviderResponse:
        """Fall back to /v1/completions for models that don't support chat."""
        prompt = self._format_prompt(messages, tools, system)

        kwargs: dict = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": 4096,
        }
        if self._supports_temperature():
            kwargs["temperature"] = 0.0

        response = await self._client.completions.create(**kwargs)
        return self._parse_completions_response(response, tools)

    def _format_prompt(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> str:
        """Convert messages and tools into a single prompt string."""
        parts: list[str] = []

        if system:
            parts.append(f"System: {system}")

        if tools:
            tool_descriptions = []
            for t in tools:
                schema = json.dumps(t.input_schema, indent=2)
                tool_descriptions.append(f"- {t.name}: {t.description}\n  Parameters: {schema}")
            parts.append(
                "Available tools:\n" + "\n".join(tool_descriptions) + "\n\n"
                "To call a tool, respond with a JSON block:\n"
                "```json\n"
                '{"tool_calls": [{"name": "tool_name", "arguments": {...}}]}\n'
                "```\n"
                "You may include reasoning text before the JSON block."
            )

        for msg in messages:
            if msg.role == Role.SYSTEM:
                parts.append(f"System: {msg.content}")
            elif msg.role == Role.USER:
                parts.append(f"User: {msg.content}")
            elif msg.role == Role.ASSISTANT:
                parts.append(f"Assistant: {msg.content}")
            elif msg.role == Role.TOOL:
                parts.append(f"Tool result ({msg.tool_call_id}): {msg.content}")

        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _parse_completions_response(
        self,
        response: openai.types.Completion,
        tools: list[ToolDefinition] | None = None,
    ) -> ProviderResponse:
        """Parse a completions response, extracting any tool calls from JSON blocks."""
        choice = response.choices[0]
        text = choice.text or ""

        tool_calls: list[ToolCall] = []
        content = text

        if tools:
            tool_calls, content = self._extract_tool_calls(text)

        return ProviderResponse(
            content=content.strip(),
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "",
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            model=response.model,
        )

    @staticmethod
    def _extract_tool_calls(text: str) -> tuple[list[ToolCall], str]:
        """Extract tool calls from JSON code blocks in completion text.

        Returns (tool_calls, remaining_text).
        """
        pattern = r"```json\s*(\{.*?\})\s*```"
        matches = re.findall(pattern, text, re.DOTALL)

        tool_calls: list[ToolCall] = []
        for match in matches:
            try:
                parsed = json.loads(match)
                calls = parsed.get("tool_calls", [])
                for tc in calls:
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:12]}",
                            name=tc["name"],
                            arguments=tc.get("arguments", {}),
                        )
                    )
            except (json.JSONDecodeError, KeyError):
                continue

        # Remove JSON blocks from content to get the reasoning text
        content = re.sub(r"```json\s*\{.*?\}\s*```", "", text, flags=re.DOTALL)
        return tool_calls, content
