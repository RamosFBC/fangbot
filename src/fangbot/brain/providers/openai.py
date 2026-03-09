"""OpenAI GPT provider."""

from __future__ import annotations

import json

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
    """OpenAI GPT provider with tool-use support.

    Tries the Chat Completions API first. If the model returns a 404
    ("not a chat model" / "not supported"), falls back automatically
    to the Responses API for the rest of the session.
    """

    # Models that don't support custom temperature (reasoning + newer GPT-5 variants)
    _NO_TEMPERATURE_PREFIXES = ("o1", "o3", "o4", "gpt-5")

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        self._use_responses_api = False

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
        if self._use_responses_api:
            return await self._call_responses(messages, tools, system)

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
            if "not a chat model" in str(e) or "not supported" in str(e):
                self._use_responses_api = True
                return await self._call_responses(messages, tools, system)
            raise

        return self._parse_response(response)

    def format_tool_result(self, result: ToolResult) -> Message:
        return Message(
            role=Role.TOOL,
            content=result.content,
            tool_call_id=result.tool_call_id,
        )

    # ------------------------------------------------------------------
    # Chat Completions formatting
    # ------------------------------------------------------------------

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
    # Responses API fallback for models that don't support chat
    # ------------------------------------------------------------------

    async def _call_responses(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> ProviderResponse:
        """Use the Responses API (/v1/responses) for Responses-only models."""
        input_items = self._format_responses_input(messages)

        kwargs: dict = {
            "model": self._model,
            "input": input_items,
        }
        if system:
            kwargs["instructions"] = system
        if self._supports_temperature():
            kwargs["temperature"] = 0.0
        if tools:
            kwargs["tools"] = [self._format_responses_tool(t) for t in tools]

        response = await self._client.responses.create(**kwargs)
        return self._parse_responses_response(response)

    def _format_responses_input(self, messages: list[Message]) -> list[dict]:
        """Convert normalized messages to Responses API input format."""
        input_items: list[dict] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # System messages become instructions; skip here (handled via kwarg)
                continue
            elif msg.tool_calls:
                # Assistant text + function_call items
                if msg.content:
                    input_items.append({"role": "assistant", "content": msg.content})
                for tc in msg.tool_calls:
                    input_items.append(
                        {
                            "type": "function_call",
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                            "call_id": tc.id,
                        }
                    )
            elif msg.tool_call_id:
                # Tool result → function_call_output
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id,
                        "output": msg.content,
                    }
                )
            else:
                input_items.append(
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                    }
                )

        return input_items

    @staticmethod
    def _format_responses_tool(tool: ToolDefinition) -> dict:
        """Convert a ToolDefinition to Responses API tool format."""
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }

    @staticmethod
    def _parse_responses_response(response) -> ProviderResponse:
        """Parse a Responses API response into normalized ProviderResponse."""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if hasattr(part, "text"):
                        content_parts.append(part.text)
            elif item.type == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=item.call_id,
                        name=item.name,
                        arguments=json.loads(item.arguments),
                    )
                )

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
            }

        return ProviderResponse(
            content="\n".join(content_parts),
            tool_calls=tool_calls,
            stop_reason=response.status if hasattr(response, "status") else "",
            usage=usage,
            model=response.model if hasattr(response, "model") else "",
        )
