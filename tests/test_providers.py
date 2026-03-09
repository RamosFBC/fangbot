"""Tests for LLM providers."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from fangbot.models import (
    Message,
    Role,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class TestOpenAIProvider:
    """Test OpenAI provider message formatting and response parsing."""

    def _make_provider(self):
        from fangbot.brain.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        return provider

    def test_format_messages_with_system(self):
        provider = self._make_provider()
        messages = [Message(role=Role.USER, content="Hello")]
        result = provider._format_messages(messages, system="You are helpful.")

        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_format_messages_with_tool_calls(self):
        provider = self._make_provider()
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="Let me search.",
                tool_calls=[ToolCall(id="tc_1", name="search", arguments={"q": "test"})],
            ),
        ]
        result = provider._format_messages(messages, system=None)

        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me search."
        assert len(result[0]["tool_calls"]) == 1
        tc = result[0]["tool_calls"][0]
        assert tc["id"] == "tc_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "search"
        assert json.loads(tc["function"]["arguments"]) == {"q": "test"}

    def test_format_messages_with_tool_result(self):
        provider = self._make_provider()
        messages = [
            Message(role=Role.TOOL, content="Result: 42", tool_call_id="tc_1"),
        ]
        result = provider._format_messages(messages, system=None)

        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "tc_1"
        assert result[0]["content"] == "Result: 42"

    def test_format_tool(self):
        provider = self._make_provider()
        tool = ToolDefinition(
            name="search_clinical_calculators",
            description="Search for calculators",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        result = provider._format_tool(tool)

        assert result["type"] == "function"
        assert result["function"]["name"] == "search_clinical_calculators"
        assert result["function"]["description"] == "Search for calculators"
        assert result["function"]["parameters"] == tool.input_schema

    def test_parse_response_text_only(self):
        provider = self._make_provider()

        mock_message = MagicMock()
        mock_message.content = "The score is 3."
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "gpt-4o"

        result = provider._parse_response(mock_response)
        assert result.content == "The score is 3."
        assert result.tool_calls == []
        assert result.stop_reason == "stop"
        assert result.usage == {"input_tokens": 100, "output_tokens": 50}

    def test_parse_response_with_tool_calls(self):
        provider = self._make_provider()

        mock_fn = MagicMock()
        mock_fn.name = "execute_clinical_calculator"
        mock_fn.arguments = json.dumps({"calculator_id": "chadsvasc"})

        mock_tc = MagicMock()
        mock_tc.id = "call_abc123"
        mock_tc.function = mock_fn

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=80, completion_tokens=20)
        mock_response.model = "gpt-4o"

        result = provider._parse_response(mock_response)
        assert result.content == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "execute_clinical_calculator"
        assert result.tool_calls[0].arguments == {"calculator_id": "chadsvasc"}
        assert result.stop_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_call_integration(self):
        """Test full call flow with mocked OpenAI client."""
        provider = self._make_provider()

        mock_fn = MagicMock()
        mock_fn.name = "search_clinical_calculators"
        mock_fn.arguments = json.dumps({"query": "CHA2DS2-VASc"})

        mock_tc = MagicMock()
        mock_tc.id = "call_xyz"
        mock_tc.function = mock_fn

        mock_message = MagicMock()
        mock_message.content = "Searching for calculator."
        mock_message.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=30)
        mock_response.model = "gpt-4o"

        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        tools = [
            ToolDefinition(
                name="search_clinical_calculators",
                description="Search",
                input_schema={"type": "object"},
            )
        ]
        result = await provider.call(
            messages=[Message(role=Role.USER, content="Calculate CHA2DS2-VASc")],
            tools=tools,
            system="You are a clinical assistant.",
        )

        assert result.content == "Searching for calculator."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_clinical_calculators"

    def test_format_tool_result(self):
        provider = self._make_provider()
        result = provider.format_tool_result(ToolResult(tool_call_id="tc_1", content="Score: 3"))
        assert result.role == Role.TOOL
        assert result.content == "Score: 3"
        assert result.tool_call_id == "tc_1"


class TestClaudeProvider:
    """Test Claude provider message formatting."""

    def _make_provider(self):
        from fangbot.brain.providers.claude import ClaudeProvider

        return ClaudeProvider(api_key="test-key")

    def test_format_tool(self):
        provider = self._make_provider()
        tool = ToolDefinition(
            name="search_clinical_calculators",
            description="Search",
            input_schema={"type": "object"},
        )
        result = provider._format_tool(tool)

        # Anthropic uses input_schema, not parameters
        assert result["name"] == "search_clinical_calculators"
        assert result["input_schema"] == {"type": "object"}
        assert "parameters" not in result

    def test_format_tool_result(self):
        provider = self._make_provider()
        result = provider.format_tool_result(ToolResult(tool_call_id="tc_1", content="Score: 3"))
        # Anthropic tool results go in a user message
        assert result.role == Role.USER
        assert result.tool_call_id == "tc_1"
