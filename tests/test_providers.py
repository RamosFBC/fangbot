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


class TestOpenAIResponsesApiFallback:
    """Test Responses API fallback for models that don't support chat completions."""

    def _make_provider(self):
        from fangbot.brain.providers.openai import OpenAIProvider

        return OpenAIProvider(api_key="test-key", model="gpt-5.4-pro")

    def test_format_responses_input_user_message(self):
        provider = self._make_provider()
        messages = [Message(role=Role.USER, content="Hello")]
        result = provider._format_responses_input(messages)

        assert result == [{"role": "user", "content": "Hello"}]

    def test_format_responses_input_with_tool_calls(self):
        provider = self._make_provider()
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="Let me search.",
                tool_calls=[ToolCall(id="call_1", name="search", arguments={"q": "test"})],
            ),
        ]
        result = provider._format_responses_input(messages)

        assert len(result) == 2
        assert result[0] == {"role": "assistant", "content": "Let me search."}
        assert result[1]["type"] == "function_call"
        assert result[1]["name"] == "search"
        assert result[1]["call_id"] == "call_1"
        assert json.loads(result[1]["arguments"]) == {"q": "test"}

    def test_format_responses_input_with_tool_result(self):
        provider = self._make_provider()
        messages = [
            Message(role=Role.TOOL, content="Result: 42", tool_call_id="call_1"),
        ]
        result = provider._format_responses_input(messages)

        assert result == [
            {"type": "function_call_output", "call_id": "call_1", "output": "Result: 42"}
        ]

    def test_format_responses_input_skips_system(self):
        provider = self._make_provider()
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hi"),
        ]
        result = provider._format_responses_input(messages)

        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hi"}

    def test_format_responses_tool(self):
        from fangbot.brain.providers.openai import OpenAIProvider

        tool = ToolDefinition(
            name="search_clinical_calculators",
            description="Search for calculators",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        result = OpenAIProvider._format_responses_tool(tool)

        assert result["type"] == "function"
        assert result["name"] == "search_clinical_calculators"
        assert result["description"] == "Search for calculators"
        assert result["parameters"] == tool.input_schema
        # Responses API tools have name at top level, NOT nested under "function"
        assert "function" not in result

    def test_parse_responses_response_text_only(self):
        from fangbot.brain.providers.openai import OpenAIProvider

        mock_text_part = MagicMock()
        mock_text_part.text = "The score is 3."

        mock_message_item = MagicMock()
        mock_message_item.type = "message"
        mock_message_item.content = [mock_text_part]

        mock_response = MagicMock()
        mock_response.output = [mock_message_item]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_response.status = "completed"
        mock_response.model = "gpt-5.4-pro"

        result = OpenAIProvider._parse_responses_response(mock_response)
        assert result.content == "The score is 3."
        assert result.tool_calls == []
        assert result.model == "gpt-5.4-pro"

    def test_parse_responses_response_with_function_call(self):
        from fangbot.brain.providers.openai import OpenAIProvider

        mock_fc_item = MagicMock()
        mock_fc_item.type = "function_call"
        mock_fc_item.call_id = "call_abc123"
        mock_fc_item.name = "search_clinical_calculators"
        mock_fc_item.arguments = json.dumps({"query": "GCS"})

        mock_response = MagicMock()
        mock_response.output = [mock_fc_item]
        mock_response.usage = MagicMock(input_tokens=80, output_tokens=20)
        mock_response.status = "completed"
        mock_response.model = "gpt-5.4-pro"

        result = OpenAIProvider._parse_responses_response(mock_response)
        assert result.content == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc123"
        assert result.tool_calls[0].name == "search_clinical_calculators"
        assert result.tool_calls[0].arguments == {"query": "GCS"}

    @pytest.mark.asyncio
    async def test_call_falls_back_to_responses_on_404(self):
        """Test that a 'not a chat model' 404 triggers Responses API fallback."""
        import openai as oai

        provider = self._make_provider()

        # chat.completions raises NotFoundError
        error_body = {
            "error": {
                "message": "This is not a chat model",
                "type": "invalid_request_error",
            }
        }
        mock_http_response = MagicMock()
        mock_http_response.status_code = 404
        mock_http_response.json.return_value = error_body
        mock_http_response.headers = {}
        mock_http_response.text = json.dumps(error_body)

        provider._client.chat.completions.create = AsyncMock(
            side_effect=oai.NotFoundError(
                message="This is not a chat model",
                response=mock_http_response,
                body=error_body,
            )
        )

        # Responses API succeeds with a function call
        mock_fc_item = MagicMock()
        mock_fc_item.type = "function_call"
        mock_fc_item.call_id = "call_resp_1"
        mock_fc_item.name = "search_clinical_calculators"
        mock_fc_item.arguments = json.dumps({"query": "GCS"})

        mock_responses_response = MagicMock()
        mock_responses_response.output = [mock_fc_item]
        mock_responses_response.usage = MagicMock(input_tokens=50, output_tokens=30)
        mock_responses_response.status = "completed"
        mock_responses_response.model = "gpt-5.4-pro"

        provider._client.responses.create = AsyncMock(return_value=mock_responses_response)

        tools = [
            ToolDefinition(
                name="search_clinical_calculators",
                description="Search",
                input_schema={"type": "object"},
            )
        ]
        result = await provider.call(
            messages=[Message(role=Role.USER, content="Calculate GCS")],
            tools=tools,
            system="You are a clinical assistant.",
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_clinical_calculators"
        assert result.model == "gpt-5.4-pro"
        assert provider._use_responses_api is True

        # Verify responses.create was called with correct format
        call_kwargs = provider._client.responses.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-5.4-pro"
        assert call_kwargs.kwargs["instructions"] == "You are a clinical assistant."
        # Tools use Responses format (name at top level)
        tool_arg = call_kwargs.kwargs["tools"][0]
        assert tool_arg["type"] == "function"
        assert tool_arg["name"] == "search_clinical_calculators"

    @pytest.mark.asyncio
    async def test_responses_direct_call(self):
        """Test direct Responses API call when _use_responses_api is already set."""
        provider = self._make_provider()
        provider._use_responses_api = True

        mock_text_part = MagicMock()
        mock_text_part.text = "The GCS score is 15."

        mock_message_item = MagicMock()
        mock_message_item.type = "message"
        mock_message_item.content = [mock_text_part]

        mock_response = MagicMock()
        mock_response.output = [mock_message_item]
        mock_response.usage = MagicMock(input_tokens=30, output_tokens=10)
        mock_response.status = "completed"
        mock_response.model = "gpt-5.4-pro"

        provider._client.responses.create = AsyncMock(return_value=mock_response)

        result = await provider.call(
            messages=[Message(role=Role.USER, content="What is the GCS?")],
            system="Be helpful.",
        )

        assert result.content == "The GCS score is 15."
        assert result.tool_calls == []
        provider._client.responses.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_not_supported_error_also_triggers_fallback(self):
        """Test that 'not supported' 404 (v1/completions rejection) also falls back."""
        import openai as oai

        provider = self._make_provider()

        error_body = {
            "error": {
                "message": "This model is not supported in the v1/completions endpoint.",
                "type": "invalid_request_error",
            }
        }
        mock_http_response = MagicMock()
        mock_http_response.status_code = 404
        mock_http_response.json.return_value = error_body
        mock_http_response.headers = {}
        mock_http_response.text = json.dumps(error_body)

        provider._client.chat.completions.create = AsyncMock(
            side_effect=oai.NotFoundError(
                message="This model is not supported in the v1/completions endpoint.",
                response=mock_http_response,
                body=error_body,
            )
        )

        mock_text_part = MagicMock()
        mock_text_part.text = "Response via Responses API."

        mock_message_item = MagicMock()
        mock_message_item.type = "message"
        mock_message_item.content = [mock_text_part]

        mock_response = MagicMock()
        mock_response.output = [mock_message_item]
        mock_response.usage = MagicMock(input_tokens=20, output_tokens=10)
        mock_response.status = "completed"
        mock_response.model = "gpt-5.4-pro"

        provider._client.responses.create = AsyncMock(return_value=mock_response)

        result = await provider.call(
            messages=[Message(role=Role.USER, content="Hello")],
        )

        assert result.content == "Response via Responses API."
        assert provider._use_responses_api is True


class TestLocalProvider:
    """Test LocalProvider — subclasses OpenAI with custom base_url."""

    def _make_provider(self, base_url="http://localhost:11434/v1", model="llama3.2"):
        from fangbot.brain.providers.local import LocalProvider

        return LocalProvider(base_url=base_url, model=model)

    def test_model_name(self):
        provider = self._make_provider(model="mistral")
        assert provider.model_name == "mistral"

    def test_supports_temperature_always_true(self):
        provider = self._make_provider(model="o1-mini")
        assert provider._supports_temperature() is True

    def test_inherits_format_tool(self):
        provider = self._make_provider()
        tool = ToolDefinition(
            name="search_clinical_calculators",
            description="Search",
            input_schema={"type": "object"},
        )
        result = provider._format_tool(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "search_clinical_calculators"

    def test_format_tool_result(self):
        provider = self._make_provider()
        result = provider.format_tool_result(ToolResult(tool_call_id="tc_1", content="Score: 3"))
        assert result.role == Role.TOOL
        assert result.content == "Score: 3"
        assert result.tool_call_id == "tc_1"

    def test_custom_base_url(self):
        provider = self._make_provider(base_url="http://myserver:9000/v1")
        assert str(provider._client.base_url).rstrip("/") == "http://myserver:9000/v1"

    def test_api_key_defaults_to_placeholder(self):
        provider = self._make_provider()
        assert provider._client.api_key == "not-needed"

    def test_custom_api_key(self):
        from fangbot.brain.providers.local import LocalProvider

        provider = LocalProvider(
            base_url="http://localhost:11434/v1",
            model="llama3.2",
            api_key="my-secret",
        )
        assert provider._client.api_key == "my-secret"

    @pytest.mark.asyncio
    async def test_call_integration(self):
        """Test full call flow with mocked client."""
        provider = self._make_provider()

        mock_fn = MagicMock()
        mock_fn.name = "search_clinical_calculators"
        mock_fn.arguments = json.dumps({"query": "GCS"})

        mock_tc = MagicMock()
        mock_tc.id = "call_local_1"
        mock_tc.function = mock_fn

        mock_message = MagicMock()
        mock_message.content = "Searching."
        mock_message.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=40, completion_tokens=10)
        mock_response.model = "llama3.2"

        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.call(
            messages=[Message(role=Role.USER, content="Calculate GCS")],
            tools=[
                ToolDefinition(
                    name="search_clinical_calculators",
                    description="Search",
                    input_schema={"type": "object"},
                )
            ],
            system="You are a clinical assistant.",
        )

        assert result.content == "Searching."
        assert len(result.tool_calls) == 1
        assert result.model == "llama3.2"


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
