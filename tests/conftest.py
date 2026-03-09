"""Shared test fixtures: mock MCP client, mock LLM provider."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from openmedicine_agent.brain.providers.base import LLMProvider
from openmedicine_agent.models import (
    Message,
    ProviderResponse,
    Role,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from openmedicine_agent.skills.mcp_client import OpenMedicineMCPClient


# -- Sample tool definitions matching OpenMedicine's 4 tools --

SAMPLE_TOOLS = [
    ToolDefinition(
        name="search_clinical_calculators",
        description="Search for clinical calculators",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    ),
    ToolDefinition(
        name="execute_clinical_calculator",
        description="Execute a clinical calculator",
        input_schema={
            "type": "object",
            "properties": {
                "calculator_id": {"type": "string"},
                "parameters": {"type": "object"},
            },
        },
    ),
    ToolDefinition(
        name="search_guidelines",
        description="Search clinical guidelines",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    ),
    ToolDefinition(
        name="retrieve_guideline",
        description="Retrieve a specific guideline",
        input_schema={"type": "object", "properties": {"guideline_id": {"type": "string"}}},
    ),
]


class MockProvider(LLMProvider):
    """Controllable mock LLM provider for testing."""

    def __init__(self, responses: list[ProviderResponse] | None = None):
        self._responses = list(responses or [])
        self._call_count = 0
        self.calls: list[dict] = []

    @property
    def model_name(self) -> str:
        return "mock-model"

    def queue_response(self, response: ProviderResponse) -> None:
        self._responses.append(response)

    async def call(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> ProviderResponse:
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return ProviderResponse(content="No more mock responses", stop_reason="end_turn")

    def format_tool_result(self, result: ToolResult) -> Message:
        return Message(role=Role.USER, content=result.content, tool_call_id=result.tool_call_id)


class MockMCPClient:
    """Mock MCP client that returns predetermined tool results."""

    def __init__(self, tool_results: dict[str, str] | None = None):
        self._tool_results = tool_results or {}
        self.calls: list[dict[str, Any]] = []

    async def list_tools(self) -> list[ToolDefinition]:
        return SAMPLE_TOOLS

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append({"name": name, "arguments": arguments})
        if name in self._tool_results:
            return self._tool_results[name]
        return f"Mock result for {name}"


@pytest.fixture
def sample_tools() -> list[ToolDefinition]:
    return SAMPLE_TOOLS


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def mock_mcp() -> MockMCPClient:
    return MockMCPClient()
