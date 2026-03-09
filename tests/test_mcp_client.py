"""Tests for the MCP client and tool registry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fangbot.models import ToolDefinition
from fangbot.skills.mcp_client import MCPToolError, OpenMedicineMCPClient
from fangbot.skills.tool_registry import ToolRegistry


class TestOpenMedicineMCPClient:
    def test_not_connected_raises(self):
        client = OpenMedicineMCPClient()
        with pytest.raises(RuntimeError, match="not connected"):
            client._ensure_connected()

    @pytest.mark.asyncio
    async def test_list_tools_converts_to_tool_definitions(self):
        """Verify MCP tools are converted to our ToolDefinition format."""
        mock_tool = MagicMock()
        mock_tool.name = "search_clinical_calculators"
        mock_tool.description = "Search calculators"
        mock_tool.inputSchema = {"type": "object", "properties": {"query": {"type": "string"}}}

        mock_session = AsyncMock()
        mock_session.list_tools.return_value = MagicMock(tools=[mock_tool])

        client = OpenMedicineMCPClient()
        client._session = mock_session

        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "search_clinical_calculators"
        assert isinstance(tools[0], ToolDefinition)

    @pytest.mark.asyncio
    async def test_call_tool_returns_text(self):
        """Verify successful tool call extracts text content."""
        from mcp.types import TextContent

        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [TextContent(type="text", text="Score: 3")]

        mock_session = AsyncMock()
        mock_session.call_tool.return_value = mock_result

        client = OpenMedicineMCPClient()
        client._session = mock_session

        result = await client.call_tool("execute_clinical_calculator", {"id": "chadsvasc"})
        assert result == "Score: 3"

    @pytest.mark.asyncio
    async def test_call_tool_raises_on_error(self):
        """Verify MCPToolError is raised when tool returns an error."""
        from mcp.types import TextContent

        mock_result = MagicMock()
        mock_result.isError = True
        mock_result.content = [TextContent(type="text", text="Invalid parameters")]

        mock_session = AsyncMock()
        mock_session.call_tool.return_value = mock_result

        client = OpenMedicineMCPClient()
        client._session = mock_session

        with pytest.raises(MCPToolError, match="Invalid parameters"):
            await client.call_tool("execute_clinical_calculator", {})


class TestToolRegistry:
    @pytest.mark.asyncio
    async def test_caches_tools(self, mock_mcp):
        """Verify tools are discovered once and cached."""
        registry = ToolRegistry(mock_mcp)

        tools1 = await registry.get_tools()
        tools2 = await registry.get_tools()

        assert tools1 == tools2
        assert len(tools1) == 4

    @pytest.mark.asyncio
    async def test_refresh_clears_cache(self, mock_mcp):
        """Verify refresh forces re-discovery."""
        registry = ToolRegistry(mock_mcp)

        await registry.get_tools()
        assert registry._tools is not None

        await registry.refresh()
        assert registry._tools is not None  # re-populated
