"""Tool registry for discovering and caching MCP tools."""

from __future__ import annotations

import logging

from fangbot.models import ToolDefinition
from fangbot.skills.mcp_client import OpenMedicineMCPClient

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Discovers and caches tool definitions from the MCP server."""

    def __init__(self, mcp_client: OpenMedicineMCPClient):
        self._client = mcp_client
        self._tools: list[ToolDefinition] | None = None

    async def get_tools(self) -> list[ToolDefinition]:
        """Return cached tools, discovering them on first call."""
        if self._tools is None:
            self._tools = await self._client.list_tools()
            logger.info(f"Discovered {len(self._tools)} MCP tools: {[t.name for t in self._tools]}")
        return self._tools

    async def refresh(self) -> list[ToolDefinition]:
        """Force re-discovery of tools."""
        self._tools = None
        return await self.get_tools()
