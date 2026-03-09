"""MCP client for connecting to the OpenMedicine server."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent

from fangbot.models import ToolDefinition

logger = logging.getLogger(__name__)


class MCPToolError(Exception):
    """Raised when an MCP tool call returns an error."""


class OpenMedicineMCPClient:
    """Async context manager wrapping a long-lived MCP stdio connection."""

    def __init__(
        self,
        command: str = "open-medicine-mcp",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        self._server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )
        self._session: ClientSession | None = None
        self._stdio_context = None
        self._session_context = None

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[OpenMedicineMCPClient]:
        """Open a long-lived connection to the MCP server."""
        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                logger.info("MCP connection established")
                try:
                    yield self
                finally:
                    self._session = None
                    logger.info("MCP connection closed")

    def _ensure_connected(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCP client not connected. Use `async with client.connect():`")
        return self._session

    async def list_tools(self) -> list[ToolDefinition]:
        """Discover available tools from the MCP server."""
        session = self._ensure_connected()
        result = await session.list_tools()
        return [
            ToolDefinition(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema if tool.inputSchema else {},
            )
            for tool in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool on the MCP server and return the text result."""
        session = self._ensure_connected()
        result = await session.call_tool(name, arguments)

        if result.isError:
            error_text = self._extract_text(result.content)
            raise MCPToolError(f"Tool '{name}' returned error: {error_text}")

        return self._extract_text(result.content)

    @staticmethod
    def _extract_text(content: list) -> str:
        """Extract text from MCP content blocks."""
        parts = []
        for block in content:
            if isinstance(block, TextContent):
                parts.append(block.text)
            elif hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)
