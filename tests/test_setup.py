"""Tests for the fangbot init setup wizard."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from fangbot.config import FANGBOT_HOME
from fangbot.gateway.cli import app
from fangbot.gateway.setup import _check_mcp_server, _write_env_file

runner = CliRunner()


class TestCheckMCPServer:
    def test_finds_direct_binary(self):
        with patch("shutil.which", side_effect=lambda cmd: "/usr/bin/open-medicine-mcp" if cmd == "open-medicine-mcp" else None):
            assert _check_mcp_server() == "open-medicine-mcp"

    def test_falls_back_to_uv(self):
        def which(cmd):
            if cmd == "open-medicine-mcp":
                return None
            if cmd == "uv":
                return "/usr/bin/uv"
            return None

        with patch("shutil.which", side_effect=which):
            assert _check_mcp_server() == "uv"

    def test_returns_none_when_nothing_found(self):
        with patch("shutil.which", return_value=None):
            assert _check_mcp_server() is None


class TestWriteEnvFile:
    def test_creates_env_file(self, tmp_path):
        with patch("fangbot.gateway.setup.FANGBOT_HOME", tmp_path):
            path = _write_env_file("claude", "ANTHROPIC_API_KEY", "sk-test-123", "open-medicine-mcp")

        assert path.exists()
        content = path.read_text()
        assert "ANTHROPIC_API_KEY=sk-test-123" in content
        assert "FANGBOT_PROVIDER=claude" in content
        # No MCP override needed for direct binary
        assert "FANGBOT_MCP_COMMAND" not in content

    def test_creates_env_with_uv_fallback(self, tmp_path):
        with patch("fangbot.gateway.setup.FANGBOT_HOME", tmp_path):
            path = _write_env_file("openai", "OPENAI_API_KEY", "sk-test-456", "uv")

        content = path.read_text()
        assert "OPENAI_API_KEY=sk-test-456" in content
        assert "FANGBOT_MCP_COMMAND=uv" in content
        assert "FANGBOT_MCP_ARGS=run,open-medicine-mcp" in content

    def test_preserves_existing_keys(self, tmp_path):
        with patch("fangbot.gateway.setup.FANGBOT_HOME", tmp_path):
            env_path = tmp_path / ".env"
            env_path.write_text("EXISTING_KEY=keep-me\n")

            _write_env_file("claude", "ANTHROPIC_API_KEY", "sk-new", "open-medicine-mcp")

        content = env_path.read_text()
        assert "EXISTING_KEY=keep-me" in content
        assert "ANTHROPIC_API_KEY=sk-new" in content

    def test_file_permissions_restricted(self, tmp_path):
        with patch("fangbot.gateway.setup.FANGBOT_HOME", tmp_path):
            path = _write_env_file("claude", "ANTHROPIC_API_KEY", "sk-secret", "open-medicine-mcp")

        mode = path.stat().st_mode & 0o777
        assert mode == 0o600


class TestInitCLI:
    def test_init_help(self):
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        assert "configure" in result.output.lower() or "set up" in result.output.lower()


class TestFangbotHome:
    def test_home_is_dot_fangbot(self):
        from pathlib import Path

        assert FANGBOT_HOME == Path.home() / ".fangbot"
