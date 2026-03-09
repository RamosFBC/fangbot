"""Tests for the CLI gateway and slash commands."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from fangbot.gateway.cli import (
    ChatState,
    _commands,
    _create_provider,
    _handle_slash_command,
    app,
)
from fangbot.gateway.models_catalog import PROVIDER_DEFAULTS, PROVIDER_MODELS
from fangbot.config import Settings

runner = CliRunner()


class TestCLI:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Fangbot" in result.output

    def test_chat_help(self):
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "interactive" in result.output.lower()

    def test_run_stub(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("study: test")

        result = runner.invoke(app, ["run", str(config_file)])
        assert result.exit_code == 0
        assert "not yet implemented" in result.output.lower()

    def test_report_stub(self, tmp_path):
        result = runner.invoke(app, ["report", str(tmp_path)])
        assert result.exit_code == 0
        assert "not yet implemented" in result.output.lower()


class TestProviderFactory:
    def test_create_claude_provider(self):
        settings = Settings(anthropic_api_key="test-key")
        provider = _create_provider("claude", settings)
        assert provider.model_name == PROVIDER_DEFAULTS["claude"]

    def test_create_openai_provider(self):
        settings = Settings(openai_api_key="test-key")
        provider = _create_provider("openai", settings)
        assert provider.model_name == PROVIDER_DEFAULTS["openai"]

    def test_create_with_custom_model(self):
        settings = Settings(anthropic_api_key="test-key")
        provider = _create_provider("claude", settings, model="claude-haiku-4-5-20251001")
        assert provider.model_name == "claude-haiku-4-5-20251001"

    def test_unknown_provider_raises(self):
        settings = Settings()
        with pytest.raises(ValueError, match="Unknown provider"):
            _create_provider("llama", settings)


class TestSlashCommands:
    """Test slash command handlers using a minimal ChatState."""

    @pytest.fixture
    def chat_state(self, tmp_path):
        from fangbot.memory.audit import AuditLogger
        from fangbot.memory.session import SessionContext

        settings = Settings(anthropic_api_key="test", openai_api_key="test")
        provider = _create_provider("claude", settings)
        audit = AuditLogger(log_dir=str(tmp_path))
        audit.start_session()
        session = SessionContext(system_prompt="test")

        from fangbot.brain.react import ReActLoop
        from tests.conftest import MockMCPClient

        mcp = MockMCPClient()
        react = ReActLoop(provider=provider, mcp_client=mcp, audit_logger=audit)

        return ChatState(
            settings=settings,
            provider=provider,
            provider_name="claude",
            react=react,
            session=session,
            audit=audit,
            tools=[],
            mcp=mcp,
        )

    def test_all_expected_commands_registered(self):
        expected = {
            "help",
            "status",
            "claude",
            "openai",
            "model",
            "models",
            "clear",
            "history",
            "compact",
        }
        assert expected.issubset(set(_commands.keys()))

    @pytest.mark.asyncio
    async def test_cmd_claude_switches_provider(self, chat_state):
        handler, _ = _commands["claude"]
        await handler(chat_state, ["claude-sonnet-4-20250514"])

        assert chat_state.provider_name == "claude"
        assert "claude" in chat_state.provider.model_name.lower()

    @pytest.mark.asyncio
    async def test_cmd_openai_switches_provider(self, chat_state):
        handler, _ = _commands["openai"]
        await handler(chat_state, ["gpt-4o"])

        assert chat_state.provider_name == "openai"
        assert chat_state.provider.model_name == "gpt-4o"

    @pytest.mark.asyncio
    async def test_cmd_openai_with_custom_model(self, chat_state):
        handler, _ = _commands["openai"]
        await handler(chat_state, ["gpt-4-turbo"])

        assert chat_state.provider.model_name == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_cmd_model_switches_model(self, chat_state):
        handler, _ = _commands["model"]
        await handler(chat_state, ["claude-haiku-4-5-20251001"])

        assert chat_state.provider.model_name == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_cmd_clear_resets_session(self, chat_state):
        chat_state.session.add_user_message("hello")
        assert len(chat_state.session.messages) == 1

        handler, _ = _commands["clear"]
        await handler(chat_state, [])

        assert len(chat_state.session.messages) == 0

    @pytest.mark.asyncio
    async def test_cmd_compact_preserves_tool_history(self, chat_state):
        chat_state.session.add_user_message("msg1")
        chat_state.session.add_user_message("msg2")
        chat_state.session.add_user_message("msg3")
        chat_state.session.record_tool_call("search_clinical_calculators")

        handler, _ = _commands["compact"]
        await handler(chat_state, [])

        assert len(chat_state.session.messages) == 0
        assert chat_state.session.tool_calls_made == ["search_clinical_calculators"]

    @pytest.mark.asyncio
    async def test_handle_unknown_command(self, chat_state):
        result = await _handle_slash_command("/nonexistent", chat_state)
        assert result is True  # handled (printed error)

    @pytest.mark.asyncio
    async def test_handle_fuzzy_match(self, chat_state):
        """Prefix match: /stat should resolve to /status."""
        result = await _handle_slash_command("/stat", chat_state)
        assert result is True

    @pytest.mark.asyncio
    async def test_provider_switch_preserves_session(self, chat_state):
        """Switching provider should NOT clear conversation history."""
        chat_state.session.add_user_message("important context")

        handler, _ = _commands["openai"]
        await handler(chat_state, ["gpt-4o"])

        assert len(chat_state.session.messages) == 1
        assert chat_state.session.messages[0].content == "important context"


class TestModelsCatalog:
    def test_all_providers_have_models(self):
        for provider in PROVIDER_DEFAULTS:
            assert provider in PROVIDER_MODELS
            assert len(PROVIDER_MODELS[provider]) > 0

    def test_defaults_exist_in_catalog(self):
        for provider, default_model in PROVIDER_DEFAULTS.items():
            model_ids = {m.id for m in PROVIDER_MODELS[provider]}
            assert default_model in model_ids, f"Default {default_model} not in {provider} catalog"

    def test_openai_models_include_key_families(self):
        ids = {m.id for m in PROVIDER_MODELS["openai"]}
        assert "gpt-4o" in ids
        assert "o3" in ids
        assert "o4-mini" in ids
        assert "gpt-5" in ids

    def test_claude_models_include_key_families(self):
        ids = {m.id for m in PROVIDER_MODELS["claude"]}
        assert "claude-opus-4-6" in ids
        assert "claude-sonnet-4-6" in ids
        assert "claude-haiku-4-5-20251001" in ids
