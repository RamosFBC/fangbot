"""Tests for the CLI gateway."""

from __future__ import annotations

from typer.testing import CliRunner

from openmedicine_agent.gateway.cli import app

runner = CliRunner()


class TestCLI:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "OpenMedicine Agent CLI" in result.output

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
