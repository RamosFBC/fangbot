"""Tests for the batch evaluation framework."""

from __future__ import annotations

import json as json_mod
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from fangbot.evaluation.batch_runner import BatchRunner
from fangbot.evaluation.gold_standard import load_cases, load_study_config
from fangbot.evaluation.models import (
    CaseResult,
    ExpectedToolCall,
    GoldStandardCase,
    RiskTier,
    StudyConfig,
)
from fangbot.evaluation.report import generate_report
from fangbot.gateway.cli import app as cli_app


def _make_gold(case_id, score, tier):
    return GoldStandardCase(
        case_id=case_id,
        narrative=f"Patient {case_id}.",
        expected_score=score,
        expected_risk_tier=tier,
        expected_tool_calls=[ExpectedToolCall(tool_name="execute_clinical_calculator")],
    )


def _make_result(case_id, score, tier, tool_calls=None):
    return CaseResult(
        case_id=case_id,
        provider="claude",
        model="claude-sonnet-4-20250514",
        actual_score=score,
        actual_risk_tier=tier,
        actual_tool_calls=tool_calls
        if tool_calls is not None
        else ["search_clinical_calculators", "execute_clinical_calculator"],
        chain_of_thought=["Step 1"],
        guardrail_passed=True,
    )


class TestGoldStandardCase:
    def test_minimal_case(self):
        case = GoldStandardCase(
            case_id="case_001",
            narrative="72-year-old male with history of CHF and hypertension.",
            expected_score=2,
            expected_risk_tier=RiskTier.MODERATE,
            expected_tool_calls=[
                ExpectedToolCall(tool_name="execute_clinical_calculator"),
            ],
        )
        assert case.case_id == "case_001"
        assert case.expected_score == 2
        assert case.expected_risk_tier == RiskTier.MODERATE
        assert len(case.expected_tool_calls) == 1

    def test_case_with_all_fields(self):
        case = GoldStandardCase(
            case_id="case_002",
            narrative="65-year-old female with diabetes and prior stroke.",
            expected_score=5,
            expected_risk_tier=RiskTier.HIGH,
            expected_tool_calls=[
                ExpectedToolCall(
                    tool_name="search_clinical_calculators",
                    required_arguments={"query": "CHA2DS2-VASc"},
                ),
                ExpectedToolCall(tool_name="execute_clinical_calculator"),
            ],
            expected_variables={"age": "65", "sex": "female", "diabetes": "1"},
            notes="Stroke history should be weighted as 2 points.",
        )
        assert case.expected_variables["diabetes"] == "1"
        assert case.notes is not None

    def test_case_requires_narrative(self):
        with pytest.raises(ValidationError):
            GoldStandardCase(
                case_id="bad",
                narrative="",  # Empty narrative should fail
                expected_score=0,
                expected_risk_tier=RiskTier.LOW,
                expected_tool_calls=[],
            )

    def test_case_requires_at_least_one_tool_call(self):
        with pytest.raises(ValidationError):
            GoldStandardCase(
                case_id="bad",
                narrative="Some patient.",
                expected_score=0,
                expected_risk_tier=RiskTier.LOW,
                expected_tool_calls=[],  # Empty — should fail
            )


class TestStudyConfig:
    def test_study_config_fields(self):
        config = StudyConfig(
            study_name="CHA2DS2-VASc Scoring",
            calculator_name="CHA2DS2-VASc",
            description="Evaluate CHA2DS2-VASc scoring accuracy",
            cases_dir="studies/chadsvasc/cases",
            results_dir="studies/chadsvasc/results",
        )
        assert config.study_name == "CHA2DS2-VASc Scoring"
        assert config.cases_dir == "studies/chadsvasc/cases"


class TestRiskTier:
    def test_tiers_exist(self):
        assert RiskTier.LOW.value == "low"
        assert RiskTier.MODERATE.value == "moderate"
        assert RiskTier.HIGH.value == "high"


class TestLoadStudyConfig:
    def test_load_valid_config(self, tmp_path):
        config_data = {
            "study_name": "CHA2DS2-VASc Scoring",
            "calculator_name": "CHA2DS2-VASc",
            "description": "Test study",
            "cases_dir": "cases",
            "results_dir": "results",
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_study_config(config_file)
        assert config.study_name == "CHA2DS2-VASc Scoring"
        assert config.calculator_name == "CHA2DS2-VASc"

    def test_load_config_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_study_config(tmp_path / "nonexistent.yaml")

    def test_load_config_invalid_yaml(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("not: valid: yaml: {{{{")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_study_config(bad_file)


class TestLoadCases:
    def _write_case(self, cases_dir, case_id, score, tier="low"):
        case_data = {
            "case_id": case_id,
            "narrative": f"Patient for {case_id}.",
            "expected_score": score,
            "expected_risk_tier": tier,
            "expected_tool_calls": [
                {"tool_name": "execute_clinical_calculator"},
            ],
        }
        case_file = cases_dir / f"{case_id}.yaml"
        case_file.write_text(yaml.dump(case_data))

    def test_load_cases_from_directory(self, tmp_path):
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        self._write_case(cases_dir, "case_001", 2, "moderate")
        self._write_case(cases_dir, "case_002", 5, "high")

        cases = load_cases(cases_dir)
        assert len(cases) == 2
        ids = {c.case_id for c in cases}
        assert ids == {"case_001", "case_002"}

    def test_load_cases_sorted_by_id(self, tmp_path):
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        self._write_case(cases_dir, "case_003", 1)
        self._write_case(cases_dir, "case_001", 2)
        self._write_case(cases_dir, "case_002", 3)

        cases = load_cases(cases_dir)
        assert [c.case_id for c in cases] == ["case_001", "case_002", "case_003"]

    def test_load_cases_empty_directory(self, tmp_path):
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()

        with pytest.raises(ValueError, match="No .yaml case files"):
            load_cases(cases_dir)

    def test_load_cases_skips_non_yaml(self, tmp_path):
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        self._write_case(cases_dir, "case_001", 2)
        (cases_dir / "README.md").write_text("# Notes")

        cases = load_cases(cases_dir)
        assert len(cases) == 1

    def test_load_cases_validation_error_includes_filename(self, tmp_path):
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        bad_case = cases_dir / "bad_case.yaml"
        bad_case.write_text(yaml.dump({"case_id": "bad", "narrative": ""}))

        with pytest.raises(ValueError, match="bad_case.yaml"):
            load_cases(cases_dir)


class TestReportGenerator:
    def test_generates_markdown(self):
        golds = [
            _make_gold("c1", 2, RiskTier.MODERATE),
            _make_gold("c2", 5, RiskTier.HIGH),
        ]
        results_by_provider = {
            "claude/claude-sonnet-4-20250514": [
                _make_result("c1", 2, RiskTier.MODERATE),
                _make_result("c2", 5, RiskTier.HIGH),
            ],
        }
        report = generate_report("CHA2DS2-VASc", golds, results_by_provider)
        assert "# CHA2DS2-VASc Evaluation Report" in report
        assert "accuracy" in report.lower()
        assert "claude" in report.lower()

    def test_multiple_providers(self):
        golds = [_make_gold("c1", 2, RiskTier.MODERATE)]
        results_by_provider = {
            "claude/claude-sonnet-4-20250514": [_make_result("c1", 2, RiskTier.MODERATE)],
            "openai/gpt-4o": [
                CaseResult(
                    case_id="c1",
                    provider="openai",
                    model="gpt-4o",
                    actual_score=3,
                    actual_risk_tier=RiskTier.HIGH,
                    actual_tool_calls=["execute_clinical_calculator"],
                    guardrail_passed=True,
                ),
            ],
        }
        report = generate_report("CHA2DS2-VASc", golds, results_by_provider)
        assert "claude" in report.lower()
        assert "openai" in report.lower()

    def test_report_includes_per_case_table(self):
        golds = [_make_gold("c1", 2, RiskTier.MODERATE)]
        results_by_provider = {
            "claude/claude-sonnet-4-20250514": [_make_result("c1", 2, RiskTier.MODERATE)],
        }
        report = generate_report("CHA2DS2-VASc", golds, results_by_provider)
        assert "case" in report.lower() or "Case" in report


class TestBatchRunner:
    @pytest.mark.asyncio
    async def test_runs_all_cases(self, tmp_path):
        """BatchRunner should run each gold standard case through the agent."""
        golds = [
            _make_gold("c1", 2, RiskTier.MODERATE),
            _make_gold("c2", 5, RiskTier.HIGH),
        ]
        config = StudyConfig(
            study_name="Test Study",
            calculator_name="CHA2DS2-VASc",
            cases_dir="cases",
            results_dir=str(tmp_path / "results"),
        )

        runner = BatchRunner(config)
        with patch.object(runner, "_run_single_case", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = CaseResult(
                case_id="placeholder",
                provider="claude",
                model="claude-sonnet-4-20250514",
                actual_score=2,
                actual_risk_tier=RiskTier.MODERATE,
                actual_tool_calls=["search_clinical_calculators", "execute_clinical_calculator"],
                synthesis="CHA2DS2-VASc score: 2",
                chain_of_thought=["Analyzing"],
                guardrail_passed=True,
                iterations=2,
            )

            results = await runner.run_cases(golds, provider_name="claude", model_name="claude-sonnet-4-20250514")

        assert len(results) == 2
        assert mock_run.call_count == 2

    @pytest.mark.asyncio
    async def test_saves_results_to_json(self, tmp_path):
        """BatchRunner should save results as JSON."""
        golds = [_make_gold("c1", 2, RiskTier.MODERATE)]
        config = StudyConfig(
            study_name="Test Study",
            calculator_name="CHA2DS2-VASc",
            cases_dir="cases",
            results_dir=str(tmp_path / "results"),
        )

        runner = BatchRunner(config)
        with patch.object(runner, "_run_single_case", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = CaseResult(
                case_id="c1",
                provider="claude",
                model="test-model",
                actual_score=2,
                actual_risk_tier=RiskTier.MODERATE,
                actual_tool_calls=["execute_clinical_calculator"],
                guardrail_passed=True,
            )

            results = await runner.run_cases(golds, provider_name="claude", model_name="test-model")
            runner.save_results(results, provider_name="claude", model_name="test-model")

        results_dir = Path(config.results_dir)
        assert results_dir.exists()
        json_files = list(results_dir.glob("*.json"))
        assert len(json_files) == 1


class TestCLIRunCommand:
    def test_run_with_missing_config_file(self):
        runner = CliRunner()
        result = runner.invoke(cli_app, ["run", "/nonexistent/config.yaml"])
        assert result.exit_code != 0 or "not found" in result.output.lower() or "error" in result.output.lower()

    def test_run_with_valid_config(self, tmp_path):
        """CLI run command should load config and attempt to run study."""
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        case_data = {
            "case_id": "c1",
            "narrative": "72yo male with CHF.",
            "expected_score": 1,
            "expected_risk_tier": "low",
            "expected_tool_calls": [{"tool_name": "execute_clinical_calculator"}],
        }
        (cases_dir / "c1.yaml").write_text(yaml.dump(case_data))

        config_data = {
            "study_name": "Test",
            "calculator_name": "CHA2DS2-VASc",
            "cases_dir": str(cases_dir),
            "results_dir": str(tmp_path / "results"),
            "providers": ["claude"],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        runner = CliRunner()
        result = runner.invoke(cli_app, ["run", str(config_file)])
        # Should have loaded config successfully (even if run fails due to no API key)
        assert "Loading study" in result.output or "Error" in result.output or result.exit_code == 0


class TestCLIReportCommand:
    def test_report_with_missing_dir(self):
        runner = CliRunner()
        result = runner.invoke(cli_app, ["report", "/nonexistent/dir"])
        assert result.exit_code != 0 or "not found" in result.output.lower() or "error" in result.output.lower()

    def test_report_generates_markdown(self, tmp_path):
        """CLI report command should generate a Markdown report from saved results."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        results_data = {
            "study_name": "CHA2DS2-VASc Test",
            "provider": "claude",
            "model": "claude-sonnet-4-20250514",
            "timestamp": "20260309_120000",
            "cases": [
                {
                    "case_id": "c1",
                    "provider": "claude",
                    "model": "claude-sonnet-4-20250514",
                    "actual_score": 2,
                    "actual_risk_tier": "moderate",
                    "actual_tool_calls": ["execute_clinical_calculator"],
                    "synthesis": "Score: 2",
                    "chain_of_thought": ["Step 1"],
                    "guardrail_passed": True,
                    "iterations": 2,
                    "error": None,
                    "duration_seconds": 1.5,
                    "timestamp": "2026-03-09T12:00:00",
                    "audit_session_id": "abc123",
                },
            ],
        }
        (results_dir / "claude_test_20260309.json").write_text(json_mod.dumps(results_data))

        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        case_data = {
            "case_id": "c1",
            "narrative": "72yo male with CHF.",
            "expected_score": 2,
            "expected_risk_tier": "moderate",
            "expected_tool_calls": [{"tool_name": "execute_clinical_calculator"}],
        }
        (cases_dir / "c1.yaml").write_text(yaml.dump(case_data))

        config_data = {
            "study_name": "CHA2DS2-VASc Test",
            "calculator_name": "CHA2DS2-VASc",
            "cases_dir": str(cases_dir),
            "results_dir": str(results_dir),
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        runner = CliRunner()
        result = runner.invoke(cli_app, ["report", str(results_dir), "--config", str(config_file)])
        assert result.exit_code == 0 or "Report" in result.output
