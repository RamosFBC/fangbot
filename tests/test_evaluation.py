"""Tests for the batch evaluation framework."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fangbot.evaluation.models import (
    ExpectedToolCall,
    GoldStandardCase,
    RiskTier,
    StudyConfig,
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
