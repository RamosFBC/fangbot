"""Tests for encounter-level evaluation models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fangbot.evaluation.encounter_models import (
    DecisionCategory,
    EncounterCaseResult,
    EncounterGoldStandard,
    ExpectedDecision,
    ExpectedToolInvocation,
)


class TestExpectedDecision:
    def test_valid_decision(self) -> None:
        d = ExpectedDecision(
            category=DecisionCategory.MEDICATION,
            decision="Initiate oral anticoagulation",
            acceptable=["apixaban", "rivaroxaban"],
            contraindicated=["aspirin monotherapy"],
            reasoning="CHA2DS2-VASc >= 2",
        )
        assert d.category == DecisionCategory.MEDICATION
        assert len(d.acceptable) == 2

    def test_category_values(self) -> None:
        for cat in [
            "medication",
            "disposition",
            "referral",
            "follow_up",
            "diagnostic",
            "procedure",
        ]:
            d = ExpectedDecision(category=cat, decision="test")
            assert d.category == cat


class TestEncounterGoldStandard:
    def test_valid_case(self) -> None:
        case = EncounterGoldStandard(
            case_id="test_001",
            encounter_type="initial_consultation",
            narrative="Patient presents with...",
            expected_tool_invocations=[
                ExpectedToolInvocation(tool="CHA2DS2-VASc", reason="AFib risk")
            ],
            expected_reasoning=["Identified AFib"],
            expected_decisions=[
                ExpectedDecision(
                    category="medication",
                    decision="Anticoagulation",
                    acceptable=["apixaban"],
                )
            ],
            required_elements=["medication_reconciliation"],
            skill_loaded="initial_consultation",
        )
        assert case.case_id == "test_001"

    def test_empty_narrative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EncounterGoldStandard(
                case_id="bad",
                encounter_type="initial_consultation",
                narrative="",
                expected_decisions=[],
                skill_loaded="initial_consultation",
            )

    def test_forbidden_elements_default_empty(self) -> None:
        case = EncounterGoldStandard(
            case_id="test",
            encounter_type="follow_up",
            narrative="Patient returns for follow-up.",
            expected_decisions=[],
            skill_loaded="follow_up",
        )
        assert case.forbidden_elements == []


class TestEncounterCaseResult:
    def test_valid_result(self) -> None:
        r = EncounterCaseResult(
            case_id="test_001",
            provider="claude",
            model="claude-sonnet-4-20250514",
            actual_tool_calls=["search_clinical_calculators", "execute_clinical_calculator"],
            actual_decisions=[{"category": "medication", "decision": "apixaban"}],
            synthesis="Assessment complete.",
            skill_loaded="initial_consultation",
        )
        assert r.case_id == "test_001"
        assert r.skill_loaded == "initial_consultation"
