"""Tests for encounter-level evaluation metrics."""

from __future__ import annotations

from fangbot.evaluation.encounter_metrics import (
    compute_decision_accuracy,
    compute_decision_completeness,
    compute_decision_safety,
    compute_encounter_metrics,
    compute_forbidden_elements_absence,
    compute_reasoning_quality,
    compute_required_elements_coverage,
    compute_skill_appropriateness,
)
from fangbot.evaluation.encounter_models import (
    EncounterCaseResult,
    EncounterGoldStandard,
    ExpectedDecision,
)


def _make_gold(
    case_id: str = "c1",
    decisions: list | None = None,
    reasoning: list | None = None,
    required: list | None = None,
    forbidden: list | None = None,
    skill: str = "initial_consultation",
) -> EncounterGoldStandard:
    return EncounterGoldStandard(
        case_id=case_id,
        encounter_type="initial_consultation",
        narrative="Test patient narrative.",
        expected_decisions=decisions or [],
        expected_reasoning=reasoning or [],
        required_elements=required or [],
        forbidden_elements=forbidden or [],
        skill_loaded=skill,
    )


def _make_result(
    case_id: str = "c1",
    synthesis: str = "",
    decisions: list | None = None,
    skill: str | None = "initial_consultation",
) -> EncounterCaseResult:
    return EncounterCaseResult(
        case_id=case_id,
        provider="claude",
        model="test",
        synthesis=synthesis,
        actual_decisions=decisions or [],
        skill_loaded=skill,
    )


class TestDecisionAccuracy:
    def test_matching_decision(self) -> None:
        gold = _make_gold(
            decisions=[
                ExpectedDecision(
                    category="medication",
                    decision="Initiate anticoagulation",
                    acceptable=["apixaban", "rivaroxaban"],
                )
            ]
        )
        result = _make_result(
            synthesis="I recommend starting apixaban for stroke prevention."
        )
        score = compute_decision_accuracy([gold], [result])
        assert score > 0.0

    def test_missing_decision(self) -> None:
        gold = _make_gold(
            decisions=[
                ExpectedDecision(
                    category="medication",
                    decision="Initiate anticoagulation",
                    acceptable=["apixaban"],
                )
            ]
        )
        result = _make_result(synthesis="No medication changes recommended.")
        score = compute_decision_accuracy([gold], [result])
        assert score == 0.0

    def test_empty_cases(self) -> None:
        assert compute_decision_accuracy([], []) == 0.0


class TestDecisionSafety:
    def test_safe_output(self) -> None:
        gold = _make_gold(
            decisions=[
                ExpectedDecision(
                    category="medication",
                    decision="Anticoagulation",
                    contraindicated=["aspirin monotherapy for stroke prevention"],
                )
            ]
        )
        result = _make_result(synthesis="Starting apixaban 5mg twice daily.")
        score = compute_decision_safety([gold], [result])
        assert score == 1.0

    def test_unsafe_output(self) -> None:
        gold = _make_gold(
            decisions=[
                ExpectedDecision(
                    category="medication",
                    decision="Anticoagulation",
                    contraindicated=["aspirin monotherapy for stroke prevention"],
                )
            ]
        )
        result = _make_result(
            synthesis="I recommend aspirin monotherapy for stroke prevention."
        )
        score = compute_decision_safety([gold], [result])
        assert score < 1.0


class TestDecisionCompleteness:
    def test_all_decisions_present(self) -> None:
        gold = _make_gold(
            decisions=[
                ExpectedDecision(category="medication", decision="Anticoagulation"),
                ExpectedDecision(category="referral", decision="Cardiology referral"),
            ]
        )
        result = _make_result(
            synthesis="Starting anticoagulation. Cardiology referral placed."
        )
        score = compute_decision_completeness([gold], [result])
        assert score == 1.0

    def test_partial_decisions(self) -> None:
        gold = _make_gold(
            decisions=[
                ExpectedDecision(category="medication", decision="Anticoagulation"),
                ExpectedDecision(category="referral", decision="Cardiology referral"),
            ]
        )
        result = _make_result(synthesis="Starting anticoagulation.")
        score = compute_decision_completeness([gold], [result])
        assert 0.0 < score < 1.0


class TestReasoningQuality:
    def test_all_reasoning_present(self) -> None:
        gold = _make_gold(reasoning=["Identified atrial fibrillation", "Assessed stroke risk"])
        result = _make_result(
            synthesis="Identified atrial fibrillation. Assessed stroke risk using CHA2DS2-VASc."
        )
        score = compute_reasoning_quality([gold], [result])
        assert score == 1.0

    def test_partial_reasoning(self) -> None:
        gold = _make_gold(reasoning=["Identified atrial fibrillation", "Assessed stroke risk"])
        result = _make_result(synthesis="Identified atrial fibrillation.")
        score = compute_reasoning_quality([gold], [result])
        assert 0.0 < score < 1.0


class TestRequiredElements:
    def test_all_elements_present(self) -> None:
        gold = _make_gold(required=["medication_reconciliation", "allergy_check"])
        result = _make_result(
            synthesis="Performed medication reconciliation. Verified allergy check."
        )
        score = compute_required_elements_coverage([gold], [result])
        assert score == 1.0


class TestForbiddenElements:
    def test_no_forbidden_present(self) -> None:
        gold = _make_gold(forbidden=["Prescribed without risk score"])
        result = _make_result(synthesis="Calculated CHA2DS2-VASc score of 3.")
        score = compute_forbidden_elements_absence([gold], [result])
        assert score == 1.0

    def test_forbidden_present(self) -> None:
        gold = _make_gold(forbidden=["Prescribed without risk score"])
        result = _make_result(
            synthesis="Prescribed without risk score calculation."
        )
        score = compute_forbidden_elements_absence([gold], [result])
        assert score < 1.0


class TestSkillAppropriateness:
    def test_correct_skill(self) -> None:
        gold = _make_gold(skill="initial_consultation")
        result = _make_result(skill="initial_consultation")
        score = compute_skill_appropriateness([gold], [result])
        assert score == 1.0

    def test_wrong_skill(self) -> None:
        gold = _make_gold(skill="initial_consultation")
        result = _make_result(skill="follow_up")
        score = compute_skill_appropriateness([gold], [result])
        assert score == 0.0


class TestComputeAllEncounterMetrics:
    def test_returns_all_metrics(self) -> None:
        gold = _make_gold(
            decisions=[ExpectedDecision(category="medication", decision="Test")],
            reasoning=["Test reasoning"],
            required=["med_recon"],
            skill="initial_consultation",
        )
        result = _make_result(synthesis="Test output.", skill="initial_consultation")
        metrics = compute_encounter_metrics([gold], [result])
        assert "decision_accuracy" in metrics
        assert "decision_safety" in metrics
        assert "decision_completeness" in metrics
        assert "reasoning_quality" in metrics
        assert "required_elements_coverage" in metrics
        assert "forbidden_elements_absence" in metrics
        assert "skill_appropriateness" in metrics
