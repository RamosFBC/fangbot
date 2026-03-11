"""Tests for the evaluation metrics engine."""

from __future__ import annotations

from fangbot.evaluation.metrics import (
    compute_accuracy,
    compute_cot_quality,
    compute_kappa,
    compute_mae,
    compute_protocol_adherence,
    compute_sensitivity_specificity,
    compute_all_metrics,
)
from fangbot.evaluation.models import (
    CaseResult,
    GoldStandardCase,
    ExpectedToolCall,
    RiskTier,
)


def _make_gold(case_id: str, score: int, tier: RiskTier) -> GoldStandardCase:
    return GoldStandardCase(
        case_id=case_id,
        narrative=f"Patient {case_id}.",
        expected_score=score,
        expected_risk_tier=tier,
        expected_tool_calls=[ExpectedToolCall(tool_name="execute_clinical_calculator")],
    )


def _make_result(
    case_id: str,
    score: int | None,
    tier: RiskTier | None,
    tool_calls: list[str] | None = None,
    cot: list[str] | None = None,
    guardrail_passed: bool = True,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        provider="claude",
        model="claude-sonnet-4-20250514",
        actual_score=score,
        actual_risk_tier=tier,
        actual_tool_calls=["search_clinical_calculators", "execute_clinical_calculator"] if tool_calls is None else tool_calls,
        chain_of_thought=["Identified risk factors", "Calling calculator"] if cot is None else cot,
        guardrail_passed=guardrail_passed,
    )


class TestAccuracy:
    def test_perfect_scores(self):
        golds = [_make_gold("c1", 2, RiskTier.MODERATE), _make_gold("c2", 5, RiskTier.HIGH)]
        results = [_make_result("c1", 2, RiskTier.MODERATE), _make_result("c2", 5, RiskTier.HIGH)]
        assert compute_accuracy(golds, results) == 1.0

    def test_zero_accuracy(self):
        golds = [_make_gold("c1", 2, RiskTier.MODERATE)]
        results = [_make_result("c1", 3, RiskTier.HIGH)]
        assert compute_accuracy(golds, results) == 0.0

    def test_partial_accuracy(self):
        golds = [_make_gold("c1", 2, RiskTier.MODERATE), _make_gold("c2", 5, RiskTier.HIGH)]
        results = [_make_result("c1", 2, RiskTier.MODERATE), _make_result("c2", 3, RiskTier.LOW)]
        assert compute_accuracy(golds, results) == 0.5

    def test_none_score_counts_as_wrong(self):
        golds = [_make_gold("c1", 2, RiskTier.MODERATE)]
        results = [_make_result("c1", None, None)]
        assert compute_accuracy(golds, results) == 0.0


class TestMAE:
    def test_perfect_mae(self):
        golds = [_make_gold("c1", 2, RiskTier.MODERATE)]
        results = [_make_result("c1", 2, RiskTier.MODERATE)]
        assert compute_mae(golds, results) == 0.0

    def test_mae_calculation(self):
        golds = [_make_gold("c1", 2, RiskTier.LOW), _make_gold("c2", 5, RiskTier.HIGH)]
        results = [_make_result("c1", 3, RiskTier.LOW), _make_result("c2", 3, RiskTier.HIGH)]
        # |2-3| + |5-3| = 1 + 2 = 3, MAE = 3/2 = 1.5
        assert compute_mae(golds, results) == 1.5

    def test_none_score_excluded(self):
        golds = [_make_gold("c1", 2, RiskTier.LOW), _make_gold("c2", 5, RiskTier.HIGH)]
        results = [_make_result("c1", None, None), _make_result("c2", 4, RiskTier.HIGH)]
        # Only c2: |5-4| = 1, MAE = 1/1 = 1.0
        assert compute_mae(golds, results) == 1.0


class TestKappa:
    def test_perfect_agreement(self):
        golds = [
            _make_gold("c1", 1, RiskTier.LOW),
            _make_gold("c2", 3, RiskTier.MODERATE),
            _make_gold("c3", 6, RiskTier.HIGH),
        ]
        results = [
            _make_result("c1", 1, RiskTier.LOW),
            _make_result("c2", 3, RiskTier.MODERATE),
            _make_result("c3", 6, RiskTier.HIGH),
        ]
        assert compute_kappa(golds, results) == 1.0

    def test_kappa_with_disagreement(self):
        golds = [
            _make_gold("c1", 1, RiskTier.LOW),
            _make_gold("c2", 3, RiskTier.LOW),
        ]
        results = [
            _make_result("c1", 1, RiskTier.LOW),
            _make_result("c2", 3, RiskTier.HIGH),
        ]
        kappa = compute_kappa(golds, results)
        assert -1.0 <= kappa <= 1.0


class TestSensitivitySpecificity:
    def test_perfect_classification(self):
        golds = [
            _make_gold("c1", 1, RiskTier.LOW),
            _make_gold("c2", 1, RiskTier.LOW),
            _make_gold("c3", 3, RiskTier.MODERATE),
            _make_gold("c4", 6, RiskTier.HIGH),
        ]
        results = [
            _make_result("c1", 1, RiskTier.LOW),
            _make_result("c2", 1, RiskTier.LOW),
            _make_result("c3", 3, RiskTier.MODERATE),
            _make_result("c4", 6, RiskTier.HIGH),
        ]
        ss = compute_sensitivity_specificity(golds, results)
        for tier in RiskTier:
            assert ss[tier]["sensitivity"] == 1.0
            assert ss[tier]["specificity"] == 1.0

    def test_returns_all_tiers(self):
        golds = [_make_gold("c1", 1, RiskTier.LOW)]
        results = [_make_result("c1", 1, RiskTier.LOW)]
        ss = compute_sensitivity_specificity(golds, results)
        assert set(ss.keys()) == {RiskTier.LOW, RiskTier.MODERATE, RiskTier.HIGH}


class TestProtocolAdherence:
    def test_all_used_tools(self):
        golds = [
            _make_gold("c1", 2, RiskTier.MODERATE),
            _make_gold("c2", 5, RiskTier.HIGH),
        ]
        results = [
            _make_result("c1", 2, RiskTier.MODERATE),
            _make_result("c2", 5, RiskTier.HIGH),
        ]
        assert compute_protocol_adherence(golds, results) == 1.0

    def test_missing_tool_call(self):
        golds = [_make_gold("c1", 2, RiskTier.MODERATE)]
        results = [_make_result("c1", 2, RiskTier.MODERATE, tool_calls=["search_clinical_calculators"])]
        assert compute_protocol_adherence(golds, results) == 0.0

    def test_no_tool_calls_at_all(self):
        golds = [_make_gold("c1", 2, RiskTier.MODERATE)]
        results = [_make_result("c1", 2, RiskTier.MODERATE, tool_calls=[])]
        assert compute_protocol_adherence(golds, results) == 0.0


class TestCotQuality:
    def test_good_cot(self):
        golds = [_make_gold("c1", 2, RiskTier.MODERATE)]
        results = [_make_result("c1", 2, RiskTier.MODERATE, cot=["Step 1", "Step 2"])]
        quality = compute_cot_quality(golds, results)
        assert 0.0 <= quality <= 1.0

    def test_empty_cot(self):
        golds = [_make_gold("c1", 2, RiskTier.MODERATE)]
        results = [_make_result("c1", 2, RiskTier.MODERATE, cot=[])]
        assert compute_cot_quality(golds, results) == 0.0


class TestComputeAllMetrics:
    def test_returns_all_metric_keys(self):
        golds = [_make_gold("c1", 2, RiskTier.MODERATE)]
        results = [_make_result("c1", 2, RiskTier.MODERATE)]
        metrics = compute_all_metrics(golds, results)
        assert "accuracy" in metrics
        assert "mae" in metrics
        assert "kappa" in metrics
        assert "sensitivity_specificity" in metrics
        assert "protocol_adherence" in metrics
        assert "cot_quality" in metrics
