"""Evaluation metrics for clinical agent benchmark studies."""

from __future__ import annotations

from typing import Any

from fangbot.evaluation.models import CaseResult, GoldStandardCase, RiskTier


def _pair_by_case_id(
    golds: list[GoldStandardCase],
    results: list[CaseResult],
) -> list[tuple[GoldStandardCase, CaseResult]]:
    """Pair gold standard cases with results by case_id."""
    result_map = {r.case_id: r for r in results}
    pairs = []
    for gold in golds:
        if gold.case_id in result_map:
            pairs.append((gold, result_map[gold.case_id]))
    return pairs


def compute_accuracy(
    golds: list[GoldStandardCase],
    results: list[CaseResult],
) -> float:
    """Exact-match accuracy: fraction of cases where score matches exactly."""
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 0.0
    correct = sum(
        1 for g, r in pairs if r.actual_score is not None and r.actual_score == g.expected_score
    )
    return correct / len(pairs)


def compute_mae(
    golds: list[GoldStandardCase],
    results: list[CaseResult],
) -> float:
    """Mean Absolute Error between expected and actual scores.

    Cases where actual_score is None are excluded.
    """
    pairs = _pair_by_case_id(golds, results)
    scored = [(g, r) for g, r in pairs if r.actual_score is not None]
    if not scored:
        return float("inf")
    total_error = sum(abs(g.expected_score - r.actual_score) for g, r in scored)
    return total_error / len(scored)


def compute_kappa(
    golds: list[GoldStandardCase],
    results: list[CaseResult],
) -> float:
    """Cohen's Kappa for risk tier agreement.

    Measures inter-rater reliability between gold standard and agent.
    """
    pairs = _pair_by_case_id(golds, results)
    rated = [(g, r) for g, r in pairs if r.actual_risk_tier is not None]
    if not rated:
        return 0.0

    n = len(rated)
    tiers = list(RiskTier)

    # Build confusion matrix
    observed_agreement = sum(
        1 for g, r in rated if g.expected_risk_tier == r.actual_risk_tier
    )
    p_o = observed_agreement / n

    # Expected agreement by chance
    p_e = 0.0
    for tier in tiers:
        gold_count = sum(1 for g, _ in rated if g.expected_risk_tier == tier)
        result_count = sum(1 for _, r in rated if r.actual_risk_tier == tier)
        p_e += (gold_count / n) * (result_count / n)

    if p_e == 1.0:
        return 1.0  # Perfect agreement edge case

    return (p_o - p_e) / (1.0 - p_e)


def compute_sensitivity_specificity(
    golds: list[GoldStandardCase],
    results: list[CaseResult],
) -> dict[RiskTier, dict[str, float]]:
    """Per-tier sensitivity and specificity.

    For each tier, treats it as binary: "is this tier" vs. "is not this tier".
    """
    pairs = _pair_by_case_id(golds, results)
    rated = [(g, r) for g, r in pairs if r.actual_risk_tier is not None]

    output: dict[RiskTier, dict[str, float]] = {}
    for tier in RiskTier:
        tp = sum(1 for g, r in rated if g.expected_risk_tier == tier and r.actual_risk_tier == tier)
        fn = sum(1 for g, r in rated if g.expected_risk_tier == tier and r.actual_risk_tier != tier)
        fp = sum(1 for g, r in rated if g.expected_risk_tier != tier and r.actual_risk_tier == tier)
        tn = sum(1 for g, r in rated if g.expected_risk_tier != tier and r.actual_risk_tier != tier)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        output[tier] = {"sensitivity": sensitivity, "specificity": specificity}

    return output


def compute_protocol_adherence(
    golds: list[GoldStandardCase],
    results: list[CaseResult],
) -> float:
    """Fraction of cases where all expected tool calls were made."""
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 0.0

    adherent = 0
    for gold, result in pairs:
        expected_tools = {tc.tool_name for tc in gold.expected_tool_calls}
        actual_tools = set(result.actual_tool_calls)
        if expected_tools.issubset(actual_tools):
            adherent += 1

    return adherent / len(pairs)


def compute_cot_quality(
    golds: list[GoldStandardCase],
    results: list[CaseResult],
) -> float:
    """Chain-of-thought quality score.

    Simple heuristic: fraction of cases with non-empty chain of thought.
    Future: LLM-based quality assessment.
    """
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 0.0

    with_cot = sum(1 for _, r in pairs if len(r.chain_of_thought) > 0)
    return with_cot / len(pairs)


def compute_all_metrics(
    golds: list[GoldStandardCase],
    results: list[CaseResult],
) -> dict[str, Any]:
    """Compute all evaluation metrics."""
    return {
        "accuracy": compute_accuracy(golds, results),
        "mae": compute_mae(golds, results),
        "kappa": compute_kappa(golds, results),
        "sensitivity_specificity": compute_sensitivity_specificity(golds, results),
        "protocol_adherence": compute_protocol_adherence(golds, results),
        "cot_quality": compute_cot_quality(golds, results),
    }
