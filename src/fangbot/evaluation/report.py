"""Generate Markdown evaluation reports comparing providers/models."""

from __future__ import annotations

from typing import Any

from fangbot.evaluation.metrics import compute_all_metrics
from fangbot.evaluation.models import CaseResult, GoldStandardCase, RiskTier


def generate_report(
    study_name: str,
    golds: list[GoldStandardCase],
    results_by_provider: dict[str, list[CaseResult]],
) -> str:
    """Generate a Markdown comparison report."""
    lines: list[str] = []
    lines.append(f"# {study_name} Evaluation Report\n")
    lines.append(f"**Cases:** {len(golds)}\n")
    lines.append(f"**Providers:** {len(results_by_provider)}\n")
    lines.append("---\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Provider/Model | Accuracy | MAE | Kappa | Protocol Adherence | CoT Quality |")
    lines.append("|---|---|---|---|---|---|")

    all_metrics: dict[str, dict[str, Any]] = {}
    for provider_key, results in results_by_provider.items():
        metrics = compute_all_metrics(golds, results)
        all_metrics[provider_key] = metrics
        lines.append(
            f"| {provider_key} "
            f"| {metrics['accuracy']:.2%} "
            f"| {metrics['mae']:.2f} "
            f"| {metrics['kappa']:.3f} "
            f"| {metrics['protocol_adherence']:.2%} "
            f"| {metrics['cot_quality']:.2%} |"
        )
    lines.append("")

    # Sensitivity/Specificity per tier
    lines.append("## Sensitivity / Specificity by Risk Tier\n")
    for provider_key, metrics in all_metrics.items():
        lines.append(f"### {provider_key}\n")
        lines.append("| Tier | Sensitivity | Specificity |")
        lines.append("|---|---|---|")
        ss = metrics["sensitivity_specificity"]
        for tier in RiskTier:
            s = ss[tier]["sensitivity"]
            sp = ss[tier]["specificity"]
            lines.append(f"| {tier.value} | {s:.2%} | {sp:.2%} |")
        lines.append("")

    # Per-case detail
    lines.append("## Per-Case Results\n")
    for provider_key, results in results_by_provider.items():
        lines.append(f"### {provider_key}\n")
        lines.append("| Case | Expected Score | Actual Score | Expected Tier | Actual Tier | Match | Tools Used |")
        lines.append("|---|---|---|---|---|---|---|")
        gold_map = {g.case_id: g for g in golds}
        for result in sorted(results, key=lambda r: r.case_id):
            gold = gold_map.get(result.case_id)
            if not gold:
                continue
            match = "Y" if (result.actual_score == gold.expected_score) else "N"
            actual_tier = result.actual_risk_tier.value if result.actual_risk_tier else "N/A"
            tools = ", ".join(result.actual_tool_calls) if result.actual_tool_calls else "none"
            lines.append(
                f"| {result.case_id} "
                f"| {gold.expected_score} "
                f"| {result.actual_score if result.actual_score is not None else 'N/A'} "
                f"| {gold.expected_risk_tier.value} "
                f"| {actual_tier} "
                f"| {match} "
                f"| {tools} |"
            )
        lines.append("")

    return "\n".join(lines)
