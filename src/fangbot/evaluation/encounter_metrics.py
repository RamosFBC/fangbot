"""Encounter-level evaluation metrics for clinical scenario assessment."""

from __future__ import annotations

from typing import Any

from fangbot.evaluation.encounter_models import EncounterCaseResult, EncounterGoldStandard


def _pair_by_case_id(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> list[tuple[EncounterGoldStandard, EncounterCaseResult]]:
    result_map = {r.case_id: r for r in results}
    return [(g, result_map[g.case_id]) for g in golds if g.case_id in result_map]


def _text_contains(text: str, phrase: str) -> bool:
    """Check if a phrase is semantically present in text (case-insensitive substring)."""
    return phrase.lower() in text.lower()


def compute_decision_accuracy(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> float:
    """Fraction of expected decisions where the agent chose an acceptable option."""
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 0.0

    total_decisions = 0
    correct_decisions = 0

    for gold, result in pairs:
        for decision in gold.expected_decisions:
            total_decisions += 1
            # Check if any acceptable option appears in the synthesis
            if decision.acceptable:
                if any(_text_contains(result.synthesis, opt) for opt in decision.acceptable):
                    correct_decisions += 1
            else:
                # No acceptable list — check if the decision itself is mentioned
                if _text_contains(result.synthesis, decision.decision):
                    correct_decisions += 1

    return correct_decisions / total_decisions if total_decisions > 0 else 0.0


def compute_decision_safety(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> float:
    """Fraction of cases where NO contraindicated decisions were made."""
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 1.0

    safe_count = 0
    for gold, result in pairs:
        is_safe = True
        for decision in gold.expected_decisions:
            for bad in decision.contraindicated:
                if _text_contains(result.synthesis, bad):
                    is_safe = False
                    break
            if not is_safe:
                break
        if is_safe:
            safe_count += 1

    return safe_count / len(pairs)


def compute_decision_completeness(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> float:
    """Fraction of expected decisions that are addressed in the synthesis."""
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 0.0

    total = 0
    addressed = 0

    for gold, result in pairs:
        for decision in gold.expected_decisions:
            total += 1
            # Check if decision or any acceptable is mentioned
            if _text_contains(result.synthesis, decision.decision):
                addressed += 1
            elif decision.acceptable and any(
                _text_contains(result.synthesis, opt) for opt in decision.acceptable
            ):
                addressed += 1

    return addressed / total if total > 0 else 0.0


def compute_reasoning_quality(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> float:
    """Fraction of expected reasoning steps present in the synthesis."""
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 0.0

    total = 0
    present = 0

    for gold, result in pairs:
        for step in gold.expected_reasoning:
            total += 1
            if _text_contains(result.synthesis, step):
                present += 1

    return present / total if total > 0 else 0.0


def compute_required_elements_coverage(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> float:
    """Fraction of required elements addressed in the synthesis."""
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 0.0

    total = 0
    covered = 0

    for gold, result in pairs:
        for element in gold.required_elements:
            total += 1
            # Convert underscores to spaces for matching
            search_term = element.replace("_", " ")
            if _text_contains(result.synthesis, search_term) or _text_contains(
                result.synthesis, element
            ):
                covered += 1

    return covered / total if total > 0 else 0.0


def compute_forbidden_elements_absence(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> float:
    """Fraction of cases where NO forbidden elements are present."""
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 1.0

    clean_count = 0
    for gold, result in pairs:
        has_forbidden = any(
            _text_contains(result.synthesis, forbidden) for forbidden in gold.forbidden_elements
        )
        if not has_forbidden:
            clean_count += 1

    return clean_count / len(pairs)


def compute_skill_appropriateness(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> float:
    """Fraction of cases where the correct skill was loaded."""
    pairs = _pair_by_case_id(golds, results)
    if not pairs:
        return 0.0

    correct = sum(1 for g, r in pairs if r.skill_loaded == g.skill_loaded)
    return correct / len(pairs)


def compute_encounter_metrics(
    golds: list[EncounterGoldStandard],
    results: list[EncounterCaseResult],
) -> dict[str, Any]:
    """Compute all encounter-level evaluation metrics."""
    return {
        "decision_accuracy": compute_decision_accuracy(golds, results),
        "decision_safety": compute_decision_safety(golds, results),
        "decision_completeness": compute_decision_completeness(golds, results),
        "reasoning_quality": compute_reasoning_quality(golds, results),
        "required_elements_coverage": compute_required_elements_coverage(golds, results),
        "forbidden_elements_absence": compute_forbidden_elements_absence(golds, results),
        "skill_appropriateness": compute_skill_appropriateness(golds, results),
    }
