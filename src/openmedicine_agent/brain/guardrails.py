"""Guardrails enforcing mandatory MCP tool use."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GuardrailResult:
    passed: bool
    violations: list[str]
    corrective_message: str | None = None


CORRECTIVE_PROMPT = (
    "You MUST use the available MCP tools to perform clinical calculations. "
    "Do NOT compute scores or risk stratifications from your own knowledge. "
    "Please search for the appropriate calculator using `search_clinical_calculators` "
    "and then use `execute_clinical_calculator` to get the result."
)


def check_tool_use(tool_calls_made: list[str]) -> GuardrailResult:
    """Verify that the agent called at least one MCP tool during the interaction.

    This is the core protocol adherence check — responses without tool-call
    traces are failures regardless of whether the answer is correct.
    """
    if not tool_calls_made:
        return GuardrailResult(
            passed=False,
            violations=["No MCP tools were called during this interaction"],
            corrective_message=CORRECTIVE_PROMPT,
        )
    return GuardrailResult(passed=True, violations=[])


def check_calculator_use(tool_calls_made: list[str]) -> GuardrailResult:
    """Verify that execute_clinical_calculator was called (not just search)."""
    calculator_tools = {"execute_clinical_calculator"}
    used_calculators = [t for t in tool_calls_made if t in calculator_tools]

    if not used_calculators:
        search_only = [t for t in tool_calls_made if "search" in t]
        if search_only:
            return GuardrailResult(
                passed=False,
                violations=["Searched for calculators but never executed one"],
                corrective_message=(
                    "You searched for calculators but did not execute any. "
                    "Please use `execute_clinical_calculator` with the appropriate parameters."
                ),
            )
    return GuardrailResult(passed=True, violations=[])


def run_all_guardrails(tool_calls_made: list[str]) -> GuardrailResult:
    """Run all guardrail checks and return combined result."""
    checks = [
        check_tool_use(tool_calls_made),
        check_calculator_use(tool_calls_made),
    ]

    all_violations = []
    corrective = None
    for check in checks:
        if not check.passed:
            all_violations.extend(check.violations)
            if check.corrective_message and corrective is None:
                corrective = check.corrective_message

    return GuardrailResult(
        passed=len(all_violations) == 0,
        violations=all_violations,
        corrective_message=corrective,
    )
