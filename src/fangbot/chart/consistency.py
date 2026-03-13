"""Chart inconsistency detection — rule-based checks on structured chart data."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

import re
from collections import defaultdict

from fangbot.chart.models import ChartFact, FactCategory, FactStatus, PatientChart


class InconsistencyType(str, Enum):
    DUPLICATE_VALUE = "duplicate_value"
    CONFLICTING_VALUE = "conflicting_value"
    IMPOSSIBLE_VALUE = "impossible_value"
    ALLERGY_VIOLATION = "allergy_violation"
    STATUS_CONFLICT = "status_conflict"
    COPY_FORWARD = "copy_forward"


class InconsistencySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, InconsistencySeverity):
            return NotImplemented
        order = {self.INFO: 0, self.WARNING: 1, self.CRITICAL: 2}
        return order[self] < order[other]

    def __le__(self, other: object) -> bool:
        if not isinstance(other, InconsistencySeverity):
            return NotImplemented
        order = {self.INFO: 0, self.WARNING: 1, self.CRITICAL: 2}
        return order[self] <= order[other]

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, InconsistencySeverity):
            return NotImplemented
        order = {self.INFO: 0, self.WARNING: 1, self.CRITICAL: 2}
        return order[self] > order[other]

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, InconsistencySeverity):
            return NotImplemented
        order = {self.INFO: 0, self.WARNING: 1, self.CRITICAL: 2}
        return order[self] >= order[other]


class Inconsistency(BaseModel):
    """A detected inconsistency between chart facts."""

    type: InconsistencyType
    severity: InconsistencySeverity
    description: str
    fact_a: ChartFact
    fact_b: ChartFact | None = None
    recommendation: str = ""


class ConsistencyReport(BaseModel):
    """Result of running all consistency checks on a patient chart."""

    inconsistencies: list[Inconsistency] = Field(default_factory=list)
    facts_checked: int = 0
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def is_clean(self) -> bool:
        """True if no inconsistencies were found."""
        return len(self.inconsistencies) == 0

    def by_severity(self, severity: InconsistencySeverity) -> list[Inconsistency]:
        """Return inconsistencies filtered by severity level."""
        return [i for i in self.inconsistencies if i.severity == severity]


def _extract_number(value: str) -> float | None:
    """Extract the first numeric value from a string like '72 bpm' or '37.2 C'."""
    match = re.search(r"-?\d+\.?\d*", value)
    if match:
        return float(match.group())
    return None


def _extract_bp(value: str) -> tuple[float, float] | None:
    """Extract systolic/diastolic from a BP string like '120/80 mmHg'."""
    match = re.search(r"(\d+)\s*/\s*(\d+)", value)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None


# Vital sign ranges: (min_valid, max_valid)
# Values outside these ranges are flagged as impossible.
_VITAL_RANGES: dict[str, tuple[float, float]] = {
    "hr": (0, 300),
    "heart rate": (0, 300),
    "spo2": (0, 100),
    "o2 sat": (0, 100),
    "oxygen saturation": (0, 100),
    "temp": (25, 45),
    "temperature": (25, 45),
    "rr": (0, 80),
    "respiratory rate": (0, 80),
}

# Names that represent blood pressure
_BP_NAMES: set[str] = {"bp", "blood pressure", "nibp"}


def check_impossible_vitals(chart: PatientChart) -> list[Inconsistency]:
    """Check for physiologically impossible vital sign values."""
    results: list[Inconsistency] = []
    vitals = chart.facts_by_category(FactCategory.VITAL)

    for fact in vitals:
        name_lower = fact.name.lower().strip()

        # Check blood pressure separately (systolic must exceed diastolic)
        if name_lower in _BP_NAMES:
            bp = _extract_bp(fact.value)
            if bp is not None:
                systolic, diastolic = bp
                if systolic <= diastolic:
                    results.append(
                        Inconsistency(
                            type=InconsistencyType.IMPOSSIBLE_VALUE,
                            severity=InconsistencySeverity.CRITICAL,
                            description=(
                                f"BP systolic ({systolic}) <= diastolic ({diastolic}): "
                                f"{fact.value}"
                            ),
                            fact_a=fact,
                            recommendation="Verify blood pressure reading — systolic must exceed diastolic",
                        )
                    )
                if systolic < 0 or diastolic < 0 or systolic > 350 or diastolic > 250:
                    results.append(
                        Inconsistency(
                            type=InconsistencyType.IMPOSSIBLE_VALUE,
                            severity=InconsistencySeverity.CRITICAL,
                            description=(
                                f"BP values out of physiological range: {fact.value}"
                            ),
                            fact_a=fact,
                            recommendation="Review blood pressure entry for data error",
                        )
                    )
            continue

        # Check numeric vitals against known ranges
        if name_lower in _VITAL_RANGES:
            num = _extract_number(fact.value)
            if num is None:
                continue
            low, high = _VITAL_RANGES[name_lower]
            if num <= low or num > high:
                results.append(
                    Inconsistency(
                        type=InconsistencyType.IMPOSSIBLE_VALUE,
                        severity=InconsistencySeverity.CRITICAL,
                        description=(
                            f"{fact.name} value {num} is outside physiological range "
                            f"({low}-{high}): {fact.value}"
                        ),
                        fact_a=fact,
                        recommendation=f"Review {fact.name} entry for data entry error",
                    )
                )

    return results


def check_duplicate_facts(chart: PatientChart) -> list[Inconsistency]:
    """Detect duplicate or conflicting facts with the same name and category.

    Facts with the same name at different timestamps are treated as a trend
    (legitimate clinical data), not a conflict. Only facts with the same
    timestamp (or no timestamp) are compared.
    """
    results: list[Inconsistency] = []

    # Group facts by (name_lower, category)
    groups: dict[tuple[str, FactCategory], list[ChartFact]] = defaultdict(list)
    for fact in chart.facts:
        key = (fact.name.lower().strip(), fact.category)
        groups[key].append(fact)

    for (_name, _cat), facts in groups.items():
        if len(facts) < 2:
            continue

        # Compare pairs, but only if they share a timestamp (or both lack one)
        for i, fa in enumerate(facts):
            for fb in facts[i + 1 :]:
                # If both have distinct timestamps, it's a trend — skip
                if (
                    fa.timestamp is not None
                    and fb.timestamp is not None
                    and fa.timestamp != fb.timestamp
                ):
                    continue

                if fa.value.strip().lower() == fb.value.strip().lower():
                    results.append(
                        Inconsistency(
                            type=InconsistencyType.DUPLICATE_VALUE,
                            severity=InconsistencySeverity.INFO,
                            description=(
                                f"Duplicate {fa.name}: '{fa.value}' appears in both "
                                f"'{fa.source}' and '{fb.source}'"
                            ),
                            fact_a=fa,
                            fact_b=fb,
                            recommendation="Verify if this is an intentional duplicate or copy-paste error",
                        )
                    )
                else:
                    results.append(
                        Inconsistency(
                            type=InconsistencyType.CONFLICTING_VALUE,
                            severity=InconsistencySeverity.WARNING,
                            description=(
                                f"Conflicting {fa.name}: '{fa.value}' ({fa.source}) "
                                f"vs '{fb.value}' ({fb.source})"
                            ),
                            fact_a=fa,
                            fact_b=fb,
                            recommendation=f"Clarify which {fa.name} value is correct",
                        )
                    )

    return results
