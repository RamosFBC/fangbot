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


def check_allergy_medication_conflict(chart: PatientChart) -> list[Inconsistency]:
    """Detect medications that match a documented allergy.

    Uses case-insensitive substring matching: if the allergy name appears
    within the medication name, or vice versa, it's flagged. Cross-reactivity
    (e.g., penicillin allergy + amoxicillin) is NOT handled — that requires
    a drug ontology and is a future enhancement.
    """
    results: list[Inconsistency] = []
    allergies = chart.facts_by_category(FactCategory.ALLERGY)
    medications = chart.facts_by_category(FactCategory.MEDICATION)

    for allergy in allergies:
        allergy_name = allergy.name.lower().strip()
        for med in medications:
            med_name = med.name.lower().strip()

            # Check if allergy name is a substring of med name or vice versa
            if allergy_name in med_name or med_name in allergy_name:
                # Resolved allergy gets WARNING, active/unknown gets CRITICAL
                if allergy.status == FactStatus.RESOLVED:
                    severity = InconsistencySeverity.WARNING
                else:
                    severity = InconsistencySeverity.CRITICAL

                results.append(
                    Inconsistency(
                        type=InconsistencyType.ALLERGY_VIOLATION,
                        severity=severity,
                        description=(
                            f"Allergy to '{allergy.name}' ({allergy.value}) conflicts "
                            f"with active medication '{med.name}' ({med.value})"
                        ),
                        fact_a=allergy,
                        fact_b=med,
                        recommendation=(
                            f"Verify allergy status and medication safety — "
                            f"allergy '{allergy.name}' documented with reaction: {allergy.value}"
                        ),
                    )
                )

    return results


def check_status_conflicts(chart: PatientChart) -> list[Inconsistency]:
    """Detect facts with the same name and category but conflicting statuses.

    For example, a diagnosis listed as both ACTIVE and RESOLVED indicates
    a documentation error that needs resolution.
    """
    results: list[Inconsistency] = []

    # Only check facts that have a status set
    facts_with_status = [f for f in chart.facts if f.status is not None]

    # Group by (name_lower, category)
    groups: dict[tuple[str, FactCategory], list[ChartFact]] = defaultdict(list)
    for fact in facts_with_status:
        key = (fact.name.lower().strip(), fact.category)
        groups[key].append(fact)

    for (_name, _cat), facts in groups.items():
        if len(facts) < 2:
            continue

        statuses = {f.status for f in facts}
        if len(statuses) > 1:
            # Find the conflicting pair
            by_status: dict[FactStatus, ChartFact] = {}
            for f in facts:
                assert f.status is not None  # guaranteed by filter above
                if f.status not in by_status:
                    by_status[f.status] = f

            status_list = list(by_status.keys())
            for i, s1 in enumerate(status_list):
                for s2 in status_list[i + 1 :]:
                    results.append(
                        Inconsistency(
                            type=InconsistencyType.STATUS_CONFLICT,
                            severity=InconsistencySeverity.WARNING,
                            description=(
                                f"'{by_status[s1].name}' has conflicting statuses: "
                                f"{s1.value} ({by_status[s1].source}) vs "
                                f"{s2.value} ({by_status[s2].source})"
                            ),
                            fact_a=by_status[s1],
                            fact_b=by_status[s2],
                            recommendation=(
                                f"Clarify current status of {by_status[s1].name} — "
                                f"is it {s1.value} or {s2.value}?"
                            ),
                        )
                    )

    return results


def check_copy_forward(chart: PatientChart) -> list[Inconsistency]:
    """Detect likely copy-forward errors: identical values across 3+ timestamps.

    Copy-forward is a common EHR error where a clinician copies a previous
    note without updating it. We flag when the same fact name has the exact
    same value at 3 or more distinct timestamps.
    """
    results: list[Inconsistency] = []

    # Group facts by (name_lower, category), only those with timestamps
    groups: dict[tuple[str, FactCategory], list[ChartFact]] = defaultdict(list)
    for fact in chart.facts:
        if fact.timestamp is not None:
            key = (fact.name.lower().strip(), fact.category)
            groups[key].append(fact)

    for (_name, _cat), facts in groups.items():
        if len(facts) < 3:
            continue

        # Sort by timestamp
        sorted_facts = sorted(facts, key=lambda f: f.timestamp)  # type: ignore[arg-type]

        # Group by value
        by_value: dict[str, list[ChartFact]] = defaultdict(list)
        for f in sorted_facts:
            by_value[f.value.strip()].append(f)

        for value, occurrences in by_value.items():
            # Only flag if 3+ distinct timestamps have the same value
            distinct_timestamps = {f.timestamp for f in occurrences}
            if len(distinct_timestamps) >= 3:
                results.append(
                    Inconsistency(
                        type=InconsistencyType.COPY_FORWARD,
                        severity=InconsistencySeverity.WARNING,
                        description=(
                            f"Possible copy-forward: '{occurrences[0].name}' has identical "
                            f"value '{value}' across {len(distinct_timestamps)} dates "
                            f"({occurrences[0].source} through {occurrences[-1].source})"
                        ),
                        fact_a=occurrences[0],
                        fact_b=occurrences[-1],
                        recommendation=(
                            f"Review whether '{occurrences[0].name}' was actually "
                            f"reassessed on each date or copy-forwarded"
                        ),
                    )
                )

    return results


def run_all_checks(chart: PatientChart) -> ConsistencyReport:
    """Run all consistency checks and return a consolidated report."""
    all_inconsistencies: list[Inconsistency] = []

    all_inconsistencies.extend(check_impossible_vitals(chart))
    all_inconsistencies.extend(check_duplicate_facts(chart))
    all_inconsistencies.extend(check_allergy_medication_conflict(chart))
    all_inconsistencies.extend(check_status_conflicts(chart))
    all_inconsistencies.extend(check_copy_forward(chart))

    return ConsistencyReport(
        inconsistencies=all_inconsistencies,
        facts_checked=len(chart.facts),
    )
