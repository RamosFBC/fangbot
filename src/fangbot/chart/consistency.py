"""Chart inconsistency detection — rule-based checks on structured chart data."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

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
