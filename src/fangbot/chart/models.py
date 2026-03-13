"""Pydantic models for structured chart data with provenance tracking."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class FactCategory(str, Enum):
    LAB = "lab"
    VITAL = "vital"
    MEDICATION = "medication"
    DIAGNOSIS = "diagnosis"
    PROCEDURE = "procedure"
    ALLERGY = "allergy"
    IMAGING = "imaging"
    CULTURE = "culture"


class FactStatus(str, Enum):
    ACTIVE = "active"
    HISTORICAL = "historical"
    RESOLVED = "resolved"


class ChartFact(BaseModel):
    """A single fact extracted from the chart with provenance."""

    name: str
    value: str
    category: FactCategory
    source: str
    timestamp: datetime | None = None
    source_location: str | None = None
    status: FactStatus | None = None
    confidence: float = Field(default=1.0)

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v


class PatientChart(BaseModel):
    """Structured patient chart with provenance-tracked facts."""

    facts: list[ChartFact] = Field(default_factory=list)
    raw_text: str = Field(min_length=1)
    parse_warnings: list[str] = Field(default_factory=list)

    def facts_by_category(self, category: FactCategory) -> list[ChartFact]:
        """Return all facts of a given category."""
        return [f for f in self.facts if f.category == category]

    def active_facts(self) -> list[ChartFact]:
        """Return facts with status == ACTIVE."""
        return [f for f in self.facts if f.status == FactStatus.ACTIVE]

    def latest_value(self, name: str) -> ChartFact | None:
        """Return the most recent fact with the given name.

        Uses timestamp if available, otherwise returns the last occurrence.
        """
        matching = [f for f in self.facts if f.name == name]
        if not matching:
            return None
        with_ts = [f for f in matching if f.timestamp is not None]
        if with_ts:
            return max(with_ts, key=lambda f: f.timestamp)  # type: ignore[arg-type]
        return matching[-1]
