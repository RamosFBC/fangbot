"""Uncertainty calibration: confidence levels and structured assessment."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, computed_field


class ConfidenceLevel(str, Enum):
    """Clinical confidence levels for uncertainty calibration."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INSUFFICIENT_DATA = "insufficient_data"

    @property
    def numeric_value(self) -> float:
        """Numeric value for quantitative comparisons."""
        return _NUMERIC_VALUES[self]


_NUMERIC_VALUES: dict[ConfidenceLevel, float] = {
    ConfidenceLevel.HIGH: 1.0,
    ConfidenceLevel.MODERATE: 0.7,
    ConfidenceLevel.LOW: 0.4,
    ConfidenceLevel.INSUFFICIENT_DATA: 0.0,
}


class UncertaintyAssessment(BaseModel):
    """Structured uncertainty assessment extracted from agent synthesis."""

    confidence: ConfidenceLevel
    reasoning: str
    missing_data: list[str] = []
    contradictions: list[str] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def escalation_recommended(self) -> bool:
        """Escalation is recommended for LOW or INSUFFICIENT_DATA confidence."""
        return self.confidence in (
            ConfidenceLevel.LOW,
            ConfidenceLevel.INSUFFICIENT_DATA,
        )
