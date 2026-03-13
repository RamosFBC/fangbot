"""Uncertainty calibration: confidence levels and structured assessment."""

from __future__ import annotations

import logging
import re
from enum import Enum

from pydantic import BaseModel, computed_field

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Parser: extract UncertaintyAssessment from synthesis text
# ---------------------------------------------------------------------------

_UNCERTAINTY_BLOCK_RE = re.compile(
    r"---\s*\n"
    r"Confidence:\s*(?P<confidence>\S+)\s*\n"
    r"Reasoning:\s*(?P<reasoning>.+?)\s*\n"
    r"Missing data:\s*(?P<missing>.+?)\s*\n"
    r"Contradictions:\s*(?P<contradictions>.+?)\s*\n"
    r"---",
    re.IGNORECASE,
)


def _parse_list_field(raw: str) -> list[str]:
    """Parse a semicolon-separated list, filtering 'None' variants."""
    items = [item.strip() for item in raw.split(";")]
    return [
        item
        for item in items
        if item and item.lower() not in ("none", "n/a", "none reported")
    ]


def parse_uncertainty_assessment(text: str) -> UncertaintyAssessment | None:
    """Extract an UncertaintyAssessment from a synthesis text block."""
    match = _UNCERTAINTY_BLOCK_RE.search(text)
    if not match:
        return None

    confidence_raw = match.group("confidence").strip().upper()
    try:
        confidence = ConfidenceLevel(confidence_raw.lower())
    except ValueError:
        logger.warning(f"Unknown confidence level: {confidence_raw}")
        return None

    return UncertaintyAssessment(
        confidence=confidence,
        reasoning=match.group("reasoning").strip(),
        missing_data=_parse_list_field(match.group("missing")),
        contradictions=_parse_list_field(match.group("contradictions")),
    )


def strip_uncertainty_block(text: str) -> str:
    """Remove the uncertainty block from synthesis text."""
    stripped = _UNCERTAINTY_BLOCK_RE.sub("", text).strip()
    return stripped
