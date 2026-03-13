"""Evidence retrieval orchestrator — structured guideline extraction and conflict detection."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EvidenceSource(str, Enum):
    """Type of evidence source."""

    GUIDELINE = "guideline"
    LANDMARK_TRIAL = "landmark_trial"
    HOSPITAL_PROTOCOL = "hospital_protocol"


class EvidenceStrength(str, Enum):
    """Strength of evidence / recommendation grade."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    EXPERT_OPINION = "expert_opinion"


class EvidenceCitation(BaseModel):
    """A single evidence citation extracted from a tool result."""

    recommendation: str
    source: EvidenceSource
    doi: str | None = None
    pmid: str | None = None
    strength: EvidenceStrength | None = None
    guideline_id: str | None = None
    section: str | None = None
    organization: str | None = None


class GuidelineReference(BaseModel):
    """A guideline that was consulted during the session."""

    guideline_id: str
    title: str
    organization: str | None = None
    year: int | None = None
    sections_consulted: list[str] = []


class EvidenceConflict(BaseModel):
    """A detected conflict between two or more evidence sources."""

    topic: str
    citations: list[EvidenceCitation] = Field(min_length=2)
    description: str
