"""Temporal abstraction engine — classifies findings as new/chronic, builds timelines."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from fangbot.chart.models import ChartFact, FactCategory, PatientChart


class TemporalClassification(str, Enum):
    NEW = "new"
    CHRONIC = "chronic"
    WORSENING = "worsening"
    IMPROVING = "improving"
    RESOLVED = "resolved"


class TemporalFact(BaseModel):
    """A chart fact annotated with its temporal classification."""

    fact: ChartFact
    classification: TemporalClassification
    rationale: str = ""


class TimelineEntry(BaseModel):
    """A single entry in a problem-oriented timeline."""

    timestamp: datetime
    description: str
    category: FactCategory
    fact_name: str
    classification: TemporalClassification | None = None


class PatientTimeline(BaseModel):
    """Problem-oriented timeline with entries sorted chronologically."""

    entries: list[TimelineEntry] = Field(default_factory=list)
    start: datetime | None = None
    end: datetime | None = None
    summary: str = ""
