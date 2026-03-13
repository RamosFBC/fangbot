"""Trend detection engine — identifies rising/falling/stable patterns in numeric series."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from fangbot.chart.models import FactCategory, PatientChart


class TrendDirection(str, Enum):
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    INSUFFICIENT_DATA = "insufficient_data"


class TrendPoint(BaseModel):
    """A single data point in a trend series."""

    timestamp: datetime
    value: float


class Trend(BaseModel):
    """A detected trend for a named fact across time."""

    fact_name: str
    category: FactCategory
    direction: TrendDirection
    points: list[TrendPoint] = Field(min_length=1)
    rate_of_change: float | None = None
    summary: str = ""
