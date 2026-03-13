"""Tests for trend detection engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fangbot.chart.models import ChartFact, FactCategory, FactStatus, PatientChart
from fangbot.chart.trends import (
    Trend,
    TrendDirection,
    TrendPoint,
)


# -- Helpers --

BASE_TIME = datetime(2026, 3, 7, 8, 0, tzinfo=timezone.utc)


def _lab_fact(name: str, value: str, hours_offset: float) -> ChartFact:
    """Create a lab fact at BASE_TIME + hours_offset."""
    return ChartFact(
        name=name,
        value=value,
        category=FactCategory.LAB,
        source=f"BMP {hours_offset}h",
        timestamp=BASE_TIME + timedelta(hours=hours_offset),
        status=FactStatus.ACTIVE,
    )


def _vital_fact(name: str, value: str, hours_offset: float) -> ChartFact:
    """Create a vital fact at BASE_TIME + hours_offset."""
    return ChartFact(
        name=name,
        value=value,
        category=FactCategory.VITAL,
        source=f"Vitals {hours_offset}h",
        timestamp=BASE_TIME + timedelta(hours=hours_offset),
        status=FactStatus.ACTIVE,
    )


def _chart_with_facts(facts: list[ChartFact]) -> PatientChart:
    return PatientChart(facts=facts, raw_text="synthetic test chart")


# -- Model tests --


class TestTrendDirection:
    def test_enum_values(self):
        assert TrendDirection.RISING == "rising"
        assert TrendDirection.FALLING == "falling"
        assert TrendDirection.STABLE == "stable"
        assert TrendDirection.INSUFFICIENT_DATA == "insufficient_data"


class TestTrendPoint:
    def test_creation(self):
        ts = BASE_TIME
        pt = TrendPoint(timestamp=ts, value=1.8)
        assert pt.timestamp == ts
        assert pt.value == 1.8


class TestTrendModel:
    def test_trend_creation(self):
        points = [
            TrendPoint(timestamp=BASE_TIME, value=1.2),
            TrendPoint(timestamp=BASE_TIME + timedelta(hours=24), value=1.8),
        ]
        trend = Trend(
            fact_name="Creatinine",
            category=FactCategory.LAB,
            direction=TrendDirection.RISING,
            points=points,
            rate_of_change=0.025,
            summary="Creatinine rising over 24h: 1.2 -> 1.8",
        )
        assert trend.fact_name == "Creatinine"
        assert trend.direction == TrendDirection.RISING
        assert len(trend.points) == 2
        assert trend.rate_of_change == pytest.approx(0.025)
