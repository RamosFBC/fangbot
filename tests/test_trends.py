"""Tests for trend detection engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fangbot.chart.models import ChartFact, FactCategory, FactStatus, PatientChart
from fangbot.chart.trends import (
    Trend,
    TrendDirection,
    TrendPoint,
    detect_trends,
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


class TestDetectTrends:
    def test_rising_creatinine(self):
        """Creatinine 1.2 -> 1.8 -> 2.1 over 72h should be RISING."""
        facts = [
            _lab_fact("Creatinine", "1.2 mg/dL", 0),
            _lab_fact("Creatinine", "1.8 mg/dL", 24),
            _lab_fact("Creatinine", "2.1 mg/dL", 72),
        ]
        chart = _chart_with_facts(facts)
        trends = detect_trends(chart)

        assert len(trends) == 1
        t = trends[0]
        assert t.fact_name == "Creatinine"
        assert t.direction == TrendDirection.RISING
        assert len(t.points) == 3
        assert t.points[0].value == pytest.approx(1.2)
        assert t.points[1].value == pytest.approx(1.8)
        assert t.points[2].value == pytest.approx(2.1)
        assert t.rate_of_change is not None
        assert t.rate_of_change > 0
        assert "1.2" in t.summary and "2.1" in t.summary

    def test_falling_hemoglobin(self):
        """Hemoglobin 12.0 -> 10.5 -> 8.2 should be FALLING."""
        facts = [
            _lab_fact("Hemoglobin", "12.0 g/dL", 0),
            _lab_fact("Hemoglobin", "10.5 g/dL", 12),
            _lab_fact("Hemoglobin", "8.2 g/dL", 24),
        ]
        chart = _chart_with_facts(facts)
        trends = detect_trends(chart)

        assert len(trends) == 1
        assert trends[0].direction == TrendDirection.FALLING
        assert trends[0].rate_of_change is not None
        assert trends[0].rate_of_change < 0

    def test_stable_sodium(self):
        """Sodium 139 -> 140 -> 139 should be STABLE."""
        facts = [
            _lab_fact("Sodium", "139 mEq/L", 0),
            _lab_fact("Sodium", "140 mEq/L", 24),
            _lab_fact("Sodium", "139 mEq/L", 48),
        ]
        chart = _chart_with_facts(facts)
        trends = detect_trends(chart)

        assert len(trends) == 1
        assert trends[0].direction == TrendDirection.STABLE

    def test_single_value_insufficient_data(self):
        """A single measurement cannot determine a trend."""
        facts = [_lab_fact("Creatinine", "1.8 mg/dL", 0)]
        chart = _chart_with_facts(facts)
        trends = detect_trends(chart)

        assert len(trends) == 1
        assert trends[0].direction == TrendDirection.INSUFFICIENT_DATA

    def test_non_numeric_values_skipped(self):
        """Facts with non-numeric values should not produce trends."""
        facts = [
            ChartFact(
                name="Blood Culture",
                value="pending",
                category=FactCategory.CULTURE,
                source="Micro lab",
                timestamp=BASE_TIME,
                status=FactStatus.ACTIVE,
            ),
        ]
        chart = _chart_with_facts(facts)
        trends = detect_trends(chart)
        assert len(trends) == 0

    def test_multiple_labs_detected_separately(self):
        """Multiple different lab names each get their own trend."""
        facts = [
            _lab_fact("Creatinine", "1.2 mg/dL", 0),
            _lab_fact("Creatinine", "1.8 mg/dL", 24),
            _lab_fact("BUN", "20 mg/dL", 0),
            _lab_fact("BUN", "35 mg/dL", 24),
        ]
        chart = _chart_with_facts(facts)
        trends = detect_trends(chart)

        assert len(trends) == 2
        names = {t.fact_name for t in trends}
        assert names == {"Creatinine", "BUN"}

    def test_vitals_also_detected(self):
        """Trend detection works for vitals, not just labs."""
        facts = [
            _vital_fact("Heart Rate", "72 bpm", 0),
            _vital_fact("Heart Rate", "88 bpm", 4),
            _vital_fact("Heart Rate", "110 bpm", 8),
        ]
        chart = _chart_with_facts(facts)
        trends = detect_trends(chart)

        assert len(trends) == 1
        assert trends[0].direction == TrendDirection.RISING
        assert trends[0].category == FactCategory.VITAL

    def test_facts_without_timestamps_excluded(self):
        """Facts missing timestamps are excluded from trend analysis."""
        facts = [
            ChartFact(
                name="Creatinine",
                value="1.8 mg/dL",
                category=FactCategory.LAB,
                source="BMP",
            ),
            ChartFact(
                name="Creatinine",
                value="2.1 mg/dL",
                category=FactCategory.LAB,
                source="BMP",
            ),
        ]
        chart = _chart_with_facts(facts)
        trends = detect_trends(chart)
        assert len(trends) == 0

    def test_stable_threshold_parameter(self):
        """Custom stable_threshold changes what counts as stable."""
        facts = [
            _lab_fact("Sodium", "139 mEq/L", 0),
            _lab_fact("Sodium", "142 mEq/L", 24),
        ]
        chart = _chart_with_facts(facts)

        # With high threshold, 139->142 is stable
        trends_strict = detect_trends(chart, stable_threshold=0.05)
        assert trends_strict[0].direction == TrendDirection.STABLE

        # With low threshold, 139->142 is rising
        trends_loose = detect_trends(chart, stable_threshold=0.0005)
        assert trends_loose[0].direction == TrendDirection.RISING
