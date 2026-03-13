"""Tests for temporal abstraction engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fangbot.chart.models import ChartFact, FactCategory, FactStatus, PatientChart
from fangbot.chart.temporal import (
    TemporalClassification,
    TemporalFact,
    TimelineEntry,
    PatientTimeline,
)


BASE_TIME = datetime(2026, 3, 7, 8, 0, tzinfo=timezone.utc)


def _fact(
    name: str,
    value: str,
    category: FactCategory,
    hours_offset: float,
    status: FactStatus = FactStatus.ACTIVE,
) -> ChartFact:
    return ChartFact(
        name=name,
        value=value,
        category=category,
        source=f"Chart note {hours_offset}h",
        timestamp=BASE_TIME + timedelta(hours=hours_offset),
        status=status,
    )


def _chart_with_facts(facts: list[ChartFact]) -> PatientChart:
    return PatientChart(facts=facts, raw_text="synthetic test chart")


# -- Model tests --


class TestTemporalClassification:
    def test_enum_values(self):
        assert TemporalClassification.NEW == "new"
        assert TemporalClassification.CHRONIC == "chronic"
        assert TemporalClassification.WORSENING == "worsening"
        assert TemporalClassification.IMPROVING == "improving"
        assert TemporalClassification.RESOLVED == "resolved"


class TestTemporalFact:
    def test_creation(self):
        fact = _fact("O2 requirement", "2L NC", FactCategory.VITAL, 0)
        tf = TemporalFact(
            fact=fact,
            classification=TemporalClassification.NEW,
            rationale="First occurrence in chart, no prior history",
        )
        assert tf.classification == TemporalClassification.NEW
        assert tf.fact.name == "O2 requirement"
        assert "First occurrence" in tf.rationale


class TestTimelineEntry:
    def test_creation(self):
        entry = TimelineEntry(
            timestamp=BASE_TIME,
            description="Creatinine 1.2 mg/dL",
            category=FactCategory.LAB,
            fact_name="Creatinine",
        )
        assert entry.timestamp == BASE_TIME
        assert entry.category == FactCategory.LAB


class TestPatientTimeline:
    def test_creation(self):
        entries = [
            TimelineEntry(
                timestamp=BASE_TIME,
                description="Creatinine 1.2",
                category=FactCategory.LAB,
                fact_name="Creatinine",
            ),
            TimelineEntry(
                timestamp=BASE_TIME + timedelta(hours=24),
                description="Creatinine 1.8",
                category=FactCategory.LAB,
                fact_name="Creatinine",
            ),
        ]
        timeline = PatientTimeline(
            entries=entries,
            start=BASE_TIME,
            end=BASE_TIME + timedelta(hours=24),
            summary="24h timeline with 2 events",
        )
        assert len(timeline.entries) == 2
        assert timeline.summary == "24h timeline with 2 events"
