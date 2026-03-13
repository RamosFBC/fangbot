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
    classify_facts,
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


class TestClassifyFacts:
    def test_new_finding_single_occurrence(self):
        """A diagnosis appearing only once with ACTIVE status is NEW."""
        facts = [
            _fact("Pneumonia", "right lower lobe", FactCategory.DIAGNOSIS, 0),
        ]
        chart = _chart_with_facts(facts)
        classified = classify_facts(chart)

        assert len(classified) == 1
        assert classified[0].classification == TemporalClassification.NEW
        assert classified[0].fact.name == "Pneumonia"

    def test_chronic_finding_historical_status(self):
        """A fact with HISTORICAL status is CHRONIC."""
        facts = [
            _fact(
                "Hypertension", "controlled", FactCategory.DIAGNOSIS, 0,
                status=FactStatus.HISTORICAL,
            ),
        ]
        chart = _chart_with_facts(facts)
        classified = classify_facts(chart)

        assert len(classified) == 1
        assert classified[0].classification == TemporalClassification.CHRONIC

    def test_resolved_finding(self):
        """A fact with RESOLVED status maps to RESOLVED classification."""
        facts = [
            _fact(
                "UTI", "resolved", FactCategory.DIAGNOSIS, 0,
                status=FactStatus.RESOLVED,
            ),
        ]
        chart = _chart_with_facts(facts)
        classified = classify_facts(chart)

        assert len(classified) == 1
        assert classified[0].classification == TemporalClassification.RESOLVED

    def test_worsening_numeric_increase_for_lab(self):
        """Rising lab values with multiple readings indicate WORSENING."""
        facts = [
            _fact("Creatinine", "1.2 mg/dL", FactCategory.LAB, 0),
            _fact("Creatinine", "1.8 mg/dL", FactCategory.LAB, 24),
            _fact("Creatinine", "2.4 mg/dL", FactCategory.LAB, 48),
        ]
        chart = _chart_with_facts(facts)
        classified = classify_facts(chart)

        # The latest fact (2.4) should be classified as WORSENING
        latest = [c for c in classified if c.fact.value == "2.4 mg/dL"]
        assert len(latest) == 1
        assert latest[0].classification == TemporalClassification.WORSENING

    def test_improving_numeric_decrease_for_lab(self):
        """Falling lab values (where high is bad) indicate IMPROVING."""
        facts = [
            _fact("Creatinine", "2.4 mg/dL", FactCategory.LAB, 0),
            _fact("Creatinine", "1.8 mg/dL", FactCategory.LAB, 24),
            _fact("Creatinine", "1.2 mg/dL", FactCategory.LAB, 48),
        ]
        chart = _chart_with_facts(facts)
        classified = classify_facts(chart)

        latest = [c for c in classified if c.fact.value == "1.2 mg/dL"]
        assert len(latest) == 1
        assert latest[0].classification == TemporalClassification.IMPROVING

    def test_mixed_categories(self):
        """Facts from different categories are all classified."""
        facts = [
            _fact("Pneumonia", "bilateral", FactCategory.DIAGNOSIS, 0),
            _fact("Heart Rate", "110 bpm", FactCategory.VITAL, 0),
            _fact("Vancomycin", "1g IV q12h", FactCategory.MEDICATION, 0),
        ]
        chart = _chart_with_facts(facts)
        classified = classify_facts(chart)

        assert len(classified) == 3
        names = {c.fact.name for c in classified}
        assert names == {"Pneumonia", "Heart Rate", "Vancomycin"}

    def test_facts_without_timestamps_still_classified(self):
        """Facts without timestamps are classified based on status alone."""
        fact = ChartFact(
            name="Diabetes",
            value="Type 2",
            category=FactCategory.DIAGNOSIS,
            source="PMH",
            status=FactStatus.HISTORICAL,
        )
        chart = _chart_with_facts([fact])
        classified = classify_facts(chart)

        assert len(classified) == 1
        assert classified[0].classification == TemporalClassification.CHRONIC
