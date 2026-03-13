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
    build_timeline,
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


from fangbot.chart.episodes import ClinicalEpisode, segment_episodes


class TestClinicalEpisode:
    def test_creation(self):
        facts = [
            _fact("Creatinine", "1.8 mg/dL", FactCategory.LAB, 0),
            _fact("BUN", "35 mg/dL", FactCategory.LAB, 0),
        ]
        ep = ClinicalEpisode(
            label="Renal panel",
            facts=facts,
            start=BASE_TIME,
            end=BASE_TIME,
            category=FactCategory.LAB,
        )
        assert ep.label == "Renal panel"
        assert len(ep.facts) == 2
        assert ep.start == BASE_TIME


class TestSegmentEpisodes:
    def test_single_cluster(self):
        """Facts within the time window are grouped into one episode."""
        facts = [
            _fact("Creatinine", "1.8 mg/dL", FactCategory.LAB, 0),
            _fact("BUN", "35 mg/dL", FactCategory.LAB, 1),
            _fact("Potassium", "4.2 mEq/L", FactCategory.LAB, 2),
        ]
        chart = _chart_with_facts(facts)
        episodes = segment_episodes(chart, window_hours=6)

        assert len(episodes) == 1
        assert len(episodes[0].facts) == 3
        assert episodes[0].category == FactCategory.LAB

    def test_two_clusters_by_time(self):
        """Facts separated by > window_hours form separate episodes."""
        facts = [
            _fact("Creatinine", "1.2 mg/dL", FactCategory.LAB, 0),
            _fact("Creatinine", "1.8 mg/dL", FactCategory.LAB, 24),
        ]
        chart = _chart_with_facts(facts)
        episodes = segment_episodes(chart, window_hours=6)

        assert len(episodes) == 2

    def test_separate_categories_separate_episodes(self):
        """Different categories at the same time form separate episodes."""
        facts = [
            _fact("Creatinine", "1.8 mg/dL", FactCategory.LAB, 0),
            _fact("Heart Rate", "88 bpm", FactCategory.VITAL, 0),
        ]
        chart = _chart_with_facts(facts)
        episodes = segment_episodes(chart, window_hours=6)

        assert len(episodes) == 2
        categories = {ep.category for ep in episodes}
        assert categories == {FactCategory.LAB, FactCategory.VITAL}

    def test_empty_chart(self):
        """Empty chart produces no episodes."""
        chart = _chart_with_facts([])
        episodes = segment_episodes(chart, window_hours=6)
        assert len(episodes) == 0

    def test_facts_without_timestamps_excluded(self):
        """Facts without timestamps are not included in episodes."""
        facts = [
            ChartFact(
                name="Aspirin",
                value="81mg daily",
                category=FactCategory.MEDICATION,
                source="Med list",
            ),
        ]
        chart = _chart_with_facts(facts)
        episodes = segment_episodes(chart, window_hours=6)
        assert len(episodes) == 0

    def test_episode_start_end_correct(self):
        """Episode start/end bracket the contained facts."""
        facts = [
            _fact("Creatinine", "1.2 mg/dL", FactCategory.LAB, 0),
            _fact("BUN", "20 mg/dL", FactCategory.LAB, 2),
            _fact("Potassium", "4.5 mEq/L", FactCategory.LAB, 4),
        ]
        chart = _chart_with_facts(facts)
        episodes = segment_episodes(chart, window_hours=6)

        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.start == BASE_TIME
        assert ep.end == BASE_TIME + timedelta(hours=4)


class TestBuildTimeline:
    def test_basic_timeline(self):
        """Build a timeline from a chart with multiple timestamped facts."""
        facts = [
            _fact("Creatinine", "1.2 mg/dL", FactCategory.LAB, 0),
            _fact("Creatinine", "1.8 mg/dL", FactCategory.LAB, 24),
            _fact("Pneumonia", "RLL infiltrate", FactCategory.DIAGNOSIS, 2),
            _fact("Heart Rate", "110 bpm", FactCategory.VITAL, 0),
        ]
        chart = _chart_with_facts(facts)
        timeline = build_timeline(chart)

        assert len(timeline.entries) == 4
        assert timeline.start == BASE_TIME
        assert timeline.end == BASE_TIME + timedelta(hours=24)
        # Entries should be sorted by timestamp
        timestamps = [e.timestamp for e in timeline.entries]
        assert timestamps == sorted(timestamps)

    def test_timeline_entries_have_descriptions(self):
        """Each entry has a human-readable description."""
        facts = [
            _fact("Creatinine", "1.8 mg/dL", FactCategory.LAB, 0),
        ]
        chart = _chart_with_facts(facts)
        timeline = build_timeline(chart)

        assert len(timeline.entries) == 1
        entry = timeline.entries[0]
        assert "Creatinine" in entry.description
        assert "1.8" in entry.description
        assert entry.fact_name == "Creatinine"
        assert entry.category == FactCategory.LAB

    def test_timeline_includes_temporal_classification(self):
        """Timeline entries include temporal classification when available."""
        facts = [
            _fact("Creatinine", "1.2 mg/dL", FactCategory.LAB, 0),
            _fact("Creatinine", "2.4 mg/dL", FactCategory.LAB, 24),
        ]
        chart = _chart_with_facts(facts)
        timeline = build_timeline(chart)

        # The latest creatinine entry should be marked WORSENING
        latest = [e for e in timeline.entries if e.timestamp == BASE_TIME + timedelta(hours=24)]
        assert len(latest) == 1
        assert latest[0].classification == TemporalClassification.WORSENING

    def test_timeline_excludes_untimestamped_facts(self):
        """Facts without timestamps are not included in the timeline."""
        facts = [
            ChartFact(
                name="Diabetes",
                value="Type 2",
                category=FactCategory.DIAGNOSIS,
                source="PMH",
                status=FactStatus.HISTORICAL,
            ),
            _fact("Creatinine", "1.8 mg/dL", FactCategory.LAB, 0),
        ]
        chart = _chart_with_facts(facts)
        timeline = build_timeline(chart)

        assert len(timeline.entries) == 1
        assert timeline.entries[0].fact_name == "Creatinine"

    def test_empty_chart_empty_timeline(self):
        """Empty chart produces empty timeline."""
        chart = _chart_with_facts([])
        timeline = build_timeline(chart)

        assert len(timeline.entries) == 0
        assert timeline.start is None
        assert timeline.end is None

    def test_timeline_summary_includes_span(self):
        """Timeline summary mentions the time span."""
        facts = [
            _fact("Creatinine", "1.2 mg/dL", FactCategory.LAB, 0),
            _fact("Creatinine", "1.8 mg/dL", FactCategory.LAB, 48),
        ]
        chart = _chart_with_facts(facts)
        timeline = build_timeline(chart)

        assert "48" in timeline.summary or "2" in timeline.summary  # hours or days
        assert "2 events" in timeline.summary or "2 entries" in timeline.summary


from fangbot.chart.temporal import BaselineComparison, compare_to_baseline


class TestBaselineComparison:
    def test_creation(self):
        comp = BaselineComparison(
            fact_name="Creatinine",
            baseline_value=1.0,
            current_value=2.1,
            change_absolute=1.1,
            change_percent=110.0,
            summary="Creatinine 110% above baseline (1.0 -> 2.1)",
        )
        assert comp.change_percent == pytest.approx(110.0)


class TestCompareToBaseline:
    def test_creatinine_above_baseline(self):
        """Current value higher than earliest value (baseline)."""
        facts = [
            _fact("Creatinine", "1.0 mg/dL", FactCategory.LAB, 0),
            _fact("Creatinine", "1.5 mg/dL", FactCategory.LAB, 24),
            _fact("Creatinine", "2.1 mg/dL", FactCategory.LAB, 48),
        ]
        chart = _chart_with_facts(facts)
        comparisons = compare_to_baseline(chart)

        assert len(comparisons) == 1
        comp = comparisons[0]
        assert comp.fact_name == "Creatinine"
        assert comp.baseline_value == pytest.approx(1.0)
        assert comp.current_value == pytest.approx(2.1)
        assert comp.change_absolute == pytest.approx(1.1)
        assert comp.change_percent == pytest.approx(110.0)

    def test_hemoglobin_below_baseline(self):
        """Current value lower than baseline."""
        facts = [
            _fact("Hemoglobin", "14.0 g/dL", FactCategory.LAB, 0),
            _fact("Hemoglobin", "10.0 g/dL", FactCategory.LAB, 48),
        ]
        chart = _chart_with_facts(facts)
        comparisons = compare_to_baseline(chart)

        assert len(comparisons) == 1
        comp = comparisons[0]
        assert comp.change_absolute == pytest.approx(-4.0)
        assert comp.change_percent == pytest.approx(-28.571, abs=0.01)

    def test_no_change(self):
        """Stable values have 0% change."""
        facts = [
            _fact("Sodium", "140 mEq/L", FactCategory.LAB, 0),
            _fact("Sodium", "140 mEq/L", FactCategory.LAB, 24),
        ]
        chart = _chart_with_facts(facts)
        comparisons = compare_to_baseline(chart)

        assert len(comparisons) == 1
        assert comparisons[0].change_percent == pytest.approx(0.0)

    def test_single_value_no_comparison(self):
        """A single value has no baseline to compare against."""
        facts = [_fact("Creatinine", "1.8 mg/dL", FactCategory.LAB, 0)]
        chart = _chart_with_facts(facts)
        comparisons = compare_to_baseline(chart)

        assert len(comparisons) == 0

    def test_multiple_labs_compared(self):
        """Each lab with 2+ values gets its own comparison."""
        facts = [
            _fact("Creatinine", "1.0 mg/dL", FactCategory.LAB, 0),
            _fact("Creatinine", "2.0 mg/dL", FactCategory.LAB, 24),
            _fact("BUN", "15 mg/dL", FactCategory.LAB, 0),
            _fact("BUN", "30 mg/dL", FactCategory.LAB, 24),
        ]
        chart = _chart_with_facts(facts)
        comparisons = compare_to_baseline(chart)

        assert len(comparisons) == 2
        names = {c.fact_name for c in comparisons}
        assert names == {"Creatinine", "BUN"}
