"""Tests for chart data models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fangbot.chart.models import ChartFact, FactCategory, FactStatus, PatientChart


class TestChartFact:
    def test_minimal_fact(self):
        fact = ChartFact(
            name="Creatinine",
            value="1.8 mg/dL",
            category=FactCategory.LAB,
            source="BMP from 2026-03-07",
        )
        assert fact.name == "Creatinine"
        assert fact.value == "1.8 mg/dL"
        assert fact.category == FactCategory.LAB
        assert fact.source == "BMP from 2026-03-07"
        assert fact.timestamp is None
        assert fact.source_location is None
        assert fact.status is None
        assert fact.confidence == 1.0

    def test_full_fact(self):
        ts = datetime(2026, 3, 7, 8, 14, tzinfo=timezone.utc)
        fact = ChartFact(
            name="Metformin",
            value="500mg twice daily",
            category=FactCategory.MEDICATION,
            timestamp=ts,
            source="Medication list",
            source_location="Active medications section",
            status=FactStatus.ACTIVE,
            confidence=0.95,
        )
        assert fact.timestamp == ts
        assert fact.status == FactStatus.ACTIVE
        assert fact.confidence == 0.95

    def test_confidence_must_be_between_0_and_1(self):
        with pytest.raises(ValueError):
            ChartFact(
                name="X",
                value="Y",
                category=FactCategory.LAB,
                source="Z",
                confidence=1.5,
            )
        with pytest.raises(ValueError):
            ChartFact(
                name="X",
                value="Y",
                category=FactCategory.LAB,
                source="Z",
                confidence=-0.1,
            )

    def test_all_categories_exist(self):
        expected = {
            "lab",
            "vital",
            "medication",
            "diagnosis",
            "procedure",
            "allergy",
            "imaging",
            "culture",
        }
        assert {c.value for c in FactCategory} == expected

    def test_all_statuses_exist(self):
        expected = {"active", "historical", "resolved"}
        assert {s.value for s in FactStatus} == expected


class TestPatientChart:
    def test_empty_chart(self):
        chart = PatientChart(raw_text="Some clinical text")
        assert chart.facts == []
        assert chart.parse_warnings == []
        assert chart.raw_text == "Some clinical text"

    def test_chart_with_facts(self):
        facts = [
            ChartFact(
                name="HR",
                value="88 bpm",
                category=FactCategory.VITAL,
                source="Vitals",
            ),
            ChartFact(
                name="BP",
                value="140/90",
                category=FactCategory.VITAL,
                source="Vitals",
            ),
        ]
        chart = PatientChart(facts=facts, raw_text="HR 88, BP 140/90")
        assert len(chart.facts) == 2

    def test_chart_with_warnings(self):
        chart = PatientChart(
            raw_text="...",
            parse_warnings=["Creatinine found in two places with different values"],
        )
        assert len(chart.parse_warnings) == 1

    def test_raw_text_required(self):
        with pytest.raises(ValueError):
            PatientChart(raw_text="")

    def test_facts_by_category(self):
        facts = [
            ChartFact(
                name="HR",
                value="88",
                category=FactCategory.VITAL,
                source="Vitals",
            ),
            ChartFact(
                name="Cr",
                value="1.2",
                category=FactCategory.LAB,
                source="BMP",
            ),
            ChartFact(
                name="BP",
                value="120/80",
                category=FactCategory.VITAL,
                source="Vitals",
            ),
        ]
        chart = PatientChart(facts=facts, raw_text="...")
        vitals = chart.facts_by_category(FactCategory.VITAL)
        assert len(vitals) == 2
        assert all(f.category == FactCategory.VITAL for f in vitals)

    def test_active_facts(self):
        facts = [
            ChartFact(
                name="HTN",
                value="yes",
                category=FactCategory.DIAGNOSIS,
                source="PMH",
                status=FactStatus.ACTIVE,
            ),
            ChartFact(
                name="Appendectomy",
                value="2010",
                category=FactCategory.PROCEDURE,
                source="PSH",
                status=FactStatus.HISTORICAL,
            ),
            ChartFact(
                name="Fever",
                value="resolved",
                category=FactCategory.VITAL,
                source="HPI",
                status=FactStatus.RESOLVED,
            ),
            ChartFact(
                name="DM",
                value="yes",
                category=FactCategory.DIAGNOSIS,
                source="PMH",
            ),
        ]
        chart = PatientChart(facts=facts, raw_text="...")
        active = chart.active_facts()
        assert len(active) == 1
        assert active[0].name == "HTN"

    def test_latest_value(self):
        facts = [
            ChartFact(
                name="Creatinine",
                value="1.2 mg/dL",
                category=FactCategory.LAB,
                source="BMP 3/5",
                timestamp=datetime(2026, 3, 5, tzinfo=timezone.utc),
            ),
            ChartFact(
                name="Creatinine",
                value="1.8 mg/dL",
                category=FactCategory.LAB,
                source="BMP 3/7",
                timestamp=datetime(2026, 3, 7, tzinfo=timezone.utc),
            ),
        ]
        chart = PatientChart(facts=facts, raw_text="...")
        latest = chart.latest_value("Creatinine")
        assert latest is not None
        assert latest.value == "1.8 mg/dL"

    def test_latest_value_not_found(self):
        chart = PatientChart(facts=[], raw_text="...")
        assert chart.latest_value("Creatinine") is None

    def test_latest_value_no_timestamps_returns_last(self):
        facts = [
            ChartFact(name="Cr", value="1.0", category=FactCategory.LAB, source="A"),
            ChartFact(name="Cr", value="1.5", category=FactCategory.LAB, source="B"),
        ]
        chart = PatientChart(facts=facts, raw_text="...")
        latest = chart.latest_value("Cr")
        assert latest is not None
        assert latest.value == "1.5"
