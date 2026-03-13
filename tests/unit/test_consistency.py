"""Tests for chart inconsistency detection."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fangbot.chart.consistency import (
    ConsistencyReport,
    Inconsistency,
    InconsistencySeverity,
    InconsistencyType,
)
from fangbot.chart.models import ChartFact, FactCategory, FactStatus, PatientChart


class TestInconsistencyModels:
    def test_inconsistency_type_values(self):
        expected = {
            "duplicate_value",
            "conflicting_value",
            "impossible_value",
            "allergy_violation",
            "status_conflict",
            "copy_forward",
        }
        assert {t.value for t in InconsistencyType} == expected

    def test_severity_values(self):
        expected = {"info", "warning", "critical"}
        assert {s.value for s in InconsistencySeverity} == expected

    def test_severity_ordering(self):
        assert InconsistencySeverity.INFO < InconsistencySeverity.WARNING
        assert InconsistencySeverity.WARNING < InconsistencySeverity.CRITICAL

    def test_inconsistency_with_two_facts(self):
        fact_a = ChartFact(
            name="Creatinine",
            value="1.2 mg/dL",
            category=FactCategory.LAB,
            source="BMP 3/5",
        )
        fact_b = ChartFact(
            name="Creatinine",
            value="8.5 mg/dL",
            category=FactCategory.LAB,
            source="BMP 3/5",
        )
        inc = Inconsistency(
            type=InconsistencyType.CONFLICTING_VALUE,
            severity=InconsistencySeverity.WARNING,
            description="Creatinine has conflicting values: 1.2 mg/dL vs 8.5 mg/dL",
            fact_a=fact_a,
            fact_b=fact_b,
            recommendation="Verify which creatinine value is correct",
        )
        assert inc.fact_a.value == "1.2 mg/dL"
        assert inc.fact_b is not None
        assert inc.fact_b.value == "8.5 mg/dL"

    def test_inconsistency_with_single_fact(self):
        fact = ChartFact(
            name="HR",
            value="-10 bpm",
            category=FactCategory.VITAL,
            source="Vitals",
        )
        inc = Inconsistency(
            type=InconsistencyType.IMPOSSIBLE_VALUE,
            severity=InconsistencySeverity.CRITICAL,
            description="Heart rate cannot be negative: -10 bpm",
            fact_a=fact,
            recommendation="Review vitals entry for data entry error",
        )
        assert inc.fact_b is None

    def test_consistency_report(self):
        report = ConsistencyReport(
            inconsistencies=[],
            facts_checked=10,
        )
        assert report.inconsistencies == []
        assert report.facts_checked == 10
        assert report.checked_at is not None

    def test_consistency_report_is_clean(self):
        report = ConsistencyReport(inconsistencies=[], facts_checked=5)
        assert report.is_clean is True

    def test_consistency_report_not_clean(self):
        fact = ChartFact(
            name="HR", value="-5", category=FactCategory.VITAL, source="Vitals"
        )
        inc = Inconsistency(
            type=InconsistencyType.IMPOSSIBLE_VALUE,
            severity=InconsistencySeverity.CRITICAL,
            description="Impossible HR",
            fact_a=fact,
            recommendation="Check vitals",
        )
        report = ConsistencyReport(inconsistencies=[inc], facts_checked=1)
        assert report.is_clean is False

    def test_consistency_report_by_severity(self):
        fact = ChartFact(
            name="HR", value="500", category=FactCategory.VITAL, source="V"
        )
        critical = Inconsistency(
            type=InconsistencyType.IMPOSSIBLE_VALUE,
            severity=InconsistencySeverity.CRITICAL,
            description="Impossible HR",
            fact_a=fact,
            recommendation="Check",
        )
        warning = Inconsistency(
            type=InconsistencyType.DUPLICATE_VALUE,
            severity=InconsistencySeverity.WARNING,
            description="Duplicate",
            fact_a=fact,
            recommendation="Check",
        )
        report = ConsistencyReport(
            inconsistencies=[critical, warning], facts_checked=2
        )
        assert len(report.by_severity(InconsistencySeverity.CRITICAL)) == 1
        assert len(report.by_severity(InconsistencySeverity.WARNING)) == 1
        assert len(report.by_severity(InconsistencySeverity.INFO)) == 0
