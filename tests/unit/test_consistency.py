"""Tests for chart inconsistency detection."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fangbot.chart.consistency import (
    ConsistencyReport,
    Inconsistency,
    InconsistencySeverity,
    InconsistencyType,
    check_impossible_vitals,
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


class TestImpossibleVitals:
    def _make_vital(self, name: str, value: str, source: str = "Vitals") -> ChartFact:
        return ChartFact(
            name=name,
            value=value,
            category=FactCategory.VITAL,
            source=source,
        )

    def test_negative_heart_rate(self):
        chart = PatientChart(
            facts=[self._make_vital("HR", "-10 bpm")],
            raw_text="HR -10",
        )
        results = check_impossible_vitals(chart)
        assert len(results) == 1
        assert results[0].type == InconsistencyType.IMPOSSIBLE_VALUE
        assert results[0].severity == InconsistencySeverity.CRITICAL
        assert "HR" in results[0].description or "heart rate" in results[0].description.lower()

    def test_heart_rate_over_300(self):
        chart = PatientChart(
            facts=[self._make_vital("HR", "350 bpm")],
            raw_text="HR 350",
        )
        results = check_impossible_vitals(chart)
        assert len(results) == 1
        assert results[0].severity == InconsistencySeverity.CRITICAL

    def test_valid_heart_rate_no_flag(self):
        chart = PatientChart(
            facts=[self._make_vital("HR", "72 bpm")],
            raw_text="HR 72",
        )
        results = check_impossible_vitals(chart)
        assert len(results) == 0

    def test_spo2_over_100(self):
        chart = PatientChart(
            facts=[self._make_vital("SpO2", "105%")],
            raw_text="SpO2 105%",
        )
        results = check_impossible_vitals(chart)
        assert len(results) == 1

    def test_spo2_negative(self):
        chart = PatientChart(
            facts=[self._make_vital("SpO2", "-2%")],
            raw_text="SpO2 -2%",
        )
        results = check_impossible_vitals(chart)
        assert len(results) == 1

    def test_valid_spo2_no_flag(self):
        chart = PatientChart(
            facts=[self._make_vital("SpO2", "97%")],
            raw_text="SpO2 97",
        )
        results = check_impossible_vitals(chart)
        assert len(results) == 0

    def test_temperature_over_45(self):
        chart = PatientChart(
            facts=[self._make_vital("Temp", "46.5 C")],
            raw_text="T 46.5",
        )
        results = check_impossible_vitals(chart)
        assert len(results) == 1

    def test_temperature_below_25(self):
        chart = PatientChart(
            facts=[self._make_vital("Temp", "20.0 C")],
            raw_text="T 20.0",
        )
        results = check_impossible_vitals(chart)
        assert len(results) == 1

    def test_valid_temperature_no_flag(self):
        chart = PatientChart(
            facts=[self._make_vital("Temp", "37.2 C")],
            raw_text="T 37.2",
        )
        results = check_impossible_vitals(chart)
        assert len(results) == 0

    def test_systolic_below_diastolic(self):
        chart = PatientChart(
            facts=[self._make_vital("BP", "60/120 mmHg")],
            raw_text="BP 60/120",
        )
        results = check_impossible_vitals(chart)
        assert len(results) == 1
        assert results[0].severity == InconsistencySeverity.CRITICAL

    def test_valid_bp_no_flag(self):
        chart = PatientChart(
            facts=[self._make_vital("BP", "120/80 mmHg")],
            raw_text="BP 120/80",
        )
        results = check_impossible_vitals(chart)
        assert len(results) == 0

    def test_negative_respiratory_rate(self):
        chart = PatientChart(
            facts=[self._make_vital("RR", "-5 breaths/min")],
            raw_text="RR -5",
        )
        results = check_impossible_vitals(chart)
        assert len(results) == 1

    def test_respiratory_rate_over_80(self):
        chart = PatientChart(
            facts=[self._make_vital("RR", "100 breaths/min")],
            raw_text="RR 100",
        )
        results = check_impossible_vitals(chart)
        assert len(results) == 1

    def test_non_vital_facts_ignored(self):
        chart = PatientChart(
            facts=[
                ChartFact(
                    name="Creatinine",
                    value="-5 mg/dL",
                    category=FactCategory.LAB,
                    source="BMP",
                )
            ],
            raw_text="Cr -5",
        )
        results = check_impossible_vitals(chart)
        assert len(results) == 0

    def test_unparseable_value_no_crash(self):
        chart = PatientChart(
            facts=[self._make_vital("HR", "normal")],
            raw_text="HR normal",
        )
        results = check_impossible_vitals(chart)
        assert len(results) == 0

    def test_multiple_impossible_vitals(self):
        chart = PatientChart(
            facts=[
                self._make_vital("HR", "-10 bpm"),
                self._make_vital("SpO2", "110%"),
                self._make_vital("Temp", "37.0 C"),
            ],
            raw_text="HR -10, SpO2 110, T 37.0",
        )
        results = check_impossible_vitals(chart)
        assert len(results) == 2
