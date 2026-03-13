"""Temporal abstraction engine — classifies findings as new/chronic, builds timelines."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from fangbot.chart.models import ChartFact, FactCategory, FactStatus, PatientChart


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


class BaselineComparison(BaseModel):
    """Comparison of current value against the patient's baseline (earliest value)."""

    fact_name: str
    baseline_value: float
    current_value: float
    change_absolute: float
    change_percent: float
    summary: str = ""


import re
from collections import defaultdict


def _extract_numeric(value: str) -> float | None:
    """Extract the first numeric value from a string."""
    match = re.search(r"[-+]?\d*\.?\d+", value)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def classify_facts(chart: PatientChart) -> list[TemporalFact]:
    """Classify each chart fact with a temporal label.

    Classification rules:
    1. RESOLVED status -> RESOLVED
    2. HISTORICAL status -> CHRONIC
    3. For numeric series (labs/vitals) with 2+ timestamped values:
       - Rising values -> WORSENING (latest), earlier values -> NEW
       - Falling values -> IMPROVING (latest), earlier values -> NEW
    4. Single occurrence with ACTIVE/None status -> NEW

    Args:
        chart: Patient chart with extracted facts.

    Returns:
        List of TemporalFact, one per input fact, preserving order.
    """
    # Pre-compute trend info for numeric series
    numeric_series: dict[str, list[tuple[float, datetime, int]]] = defaultdict(list)
    for idx, fact in enumerate(chart.facts):
        if fact.timestamp is None:
            continue
        if fact.category not in (FactCategory.LAB, FactCategory.VITAL):
            continue
        num = _extract_numeric(fact.value)
        if num is not None:
            numeric_series[fact.name].append((num, fact.timestamp, idx))

    # Sort each series by timestamp
    for name in numeric_series:
        numeric_series[name].sort(key=lambda x: x[1])

    # Determine direction for each series
    series_direction: dict[str, str] = {}
    for name, points in numeric_series.items():
        if len(points) < 2:
            continue
        first_val = points[0][0]
        last_val = points[-1][0]
        diff = last_val - first_val
        mean = (first_val + last_val) / 2
        if abs(mean) < 1e-12:
            continue
        ratio = abs(diff) / abs(mean)
        if ratio < 0.02:
            series_direction[name] = "stable"
        elif diff > 0:
            series_direction[name] = "rising"
        else:
            series_direction[name] = "falling"

    # Track which index is the latest in each series
    latest_idx: dict[str, int] = {}
    for name, points in numeric_series.items():
        if len(points) >= 2:
            latest_idx[name] = points[-1][2]

    results: list[TemporalFact] = []
    for idx, fact in enumerate(chart.facts):
        classification, rationale = _classify_single(
            fact, idx, numeric_series, series_direction, latest_idx
        )
        results.append(
            TemporalFact(fact=fact, classification=classification, rationale=rationale)
        )

    return results


def _classify_single(
    fact: ChartFact,
    idx: int,
    numeric_series: dict[str, list[tuple[float, datetime, int]]],
    series_direction: dict[str, str],
    latest_idx: dict[str, int],
) -> tuple[TemporalClassification, str]:
    """Classify a single fact."""
    # Rule 1: explicit resolved status
    if fact.status == FactStatus.RESOLVED:
        return TemporalClassification.RESOLVED, "Fact has RESOLVED status"

    # Rule 2: explicit historical status
    if fact.status == FactStatus.HISTORICAL:
        return TemporalClassification.CHRONIC, "Fact has HISTORICAL status (pre-existing)"

    # Rule 3: numeric trend for latest value in a series
    name = fact.name
    if name in series_direction and idx == latest_idx.get(name):
        direction = series_direction[name]
        series = numeric_series[name]
        first_val = series[0][0]
        last_val = series[-1][0]
        if direction == "rising":
            return (
                TemporalClassification.WORSENING,
                f"{name} rising: {first_val:g} -> {last_val:g}",
            )
        elif direction == "falling":
            return (
                TemporalClassification.IMPROVING,
                f"{name} falling: {first_val:g} -> {last_val:g}",
            )

    # Rule 4: default to NEW
    return TemporalClassification.NEW, "First or single occurrence, active finding"


def build_timeline(chart: PatientChart) -> PatientTimeline:
    """Build a problem-oriented timeline from chart facts.

    Combines temporal classification with chronological ordering.
    Only includes facts that have timestamps.

    Args:
        chart: Patient chart with extracted facts.

    Returns:
        A PatientTimeline with entries sorted chronologically.
    """
    # Get temporal classifications for all facts
    classified = classify_facts(chart)

    # Build a lookup: fact index -> classification
    classification_map: dict[int, TemporalClassification] = {}
    for idx, tf in enumerate(classified):
        classification_map[idx] = tf.classification

    # Build timeline entries for timestamped facts only
    entries: list[TimelineEntry] = []
    for idx, fact in enumerate(chart.facts):
        if fact.timestamp is None:
            continue
        entries.append(
            TimelineEntry(
                timestamp=fact.timestamp,
                description=f"{fact.name}: {fact.value}",
                category=fact.category,
                fact_name=fact.name,
                classification=classification_map.get(idx),
            )
        )

    # Sort by timestamp
    entries.sort(key=lambda e: e.timestamp)

    if not entries:
        return PatientTimeline(entries=[], summary="No timestamped events")

    start = entries[0].timestamp
    end = entries[-1].timestamp
    span_hours = (end - start).total_seconds() / 3600
    n = len(entries)

    if span_hours >= 48:
        span_str = f"{span_hours / 24:.0f} days"
    else:
        span_str = f"{span_hours:.0f}h"

    summary = f"{span_str} timeline with {n} entries"

    return PatientTimeline(entries=entries, start=start, end=end, summary=summary)


def compare_to_baseline(
    chart: PatientChart,
    categories: set[FactCategory] | None = None,
) -> list[BaselineComparison]:
    """Compare current values against the patient's baseline for each numeric series.

    Baseline is defined as the earliest timestamped value for each fact name.
    Current is the latest timestamped value.

    Args:
        chart: Patient chart with extracted facts.
        categories: Which categories to analyze. Defaults to LAB and VITAL.

    Returns:
        List of BaselineComparison, one per fact name that has 2+ numeric values.
    """
    if categories is None:
        categories = {FactCategory.LAB, FactCategory.VITAL}

    # Group timestamped numeric facts by name
    series: dict[str, list[tuple[float, datetime]]] = defaultdict(list)
    for fact in chart.facts:
        if fact.category not in categories:
            continue
        if fact.timestamp is None:
            continue
        num = _extract_numeric(fact.value)
        if num is not None:
            series[fact.name].append((num, fact.timestamp))

    comparisons: list[BaselineComparison] = []
    for name, points in sorted(series.items()):
        if len(points) < 2:
            continue
        points.sort(key=lambda p: p[1])
        baseline = points[0][0]
        current = points[-1][0]
        change_abs = current - baseline

        if abs(baseline) < 1e-12:
            change_pct = 0.0
        else:
            change_pct = (change_abs / baseline) * 100.0

        if change_pct > 0:
            direction = "above"
        elif change_pct < 0:
            direction = "below"
        else:
            direction = "at"

        summary = (
            f"{name} {abs(change_pct):.0f}% {direction} baseline "
            f"({baseline:g} -> {current:g})"
        )

        comparisons.append(
            BaselineComparison(
                fact_name=name,
                baseline_value=baseline,
                current_value=current,
                change_absolute=change_abs,
                change_percent=change_pct,
                summary=summary,
            )
        )

    return comparisons
