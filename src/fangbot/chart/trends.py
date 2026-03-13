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


import re
from collections import defaultdict


def _extract_numeric(value: str) -> float | None:
    """Extract the first numeric value from a string like '1.8 mg/dL'."""
    match = re.search(r"[-+]?\d*\.?\d+", value)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def _compute_slope(points: list[TrendPoint]) -> float | None:
    """Compute slope (value change per hour) using least-squares on the series."""
    if len(points) < 2:
        return None
    t0 = points[0].timestamp
    # x = hours since first point, y = value
    xs = [(p.timestamp - t0).total_seconds() / 3600.0 for p in points]
    ys = [p.value for p in points]

    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom


def _classify_direction(
    slope: float | None,
    points: list[TrendPoint],
    stable_threshold: float,
) -> TrendDirection:
    """Classify trend direction from slope and point count."""
    if len(points) < 2:
        return TrendDirection.INSUFFICIENT_DATA
    if slope is None:
        return TrendDirection.INSUFFICIENT_DATA

    # Normalize slope relative to mean value to get proportional change rate
    mean_val = sum(p.value for p in points) / len(points)
    if abs(mean_val) < 1e-12:
        return TrendDirection.STABLE

    normalized = abs(slope) / abs(mean_val)
    if normalized <= stable_threshold:
        return TrendDirection.STABLE
    return TrendDirection.RISING if slope > 0 else TrendDirection.FALLING


def detect_trends(
    chart: PatientChart,
    stable_threshold: float = 0.005,
    categories: set[FactCategory] | None = None,
) -> list[Trend]:
    """Detect trends in numeric fact series within the chart.

    Args:
        chart: The patient chart with extracted facts.
        stable_threshold: Proportional change rate below which a trend
            is classified as STABLE. Default 0.005 (0.5% of mean per hour).
        categories: Which fact categories to analyze. Defaults to LAB and VITAL.

    Returns:
        List of Trend objects, one per fact name that has numeric values.
    """
    if categories is None:
        categories = {FactCategory.LAB, FactCategory.VITAL}

    # Group timestamped numeric facts by (name, category)
    series: dict[tuple[str, FactCategory], list[TrendPoint]] = defaultdict(list)

    for fact in chart.facts:
        if fact.category not in categories:
            continue
        if fact.timestamp is None:
            continue
        numeric = _extract_numeric(fact.value)
        if numeric is None:
            continue
        key = (fact.name, fact.category)
        series[key].append(TrendPoint(timestamp=fact.timestamp, value=numeric))

    trends: list[Trend] = []
    for (name, category), points in sorted(series.items()):
        # Sort by timestamp
        points.sort(key=lambda p: p.timestamp)

        slope = _compute_slope(points)
        direction = _classify_direction(slope, points, stable_threshold)

        first_val = points[0].value
        last_val = points[-1].value
        if len(points) >= 2:
            hours = (points[-1].timestamp - points[0].timestamp).total_seconds() / 3600
            summary = (
                f"{name} {direction.value} over {hours:.0f}h: "
                f"{first_val:g} -> {last_val:g}"
            )
        else:
            summary = f"{name}: single value {first_val:g} (insufficient data for trend)"

        trends.append(
            Trend(
                fact_name=name,
                category=category,
                direction=direction,
                points=points,
                rate_of_change=slope,
                summary=summary,
            )
        )

    return trends
