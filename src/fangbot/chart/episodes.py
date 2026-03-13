"""Episode segmentation — groups related clinical facts by time window and category."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from fangbot.chart.models import ChartFact, FactCategory, PatientChart


class ClinicalEpisode(BaseModel):
    """A cluster of related facts grouped by time proximity and category."""

    label: str
    facts: list[ChartFact] = Field(min_length=1)
    start: datetime
    end: datetime
    category: FactCategory


def segment_episodes(
    chart: PatientChart,
    window_hours: float = 6.0,
) -> list[ClinicalEpisode]:
    """Segment chart facts into clinical episodes using time-window clustering.

    Facts are grouped by category, then clustered by time proximity. Two facts
    belong to the same episode if the gap between consecutive timestamps is
    less than window_hours.

    Args:
        chart: Patient chart with extracted facts.
        window_hours: Maximum gap (in hours) between consecutive facts
            in the same episode. Default 6 hours.

    Returns:
        List of ClinicalEpisode objects, sorted by start time.
    """
    # Group by category, keep only timestamped facts
    by_category: dict[FactCategory, list[ChartFact]] = defaultdict(list)
    for fact in chart.facts:
        if fact.timestamp is not None:
            by_category[fact.category].append(fact)

    episodes: list[ClinicalEpisode] = []
    window = timedelta(hours=window_hours)

    for category, facts in by_category.items():
        # Sort by timestamp
        sorted_facts = sorted(facts, key=lambda f: f.timestamp)  # type: ignore[arg-type]

        # Cluster by time window
        clusters: list[list[ChartFact]] = []
        current_cluster: list[ChartFact] = [sorted_facts[0]] if sorted_facts else []

        for fact in sorted_facts[1:]:
            prev_ts = current_cluster[-1].timestamp
            curr_ts = fact.timestamp
            if prev_ts is None or curr_ts is None:
                continue  # skip facts without timestamps
            if curr_ts - prev_ts <= window:
                current_cluster.append(fact)
            else:
                clusters.append(current_cluster)
                current_cluster = [fact]

        if current_cluster:
            clusters.append(current_cluster)

        # Build episodes from clusters
        for cluster in clusters:
            start_ts = cluster[0].timestamp
            end_ts = cluster[-1].timestamp
            if start_ts is None or end_ts is None:
                continue  # skip clusters without timestamps

            names = sorted({f.name for f in cluster})
            if len(names) <= 3:
                label = f"{category.value}: {', '.join(names)}"
            else:
                label = f"{category.value}: {names[0]} +{len(names)-1} others"

            episodes.append(
                ClinicalEpisode(
                    label=label,
                    facts=cluster,
                    start=start_ts,
                    end=end_ts,
                    category=category,
                )
            )

    # Sort by start time
    episodes.sort(key=lambda ep: ep.start)
    return episodes
