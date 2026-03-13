"""Tests for the pre-round summary workflow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fangbot.chart.models import ChartFact, FactCategory, FactStatus, PatientChart
from fangbot.memory.audit import AuditLogger
from fangbot.models import ProviderResponse
from fangbot.workflows.engine import WorkflowContext, WorkflowEngine
from fangbot.workflows.models import StepStatus
from fangbot.workflows.pre_round_summary import PreRoundSummary

BASE_TIME = datetime(2026, 3, 7, 8, 0, tzinfo=timezone.utc)


def _build_multi_day_chart() -> PatientChart:
    """Build a synthetic 3-day chart for pre-round summary testing."""
    facts = [
        # Demographics
        ChartFact(name="Age", value="58", category=FactCategory.VITAL,
                  source="Demographics", status=FactStatus.ACTIVE, confidence=1.0),
        # AKI progression
        ChartFact(name="Creatinine", value="1.2 mg/dL", category=FactCategory.LAB,
                  source="BMP", timestamp=BASE_TIME, status=FactStatus.ACTIVE, confidence=1.0),
        ChartFact(name="Creatinine", value="1.8 mg/dL", category=FactCategory.LAB,
                  source="BMP", timestamp=BASE_TIME + timedelta(hours=24),
                  status=FactStatus.ACTIVE, confidence=1.0),
        ChartFact(name="Creatinine", value="2.4 mg/dL", category=FactCategory.LAB,
                  source="BMP", timestamp=BASE_TIME + timedelta(hours=48),
                  status=FactStatus.ACTIVE, confidence=1.0),
        # Stable vital
        ChartFact(name="HR", value="78", category=FactCategory.VITAL,
                  source="Vitals", timestamp=BASE_TIME + timedelta(hours=48),
                  status=FactStatus.ACTIVE, confidence=1.0),
        # Diagnoses
        ChartFact(name="AKI", value="Acute kidney injury",
                  category=FactCategory.DIAGNOSIS, source="Assessment",
                  status=FactStatus.ACTIVE, confidence=0.9,
                  timestamp=BASE_TIME + timedelta(hours=24)),
        ChartFact(name="Hypertension", value="Essential hypertension",
                  category=FactCategory.DIAGNOSIS, source="PMH",
                  status=FactStatus.HISTORICAL, confidence=1.0),
        # Medication
        ChartFact(name="IV Fluids", value="NS 125 mL/hr",
                  category=FactCategory.MEDICATION, source="Orders",
                  status=FactStatus.ACTIVE, confidence=1.0),
    ]
    return PatientChart(
        facts=facts,
        raw_text="58yo with HTN admitted for AKI. Cr rising 1.2->1.8->2.4. On IV fluids.",
        parse_warnings=[],
    )


@pytest.fixture
def preround_context(mock_provider, tmp_path) -> WorkflowContext:
    mock_provider.queue_response(
        ProviderResponse(
            content=(
                "# Pre-Round Summary\n\n"
                "## Active Problems\n"
                "1. **AKI** — Creatinine rising (1.2 → 1.8 → 2.4 mg/dL over 48h, "
                "+100% from baseline). Currently on NS 125 mL/hr.\n"
                "## Chronic/Background\n"
                "- Hypertension (historical)\n"
                "## Overnight Events\n"
                "- No acute events\n"
                "## Plan\n"
                "- Repeat BMP this AM\n"
                "- Consider nephrology consult if Cr continues to rise"
            ),
            stop_reason="end_turn",
        )
    )
    chart = _build_multi_day_chart()
    audit = AuditLogger(log_dir=str(tmp_path))
    audit.start_session()
    return WorkflowContext(
        chart=chart, provider=mock_provider, audit=audit, raw_text=chart.raw_text,
    )


class TestPreRoundSummary:
    def test_workflow_metadata(self):
        assert PreRoundSummary.name == "pre_round_summary"

    def test_steps_include_all_analyses(self):
        workflow = PreRoundSummary()
        step_names = [s.name for s in workflow.steps()]
        assert "classify_temporal" in step_names
        assert "detect_trends" in step_names
        assert "compare_baseline" in step_names
        assert "generate_summary" in step_names

    @pytest.mark.asyncio
    async def test_full_execution(self, preround_context):
        engine = WorkflowEngine()
        engine.register(PreRoundSummary)
        draft = await engine.run("pre_round_summary", preround_context)

        assert draft.workflow_name == "pre_round_summary"
        assert draft.is_draft is True
        headings = [s.heading for s in draft.sections]
        assert "Pre-Round Summary" in headings

    @pytest.mark.asyncio
    async def test_rising_trend_detected(self, preround_context):
        engine = WorkflowEngine()
        engine.register(PreRoundSummary)
        await engine.run("pre_round_summary", preround_context)

        trend_result = preround_context.step_results.get("detect_trends")
        assert trend_result is not None
        trends = trend_result.output.get("trends", [])
        cr_trends = [t for t in trends if t["name"] == "Creatinine"]
        assert len(cr_trends) == 1
        assert cr_trends[0]["direction"] == "rising"

    @pytest.mark.asyncio
    async def test_baseline_comparison(self, preround_context):
        engine = WorkflowEngine()
        engine.register(PreRoundSummary)
        await engine.run("pre_round_summary", preround_context)

        baseline_result = preround_context.step_results.get("compare_baseline")
        assert baseline_result is not None
        comparisons = baseline_result.output.get("comparisons", [])
        cr_comps = [c for c in comparisons if c["name"] == "Creatinine"]
        assert len(cr_comps) == 1
        assert cr_comps[0]["change_percent"] == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_temporal_classification(self, preround_context):
        engine = WorkflowEngine()
        engine.register(PreRoundSummary)
        await engine.run("pre_round_summary", preround_context)

        temporal_result = preround_context.step_results.get("classify_temporal")
        assert temporal_result is not None
        classifications = temporal_result.output.get("classifications", [])
        assert len(classifications) > 0
