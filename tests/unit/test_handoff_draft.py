"""Tests for the IPASS handoff draft workflow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fangbot.chart.models import ChartFact, FactCategory, FactStatus, PatientChart
from fangbot.memory.audit import AuditLogger
from fangbot.models import ProviderResponse
from fangbot.workflows.engine import WorkflowContext, WorkflowEngine
from fangbot.workflows.handoff_draft import HandoffDraft
from fangbot.workflows.models import StepStatus

BASE_TIME = datetime(2026, 3, 7, 8, 0, tzinfo=timezone.utc)


def _build_handoff_chart() -> PatientChart:
    """Build a synthetic multi-day chart for handoff testing."""
    facts = [
        # Demographics
        ChartFact(name="Age", value="65", category=FactCategory.VITAL,
                  source="Demographics", status=FactStatus.ACTIVE, confidence=1.0),
        ChartFact(name="Sex", value="Female", category=FactCategory.VITAL,
                  source="Demographics", status=FactStatus.ACTIVE, confidence=1.0),
        # Active diagnoses
        ChartFact(name="Pneumonia", value="Community-acquired pneumonia",
                  category=FactCategory.DIAGNOSIS, source="Assessment",
                  status=FactStatus.ACTIVE, confidence=0.9,
                  timestamp=BASE_TIME),
        ChartFact(name="Type 2 Diabetes", value="Type 2 DM",
                  category=FactCategory.DIAGNOSIS, source="PMH",
                  status=FactStatus.ACTIVE, confidence=1.0),
        # Labs trending
        ChartFact(name="WBC", value="14.2", category=FactCategory.LAB,
                  source="Labs", timestamp=BASE_TIME, status=FactStatus.ACTIVE, confidence=1.0),
        ChartFact(name="WBC", value="11.8", category=FactCategory.LAB,
                  source="Labs", timestamp=BASE_TIME + timedelta(hours=24),
                  status=FactStatus.ACTIVE, confidence=1.0),
        # Vitals
        ChartFact(name="Temperature", value="38.5", category=FactCategory.VITAL,
                  source="Vitals", timestamp=BASE_TIME, status=FactStatus.ACTIVE, confidence=1.0),
        ChartFact(name="Temperature", value="37.2", category=FactCategory.VITAL,
                  source="Vitals", timestamp=BASE_TIME + timedelta(hours=24),
                  status=FactStatus.ACTIVE, confidence=1.0),
        # Medications
        ChartFact(name="Ceftriaxone", value="1g IV daily",
                  category=FactCategory.MEDICATION, source="Orders",
                  status=FactStatus.ACTIVE, confidence=1.0),
        ChartFact(name="Metformin", value="500mg BID",
                  category=FactCategory.MEDICATION, source="Home meds",
                  status=FactStatus.ACTIVE, confidence=1.0),
    ]
    return PatientChart(
        facts=facts,
        raw_text="65F with DM2 admitted for CAP. WBC trending down 14.2->11.8. Afebrile today. On ceftriaxone.",
        parse_warnings=[],
    )


@pytest.fixture
def handoff_context(mock_provider, tmp_path) -> WorkflowContext:
    # Queue LLM response for IPASS generation
    mock_provider.queue_response(
        ProviderResponse(
            content=(
                "I: Stable, improving\n"
                "P: 65F with DM2 admitted for CAP, WBC trending down\n"
                "A: Continue ceftriaxone, check AM labs, hold metformin if NPO\n"
                "S: If temp spikes or O2 requirement increases, consider broadening coverage\n"
                "S: Day team aware, no pending consults"
            ),
            stop_reason="end_turn",
        )
    )
    chart = _build_handoff_chart()
    audit = AuditLogger(log_dir=str(tmp_path))
    audit.start_session()
    return WorkflowContext(
        chart=chart, provider=mock_provider, audit=audit, raw_text=chart.raw_text,
    )


class TestHandoffDraft:
    def test_workflow_metadata(self):
        assert HandoffDraft.name == "handoff_draft"

    def test_steps_include_analysis(self):
        workflow = HandoffDraft()
        step_names = [s.name for s in workflow.steps()]
        assert "identify_active_problems" in step_names
        assert "analyze_trends" in step_names
        assert "generate_ipass" in step_names

    @pytest.mark.asyncio
    async def test_full_execution_produces_ipass(self, handoff_context):
        engine = WorkflowEngine()
        engine.register(HandoffDraft)
        draft = await engine.run("handoff_draft", handoff_context)

        assert draft.workflow_name == "handoff_draft"
        assert draft.is_draft is True
        headings = [s.heading for s in draft.sections]
        assert "IPASS Handoff" in headings

    @pytest.mark.asyncio
    async def test_trends_detected(self, handoff_context):
        engine = WorkflowEngine()
        engine.register(HandoffDraft)
        await engine.run("handoff_draft", handoff_context)

        trend_result = handoff_context.step_results.get("analyze_trends")
        assert trend_result is not None
        assert trend_result.status == StepStatus.COMPLETED
        # WBC trending down should be detected
        trends = trend_result.output.get("trends", [])
        assert len(trends) > 0

    @pytest.mark.asyncio
    async def test_active_problems_extracted(self, handoff_context):
        engine = WorkflowEngine()
        engine.register(HandoffDraft)
        await engine.run("handoff_draft", handoff_context)

        problems_result = handoff_context.step_results.get("identify_active_problems")
        assert problems_result is not None
        problems = problems_result.output.get("problems", [])
        assert len(problems) >= 1  # At least Pneumonia
