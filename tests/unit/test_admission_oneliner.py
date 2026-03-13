"""Tests for the admission one-liner workflow."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fangbot.chart.models import ChartFact, FactCategory, FactStatus, PatientChart
from fangbot.memory.audit import AuditLogger
from fangbot.models import ProviderResponse
from fangbot.workflows.admission_oneliner import AdmissionOneLiner
from fangbot.workflows.engine import WorkflowContext, WorkflowEngine
from fangbot.workflows.models import StepStatus

BASE_TIME = datetime(2026, 3, 7, 8, 0, tzinfo=timezone.utc)


def _build_sample_chart() -> PatientChart:
    """Build a synthetic H&P chart for testing."""
    facts = [
        ChartFact(
            name="Age", value="72", category=FactCategory.VITAL,
            source="Demographics", source_location="Header",
            status=FactStatus.ACTIVE, confidence=1.0,
        ),
        ChartFact(
            name="Sex", value="Male", category=FactCategory.VITAL,
            source="Demographics", source_location="Header",
            status=FactStatus.ACTIVE, confidence=1.0,
        ),
        ChartFact(
            name="Chief Complaint", value="Shortness of breath",
            category=FactCategory.DIAGNOSIS, source="H&P note",
            source_location="Chief complaint section",
            status=FactStatus.ACTIVE, confidence=0.95,
        ),
        ChartFact(
            name="CHF", value="Congestive heart failure",
            category=FactCategory.DIAGNOSIS, source="PMH",
            source_location="Past medical history",
            status=FactStatus.ACTIVE, confidence=0.9,
        ),
        ChartFact(
            name="Hypertension", value="Hypertension",
            category=FactCategory.DIAGNOSIS, source="PMH",
            source_location="Past medical history",
            status=FactStatus.ACTIVE, confidence=0.9,
        ),
        ChartFact(
            name="BNP", value="1200 pg/mL", category=FactCategory.LAB,
            source="Labs", source_location="Lab results",
            timestamp=BASE_TIME, status=FactStatus.ACTIVE, confidence=1.0,
        ),
        ChartFact(
            name="SpO2", value="91%", category=FactCategory.VITAL,
            source="Vitals", source_location="Triage vitals",
            timestamp=BASE_TIME, status=FactStatus.ACTIVE, confidence=1.0,
        ),
        ChartFact(
            name="Furosemide", value="40mg IV", category=FactCategory.MEDICATION,
            source="Orders", source_location="Admission orders",
            status=FactStatus.ACTIVE, confidence=1.0,
        ),
    ]
    return PatientChart(
        facts=facts,
        raw_text="72yo M with PMH CHF, HTN presenting with SOB. BNP 1200. SpO2 91%. Started furosemide 40mg IV.",
        parse_warnings=[],
    )


@pytest.fixture
def sample_chart() -> PatientChart:
    return _build_sample_chart()


@pytest.fixture
def oneliner_context(mock_provider, sample_chart, tmp_path) -> WorkflowContext:
    # Queue a mock LLM response for the generation step
    mock_provider.queue_response(
        ProviderResponse(
            content="72M with PMH of CHF, HTN presenting with shortness of breath, "
            "found to have BNP 1200 pg/mL, SpO2 91%, started on furosemide 40mg IV.",
            stop_reason="end_turn",
        )
    )
    audit = AuditLogger(log_dir=str(tmp_path))
    audit.start_session()
    return WorkflowContext(
        chart=sample_chart,
        provider=mock_provider,
        audit=audit,
        raw_text=sample_chart.raw_text,
    )


class TestAdmissionOneLiner:
    def test_workflow_metadata(self):
        assert AdmissionOneLiner.name == "admission_oneliner"
        assert AdmissionOneLiner.description

    def test_steps_count(self):
        workflow = AdmissionOneLiner()
        assert len(workflow.steps()) == 4

    @pytest.mark.asyncio
    async def test_full_execution(self, oneliner_context):
        engine = WorkflowEngine()
        engine.register(AdmissionOneLiner)
        draft = await engine.run("admission_oneliner", oneliner_context)

        assert draft.workflow_name == "admission_oneliner"
        assert draft.is_draft is True
        assert len(draft.sections) >= 1
        # The one-liner section should exist
        headings = [s.heading for s in draft.sections]
        assert "One-Liner" in headings
        # Provenance should be populated
        oneliner_section = next(s for s in draft.sections if s.heading == "One-Liner")
        assert len(oneliner_section.provenance) > 0

    @pytest.mark.asyncio
    async def test_extraction_steps_populate_context(self, oneliner_context):
        engine = WorkflowEngine()
        engine.register(AdmissionOneLiner)
        await engine.run("admission_oneliner", oneliner_context)

        # Extraction steps should have completed
        assert "extract_demographics" in oneliner_context.step_results
        assert "extract_chief_complaint" in oneliner_context.step_results
        assert "extract_key_findings" in oneliner_context.step_results
        for name in ["extract_demographics", "extract_chief_complaint", "extract_key_findings"]:
            assert oneliner_context.step_results[name].status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_empty_chart_still_produces_draft(self, mock_provider, tmp_path):
        """Workflow should degrade gracefully with minimal data."""
        mock_provider.queue_response(
            ProviderResponse(
                content="Patient with limited available data.",
                stop_reason="end_turn",
            )
        )
        empty_chart = PatientChart(facts=[], raw_text="No data available", parse_warnings=[])
        audit = AuditLogger(log_dir=str(tmp_path))
        audit.start_session()
        context = WorkflowContext(
            chart=empty_chart, provider=mock_provider, audit=audit, raw_text="No data available"
        )
        engine = WorkflowEngine()
        engine.register(AdmissionOneLiner)
        draft = await engine.run("admission_oneliner", context)
        assert draft.is_draft is True
        assert draft.overall_confidence < 1.0  # Should reflect missing data
