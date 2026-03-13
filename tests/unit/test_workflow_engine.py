"""Tests for the workflow engine — registration, execution, error handling."""

from __future__ import annotations

from typing import Any

import pytest

from fangbot.chart.models import ChartFact, FactCategory, PatientChart
from fangbot.memory.audit import AuditLogger
from fangbot.workflows.engine import (
    BaseWorkflow,
    WorkflowContext,
    WorkflowEngine,
    WorkflowStep,
)
from fangbot.workflows.models import (
    DraftSection,
    StepResult,
    StepStatus,
    StepType,
    WorkflowDraft,
)



# -- Trivial test workflow --

class _EchoStep(WorkflowStep):
    name = "echo"
    step_type = StepType.EXTRACT

    async def execute(self, context: WorkflowContext) -> StepResult:
        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            status=StepStatus.COMPLETED,
            output={"echoed": context.raw_text},
            provenance=["test input"],
            confidence=1.0,
            duration_ms=1,
        )


class _FailingStep(WorkflowStep):
    name = "failing"
    step_type = StepType.ANALYZE

    async def execute(self, context: WorkflowContext) -> StepResult:
        raise ValueError("Something broke")


class _EchoWorkflow(BaseWorkflow):
    name = "echo_test"
    description = "A trivial echo workflow for testing"
    input_description = "Any text"
    output_description = "Echoed text"

    def steps(self) -> list[WorkflowStep]:
        return [_EchoStep()]

    def build_draft(self, context: WorkflowContext) -> WorkflowDraft:
        echo_output = context.step_results["echo"].output
        return WorkflowDraft(
            workflow_name=self.name,
            sections=[
                DraftSection(
                    heading="Echo",
                    content=echo_output["echoed"],
                    provenance=["test"],
                    confidence=1.0,
                )
            ],
            step_results=list(context.step_results.values()),
            overall_confidence=1.0,
            warnings=[],
        )


class _FailWorkflow(BaseWorkflow):
    name = "fail_test"
    description = "Workflow with a failing step"
    input_description = "Any"
    output_description = "Nothing"

    def steps(self) -> list[WorkflowStep]:
        return [_FailingStep(), _EchoStep()]

    def build_draft(self, context: WorkflowContext) -> WorkflowDraft:
        results = list(context.step_results.values())
        return WorkflowDraft(
            workflow_name=self.name,
            sections=[],
            step_results=results,
            overall_confidence=0.0,
            warnings=["Step 'failing' failed"],
        )


@pytest.fixture
def engine() -> WorkflowEngine:
    eng = WorkflowEngine()
    eng.register(_EchoWorkflow)
    return eng


@pytest.fixture
def context(mock_provider, tmp_path) -> WorkflowContext:
    chart = PatientChart(facts=[], raw_text="test chart", parse_warnings=[])
    audit = AuditLogger(log_dir=str(tmp_path))
    audit.start_session()
    return WorkflowContext(
        chart=chart,
        provider=mock_provider,
        audit=audit,
        raw_text="Hello from test",
    )


class TestWorkflowEngine:
    def test_register_and_list(self, engine):
        workflows = engine.list_workflows()
        assert len(workflows) == 1
        assert workflows[0].name == "echo_test"

    @pytest.mark.asyncio
    async def test_run_echo_workflow(self, engine, context):
        draft = await engine.run("echo_test", context)
        assert draft.workflow_name == "echo_test"
        assert draft.is_draft is True
        assert len(draft.sections) == 1
        assert draft.sections[0].content == "Hello from test"
        assert len(draft.step_results) == 1
        assert draft.step_results[0].status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_unknown_workflow_raises(self, engine, context):
        with pytest.raises(KeyError, match="no_such_workflow"):
            await engine.run("no_such_workflow", context)

    @pytest.mark.asyncio
    async def test_failing_step_produces_failed_status(self, context):
        engine = WorkflowEngine()
        engine.register(_FailWorkflow)
        draft = await engine.run("fail_test", context)
        # Failing step should have FAILED status
        fail_result = context.step_results["failing"]
        assert fail_result.status == StepStatus.FAILED
        assert "Something broke" in fail_result.error
        # Echo step should still run (graceful degradation)
        echo_result = context.step_results["echo"]
        assert echo_result.status == StepStatus.COMPLETED

    def test_get_tool_definition(self, engine):
        tool_def = engine.get_tool_definition()
        assert tool_def.name == "run_workflow"
        assert "echo_test" in str(tool_def.input_schema)

    @pytest.mark.asyncio
    async def test_audit_events_logged(self, engine, context):
        await engine.run("echo_test", context)
        events = context.audit.get_events()
        event_types = [e.event_type.value for e in events]
        assert "workflow_started" in event_types
        assert "workflow_step_completed" in event_types
        assert "workflow_completed" in event_types
