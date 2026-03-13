"""Tests for workflow data models."""

from fangbot.workflows.models import (
    DraftSection,
    StepResult,
    StepStatus,
    StepType,
    WorkflowDefinition,
    WorkflowDraft,
)


class TestStepResult:
    def test_minimal_construction(self):
        result = StepResult(
            step_name="extract_demographics",
            step_type=StepType.EXTRACT,
            status=StepStatus.COMPLETED,
            output={"age": 72, "sex": "male"},
            duration_ms=15,
        )
        assert result.step_name == "extract_demographics"
        assert result.confidence == 1.0  # default
        assert result.provenance == []   # default
        assert result.error is None

    def test_failed_step(self):
        result = StepResult(
            step_name="analyze_trends",
            step_type=StepType.ANALYZE,
            status=StepStatus.FAILED,
            output={},
            duration_ms=5,
            error="No timestamped facts available",
        )
        assert result.status == StepStatus.FAILED
        assert result.error is not None


class TestDraftSection:
    def test_construction(self):
        section = DraftSection(
            heading="Assessment",
            content="72M with CHF, HTN presenting with dyspnea.",
            provenance=["H&P note, chief complaint section"],
            confidence=0.9,
        )
        assert section.editable is True  # always True

    def test_editable_always_true(self):
        section = DraftSection(
            heading="Test",
            content="test",
            provenance=[],
            confidence=1.0,
            editable=False,  # attempt to override
        )
        # Validator forces it True
        assert section.editable is True


class TestWorkflowDraft:
    def test_is_draft_always_true(self):
        draft = WorkflowDraft(
            workflow_name="admission_oneliner",
            sections=[],
            step_results=[],
            overall_confidence=0.0,
            warnings=[],
            is_draft=False,  # attempt to override
        )
        assert draft.is_draft is True

    def test_metadata_default(self):
        draft = WorkflowDraft(
            workflow_name="test",
            sections=[],
            step_results=[],
            overall_confidence=1.0,
            warnings=[],
        )
        assert draft.metadata == {}

    def test_serialization_roundtrip(self):
        section = DraftSection(
            heading="Summary",
            content="Patient is stable.",
            provenance=["Progress note"],
            confidence=0.95,
        )
        draft = WorkflowDraft(
            workflow_name="test",
            sections=[section],
            step_results=[],
            overall_confidence=0.95,
            warnings=["Missing overnight labs"],
        )
        json_str = draft.model_dump_json()
        restored = WorkflowDraft.model_validate_json(json_str)
        assert restored.workflow_name == "test"
        assert len(restored.sections) == 1
        assert restored.is_draft is True


class TestWorkflowDefinition:
    def test_construction(self):
        defn = WorkflowDefinition(
            name="pre_round_summary",
            description="Generate a problem-based pre-round summary",
            input_description="Patient chart with labs, vitals, medications",
            output_description="Problem-based summary with latest values and trends",
        )
        assert defn.name == "pre_round_summary"
