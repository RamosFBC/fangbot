"""Workflow engine — executes composable clinical workflow pipelines."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from fangbot.brain.providers.base import LLMProvider
from fangbot.chart.models import PatientChart
from fangbot.memory.audit import AuditLogger, EventType
from fangbot.models import ToolDefinition
from fangbot.workflows.models import (
    StepResult,
    StepStatus,
    StepType,
    WorkflowDefinition,
    WorkflowDraft,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkflowContext:
    """Mutable context passed through workflow steps, accumulating results."""

    chart: PatientChart
    provider: LLMProvider
    audit: AuditLogger
    raw_text: str
    step_results: dict[str, StepResult] = field(default_factory=dict)


class WorkflowStep(ABC):
    """Base class for a single workflow step."""

    name: str
    step_type: StepType

    @abstractmethod
    async def execute(self, context: WorkflowContext) -> StepResult: ...


class BaseWorkflow(ABC):
    """Base class for all clinical workflows."""

    name: str
    description: str
    input_description: str
    output_description: str

    @abstractmethod
    def steps(self) -> list[WorkflowStep]: ...

    @abstractmethod
    def build_draft(self, context: WorkflowContext) -> WorkflowDraft: ...


class WorkflowEngine:
    """Executes workflows, manages registry, provides tool definitions."""

    def __init__(self) -> None:
        self._registry: dict[str, type[BaseWorkflow]] = {}

    def register(self, workflow_cls: type[BaseWorkflow]) -> None:
        self._registry[workflow_cls.name] = workflow_cls

    def list_workflows(self) -> list[WorkflowDefinition]:
        return [
            WorkflowDefinition(
                name=cls.name,
                description=cls.description,
                input_description=cls.input_description,
                output_description=cls.output_description,
            )
            for cls in self._registry.values()
        ]

    async def run(self, name: str, context: WorkflowContext) -> WorkflowDraft:
        if name not in self._registry:
            raise KeyError(f"Unknown workflow: {name}")

        workflow = self._registry[name]()

        context.audit.log(
            EventType.WORKFLOW_STARTED,
            {"workflow": name, "steps": [s.name for s in workflow.steps()]},
        )

        for step in workflow.steps():
            start = time.monotonic()
            try:
                result = await step.execute(context)
                result.duration_ms = int((time.monotonic() - start) * 1000)
                context.step_results[step.name] = result
            except Exception as e:
                duration = int((time.monotonic() - start) * 1000)
                result = StepResult(
                    step_name=step.name,
                    step_type=step.step_type,
                    status=StepStatus.FAILED,
                    output={},
                    duration_ms=duration,
                    error=str(e),
                )
                context.step_results[step.name] = result
                logger.warning(f"Workflow step '{step.name}' failed: {e}")

            context.audit.log(
                EventType.WORKFLOW_STEP_COMPLETED,
                {
                    "workflow": name,
                    "step": step.name,
                    "status": result.status.value,
                    "duration_ms": result.duration_ms,
                },
            )

        draft = workflow.build_draft(context)

        context.audit.log(
            EventType.WORKFLOW_COMPLETED,
            {
                "workflow": name,
                "sections": len(draft.sections),
                "overall_confidence": draft.overall_confidence,
                "warnings": draft.warnings,
            },
        )

        return draft

    def get_tool_definition(self) -> ToolDefinition:
        workflow_names = list(self._registry.keys())
        descriptions = "\n".join(
            f"- {w.name}: {w.description}" for w in self.list_workflows()
        )
        return ToolDefinition(
            name="run_workflow",
            description=(
                "Execute a clinical workflow to generate a structured draft. "
                f"Available workflows:\n{descriptions}"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "workflow_name": {
                        "type": "string",
                        "enum": workflow_names,
                        "description": "The workflow to execute",
                    },
                    "clinical_text": {
                        "type": "string",
                        "description": "The clinical text / chart data to process",
                    },
                },
                "required": ["workflow_name", "clinical_text"],
            },
        )
