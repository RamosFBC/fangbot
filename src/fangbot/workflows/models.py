"""Data models for the clinical workflow engine."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class StepType(str, Enum):
    EXTRACT = "extract"
    ANALYZE = "analyze"
    GENERATE = "generate"
    VALIDATE = "validate"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepResult(BaseModel):
    """Result of a single workflow step execution."""

    step_name: str
    step_type: StepType
    status: StepStatus
    output: dict[str, Any]
    provenance: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    duration_ms: int
    error: str | None = None


class DraftSection(BaseModel):
    """A single section of a workflow draft output."""

    heading: str
    content: str
    provenance: list[str]
    confidence: float
    editable: bool = True

    @model_validator(mode="after")
    def _force_editable(self) -> DraftSection:
        self.editable = True
        return self


class WorkflowDraft(BaseModel):
    """Complete workflow output — always a draft, never a hard commit."""

    workflow_name: str
    sections: list[DraftSection]
    metadata: dict[str, Any] = Field(default_factory=dict)
    step_results: list[StepResult]
    overall_confidence: float
    warnings: list[str]
    is_draft: bool = True

    @model_validator(mode="after")
    def _force_draft(self) -> WorkflowDraft:
        self.is_draft = True
        return self


class WorkflowDefinition(BaseModel):
    """Metadata describing a workflow for discovery/tool definitions."""

    name: str
    description: str
    input_description: str
    output_description: str
