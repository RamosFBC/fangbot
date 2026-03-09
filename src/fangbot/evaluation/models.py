"""Data models for the batch evaluation framework."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RiskTier(str, Enum):
    """Risk stratification tiers for clinical scores."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class ExpectedToolCall(BaseModel):
    """An expected MCP tool call in the gold standard."""

    tool_name: str
    required_arguments: dict[str, Any] = Field(default_factory=dict)


class GoldStandardCase(BaseModel):
    """A single gold standard evaluation case."""

    case_id: str
    narrative: str
    expected_score: int | float
    expected_risk_tier: RiskTier
    expected_tool_calls: list[ExpectedToolCall]
    expected_variables: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None

    @field_validator("narrative")
    @classmethod
    def narrative_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("narrative must not be empty")
        return v

    @field_validator("expected_tool_calls")
    @classmethod
    def at_least_one_tool_call(cls, v: list[ExpectedToolCall]) -> list[ExpectedToolCall]:
        if len(v) == 0:
            raise ValueError("expected_tool_calls must contain at least one entry")
        return v


class StudyConfig(BaseModel):
    """Configuration for a batch evaluation study."""

    study_name: str
    calculator_name: str
    description: str = ""
    cases_dir: str
    results_dir: str
    providers: list[str] = Field(default_factory=lambda: ["claude"])
    models: dict[str, str] = Field(default_factory=dict)
    max_iterations: int = 10
    temperature: float = 0.0


class CaseResult(BaseModel):
    """Result of running one case through the agent."""

    case_id: str
    provider: str
    model: str
    actual_score: int | float | None = None
    actual_risk_tier: RiskTier | None = None
    actual_tool_calls: list[str] = Field(default_factory=list)
    synthesis: str = ""
    chain_of_thought: list[str] = Field(default_factory=list)
    guardrail_passed: bool = False
    iterations: int = 0
    error: str | None = None
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    audit_session_id: str = ""


class StudyResults(BaseModel):
    """Aggregated results for an entire study run."""

    study_name: str
    provider: str
    model: str
    cases: list[CaseResult] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
