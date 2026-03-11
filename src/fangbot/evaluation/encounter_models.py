"""Data models for encounter-level evaluation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DecisionCategory(str, Enum):
    """Categories of clinical decisions."""

    MEDICATION = "medication"
    DISPOSITION = "disposition"
    REFERRAL = "referral"
    FOLLOW_UP = "follow_up"
    DIAGNOSTIC = "diagnostic"
    PROCEDURE = "procedure"


class ExpectedToolInvocation(BaseModel):
    """An expected tool invocation in the encounter evaluation."""

    tool: str
    reason: str = ""


class ExpectedDecision(BaseModel):
    """A clinical decision expected in the gold standard."""

    category: DecisionCategory
    decision: str
    acceptable: list[str] = Field(default_factory=list)
    contraindicated: list[str] = Field(default_factory=list)
    reasoning: str = ""


class EncounterGoldStandard(BaseModel):
    """Gold standard case for encounter-level evaluation."""

    case_id: str
    encounter_type: str
    narrative: str
    expected_tool_invocations: list[ExpectedToolInvocation] = Field(default_factory=list)
    expected_reasoning: list[str] = Field(default_factory=list)
    expected_decisions: list[ExpectedDecision]
    required_elements: list[str] = Field(default_factory=list)
    forbidden_elements: list[str] = Field(default_factory=list)
    skill_loaded: str
    notes: str | None = None

    @field_validator("narrative")
    @classmethod
    def narrative_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("narrative must not be empty")
        return v


class EncounterCaseResult(BaseModel):
    """Result of running one encounter case through the agent."""

    case_id: str
    provider: str
    model: str
    actual_tool_calls: list[str] = Field(default_factory=list)
    actual_decisions: list[dict[str, Any]] = Field(default_factory=list)
    synthesis: str = ""
    chain_of_thought: list[str] = Field(default_factory=list)
    guardrail_passed: bool = False
    iterations: int = 0
    skill_loaded: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    audit_session_id: str = ""


class EncounterStudyConfig(BaseModel):
    """Configuration for an encounter-level evaluation study."""

    study_name: str
    encounter_type: str
    description: str = ""
    cases_dir: str
    results_dir: str
    providers: list[str] = Field(default_factory=lambda: ["claude"])
    models: dict[str, str] = Field(default_factory=dict)
    max_iterations: int = 15
    temperature: float = 0.0
    evaluation_tier: str = "encounter"
