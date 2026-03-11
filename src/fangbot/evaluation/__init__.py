"""Batch evaluation framework for clinical agent benchmarking."""

from fangbot.evaluation.batch_runner import BatchRunner
from fangbot.evaluation.encounter_loader import load_encounter_cases, load_encounter_config
from fangbot.evaluation.encounter_metrics import compute_encounter_metrics
from fangbot.evaluation.encounter_models import (
    DecisionCategory,
    EncounterCaseResult,
    EncounterGoldStandard,
    EncounterStudyConfig,
    ExpectedDecision,
    ExpectedToolInvocation,
)
from fangbot.evaluation.gold_standard import load_cases, load_study_config
from fangbot.evaluation.metrics import compute_all_metrics
from fangbot.evaluation.models import (
    CaseResult,
    ExpectedToolCall,
    GoldStandardCase,
    RiskTier,
    StudyConfig,
    StudyResults,
)
from fangbot.evaluation.report import generate_report

__all__ = [
    # Calculator-level (Tier 1)
    "BatchRunner",
    "CaseResult",
    "ExpectedToolCall",
    "GoldStandardCase",
    "RiskTier",
    "StudyConfig",
    "StudyResults",
    "compute_all_metrics",
    "generate_report",
    "load_cases",
    "load_study_config",
    # Encounter-level (Tier 2)
    "DecisionCategory",
    "EncounterCaseResult",
    "EncounterGoldStandard",
    "EncounterStudyConfig",
    "ExpectedDecision",
    "ExpectedToolInvocation",
    "compute_encounter_metrics",
    "load_encounter_cases",
    "load_encounter_config",
]
