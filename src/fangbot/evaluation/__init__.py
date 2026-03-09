"""Batch evaluation framework for clinical agent benchmarking."""

from fangbot.evaluation.batch_runner import BatchRunner
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
]
