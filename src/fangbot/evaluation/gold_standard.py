"""Load and validate gold standard evaluation cases from YAML files."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from fangbot.evaluation.models import GoldStandardCase, StudyConfig

logger = logging.getLogger(__name__)


def load_study_config(path: Path) -> StudyConfig:
    """Load a study configuration from a YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Study config not found: {path}")

    raw = path.read_text()
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path.name}: {exc}") from exc

    return StudyConfig(**data)


def load_cases(cases_dir: Path) -> list[GoldStandardCase]:
    """Load all gold standard cases from a directory of YAML files.

    Returns cases sorted by case_id.
    """
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"Cases directory not found: {cases_dir}")

    yaml_files = sorted(cases_dir.glob("*.yaml"))
    if not yaml_files:
        raise ValueError(f"No .yaml case files found in {cases_dir}")

    cases: list[GoldStandardCase] = []
    for yaml_file in yaml_files:
        raw = yaml_file.read_text()
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {yaml_file.name}: {exc}") from exc

        try:
            case = GoldStandardCase(**data)
        except (ValidationError, TypeError) as exc:
            raise ValueError(f"Validation error in {yaml_file.name}: {exc}") from exc

        cases.append(case)

    cases.sort(key=lambda c: c.case_id)
    return cases
