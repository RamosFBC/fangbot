"""Load and validate encounter-level gold standard cases from YAML."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from fangbot.evaluation.encounter_models import EncounterGoldStandard, EncounterStudyConfig

logger = logging.getLogger(__name__)


def load_encounter_config(path: Path) -> EncounterStudyConfig:
    """Load an encounter study configuration from YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Study config not found: {path}")
    data = yaml.safe_load(path.read_text())
    return EncounterStudyConfig(**data)


def load_encounter_cases(cases_dir: Path) -> list[EncounterGoldStandard]:
    """Load encounter gold standard cases from a directory of YAML files."""
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"Cases directory not found: {cases_dir}")

    yaml_files = sorted(cases_dir.glob("*.yaml"))
    if not yaml_files:
        raise ValueError(f"No .yaml case files found in {cases_dir}")

    cases: list[EncounterGoldStandard] = []
    for yaml_file in yaml_files:
        try:
            data = yaml.safe_load(yaml_file.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {yaml_file.name}: {exc}") from exc

        try:
            case = EncounterGoldStandard(**data)
        except (ValidationError, TypeError) as exc:
            raise ValueError(f"Validation error in {yaml_file.name}: {exc}") from exc

        cases.append(case)

    cases.sort(key=lambda c: c.case_id)
    return cases
