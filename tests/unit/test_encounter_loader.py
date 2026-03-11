"""Tests for encounter gold standard case loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from fangbot.evaluation.encounter_loader import load_encounter_cases, load_encounter_config
from fangbot.evaluation.encounter_models import EncounterGoldStandard, EncounterStudyConfig


@pytest.fixture
def study_dir(tmp_path: Path) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        "study_name: Test Study\n"
        "encounter_type: initial_consultation\n"
        "cases_dir: cases\n"
        "results_dir: results\n"
        "evaluation_tier: encounter\n"
    )
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    case_file = cases_dir / "test_case.yaml"
    case_file.write_text(
        "case_id: test_001\n"
        "encounter_type: initial_consultation\n"
        "narrative: Patient presents with symptoms.\n"
        "expected_decisions:\n"
        "  - category: medication\n"
        "    decision: Start treatment\n"
        "skill_loaded: initial_consultation\n"
    )
    return tmp_path


class TestEncounterLoader:
    def test_load_config(self, study_dir: Path) -> None:
        config = load_encounter_config(study_dir / "config.yaml")
        assert isinstance(config, EncounterStudyConfig)
        assert config.study_name == "Test Study"
        assert config.evaluation_tier == "encounter"

    def test_load_cases(self, study_dir: Path) -> None:
        cases = load_encounter_cases(study_dir / "cases")
        assert len(cases) == 1
        assert isinstance(cases[0], EncounterGoldStandard)
        assert cases[0].case_id == "test_001"

    def test_load_cases_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_encounter_cases(tmp_path / "nonexistent")

    def test_load_cases_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No .yaml"):
            load_encounter_cases(empty)
